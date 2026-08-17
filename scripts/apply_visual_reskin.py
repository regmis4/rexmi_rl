#!/usr/bin/env python3
# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause
"""
Stage SolidWorks visual meshes into the rexmi_dog asset and (optionally) rebuild
the visual USD layer.

Workflow
--------
1. Redesign parts in SolidWorks using:
     assets/robots/visual_reskin/solidworks_in/*.stl
2. Export redesigned meshes with the exact names in MANIFEST.md into:
     assets/robots/visual_reskin/solidworks_out/
3. Run:
     ./run.sh scripts/apply_visual_reskin.py
   or (status only, no writes):
     ./run.sh scripts/apply_visual_reskin.py --status
4. Enable the reskin at runtime:
     export REXMI_ROBOT_VISUAL=rexmi_dog
   (or add that line to .env)

What this script changes
------------------------
* Copies drop-off meshes → assets/robots/rexmi_dog/meshes/
* Rewrites ONLY <visual> mesh filenames in rexmi_dog's URDF
* Leaves <collision> and <inertial> blocks pointing at the original Go2W
  geometry / values (physics-safe)
* Freezes original physics/robot/sensor USD layers
* Optionally runs Isaac Lab URDF→USD conversion and keeps the new visual
  base layer while restoring frozen physics/robot/sensor layers

What it does NOT change
-----------------------
* Joint names, joint limits, actuator cfg, init pose
* assets/robots/go2w/ (original Unitree asset stays untouched)
* Policy checkpoints / env cfgs
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
RESKIN_ROOT = REPO_ROOT / "assets" / "robots" / "visual_reskin"
SW_IN = RESKIN_ROOT / "solidworks_in"
SW_OUT = RESKIN_ROOT / "solidworks_out"
STATUS_MD = RESKIN_ROOT / "STATUS.md"

GO2W_ROOT = REPO_ROOT / "assets" / "robots" / "go2w"
REXMI_ROOT = REPO_ROOT / "assets" / "robots" / "rexmi_dog"

REXMI_MESHES = REXMI_ROOT / "meshes"
REXMI_URDF = REXMI_ROOT / "urdf" / "go2w.urdf"
REXMI_USD_ROOT = REXMI_ROOT / "urdf" / "go2w"
REXMI_CFG_DIR = REXMI_USD_ROOT / "configuration"
FROZEN_DIR = REXMI_ROOT / "_frozen_original_usd"

# Canonical part names (no extension)
REQUIRED_PARTS = [
    "base",
    "hip",
    "thigh",
    "thigh_mirror",
    "calf",
    "calf_mirror",
    "left_wheel",
    "right_wheel",
]
OPTIONAL_PARTS = [
    "foot",
]

# Preferred extension order when resolving a drop-off file
EXT_PREFERENCE = (".stl", ".obj", ".dae")

# Map: URDF visual mesh basename (without ext) → part name in solidworks_out
# Original URDF uses these basenames under ../meshes/
URDF_VISUAL_BASENAME_TO_PART = {
    "base": "base",
    "hip": "hip",
    "thigh": "thigh",
    "thigh_mirror": "thigh_mirror",
    "calf": "calf",
    "calf_mirror": "calf_mirror",
    "left_wheel": "left_wheel",
    "right_wheel": "right_wheel",
    "foot": "foot",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def resolve_dropoff(part: str) -> Path | None:
    """Return the best matching file in solidworks_out for a part name."""
    for ext in EXT_PREFERENCE:
        candidate = SW_OUT / f"{part}{ext}"
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    # Allow nested accidental drops like solidworks_out/base/base.stl
    for ext in EXT_PREFERENCE:
        matches = list(SW_OUT.rglob(f"{part}{ext}"))
        matches = [m for m in matches if m.is_file() and m.stat().st_size > 0]
        if matches:
            # Prefer shallowest path
            matches.sort(key=lambda p: (len(p.parts), str(p)))
            return matches[0]
    return None


def inventory() -> dict[str, Path | None]:
    parts = REQUIRED_PARTS + OPTIONAL_PARTS
    return {part: resolve_dropoff(part) for part in parts}


def write_status(inv: dict[str, Path | None]) -> None:
    lines = [
        f"# Visual Reskin Status",
        f"",
        f"Updated: {_now()}",
        f"",
        f"## solidworks_out inventory",
        f"",
        f"| Part | Required | Present | Path |",
        f"|------|----------|---------|------|",
    ]
    missing_req = []
    for part in REQUIRED_PARTS + OPTIONAL_PARTS:
        req = part in REQUIRED_PARTS
        path = inv.get(part)
        present = path is not None
        if req and not present:
            missing_req.append(part)
        rel = path.relative_to(REPO_ROOT).as_posix() if path else "—"
        lines.append(
            f"| `{part}` | {'yes' if req else 'no'} | "
            f"{'✅' if present else '❌'} | `{rel}` |"
        )

    lines += [
        f"",
        f"## Readiness",
        f"",
    ]
    if missing_req:
        lines.append(
            f"**Not ready** — missing required parts: "
            + ", ".join(f"`{p}`" for p in missing_req)
        )
        lines.append(f"")
        lines.append(f"Drop files into `assets/robots/visual_reskin/solidworks_out/`.")
    else:
        lines.append(f"**Ready for apply.** Run:")
        lines.append(f"")
        lines.append(f"```bash")
        lines.append(f"./run.sh scripts/apply_visual_reskin.py")
        lines.append(f"```")
        lines.append(f"")
        lines.append(f"Then enable:")
        lines.append(f"```bash")
        lines.append(f"export REXMI_ROBOT_VISUAL=rexmi_dog")
        lines.append(f"```")

    lines += [
        f"",
        f"## Runtime switch",
        f"",
        f"- `REXMI_ROBOT_VISUAL=go2w` → original Unitree visuals (default)",
        f"- `REXMI_ROBOT_VISUAL=rexmi_dog` → reskinned asset",
        f"",
        f"See `MANIFEST.md` for naming rules.",
        f"",
    ]
    STATUS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[status] Wrote {STATUS_MD.relative_to(REPO_ROOT)}")


def ensure_rexmi_skeleton() -> None:
    """Make sure rexmi_dog exists with frozen physics layers from go2w."""
    REXMI_MESHES.mkdir(parents=True, exist_ok=True)
    REXMI_CFG_DIR.mkdir(parents=True, exist_ok=True)
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)

    # Seed URDF from go2w if missing
    go2w_urdf = GO2W_ROOT / "urdf" / "go2w.urdf"
    if not REXMI_URDF.exists():
        REXMI_URDF.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(go2w_urdf, REXMI_URDF)

    # Seed / refresh frozen USD layers from original go2w (physics-safe originals)
    src_cfg = GO2W_ROOT / "urdf" / "go2w" / "configuration"
    src_root_usd = GO2W_ROOT / "urdf" / "go2w" / "go2w.usd"
    for name in [
        "go2w_physics.usd",
        "go2w_robot.usd",
        "go2w_sensor.usd",
        "go2w_base.usd",
    ]:
        src = src_cfg / name
        dst_frozen = FROZEN_DIR / name
        if src.exists() and not dst_frozen.exists():
            shutil.copy2(src, dst_frozen)
        # Ensure live configuration has physics/robot/sensor (base may be replaced later)
        if name != "go2w_base.usd" and src.exists():
            live = REXMI_CFG_DIR / name
            if not live.exists():
                shutil.copy2(src, live)

    if src_root_usd.exists():
        live_root = REXMI_USD_ROOT / "go2w.usd"
        if not live_root.exists():
            shutil.copy2(src_root_usd, live_root)
        frozen_root = FROZEN_DIR / "go2w.usd"
        if not frozen_root.exists():
            shutil.copy2(src_root_usd, frozen_root)

    # Config yaml
    cfg_src = GO2W_ROOT / "config" / "joint_names_go2w_description.yaml"
    cfg_dst_dir = REXMI_ROOT / "config"
    cfg_dst_dir.mkdir(parents=True, exist_ok=True)
    if cfg_src.exists():
        shutil.copy2(cfg_src, cfg_dst_dir / cfg_src.name)


def stage_meshes(inv: dict[str, Path | None]) -> dict[str, str]:
    """
    Copy drop-off meshes into rexmi_dog/meshes.

    Returns map: part_name -> staged filename (e.g. 'base' -> 'base.stl')
    """
    REXMI_MESHES.mkdir(parents=True, exist_ok=True)
    staged: dict[str, str] = {}
    for part, src in inv.items():
        if src is None:
            continue
        # Keep original extension
        dst_name = f"{part}{src.suffix.lower()}"
        dst = REXMI_MESHES / dst_name
        shutil.copy2(src, dst)
        staged[part] = dst_name
        print(f"[mesh] {part:14s} <- {src.relative_to(REPO_ROOT)}  -> meshes/{dst_name}")
    return staged


def _replace_visual_mesh_filenames(urdf_text: str, staged: dict[str, str]) -> str:
    """
    Replace mesh filenames that appear inside <visual>...</visual> only.
    Collision mesh paths are left unchanged (still original Go2W files via
    absolute-ish relative path rewrite below).
    """

    def visual_repl(match: re.Match[str]) -> str:
        block = match.group(0)

        def mesh_repl(m: re.Match[str]) -> str:
            path = m.group(1)
            base = Path(path).name  # e.g. base.dae
            stem = Path(base).stem  # base
            part = URDF_VISUAL_BASENAME_TO_PART.get(stem)
            if part and part in staged:
                return f'filename="../meshes/{staged[part]}"'
            return m.group(0)

        return re.sub(r'filename="([^"]+)"', mesh_repl, block)

    return re.sub(r"<visual\b[\s\S]*?</visual>", visual_repl, urdf_text)


def _retarget_collision_meshes_to_go2w(urdf_text: str) -> str:
    """
    Point any remaining collision mesh filenames at the original go2w meshes
    so wheel collision geometry stays Unitree-original even after visual swap.
    """

    def collision_repl(match: re.Match[str]) -> str:
        block = match.group(0)

        def mesh_repl(m: re.Match[str]) -> str:
            path = m.group(1)
            name = Path(path).name  # keep original basename (e.g. left_wheel.dae)
            # From rexmi_dog/urdf/go2w.urdf → ../../go2w/meshes/<name>
            return f'filename="../../go2w/meshes/{name}"'

        return re.sub(r'filename="([^"]+)"', mesh_repl, block)

    return re.sub(r"<collision\b[\s\S]*?</collision>", collision_repl, urdf_text)


def write_urdf(staged: dict[str, str]) -> None:
    """Build rexmi_dog URDF from go2w URDF with visual-only mesh retargeting."""
    src_urdf = GO2W_ROOT / "urdf" / "go2w.urdf"
    text = src_urdf.read_text(encoding="utf-8")

    # Keep robot name as go2w_description so USD layer composition
    # (physics/robot/sensor/base) continues to resolve under the same prim path.
    # Do NOT rename to rexmi_dog_description — that breaks joint/physics layers.

    text = _replace_visual_mesh_filenames(text, staged)
    text = _retarget_collision_meshes_to_go2w(text)

    # Banner comment
    banner = (
        "<!-- REXMI visual reskin URDF.\n"
        "     Link/joint names, inertial, and collision targets are physics-safe.\n"
        "     Visual meshes come from assets/robots/rexmi_dog/meshes/.\n"
        f"     Generated by scripts/apply_visual_reskin.py at {_now()}.\n"
        "-->\n"
    )
    if text.lstrip().startswith("<?xml"):
        # insert after xml declaration
        parts = text.split("\n", 1)
        text = parts[0] + "\n" + banner + (parts[1] if len(parts) > 1 else "")
    else:
        text = banner + text

    REXMI_URDF.parent.mkdir(parents=True, exist_ok=True)
    REXMI_URDF.write_text(text, encoding="utf-8")
    print(f"[urdf] Wrote {REXMI_URDF.relative_to(REPO_ROOT)}")


def restore_frozen_physics_layers() -> None:
    """Copy frozen physics/robot/sensor (and root) USD back over live rexmi_dog."""
    for name in ["go2w_physics.usd", "go2w_robot.usd", "go2w_sensor.usd"]:
        src = FROZEN_DIR / name
        dst = REXMI_CFG_DIR / name
        if src.exists():
            shutil.copy2(src, dst)
            print(f"[usd] Restored frozen {name}")
    root_src = FROZEN_DIR / "go2w.usd"
    root_dst = REXMI_USD_ROOT / "go2w.usd"
    if root_src.exists():
        shutil.copy2(root_src, root_dst)
        print("[usd] Restored frozen go2w.usd root layer")


def restore_orientation_correct_usd_stack() -> None:
    """
    Copy the original Go2W USD stack into rexmi_dog.

    Critical: DAE-sourced parts (base/hip/thigh/wheels) need the original
    importer xforms (≈90° about X). A full URDF reimport from STL drops those
    and leaves hips/wheels/trunk sideways. Calves (native STL) looked fine,
    which is the tell.
    """
    src_root = GO2W_ROOT / "urdf" / "go2w"
    src_cfg = src_root / "configuration"
    REXMI_CFG_DIR.mkdir(parents=True, exist_ok=True)
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)

    for name in [
        "go2w_base.usd",
        "go2w_physics.usd",
        "go2w_robot.usd",
        "go2w_sensor.usd",
    ]:
        src = src_cfg / name
        if src.exists():
            shutil.copy2(src, REXMI_CFG_DIR / name)
            # keep frozen backup fresh for physics/robot/sensor
            if name != "go2w_base.usd":
                shutil.copy2(src, FROZEN_DIR / name)
            elif not (FROZEN_DIR / name).exists():
                shutil.copy2(src, FROZEN_DIR / name)
            print(f"[usd] Seeded {name} from original go2w (orientation-correct)")

    src_usd = src_root / "go2w.usd"
    if src_usd.exists():
        shutil.copy2(src_usd, REXMI_USD_ROOT / "go2w.usd")
        shutil.copy2(src_usd, FROZEN_DIR / "go2w.usd")
        print("[usd] Seeded go2w.usd root from original go2w")


def swap_base_mesh_geometry(base_stl: Path, color_rgb=(0.85, 0.05, 0.05)) -> bool:
    """
    Replace ONLY the trunk visual mesh points/faces inside go2w_base.usd,
    keeping original xform ops (DAE orientation) and all other part meshes.
    Requires Isaac / pxr (run via ./run.sh).
    """
    try:
        from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, Vt  # type: ignore
        import numpy as np
        import trimesh
    except Exception as exc:
        print(f"[usd] swap_base_mesh_geometry needs Isaac pxr/trimesh: {exc}")
        return False

    if not base_stl.is_file():
        print(f"[usd] missing base mesh: {base_stl}")
        return False

    restore_orientation_correct_usd_stack()
    base_usd = REXMI_CFG_DIR / "go2w_base.usd"
    stage = Usd.Stage.Open(str(base_usd))
    if stage is None:
        print(f"[usd] could not open {base_usd}")
        return False

    mesh = trimesh.load(base_stl, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    points = Vt.Vec3fArray([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in verts])
    face_counts = Vt.IntArray([3] * len(faces))
    face_indices = Vt.IntArray(faces.reshape(-1).tolist())
    normals = None
    if mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(verts):
        normals = Vt.Vec3fArray(
            [Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in mesh.vertex_normals]
        )

    targets = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        path = str(prim.GetPath())
        if path.endswith("/visuals/base/base/mesh") or path.endswith("/meshes/base/mesh"):
            targets.append(prim)
    if not targets:
        print("[usd] no base mesh prims found in go2w_base.usd")
        return False

    # Red material
    r, g, b = color_rgb
    red = Gf.Vec3f(float(r), float(g), float(b))
    dp = stage.GetDefaultPrim()
    root = dp.GetPath() if dp else Sdf.Path("/go2w_description")
    looks = root.AppendChild("Looks")
    if not stage.GetPrimAtPath(looks):
        stage.DefinePrim(looks, "Scope")
    mat_path = looks.AppendChild("rexmi_base_red")
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path.AppendChild("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(red)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.05)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    # Stale GeomSubset children (Unitree multi-material face groups) must go —
    # otherwise old logo/body materials still paint parts of the new mesh and
    # it looks like two bases stacked.
    subset_paths = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if prim.GetTypeName() != "GeomSubset":
            continue
        if "/meshes/base/mesh/" in path or "/visuals/base/base/mesh/" in path:
            subset_paths.append(prim.GetPath())
    for sp in subset_paths:
        if stage.RemovePrim(sp):
            print(f"[usd] removed stale subset {sp}")
        else:
            prim = stage.GetPrimAtPath(sp)
            if prim and prim.IsValid():
                prim.SetActive(False)
                print(f"[usd] deactivated stale subset {sp}")

    for prim in targets:
        m = UsdGeom.Mesh(prim)
        m.GetPointsAttr().Set(points)
        m.GetFaceVertexCountsAttr().Set(face_counts)
        m.GetFaceVertexIndicesAttr().Set(face_indices)
        if normals is not None:
            m.GetNormalsAttr().Set(normals)
            m.SetNormalsInterpolation("vertex")
        for attr_name in ("primvars:st", "primvars:UVMap"):
            attr = prim.GetAttribute(attr_name)
            if attr and attr.IsValid():
                attr.Clear()
        # drop any leftover subset-family attrs
        for attr in list(prim.GetAttributes()):
            if "subset" in attr.GetName().lower():
                try:
                    attr.Clear()
                except Exception:
                    pass
        m.GetDisplayColorAttr().Set([red])
        api = UsdShade.MaterialBindingAPI.Apply(prim)
        api.UnbindAllBindings()
        api.Bind(mat)
        print(f"[usd] swapped geometry+red on {prim.GetPath()} (kept original xform)")


    stage.GetRootLayer().Save()
    print(f"[usd] wrote {base_usd.relative_to(REPO_ROOT)}")
    return True


def try_convert_usd(headless: bool = True) -> bool:
    """
    DEPRECATED for reskin: full STL URDF reimport breaks DAE orientations.

    Prefer swap_base_mesh_geometry(). Kept for emergency/manual use with
    --full-reimport.
    """

    convert_script = Path(
        os.environ.get("ISAACLAB_DIR", str(Path.home() / "IsaacLab"))
    ) / "scripts" / "tools" / "convert_urdf.py"

    if not convert_script.is_file():
        print(
            f"[usd] SKIP convert — Isaac Lab convert_urdf.py not found at {convert_script}"
        )
        print(
            "[usd] Meshes + URDF are staged. Open the URDF in Isaac Sim URDF Importer,"
        )
        print(
            "      export USD, then copy the visual base layer to:"
        )
        print(f"      {REXMI_CFG_DIR / 'go2w_base.usd'}")
        restore_frozen_physics_layers()
        return False

    out_usd = REXMI_USD_ROOT / "go2w.usd"
    # Convert into a temp sibling dir, then pick up base layer
    tmp_dir = REXMI_ROOT / "_usd_convert_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    tmp_usd = tmp_dir / "go2w.usd"

    cmd = [
        sys.executable,
        str(convert_script),
        str(REXMI_URDF),
        str(tmp_usd),
        "--joint-target-type",
        "none",
    ]
    if headless:
        cmd.append("--headless")

    print("[usd] Running URDF→USD conversion:")
    print("     ", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"[usd] Conversion failed: {exc}")
        print("[usd] Keeping frozen USD layers; meshes/URDF are still staged.")
        restore_frozen_physics_layers()
        return False

    # Find produced base layer
    candidates = list(tmp_dir.rglob("go2w_base.usd")) + list(tmp_dir.rglob("*_base.usd"))
    if not candidates:
        # Sometimes importer writes a single monolithic usd
        monolithic = list(tmp_dir.rglob("*.usd"))
        print(f"[usd] No go2w_base.usd in convert output; files={monolithic}")
        print("[usd] Copy manually if needed. Restoring frozen physics layers.")
        restore_frozen_physics_layers()
        return False

    new_base = candidates[0]
    dst_base = REXMI_CFG_DIR / "go2w_base.usd"
    # Backup previous live base
    if dst_base.exists():
        bak = REXMI_CFG_DIR / f"go2w_base.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}.usd"
        shutil.copy2(dst_base, bak)
        print(f"[usd] Backed up previous base → {bak.name}")
    shutil.copy2(new_base, dst_base)
    print(f"[usd] Installed new visual base layer from {new_base}")

    # Always re-assert frozen physics/robot/sensor/root so importer drift cannot stick
    restore_frozen_physics_layers()

    # Cleanup tmp
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"[usd] Active robot USD: {out_usd}")
    return True


def print_summary(inv: dict[str, Path | None], converted: bool | None) -> None:
    print()
    print("=" * 60)
    print("Visual reskin apply summary")
    print("=" * 60)
    missing = [p for p in REQUIRED_PARTS if inv.get(p) is None]
    if missing:
        print("Missing required:", ", ".join(missing))
    else:
        print("All required meshes present.")
    print(f"Asset root: {REXMI_ROOT.relative_to(REPO_ROOT)}")
    print(f"URDF:       {REXMI_URDF.relative_to(REPO_ROOT)}")
    print(f"USD:        {(REXMI_USD_ROOT / 'go2w.usd').relative_to(REPO_ROOT)}")
    if converted is True:
        print("USD visual layer: updated")
    elif converted is False:
        print("USD visual layer: not updated (use Isaac importer or re-run with convert)")
    print()
    print("Enable reskin:")
    print("  export REXMI_ROBOT_VISUAL=rexmi_dog")
    print("  # or add to .env")
    print("Back to Unitree look:")
    print("  export REXMI_ROBOT_VISUAL=go2w")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--status",
        action="store_true",
        help="Only refresh STATUS.md from solidworks_out (no file staging).",
    )
    p.add_argument(
        "--skip-usd",
        action="store_true",
        help="Stage meshes + URDF only; do not touch USD.",
    )
    p.add_argument(
        "--full-reimport",
        action="store_true",
        help=(
            "DANGEROUS: full URDF→USD reimport from STL. Breaks DAE orientations "
            "(hips/wheels/base go sideways). Default path is base-mesh geometry swap."
        ),
    )
    p.add_argument(
        "--base-color",
        type=str,
        default="0.85,0.05,0.05",
        help="RGB for base visual, comma-separated 0-1 floats (default red).",
    )
    p.add_argument(
        "--no-headless",
        action="store_true",
        help="Allow GUI during full-reimport USD conversion (default is headless).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not SW_OUT.is_dir():
        print(f"ERROR: drop-off folder missing: {SW_OUT}", file=sys.stderr)
        return 2

    inv = inventory()
    write_status(inv)

    if args.status:
        missing = [p for p in REQUIRED_PARTS if inv[p] is None]
        return 1 if missing else 0

    missing = [p for p in REQUIRED_PARTS if inv[p] is None]
    if missing:
        print("ERROR: missing required drop-off meshes:", ", ".join(missing), file=sys.stderr)
        print(f"Put them in {SW_OUT}", file=sys.stderr)
        return 1

    ensure_rexmi_skeleton()
    staged = stage_meshes(inv)
    write_urdf(staged)

    # Optional red (or custom) base color
    try:
        color = tuple(float(x.strip()) for x in args.base_color.split(","))
        if len(color) != 3:
            raise ValueError
    except ValueError:
        print(f"ERROR: bad --base-color {args.base_color!r}; use r,g,b", file=sys.stderr)
        return 2

    converted: bool | None
    if args.skip_usd:
        print("[usd] --skip-usd set; leaving USD layers as-is.")
        converted = None
    elif args.full_reimport:
        print("[usd] WARNING: --full-reimport can break hip/wheel/base orientations.")
        converted = try_convert_usd(headless=not args.no_headless)
    else:
        # Safe default: keep original USD xforms, swap base mesh geometry only.
        base_file = REXMI_MESHES / staged["base"]
        print("[usd] Safe path: restore orientation-correct USD + swap base mesh only.")
        ok = swap_base_mesh_geometry(base_file, color_rgb=color)
        converted = ok

    # Refresh status after apply
    write_status(inventory())
    print_summary(inv, converted)
    return 0 if converted is not False else 1



if __name__ == "__main__":
    raise SystemExit(main())
