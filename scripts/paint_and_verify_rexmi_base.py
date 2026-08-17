#!/usr/bin/env python3
"""
Paint rexmi_dog base visual red and verify USD structure.

Writes a plain-text report to:
  assets/robots/visual_reskin/VERIFY_REPORT.txt
"""

from isaaclab.app import AppLauncher
import argparse
import os
import sys
import traceback

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

REPORT = "/home/susan/rexmi_rl/assets/robots/visual_reskin/VERIFY_REPORT.txt"
USD_ROOT = "/home/susan/rexmi_rl/assets/robots/rexmi_dog/urdf/go2w/go2w.usd"
USD_BASE = "/home/susan/rexmi_rl/assets/robots/rexmi_dog/urdf/go2w/configuration/go2w_base.usd"
STL = "/home/susan/rexmi_rl/assets/robots/rexmi_dog/meshes/base.stl"

lines = []


def log(msg=""):
    lines.append(str(msg))
    print(msg, flush=True)


try:
    from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf
    import numpy as np

    log("pxr import OK")

    # ------------------------------------------------------------------
    # Paint base meshes red on the visual base layer
    # ------------------------------------------------------------------
    stage_b = Usd.Stage.Open(USD_BASE)
    if stage_b is None:
        raise RuntimeError(f"Could not open {USD_BASE}")

    # Ensure Looks scope
    looks_path = Sdf.Path("/rexmi_dog_description/Looks")
    # Find default prim root
    dp = stage_b.GetDefaultPrim()
    root_path = dp.GetPath() if dp else Sdf.Path("/rexmi_dog_description")
    looks_path = root_path.AppendChild("Looks")
    if not stage_b.GetPrimAtPath(looks_path):
        stage_b.DefinePrim(looks_path, "Scope")

    mat_path = looks_path.AppendChild("rexmi_base_red")
    mat = UsdShade.Material.Define(stage_b, mat_path)
    shader_path = mat_path.AppendChild("Shader")
    shader = UsdShade.Shader.Define(stage_b, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")
    # Bright product red
    red = Gf.Vec3f(0.85, 0.05, 0.05)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(red)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.05)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    painted = []
    for prim in stage_b.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        path = str(prim.GetPath())
        # Paint meshes under the base link (visuals), not every mesh named incidentally
        if "/base/" in path or path.endswith("/base") or "/base_" in path:
            # Prefer visual meshes
            if "collision" in path.lower():
                continue
            mesh = UsdGeom.Mesh(prim)
            mesh.GetDisplayColorAttr().Set([red])
            UsdShade.MaterialBindingAPI.Apply(prim)
            UsdShade.MaterialBindingAPI(prim).Bind(mat)
            painted.append(path)

    # If nothing matched /base/, fall back to largest mesh (trunk)
    if not painted:
        best = None
        best_n = -1
        for prim in stage_b.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            path = str(prim.GetPath())
            if "collision" in path.lower():
                continue
            pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            n = 0 if pts is None else len(pts)
            if n > best_n:
                best_n = n
                best = prim
        if best is not None:
            mesh = UsdGeom.Mesh(best)
            mesh.GetDisplayColorAttr().Set([red])
            UsdShade.MaterialBindingAPI.Apply(best)
            UsdShade.MaterialBindingAPI(best).Bind(mat)
            painted.append(str(best.GetPath()) + f" (largest n={best_n})")

    stage_b.GetRootLayer().Save()
    log(f"painted_meshes: {painted}")

    # ------------------------------------------------------------------
    # Verify composed root USD
    # ------------------------------------------------------------------
    stage = Usd.Stage.Open(USD_ROOT)
    log(f"defaultPrim: {stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else None}")
    log(f"subLayers: {list(stage.GetRootLayer().subLayerPaths)}")

    need = [
        "base",
        "FL_hip", "FL_thigh", "FL_calf", "FL_foot",
        "FR_hip", "FR_thigh", "FR_calf", "FR_foot",
        "RL_hip", "RL_thigh", "RL_calf", "RL_foot",
        "RR_hip", "RR_thigh", "RR_calf", "RR_foot",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_foot_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_foot_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RL_foot_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RR_foot_joint",
    ]
    found = {k: None for k in need}
    base_mesh_info = []
    for prim in stage.Traverse():
        name = prim.GetName()
        path = str(prim.GetPath())
        if name in found:
            found[name] = path
        if prim.IsA(UsdGeom.Mesh) and ("/base/" in path or path.endswith("/base") or "base_" in path.lower()):
            pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            n = 0 if pts is None else len(pts)
            dc = UsdGeom.Mesh(prim).GetDisplayColorAttr().Get()
            if pts is not None and n:
                a = np.array([[p[0], p[1], p[2]] for p in pts], float)
                base_mesh_info.append(
                    f"{path} n={n} ext={np.round(a.max(0)-a.min(0),4).tolist()} "
                    f"min={np.round(a.min(0),4).tolist()} max={np.round(a.max(0),4).tolist()} dc={dc}"
                )

    missing = [k for k, v in found.items() if v is None]
    log(f"missing: {missing if missing else 'NONE'}")
    log(f"joints_found: {sum(1 for k,v in found.items() if k.endswith('_joint') and v)}/16")
    log("base_meshes:")
    for m in base_mesh_info:
        log(f"  {m}")

    # STL compare without trimesh if needed
    try:
        import trimesh
        stl = trimesh.load(STL, force="mesh")
        log(
            f"stl faces={len(stl.faces)} verts={len(stl.vertices)} "
            f"ext={np.round(stl.extents,4).tolist()} bounds={np.round(stl.bounds,4).tolist()}"
        )
    except Exception as e:
        log(f"stl_load_error: {e}")

    # Confirm red material still present after save/reopen
    stage_b2 = Usd.Stage.Open(USD_BASE)
    mat_prim = stage_b2.GetPrimAtPath(mat_path)
    log(f"red_material_exists: {bool(mat_prim and mat_prim.IsValid())} path={mat_path}")
    if mat_prim:
        for p in Usd.PrimRange(mat_prim):
            for attr in p.GetAttributes():
                if "diffuse" in attr.GetName().lower():
                    log(f"  {p.GetPath()} {attr.GetName()}={attr.Get()}")

    log("STATUS: OK")

except Exception:
    log("STATUS: FAIL")
    log(traceback.format_exc())
finally:
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"WROTE {REPORT}", flush=True)
    app.close()
