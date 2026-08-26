# REXMI Visual Reskin — Mesh Manifest & Naming Convention

This folder is the handoff between **SolidWorks visual redesign** and the
Isaac Lab robot asset. Physics, joint names, actuators, and policy configs stay
on the existing Go2W kinematics. Only render meshes change.

## Folders

| Path | Purpose |
|------|---------|
| `solidworks_in/` | **Reference geometry for you to import into SolidWorks** (read-only source) |
| `solidworks_out/` | **Drop your redesigned meshes here** with exact names below |
| `docs/` | Workflow notes / export checklist |
| `../rexmi_dog/` | Runtime robot asset copy (USD stack). Visual layer gets updated after drop-off |
| `../go2w/` | Original Unitree asset — leave untouched |

## Required drop-off filenames (`solidworks_out/`)

Use these **exact** names. Prefer **binary STL**, units **meters**, frame **Z-up**.

| Filename | Link(s) that use it | Side / notes |
|----------|---------------------|--------------|
| `base.stl` | `base` | Trunk / body. Remove Unitree badge/logo geometry |
| `hip.stl` | `FL_hip`, `FR_hip`, `RL_hip`, `RR_hip` | One mesh; URDF applies per-leg visual RPY |
| `thigh.stl` | `FL_thigh`, `RL_thigh` | Left thighs |
| `thigh_mirror.stl` | `FR_thigh`, `RR_thigh` | Right thighs (mirror of left) |
| `calf.stl` | `FL_calf`, `RL_calf` | Left calves |
| `calf_mirror.stl` | `FR_calf`, `RR_calf` | Right calves |
| `left_wheel.stl` | `FL_foot`, `RL_foot` | Left wheels |
| `right_wheel.stl` | `FR_foot`, `RR_foot` | Right wheels |

### Optional

| Filename | Notes |
|----------|-------|
| `foot.stl` | Present in original mesh pack but **not** referenced by current URDF visuals. Safe to omit |

## Accepted formats

1. **Preferred:** `*.stl` (binary), meters  
2. Also OK: `*.obj`, `*.dae` with the **same basename** (e.g. `base.obj`)

If both `base.stl` and `base.obj` exist, **STL wins**.

## Coordinate / export rules (critical)

1. **Units = meters** (not mm). Wheel radius must stay ~**0.05 m**.
2. **Z-up**, same as current robot / Isaac.
3. **Origin = link frame origin**, not the mesh centroid.
   - Do not “center about mass” on export.
   - Import the reference STL from `solidworks_in/` and build around that origin.
4. **Do not rename joints or links.** Visual files map to existing link names only.
5. Keep attachment points roughly aligned with the reference:
   - hip mounts on trunk
   - thigh / calf lengths so wheels still sit near ground in the default pose
6. Left/right pairs:
   - `thigh` / `thigh_mirror`
   - `calf` / `calf_mirror`
   - `left_wheel` / `right_wheel`
7. Visual-only: collision primitives and inertias stay on the frozen physics USD.

## What each reference file is for

Copy from `solidworks_in/` into SolidWorks as a reference body, then redesign.

| Reference STL | Role |
|---------------|------|
| `base.stl` | Trunk shell + former logo region |
| `hip.stl` | Abduction housing |
| `thigh.stl` / `thigh_mirror.stl` | Upper leg armor |
| `calf.stl` / `calf_mirror.stl` | Lower leg |
| `left_wheel.stl` / `right_wheel.stl` | Wheel + tire visual (keep ~5 cm radius) |
| `foot.stl` | Unused by URDF today; reference only |
| `_original_formats/` | Raw Unitree DAE/STL copies if you need materials/history |

## URDF visual offsets (do not bake these into the mesh)

Some links apply an extra visual RPY in the URDF. Your mesh should match the
**reference STL orientation**, not pre-compensate unless you know you must.

| Link | Visual RPY (approx) | Mesh file |
|------|---------------------|-----------|
| `base` | `0 0 0` | `base` |
| `FL_hip` | `0 0 0` | `hip` |
| `FR_hip` | `π 0 0` | `hip` |
| `RL_hip` | `0 π 0` | `hip` |
| `RR_hip` | `π π 0` | `hip` |
| thighs / calves / feet | `0 0 0` | as table above |

Because hip reuses one mesh with URDF rotations, design `hip.stl` like the
reference — Isaac/URDF handles mirroring via RPY.

## After you drop files

1. Put all **8 required** meshes in:
   ```text
   assets/robots/visual_reskin/solidworks_out/
   ```
2. Tell the agent (or run):
   ```bash
   ./run.sh scripts/apply_visual_reskin.py
   # or, if already in the Isaac env:
   python scripts/apply_visual_reskin.py
   ```
3. Visual default is already `rexmi_dog` (see `source/rexmi_rl/assets/go2w.py`).
   No env var needed for the reskin. To force original Unitree look:
   ```bash
   export REXMI_ROBOT_VISUAL=go2w
   ```
   or set `REXMI_ROBOT_VISUAL=go2w` in `.env`.
4. Play any existing checkpoint — behavior should match; only look changes.

## Explicit non-goals

- Do **not** edit joint names (`FL_hip_joint`, …).
- Do **not** change collision geometry in CAD for this pass.
- Do **not** overwrite `assets/robots/go2w/` originals.
- Do **not** expect swapping files under `go2w/meshes/` alone to change the sim
  (runtime visuals are in the USD `*_base.usd` layer).

## Status flags

| File | Meaning |
|------|---------|
| `solidworks_out/DROP_FILES_HERE.txt` | Folder is waiting for your meshes |
| `solidworks_out/READY` | Optional marker you can add when upload is complete |
| `STATUS.md` | Machine-readable checklist of which outs are present |
