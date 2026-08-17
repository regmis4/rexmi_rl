# SolidWorks export checklist (REXMI visual reskin)

## 1. Import reference

1. Open SolidWorks.
2. Import each STL from:
   ```text
   assets/robots/visual_reskin/solidworks_in/
   ```
3. Treat these as **reference bodies** (lock/fix them).
4. Build new body panels **in the same coordinate system**.

## 2. Design rules

| Rule | Detail |
|------|--------|
| Units | **Meters** (or export scaled to meters). Not mm. |
| Up axis | **Z-up** |
| Origin | **Link frame origin** from the reference STL — do not recenter |
| Wheel radius | Keep ~**0.05 m** visual radius |
| Joint pivots | Do not move hip/thigh/calf attachment points |
| Naming | Exact names in `../MANIFEST.md` |
| Physics | Visual only — no need to model internal structure for collision |

## 3. Parts to deliver

Required exports → `assets/robots/visual_reskin/solidworks_out/`:

```text
base.stl
hip.stl
thigh.stl
thigh_mirror.stl
calf.stl
calf_mirror.stl
left_wheel.stl
right_wheel.stl
```

Optional: `foot.stl`

## 4. Mirrors

- `thigh_mirror` / `calf_mirror` / `right_wheel` should be true mirrors of the left-side parts, matching the reference pair envelopes.
- `hip.stl` is **one** mesh reused on all four hips. URDF applies:
  - FL: RPY `0 0 0`
  - FR: RPY `π 0 0`
  - RL: RPY `0 π 0`
  - RR: RPY `π π 0`  
  So design hip like the reference; do not pre-mirror four separate hips unless you also change the URDF (we will not).

## 5. Export settings (SolidWorks)

Recommended:
- File type: **STL (Binary)**
- Unit system: **Meters** (or output in mm and scale 0.001 — prefer native meters)
- Resolution: fine enough for demo video, not ultra-pathological density
- Do **not** “move to positive space” / recentering macros
- One body per file

## 6. Sanity checks before drop-off

Compare your export bounding boxes roughly to reference (`solidworks_in`):

| Part | Approx extents (m) from reference |
|------|-------------------------------------|
| base | ~0.46 × 0.19 × 0.19 |
| hip | ~0.12 × 0.10 × 0.08 |
| thigh / thigh_mirror | ~0.10 × 0.28 × 0.07 |
| calf / calf_mirror | ~0.10 × 0.05 × 0.31 |
| left/right_wheel | ~0.17 × 0.17 × 0.05 (radius ~0.086) |

If your wheel diameter is 2× or 0.5× reference, fix scale before drop-off.

## 7. After drop-off

```bash
# Check inventory
./run.sh scripts/apply_visual_reskin.py --status

# Stage meshes + URDF (+ optional USD rebuild)
./run.sh scripts/apply_visual_reskin.py

# Use reskinned visuals (physics/policy unchanged)
export REXMI_ROBOT_VISUAL=rexmi_dog
# add the same line to .env to make it sticky
```

## 8. If USD conversion is skipped

Meshes + URDF are still valid. In Isaac Sim:

1. File → Import → URDF  
   `assets/robots/rexmi_dog/urdf/go2w.urdf`
2. Export USD.
3. Copy the produced `*_base.usd` visual layer over:
   `assets/robots/rexmi_dog/urdf/go2w/configuration/go2w_base.usd`
4. Keep `go2w_physics.usd` / `go2w_robot.usd` / `go2w_sensor.usd` from `_frozen_original_usd/`.
