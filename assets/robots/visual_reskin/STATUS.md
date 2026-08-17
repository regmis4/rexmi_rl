# Visual Reskin Status

Updated: 2026-08-17 02:20:00Z

## solidworks_out inventory

| Part | Required | Present | Path |
|------|----------|---------|------|
| `base` | yes | ✅ | `assets/robots/visual_reskin/solidworks_out/base.stl` |
| `hip` | yes | ✅ | `assets/robots/visual_reskin/solidworks_out/hip.stl` |
| `thigh` | yes | ✅ | `assets/robots/visual_reskin/solidworks_out/thigh.stl` |
| `thigh_mirror` | yes | ✅ | `assets/robots/visual_reskin/solidworks_out/thigh_mirror.stl` |
| `calf` | yes | ✅ | `assets/robots/visual_reskin/solidworks_out/calf.stl` |
| `calf_mirror` | yes | ✅ | `assets/robots/visual_reskin/solidworks_out/calf_mirror.stl` |
| `left_wheel` | yes | ✅ | `assets/robots/visual_reskin/solidworks_out/left_wheel.stl` |
| `right_wheel` | yes | ✅ | `assets/robots/visual_reskin/solidworks_out/right_wheel.stl` |
| `foot` | no | ✅ | `assets/robots/visual_reskin/solidworks_out/foot.stl` |

## Readiness

**Ready for apply.** Run:

```bash
./run.sh scripts/apply_visual_reskin.py
```

Then enable:
```bash
export REXMI_ROBOT_VISUAL=rexmi_dog
```

## Runtime switch

- `REXMI_ROBOT_VISUAL=go2w` → original Unitree visuals (default)
- `REXMI_ROBOT_VISUAL=rexmi_dog` → reskinned asset

See `MANIFEST.md` for naming rules.

