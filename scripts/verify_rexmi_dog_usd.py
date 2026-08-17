#!/usr/bin/env python3
"""Headless USD verification for the rexmi_dog visual reskin."""

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdShade
import numpy as np
import trimesh

USD = "/home/susan/rexmi_rl/assets/robots/rexmi_dog/urdf/go2w/go2w.usd"
BASE = "/home/susan/rexmi_rl/assets/robots/rexmi_dog/urdf/go2w/configuration/go2w_base.usd"
STL = "/home/susan/rexmi_rl/assets/robots/rexmi_dog/meshes/base.stl"

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


def main():
    stage = Usd.Stage.Open(USD)
    print("defaultPrim", stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else None)
    print("subLayers", list(stage.GetRootLayer().subLayerPaths))

    found = {k: None for k in need}
    base_meshes = []
    color_hits = []

    for prim in stage.Traverse():
        name = prim.GetName()
        path = str(prim.GetPath())
        if name in found:
            found[name] = path

        if prim.IsA(UsdGeom.Mesh) and "base" in path.lower():
            mesh = UsdGeom.Mesh(prim)
            pts = mesh.GetPointsAttr().Get()
            n = 0 if pts is None else len(pts)
            dc = mesh.GetDisplayColorAttr().Get() if mesh.GetDisplayColorAttr() else None
            if pts is not None and n:
                a = np.array([[p[0], p[1], p[2]] for p in pts], float)
                base_meshes.append(
                    {
                        "path": path,
                        "n": n,
                        "min": a.min(0).round(4).tolist(),
                        "max": a.max(0).round(4).tolist(),
                        "ext": (a.max(0) - a.min(0)).round(4).tolist(),
                        "displayColor": str(dc)[:100] if dc is not None else None,
                    }
                )

        # materials / shaders
        for attr in prim.GetAttributes():
            an = attr.GetName()
            if any(k in an.lower() for k in ["diffuse", "displaycolor", "albedo", "tint"]):
                v = attr.Get()
                if v is None:
                    continue
                if any(k in path.lower() for k in ["base", "rexmi", "red", "looks"]):
                    color_hits.append((path, prim.GetTypeName(), an, str(v)[:120]))

        # binding
        if UsdShade.MaterialBindingAPI(prim):
            mat, rel = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if mat and "base" in path.lower():
                color_hits.append((path, "BoundMaterial", str(mat.GetPath()), ""))

    missing = [k for k, v in found.items() if v is None]
    print("missing", missing if missing else "NONE")
    print("joints_found", sum(1 for k, v in found.items() if k.endswith("_joint") and v))
    print("base_meshes:")
    for m in base_meshes:
        print(" ", m)
    print("color_hits:")
    for c in color_hits[:40]:
        print(" ", c)

    stl = trimesh.load(STL, force="mesh")
    print(
        "stl",
        "faces", len(stl.faces),
        "verts", len(stl.vertices),
        "ext", np.round(stl.extents, 4).tolist(),
        "bounds", np.round(stl.bounds, 4).tolist(),
    )

    # Open base layer alone too
    sb = Usd.Stage.Open(BASE)
    print("base_layer_default", sb.GetDefaultPrim().GetPath() if sb.GetDefaultPrim() else None)
    for prim in sb.Traverse():
        path = str(prim.GetPath())
        if "rexmi" in path.lower() or "red" in path.lower():
            print("named_prim", path, prim.GetTypeName())
        for attr in prim.GetAttributes():
            an = attr.GetName()
            if "diffuse" in an.lower() or an.endswith("displayColor"):
                v = attr.Get()
                if v is not None and ("base" in path.lower() or "rexmi" in path.lower() or "Looks" in path):
                    print("base_layer_color", path, an, str(v)[:120])


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
