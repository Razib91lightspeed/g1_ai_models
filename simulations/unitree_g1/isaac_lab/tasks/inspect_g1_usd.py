"""Read-only inspection of Isaac Lab's G1 29-DoF USD asset."""

from pathlib import Path

from pxr import Usd, UsdPhysics

from isaaclab_assets import G1_29DOF_CFG
from isaaclab.utils.assets import retrieve_file_path

url = G1_29DOF_CFG.spawn.usd_path

print("=== ISAAC LAB G1 USD ===")
print("Source:", url)

path = retrieve_file_path(url)

print("Local asset:", path)

stage = Usd.Stage.Open(str(path))
if stage is None:
    raise RuntimeError("Unable to open the USD stage")

joints = []

for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.Joint):
        if prim.IsA(UsdPhysics.FixedJoint):
            continue

        joints.append((
            prim.GetName(),
            prim.GetPath().pathString,
            prim.GetTypeName(),
        ))

print("\n=== JOINTS FOUND ===")

for index, (name, path, joint_type) in enumerate(joints):
    print(f"{index:02d} | {name} | {joint_type} | {path}")

print("\nTotal non-fixed joint prims:", len(joints))
print(
    "Note: USD joint-prims are not necessarily equal to "
    "the final number of actuated degrees of freedom."
)
