"""Compare the 29 MuJoCo G1 body joints against Isaac Lab USD."""

from pathlib import Path
from pxr import Usd, UsdPhysics

asset = Path(
    "/workspace/local/isaac_lab/"
    "isaaclab_usd_141627363c44411ebba4c246fbe47b4f/"
    "1/g1.usd"
)

if not asset.is_file():
    raise FileNotFoundError(f"USD asset not found: {asset}")

stage = Usd.Stage.Open(str(asset))

if stage is None:
    raise RuntimeError("Could not open the USD asset")

usd_joints = {
    prim.GetName()
    for prim in stage.Traverse()
    if prim.IsA(UsdPhysics.Joint)
    and not prim.IsA(UsdPhysics.FixedJoint)
}

mujoco_joints = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll",
    "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll",
    "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch",
    "right_wrist_yaw",
]

missing = []

print("=== MUJOCO TO ISAAC LAB JOINT MAPPING ===")

for index, name in enumerate(mujoco_joints):
    usd_name = f"{name}_joint"
    found = usd_name in usd_joints

    status = "OK" if found else "MISSING"

    print(f"{index:02d} | {name:25s} | {status}")

    if not found:
        missing.append(usd_name)

hand_joints = [
    name for name in usd_joints
    if "_hand_" in name
]

print()
print("=== SUMMARY ===")
print("MuJoCo body joints:", len(mujoco_joints))
print("Matching body joints:", len(mujoco_joints) - len(missing))
print("Additional hand joints:", len(hand_joints))
print("Total USD joints:", len(usd_joints))
print("Missing joints:", missing)

if missing:
    print("FAIL: Some body joints do not match.")
    raise SystemExit(1)

print("PASS: All 29 body-joint names match.")
