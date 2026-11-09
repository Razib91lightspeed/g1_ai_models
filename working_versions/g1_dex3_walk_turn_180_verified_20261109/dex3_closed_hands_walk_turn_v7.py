from pathlib import Path
#!/usr/bin/env python3
"""Simulation-only Dex3 ONNX standing and finger closure V2."""

import json
import importlib.util
import onnxruntime as ort
import math
import sys

import mujoco
import numpy as np

sys.path.insert(
    0, "/workspace/obstacle_navigation/autonomous"
)
import live_demo_v8 as nav

MODEL = "/workspace/dex3/assets/scene_43dof.xml"
POSE = "/workspace/hand_reference/physical_dex3_angles.json"

model = mujoco.MjModel.from_xml_path(MODEL)
data = mujoco.MjData(model)

with open(POSE) as f:
    physical = json.load(f)["angles"]

assert model.nu == 43
assert len(physical) == 14

# Obtain the existing controller's standing reference.
nav.reset_sim()

body_act = []
body_qpos = []
body_qvel = []

for old_idx in nav.ids:
    old_joint = int(
        nav.model.actuator_trnid[int(old_idx), 0]
    )
    name = nav.model.joint(old_joint).name
    new_joint = model.joint(name).id

    matches = [
        i for i in range(model.nu)
        if int(model.actuator_trnid[i, 0]) == new_joint
    ]
    assert len(matches) == 1, name

    body_act.append(matches[0])
    body_qpos.append(int(model.jnt_qposadr[new_joint]))
    body_qvel.append(int(model.jnt_dofadr[new_joint]))

body_act = np.array(body_act, dtype=int)
body_qpos = np.array(body_qpos, dtype=int)
body_qvel = np.array(body_qvel, dtype=int)

hand_act = []
hand_qpos = []
hand_qvel = []
hand_goal = []

for name, degrees in physical.items():
    joint = model.joint(name).id

    matches = [
        i for i in range(model.nu)
        if int(model.actuator_trnid[i, 0]) == joint
    ]
    assert len(matches) == 1, name

    low, high = model.jnt_range[joint]
    goal = np.clip(
        math.radians(degrees),
        low + 0.02,
        high - 0.02
    )

    hand_act.append(matches[0])
    hand_qpos.append(int(model.jnt_qposadr[joint]))
    hand_qvel.append(int(model.jnt_dofadr[joint]))
    hand_goal.append(float(goal))

hand_act = np.array(hand_act, dtype=int)
hand_qpos = np.array(hand_qpos, dtype=int)
hand_qvel = np.array(hand_qvel, dtype=int)
hand_goal = np.array(hand_goal, dtype=float)

assert len(set(body_act) | set(hand_act)) == 43
assert not (set(body_act) & set(hand_act))

# Initialize the floating base and 29 body joints from
# the previously working MuJoCo standing configuration.
data.qpos[:7] = nav.data.qpos[:7]
data.qpos[body_qpos] = nav.data.qpos[nav.qids]
data.qvel[:] = 0.0

# The initial hand pose is neutral (zero) and the
# fingers subsequently move toward the recorded pose.
initial_hands = data.qpos[hand_qpos].copy()

mujoco.mj_forward(model, data)

pelvis = model.body("pelvis").id

def measurements():
    rotation = data.xmat[pelvis].reshape(3, 3)
    tilt = math.degrees(
        math.acos(
            float(np.clip(rotation[2, 2], -1.0, 1.0))
        )
    )
    height = float(data.qpos[2])
    hand_error = float(np.max(
        np.abs(data.qpos[hand_qpos] - hand_goal)
    ))
    return height, tilt, hand_error


OMNI = Path("/workspace/omni")
spec = importlib.util.spec_from_file_location(
    "dex3_omni_remap",
    OMNI / "deploy/joint_remap.py"
)
remap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remap)

omni_session = ort.InferenceSession(
    str(OMNI / "weights/g1_deploy_walk_policy.onnx"),
    providers=["CPUExecutionProvider"],
)
omni_scan = np.load(
    OMNI / "weights/g1_deploy_scan_mean.npy",
    allow_pickle=False
).astype(np.float32).reshape(-1)

assert omni_scan.shape == (143,)
assert omni_session.get_inputs()[0].shape[-1] == 242
assert omni_session.get_outputs()[0].shape[-1] == 29

actual_names = []
for actuator in body_act:
    jid = int(model.actuator_trnid[int(actuator), 0])
    actual_names.append(
        model.joint(jid).name.replace("_joint", "")
    )
assert actual_names == remap.SIM_ORDER_NAMES

def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))

def omni_policy(command, previous):
    mujoco.mj_forward(model, data)

    velocity = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model, data,
        mujoco.mjtObj.mjOBJ_BODY,
        pelvis, velocity, 1
    )

    rotation = data.xmat[pelvis].reshape(3, 3)
    gravity = rotation.T @ np.array([0., 0., -1.])

    obs = np.concatenate([
        velocity[3:],
        velocity[:3],
        gravity,
        command,
        data.qpos[body_qpos] - remap.DEFAULT_SIM,
        data.qvel[body_qvel],
        previous,
        omni_scan,
    ]).astype(np.float32)

    assert obs.shape == (242,)
    if not np.isfinite(obs).all():
        raise RuntimeError("Invalid omni observation")

    raw = omni_session.run(
        None, {"obs": obs.reshape(1, 242)}
    )[0][0].astype(np.float32)

    if (not np.isfinite(raw).all()
            or np.max(np.abs(raw)) > 3.0):
        raise RuntimeError("Omni action safety stop")

    targets = remap.DEFAULT_SIM + 0.5 * raw

    for i, actuator in enumerate(body_act):
        jid = int(model.actuator_trnid[int(actuator), 0])
        if model.jnt_limited[jid]:
            targets[i] = np.clip(
                targets[i],
                model.jnt_range[jid, 0],
                model.jnt_range[jid, 1],
            )

    return raw, targets

print("===== G1 DEX3 STANDING TEST =====", flush=True)
print("Body actuators:", len(body_act), flush=True)
print("Hand actuators:", len(hand_act), flush=True)
print("Control period: 0.002 s physics", flush=True)
print("Finger closing interval: 4 to 8 seconds", flush=True)
print("Walking starts at 12 seconds", flush=True)
print("Target forward displacement: 0.50 m", flush=True)
print("Next stage: omnidirectional 180-degree turn", flush=True)
print("Walking command: 0.40 m/s", flush=True)
print("Initial stop threshold: 0.45 m", flush=True)

dt = float(model.opt.timestep)
duration = 43.0
next_report = 0.0
result = "INCOMPLETE"

# Experimental hand PD gains, subject to stability testing.
hand_kp = 2.0
hand_kd = 0.12
hand_saturation_counts = np.zeros(len(hand_act), dtype=int)

walk_origin = None
walk_heading = None
walk_progress = 0.0
lateral_error = 0.0
walk_finished_at = None
target_distance = 0.50

turn_started_at = None
turn_last_yaw = None
turn_accumulated = 0.0
turn_settled_at = None
turn_previous_action = np.zeros(29, dtype=np.float32)
turn_initial_targets = None
turn_result = None


# Reuse the verified ONNX walker and its observation history.
# Only the simulation model and joint indexing change.
# The ONNX model, actions, history and gains are preserved.
nav.model = model
nav.data = data
nav.ids = body_act
nav.qids = body_qpos
nav.vids = body_qvel
nav.pelvis = pelvis

nav.command[:] = 0.0
nav.target = nav.defaults.copy()

print(
    "ONNX BODY CONTROLLER READY:",
    len(nav.ids), "body joints",
    flush=True
)

for step in range(int(duration / dt)):
    t = float(data.time)

    # Run the existing walking ONNX policy every 20 ms.
    # Its observations now come from the new Dex3 model.
    if t >= 12.0 and walk_origin is None:
        walk_origin = data.qpos[:2].copy()
        walk_heading = float(nav.yaw_angle())
        print(
            "WALK START: hands remain closed",
            flush=True
        )

    if walk_origin is not None:
        delta_xy = data.qpos[:2] - walk_origin
        walk_progress = float(
            delta_xy[0] * math.cos(walk_heading)
            + delta_xy[1] * math.sin(walk_heading)
        )
        lateral_error = float(
            -delta_xy[0] * math.sin(walk_heading)
            + delta_xy[1] * math.cos(walk_heading)
        )

    if walk_origin is None or walk_finished_at is not None:
        nav.command[:] = 0.0
    else:
        remaining = target_distance - walk_progress

        # Gradually reduce commanded speed near the target.
        # Constant walking command verified on the 29-joint G1.
        # Stop slightly before the goal to measure stopping drift.
        desired_vx = 0.40

        # No walking beyond the target threshold.
        if remaining <= 0.05:
            nav.command[:] = 0.0
            walk_finished_at = t
            print(
                f"WALK TARGET REACHED: "
                f"{walk_progress:.3f} m",
                flush=True
            )
        else:
            nav.command[:] = [
                desired_vx, 0.0, 0.0
            ]

    if step % nav.interval == 0:
        if (walk_finished_at is not None
                and t - walk_finished_at >= 1.5
                and turn_started_at is None):

            if not (0.40 <= walk_progress <= 0.65
                    and abs(lateral_error) <= 0.30):
                result = (
                    "FAIL: unsafe walking-to-turn transition; "
                    f"forward={walk_progress:.3f}m"
                )
                break

            turn_started_at = t
            turn_last_yaw = float(nav.yaw_angle())
            turn_initial_targets = nav.target.copy()

            print(
                "TURN START: walking stopped; "
                "Dex3 hands remain closed",
                flush=True
            )

        if turn_started_at is None:
            nav.update_policy()
        else:
            yaw = float(nav.yaw_angle())
            turn_accumulated += wrap(yaw - turn_last_yaw)
            turn_last_yaw = yaw

            if (turn_settled_at is None
                    and turn_accumulated >= math.radians(183)):
                turn_settled_at = t
                print(
                    "TURN THRESHOLD REACHED: "
                    f"{math.degrees(turn_accumulated):.2f}deg",
                    flush=True
                )

            if (turn_settled_at is None
                    and t - turn_started_at > 20.0):
                result = (
                    "FAIL: turning timeout; "
                    f"angle={math.degrees(turn_accumulated):.2f}deg"
                )
                break

            yaw_command = 0.60 if turn_settled_at is None else 0.0

            turn_previous_action, omni_target = omni_policy(
                np.array([0., 0., yaw_command], dtype=np.float32),
                turn_previous_action
            )

            blend = min(
                1.0, (t - turn_started_at) / 1.0
            )
            nav.target = (
                (1.0 - blend) * turn_initial_targets
                + blend * omni_target
            )

    body_torque = (
        nav.kp * (nav.target - data.qpos[body_qpos])
        - nav.kd * data.qvel[body_qvel]
    )

    # Close fingers gradually after initial stabilization.
    fraction = float(np.clip((t - 4.0) / 4.0, 0, 1))
    fraction = fraction * fraction * (3 - 2 * fraction)

    hand_target = (
        initial_hands
        + fraction * (hand_goal - initial_hands)
    )

    hand_torque = (
        hand_kp * (hand_target - data.qpos[hand_qpos])
        - hand_kd * data.qvel[hand_qvel]
    )

    data.ctrl[body_act] = np.clip(
        body_torque,
        model.actuator_ctrlrange[body_act, 0],
        model.actuator_ctrlrange[body_act, 1]
    )
    data.ctrl[hand_act] = np.clip(
        hand_torque,
        model.actuator_ctrlrange[hand_act, 0],
        model.actuator_ctrlrange[hand_act, 1]
    )

    hand_saturation_counts += (
        np.abs(hand_torque - data.ctrl[hand_act]) > 1e-6
    ).astype(int)

    if not np.isfinite(data.ctrl).all():
        result = "FAIL: nonfinite actuator command"
        break

    mujoco.mj_step(model, data)

    height, tilt, error = measurements()

    if walk_origin is not None:
        delta_xy = data.qpos[:2] - walk_origin

        walk_progress = float(
            delta_xy[0] * math.cos(walk_heading)
            + delta_xy[1] * math.sin(walk_heading)
        )

        lateral_error = float(
            -delta_xy[0] * math.sin(walk_heading)
            + delta_xy[1] * math.cos(walk_heading)
        )

        if (abs(lateral_error) > 0.40
                or walk_progress > 0.70
                or walk_progress < -0.15):
            result = (
                f"FAIL: walking boundary exceeded; "
                f"forward={walk_progress:.3f}m, "
                f"lateral={lateral_error:.3f}m"
            )
            break

        if (turn_settled_at is not None
                and t - turn_settled_at >= 0.8):
            final_turn = math.degrees(turn_accumulated)
            angle_error = 180.0 - final_turn

            print(
                f"SETTLED TURN: {final_turn:.2f}deg "
                f"error={angle_error:+.2f}deg",
                flush=True
            )

            if abs(angle_error) <= 3.0:
                result = (
                    "PASS: Dex3 0.5 m walk and 180-degree turn; "
                    f"settled_turn={final_turn:.2f}deg"
                )
            else:
                result = (
                    "FAIL: settled turn outside tolerance; "
                    f"angle={final_turn:.2f}deg"
                )
            break

    if not np.isfinite([height, tilt, error]).all():
        result = "FAIL: nonfinite simulation state"
        break

    if t >= 1.0 and (height < 0.68 or tilt > 15.0):
        result = (
            f"FAIL: stability stop at {t:.2f}s, "
            f"height={height:.3f}m, tilt={tilt:.2f}deg"
        )
        break

    if t >= next_report:
        next_report += 1.0
        print(
            f"t={t:5.2f}s "
            f"height={height:.3f}m "
            f"tilt={tilt:.2f}deg "
            f"hand_max_error={math.degrees(error):.2f}deg "
            f"closure={fraction * 100:.0f}% "
            f"forward={walk_progress:.3f}m "
            f"lateral={lateral_error:+.3f}m",
            flush=True
        )
else:
    height, tilt, error = measurements()

    if walk_finished_at is None:
        result = (
            "FAIL: walking target not reached; "
            f"forward={walk_progress:.3f}m"
        )
    elif turn_settled_at is None:
        result = "FAIL: turning phase did not complete"
    elif error <= math.radians(5):
        result = "FAIL: turn completion not verified"
    else:
        result = (
            "PARTIAL: standing completed, "
            f"hand maximum error={math.degrees(error):.2f}deg"
        )

print("\n===== PER-FINGER DIAGNOSTICS =====", flush=True)

total_steps = max(1, step + 1)

for name, qindex, aindex, goal, saturated in zip(
    physical.keys(),
    hand_qpos,
    hand_act,
    hand_goal,
    hand_saturation_counts,
):
    actual = float(data.qpos[qindex])
    error = math.degrees(goal - actual)

    print(
        f"{name:30s} "
        f"target={math.degrees(goal):+8.2f}deg "
        f"actual={math.degrees(actual):+8.2f}deg "
        f"error={error:+7.2f}deg "
        f"torque_saturation={100*saturated/total_steps:5.1f}%",
        flush=True
    )

print("\n===== FINAL RESULT =====", flush=True)
print(result, flush=True)
print("Simulation only; physical G1 not involved.", flush=True)
