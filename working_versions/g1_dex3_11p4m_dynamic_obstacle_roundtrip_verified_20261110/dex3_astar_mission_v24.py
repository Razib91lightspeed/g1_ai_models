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

from dex3_dynamic_obstacles_v3 import DynamicObstacles
import dex3_runtime_control_v1 as runtime_control
import os
world = DynamicObstacles(
    count=int(os.environ.get("DEX3_OBSTACLE_COUNT", "10")),
    seed=int(os.environ.get("DEX3_OBSTACLE_SEED", "42"))
)
model = world.model
data = world.data

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

# Experimental left-thumb base target.
# The original verified V10 target remains unchanged.
thumb_name = "left_hand_thumb_0_joint"
thumb_index = list(physical).index(thumb_name)
thumb_joint = model.joint(thumb_name).id

original_degrees = float(physical[thumb_name])
import os
thumb_offset = float(
    os.environ.get("DEX3_THUMB0_OFFSET_DEG", "-8")
)

if not -15 <= thumb_offset <= 15:
    raise ValueError("Thumb offset exceeds experimental limit")

low, high = model.jnt_range[thumb_joint]
hand_goal[thumb_index] = np.clip(
    math.radians(original_degrees + thumb_offset),
    low + 0.02,
    high - 0.02
)

print(
    f"LEFT THUMB BASE: original={original_degrees:.2f}deg "
    f"offset={thumb_offset:+.2f}deg "
    f"target={math.degrees(hand_goal[thumb_index]):.2f}deg",
    flush=True
)

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
print("Target distance: configurable", flush=True)
print("Next stage: omnidirectional 180-degree turn", flush=True)
print("Final stage: return to saved home", flush=True)
print("Walking command: 0.40 m/s", flush=True)
print("Stop threshold: target minus 0.05 m", flush=True)

dt = float(model.opt.timestep)
duration = 150.0
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
import os
target_distance = float(
    os.environ.get("DEX3_MISSION_DISTANCE", "2.0")
)
if not 0.5 <= target_distance <= 11.4:
    raise ValueError("Distance must be 0.5 to 11.4 m")
print(f"MISSION DISTANCE: {target_distance:.2f} m", flush=True)

turn_started_at = None
turn_last_yaw = None
turn_accumulated = 0.0
turn_settled_at = None
turn_previous_action = np.zeros(29, dtype=np.float32)
turn_initial_targets = None
turn_result = None

return_started_at = None
return_stop_at = None
return_initial_targets = None
return_home_distance = float("inf")


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

from dex3_route_follower_v3 import RouteFollower

route_follower = RouteFollower(model)

from dex3_return_follower_v3 import ReturnFollower
return_follower = ReturnFollower(route_follower.mapper)
next_auto_spawn_progress = 0.40

return_path_distance = 0.0
return_next_spawn_at = 0.40
return_last_xy = None
return_spawn_count = 0

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
        if (
                np.linalg.norm(
                    data.qpos[:2] - (
                        walk_origin + target_distance *
                        np.array([
                            math.cos(walk_heading),
                            math.sin(walk_heading)
                        ])
                    )
                ) <= 0.25
            ):
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

    # Experimental sensor-derived A* waypoint command.
    # Override outbound command only; preserve existing
    # turn and return controller logic.
    if (walk_origin is not None
            and walk_finished_at is None):
        route_command = route_follower.command(
            data, nav, walk_origin, walk_heading,
            target_distance, t
        )
        if route_command is not None:
            nav.command[:] = route_command

    if step % nav.interval == 0:
        if (walk_finished_at is not None
                and t - walk_finished_at >= 1.5
                and turn_started_at is None):

            if (
    np.linalg.norm(
        data.qpos[:2] - (
            walk_origin + target_distance *
            np.array([
                math.cos(walk_heading),
                math.sin(walk_heading)
            ])
        )
    ) > 0.40
):
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

        if return_started_at is not None:
            home_delta = walk_origin - data.qpos[:2]
            return_home_distance = float(
                np.linalg.norm(home_delta)
            )

            if (return_stop_at is None
                    and return_home_distance <= 0.20):
                return_stop_at = t
                print(
                    f"HOME STOP REQUESTED: "
                    f"distance={return_home_distance:.3f}m",
                    flush=True
                )

            if return_stop_at is not None:
                nav.command[:] = 0.0
            elif t - return_started_at < 0.7:
                nav.command[:] = 0.0
            else:
                bearing = math.atan2(
                    float(home_delta[1]),
                    float(home_delta[0])
                )
                heading_error = wrap(
                    bearing - float(nav.yaw_angle())
                )

                if abs(heading_error) > 0.40:
                    # Align before translating if necessary.
                    nav.command[:] = [
                        0.0, 0.0,
                        float(np.clip(
                            0.8 * heading_error, -0.20, 0.20
                        ))
                    ]
                else:
                    nav.command[:] = [
                        0.40, 0.0,
                        float(np.clip(
                            0.35 * heading_error, -0.15, 0.15
                        ))
                    ]

            if (return_stop_at is None and
                    t-return_started_at >= 0.7):
                nav.command[:] = return_follower.command(
                    data, nav, walk_origin, t
                )
            if return_follower.aligning:
                if (return_follower.align_since is not None
                        and t - return_follower.align_since > 8.0):
                    result = ("FAIL: return heading alignment stalled; "
                              f"error={return_follower.heading_error_deg:+.1f}deg")
                    print(result, flush=True)
                    break
                turn_previous_action, omni_target = omni_policy(
                    np.array([0.0, 0.0,
                              return_follower.turn_command],
                             dtype=np.float32),
                    turn_previous_action
                )
                nav.target = omni_target
                return_omni_active = True
            else:
                if return_omni_active:
                    for history in nav.history.values():
                        history.clear()
                    nav.last_action = np.zeros(29)
                    nav.command[:] = 0.0
                    return_omni_active = False
                    print("RETURN WALKING POLICY RESUMED", flush=True)
                nav.update_policy()

            blend = min(
                1.0,
                (t - return_started_at) / 0.7
            )
            nav.target = (
                (1.0 - blend) * return_initial_targets
                + blend * nav.target
            )

        elif turn_started_at is None:
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


    # Spawn during locomotion, not during scene initialization.
    if walk_origin is not None and step % 10 == 0:
        manual = runtime_control.spawn_request.is_set()
        automatic = (
            t >= 13.5
            and walk_progress >= next_auto_spawn_progress
            and walk_finished_at is None
            and walk_progress + 2.8 < target_distance - 0.5
            and len(world.active) < world.count
            and os.environ.get("DEX3_AUTO_SPAWN", "1") == "1"
        )

        if manual or automatic:
            runtime_control.spawn_request.clear()

            event = world.spawn_ahead(
                robot_xy=data.qpos[:2].copy(),
                heading=float(nav.yaw_angle()),
                distance=2.8
            )

            if event is not None:
                print(
                    "DYNAMIC OBSTACLE SPAWNED:",
                    event,
                    flush=True
                )

            if automatic:
                next_auto_spawn_progress += 3.0
                print(
                    f"NEXT AUTO SPAWN AT: "
                    f"{next_auto_spawn_progress:.2f}m",
                    flush=True
                )


    # Random return obstacles, spaced along actual
    # travelled ground distance.
    if (
        return_started_at is not None
        and return_stop_at is None
        and step % 10 == 0
        and os.environ.get("DEX3_AUTO_SPAWN", "1") == "1"
        and return_path_distance >= return_next_spawn_at
        and len(world.active) < world.count
    ):
        pos = data.qpos[:2].copy()
        to_home = walk_origin - pos
        home_remaining = float(np.linalg.norm(to_home))

        # Do not put a new obstacle next to HOME.
        if home_remaining > 3.5:
            heading_home = math.atan2(
                float(to_home[1]),
                float(to_home[0])
            )

            event = world.spawn_ahead(
                robot_xy=pos,
                heading=heading_home,
                distance=float(world.rng.uniform(2.6, 3.0))
            )

            if event is not None:
                return_spawn_count += 1
                print(
                    f"RETURN OBSTACLE #{return_spawn_count}: "
                    f"{event}",
                    flush=True
                )
            else:
                print(
                    "RETURN SPAWN SKIPPED: "
                    "no valid non-overlapping position",
                    flush=True
                )

            # Randomize separation between arrivals.
            # Each event is triggered by travelled distance,
            # not by wall-clock time.
            return_next_spawn_at += float(
                world.rng.uniform(2.5, 3.4)
            )

    # Experimental forward-facing LiDAR safety gate.
    # Reads ray intersections, not obstacle coordinates.
    if walk_origin is not None and step % 10 == 0:
        mujoco.mj_forward(model, data)

        groups = np.zeros(6, dtype=np.uint8)
        groups[3] = 1

        yaw = float(nav.yaw_angle())
        ray_origin = np.array([
            float(data.qpos[0]),
            float(data.qpos[1]),
            0.45
        ], dtype=np.float64)

        nearest = float("inf")

        for degrees in range(-35, 36, 5):
            angle = yaw + math.radians(degrees)
            direction = np.array([
                math.cos(angle),
                math.sin(angle),
                0.0
            ], dtype=np.float64)

            hit_id = np.array([-1], dtype=np.int32)
            distance = float(mujoco.mj_ray(
                model, data, ray_origin,
                direction, groups, 1, -1, hit_id
            ))

            if hit_id[0] >= 0 and distance >= 0:
                nearest = min(nearest, distance)

        # Last-resort simulation collision-proximity stop.
        # Navigation normally reacts earlier using A*.
        if nearest < 0.60:
            result = (
                "FAIL: obstacle inside emergency clearance; "
                f"distance={nearest:.3f}m"
            )
            print(result, flush=True)
            break

    mujoco.mj_step(model, data)

    if return_started_at is not None:
        current_xy = data.qpos[:2].copy()

        if return_last_xy is not None:
            return_path_distance += float(
                np.linalg.norm(
                    current_xy - return_last_xy
                )
            )

        return_last_xy = current_xy



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

        # Apply different spatial limits to each mission phase.
        if turn_started_at is None:
            phase_name = "OUTBOUND"
            out_of_bounds = (
                False
                or walk_progress > target_distance + 0.50
                or walk_progress < -0.15
            )

        elif return_started_at is None:
            phase_name = "TURN"
            out_of_bounds = (
                abs(lateral_error) > 0.60
                or walk_progress > target_distance + 0.80
                or walk_progress < -0.30
            )

        else:
            phase_name = "RETURN"
            return_home_distance = float(
                np.linalg.norm(data.qpos[:2] - walk_origin)
            )
            out_of_bounds = (
                return_home_distance > target_distance + 1.50
                or False
                or walk_progress > target_distance + 1.20
                or walk_progress < -0.40
            )

        if out_of_bounds:
            result = (
                f"FAIL: {phase_name} boundary exceeded; "
                f"forward={walk_progress:.3f}m, "
                f"lateral={lateral_error:.3f}m, "
                f"home_distance={return_home_distance:.3f}m"
            )
            break

        if (turn_settled_at is not None
                and return_started_at is None
                and t - turn_settled_at >= 0.8):

            final_turn = math.degrees(turn_accumulated)
            angle_error = 180.0 - final_turn

            print(
                f"SETTLED TURN: {final_turn:.2f}deg "
                f"error={angle_error:+.2f}deg",
                flush=True
            )

            if abs(angle_error) > 3.0:
                result = (
                    "FAIL: turn outside tolerance; "
                    f"angle={final_turn:.2f}deg"
                )
                break

            return_started_at = t
            return_omni_active = False
            turn_previous_action = np.zeros(29)

            # New randomized environment for HOME journey.
            world.clear_all()
            route_follower.mapper.reset()

            return_follower.path = None
            return_follower.next_scan = 0.0
            return_follower.next_plan = 0.0

            return_path_distance = 0.0
            return_next_spawn_at = 0.40
            return_last_xy = data.qpos[:2].copy()
            return_spawn_count = 0

            print(
                "RETURN ENVIRONMENT READY: "
                "outbound obstacles cleared; "
                "new obstacles will spawn while walking",
                flush=True
            )
            return_initial_targets = nav.target.copy()

            # Restart walking-policy observation history
            # after the omnidirectional policy transition.
            for h in nav.history.values():
                h.clear()
            nav.last_action = np.zeros(29)
            nav.command[:] = 0.0

            print(
                "RETURN START: navigating to saved home",
                flush=True
            )

        if return_started_at is not None:
            return_home_distance = float(
                np.linalg.norm(
                    data.qpos[:2] - walk_origin
                )
            )

            if (
                return_stop_at is None
                and t - return_started_at > max(20.0, 5.0 * target_distance + 10.0)
            ):
                result = (
                    "FAIL: return timeout; "
                    f"home_distance={return_home_distance:.3f}m"
                )
                break

            if (
                return_stop_at is not None
                and t - return_stop_at >= 1.5
            ):
                if return_home_distance <= 0.25:
                    result = (
                        "PASS: Dex3 walk, turn and return home; "
                        f"home_error={return_home_distance:.3f}m"
                    )
                else:
                    result = (
                        "FAIL: return settling error; "
                        f"home_distance={return_home_distance:.3f}m"
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
    elif return_started_at is None:
        result = "FAIL: return phase did not begin"
    elif return_stop_at is None:
        result = "FAIL: home position not reached"
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
