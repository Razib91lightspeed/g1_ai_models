"""Four independent G1 robots using the verified ONNX locomotion policy.

Isaac Lab 3.0 / Newton / Viser.
Simulation only. Never sends physical robot commands.
"""

import math
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import isaaclab_tasks
from isaaclab.cli import cli

PROJECT = Path("/workspace/local")
BASE = PROJECT / (
    "working_versions/"
    "4G1_walking_forward_60s_working"
)

CONFIG = BASE / "policy/params/deploy.yaml"
MODEL = BASE / "policy/exported/policy.onnx"

sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "configs"))

from g1_onnx_adapter import G1PolicyAdapter
from g1_4env_onnx_walking_cfg import G14EnvONNXWalkingCfg

from isaaclab.utils import configclass

@configclass
class G14EnvDex3RouteCfg(G14EnvONNXWalkingCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 240.0

TASK_ID = "Project-G1-Four-Robot-Dex3-Route"

MOTOR_NAMES = [
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
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]

original_spec = gym.spec("Isaac-Velocity-Flat-G1")
kwargs = dict(original_spec.kwargs)
kwargs["env_cfg_entry_point"] = G14EnvDex3RouteCfg

gym.register(
    id=TASK_ID,
    entry_point=original_spec.entry_point,
    disable_env_checker=True,
    kwargs=kwargs,
)

original_gym_make = gym.make


def make_four_robot_environment(*args, **kwargs):
    env = original_gym_make(*args, **kwargs)
    sim_env = env.unwrapped
    robot = sim_env.scene["robot"]

    n = int(sim_env.cfg.scene.num_envs)
    if n != 4:
        raise RuntimeError(f"Expected four robots, got {n}")

    # A separate history and last-action vector for every robot.
    policies = [
        G1PolicyAdapter(str(CONFIG), str(MODEL))
        for _ in range(n)
    ]

    runtime_names = list(robot.joint_names)
    action_term = sim_env.action_manager.get_term("joint_pos")
    action_names = list(action_term._joint_names)
    configured_names = list(sim_env.cfg.actions.joint_pos.joint_names)

    policy_names = [
        MOTOR_NAMES[int(index)] + "_joint"
        for index in policies[0].ids
    ]

    if len(runtime_names) != 43:
        raise RuntimeError(
            f"Expected 43 physical simulation joints; got {len(runtime_names)}"
        )

    if len(action_names) != 29 or action_names != configured_names:
        raise RuntimeError("29-joint action order mismatch")

    if len(set(policy_names)) != 29:
        raise RuntimeError("Duplicate policy joint names")

    if set(policy_names) != set(action_names):
        raise RuntimeError("Policy and action joint sets differ")

    state_indices = [runtime_names.index(name) for name in policy_names]
    action_indices = [policy_names.index(name) for name in action_names]

    # Never rely on matching action dimensions alone.
    expected_omni_names = [
        name + "_joint"
        for name in _omni_remap.SIM_ORDER_NAMES
    ]

    if policy_names != expected_omni_names:
        raise RuntimeError(
            "OMNI_JOINT_ORDER_MISMATCH: "
            f"current={policy_names} "
            f"expected={expected_omni_names}"
        )

    omni_last_actions = np.zeros(
        (n, 29), dtype=np.float32
    )

    omni_turn_started = [None] * n
    omni_settle_started = [None] * n

    print(
        "OMNI_JOINT_ORDER_PASS: all 29 joints",
        flush=True,
    )


    defaults = robot.data.default_joint_pos.torch.detach().cpu().numpy()
    if defaults.shape[0] != n:
        raise RuntimeError("Default joint data batch mismatch")

    action_defaults = defaults[
        :, [runtime_names.index(name) for name in action_names]
    ]

    for i, policy in enumerate(policies):
        pose = defaults[i, state_indices]
        error = float(np.max(np.abs(pose - policy.offset)))
        print(
            f"G1_{i+1} reference_pose_error={error:.4f}rad",
            flush=True,
        )
        if error > 0.35:
            raise RuntimeError(
                f"G1_{i+1}: ONNX reference pose mismatch"
            )

    action_scale = float(sim_env.cfg.actions.joint_pos.scale)
    dt = float(sim_env.cfg.sim.dt * sim_env.cfg.decimation)

    if action_scale <= 0:
        raise RuntimeError("Invalid action scale")

    if abs(dt - float(policies[0].cfg["step_dt"])) > 1e-6:
        raise RuntimeError("ONNX control period mismatch")


    # Dex3 physical resting-hand targets.
    # Body locomotion remains controlled by the 29-action ONNX policy.
    import json

    hand_reference_path = PROJECT / (
        "working_versions/"
        "g1_dex3_physical_hand_rest_pose_visual_verified_20261102/"
        "physical_dex3_angles.json"
    )

    with hand_reference_path.open() as stream:
        physical_degrees = json.load(stream)["angles"]

    hand_names = [
        name for name in runtime_names
        if "_hand_" in name
    ]

    if len(hand_names) != 14:
        raise RuntimeError(
            f"Expected 14 Dex3 joints, found {len(hand_names)}"
        )

    if set(hand_names) != set(physical_degrees):
        raise RuntimeError("Dex3 physical joint mapping mismatch")

    hand_indices = [runtime_names.index(name) for name in hand_names]

    hand_targets_base = (
        robot.data.default_joint_pos.torch[:, hand_indices]
        .detach().clone()
    )

    hand_targets_goal = hand_targets_base.clone()
    hand_targets = hand_targets_base.clone()

    limit_field = robot.data.soft_joint_pos_limits
    limits = (
        limit_field.torch
        if hasattr(limit_field, "torch")
        else limit_field
    )

    if tuple(limits.shape) != (n, len(runtime_names), 2):
        raise RuntimeError("Unexpected Dex3 joint limit shape")

    for rid in range(n):
        for name, measured_deg in physical_degrees.items():
            ji = runtime_names.index(name)
            hi = hand_names.index(name)

            low = float(limits[rid, ji, 0])
            high = float(limits[rid, ji, 1])

            if high - low <= 0.04:
                raise RuntimeError(
                    f"Invalid usable joint range: {name}"
                )

            requested = math.radians(measured_deg)
            applied = min(
                high - 0.02,
                max(low + 0.02, requested)
            )

            hand_targets_goal[rid, hi] = applied

            if rid == 0:
                print(
                    f"WALK_DEX3_TARGET {name} "
                    f"physical={measured_deg:+.3f}deg "
                    f"applied={math.degrees(applied):+.3f}deg",
                    flush=True,
                )

    if not torch.allclose(
        hand_targets_goal,
        hand_targets_goal[0:1].expand_as(hand_targets_goal),
        atol=1e-6,
    ):
        raise RuntimeError("Four robots have different hand targets")

    print(
        "WALK_DEX3_READY: 14 hand joints, "
        "29 independent ONNX body actions, free-root simulation",
        flush=True,
    )

    route_origin = [None] * n
    route_yaw0 = [None] * n
    route_phase = ["OUT"] * n
    route_turn_accum = [0.0] * n
    route_last_yaw = [None] * n
    route_success = [False] * n
    route_wave_start = [None] * n
    wave_verified = [False] * n
    omni_return_started = [None] * n
    route_pause_start = [None] * n
    route_turn_xy = [None] * n

    def wrap_angle(a):
        return math.atan2(math.sin(a), math.cos(a))

    def yaw_from_quat(q):
        x, y, z, w = [float(v) for v in q]
        return math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z)
        )

    import csv as _v16_csv

    _v16_wave_file = PROJECT / (
        "simulations/unitree_g1/parallel_rl/experiments/"
        "g1_realdata_turn_return/recordings/"
        "left_arm_wave_reference_v1/"
        "left_arm_wave_softlimited.csv"
    )

    _v16_names = [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ]

    with _v16_wave_file.open(newline="") as _f:
        _v16_rows = list(_v16_csv.DictReader(_f))

    if len(_v16_rows) != 274:
        raise RuntimeError("Expected 274 measured-wave frames")

    _v16_times = np.array(
        [float(r["relative_time_s"]) for r in _v16_rows],
        dtype=np.float64,
    )

    _v16_q = np.array(
        [
            [
                float(r[name.removesuffix("_joint")])
                for name in _v16_names
            ]
            for r in _v16_rows
        ],
        dtype=np.float64,
    )

    if not np.isfinite(_v16_times).all():
        raise RuntimeError("Invalid recorded-wave timestamps")

    if not np.isfinite(_v16_q).all():
        raise RuntimeError("Invalid recorded-wave positions")

    if not np.all(np.diff(_v16_times) > 0):
        raise RuntimeError("Wave timestamps not increasing")

    _v16_duration = float(_v16_times[-1])

    if not 5.0 < _v16_duration < 6.0:
        raise RuntimeError("Unexpected recorded-wave duration")

    _v16_action_ids = [
        action_names.index(name)
        for name in _v16_names
    ]

    _v16_runtime_ids = [
        runtime_names.index(name)
        for name in _v16_names
    ]

    # The free-standing robot must remain free.
    if sim_env.cfg.scene.robot.spawn.fix_root_link:
        raise RuntimeError(
            "STOP: Walking robot unexpectedly has a fixed root"
        )

    _v16_soft = robot.data.soft_joint_pos_limits
    _v16_soft = getattr(
        _v16_soft, "torch", _v16_soft
    ).detach().cpu().numpy()

    for _rid in range(n):
        for _j, _jid in enumerate(_v16_runtime_ids):
            _lo, _hi = _v16_soft[_rid, _jid]
            if (
                _v16_q[:, _j].min() < float(_lo) - 0.001
                or _v16_q[:, _j].max() > float(_hi) + 0.001
            ):
                raise RuntimeError(
                    f"Recorded-wave joint limit: {_v16_names[_j]}"
                )

    # Apply the exact v15 physical Dex3 resting pose
    # using the simulator's actual soft joint limits.
    _v16_hand_degrees = {'left_hand_thumb_0_joint': 30.6766, 'left_hand_thumb_1_joint': 60.3186, 'left_hand_thumb_2_joint': 85.4347, 'left_hand_index_0_joint': -91.9567, 'left_hand_index_1_joint': -100.0102, 'left_hand_middle_0_joint': -91.6749, 'left_hand_middle_1_joint': -98.559, 'right_hand_thumb_0_joint': -2.45, 'right_hand_thumb_1_joint': -55.8133, 'right_hand_thumb_2_joint': -94.3222, 'right_hand_index_0_joint': 87.8579, 'right_hand_index_1_joint': 97.9258, 'right_hand_middle_0_joint': 88.3624, 'right_hand_middle_1_joint': 98.5291}

    if set(_v16_hand_degrees) != set(hand_names):
        raise RuntimeError("Dex3 hand-joint set mismatch")

    for _rid in range(n):
        for _name, _degrees in _v16_hand_degrees.items():
            _jid = runtime_names.index(_name)
            _hid = hand_names.index(_name)

            _lo = float(limits[_rid, _jid, 0])
            _hi = float(limits[_rid, _jid, 1])

            _requested = math.radians(_degrees)
            _applied = min(
                _hi - 0.02,
                max(_lo + 0.02, _requested),
            )

            hand_targets_goal[_rid, _hid] = _applied

            if _rid == 0:
                print(
                    f"V16_DEX3 {_name} "
                    f"target={_applied:+.4f}rad "
                    f"measured={_requested:+.4f}rad",
                    flush=True,
                )

    print(
        "V16_READY: route=v6 wave=v15 "
        "dex3=physical_measured fixed_root=False",
        flush=True,
    )
    print(
        f"V16_WAVE_DURATION={_v16_duration:.3f}s",
        flush=True,
    )


    _v23_targets = {'left_hip_pitch_joint': -0.180584, 'left_knee_joint': 0.446329, 'left_ankle_pitch_joint': -0.238329, 'right_hip_pitch_joint': -0.157156, 'right_knee_joint': 0.42314, 'right_ankle_pitch_joint': -0.243608}
    _v23_amount = 0.25

    if set(_v23_targets) - set(action_names):
        raise RuntimeError("V23 action joint mapping mismatch")

    print("V23_READY: measured physical leg posture", flush=True)
    print("V23_MODE: G1_1 posture test, arms unchanged", flush=True)

    for _name, _value in _v23_targets.items():
        print(
            f"V23_PHYSICAL_TARGET {_name}={_value:+.4f}rad",
            flush=True,
        )

    original_step = env.step

    original_reset = env.reset
    step_count = 0

    def reset_four_robots(*args, **kwargs):
        nonlocal step_count
        step_count = 0
        omni_last_actions.fill(0.0)
        for j in range(n):
            omni_turn_started[j] = None
            omni_settle_started[j] = None
        for policy in policies:
            policy.reset()
        return original_reset(*args, **kwargs)

    def step_four_robots(_zero_actions):
        nonlocal step_count

        state = robot.data

        joint_pos = state.joint_pos.torch[:, state_indices]
        joint_vel = state.joint_vel.torch[:, state_indices]
        angular = state.root_link_ang_vel_b.torch
        gravity = state.projected_gravity_b.torch
        roots = state.root_link_pos_w.torch

        positions = joint_pos.detach().cpu().numpy()
        velocities = joint_vel.detach().cpu().numpy()
        ang = angular.detach().cpu().numpy()
        grav = gravity.detach().cpu().numpy()

        root_positions = roots.detach().cpu().numpy()
        root_quaternions = (
            state.root_link_quat_w.torch.detach().cpu().numpy()
        )

        if step_count == 0:
            for policy in policies:
                policy.reset()

        if step_count == 0:
            for rid in range(n):
                route_origin[rid] = None
                route_yaw0[rid] = None
                route_phase[rid] = "OUT"
                route_pause_start[rid] = None
                route_turn_xy[rid] = None
                route_wave_start[rid] = None
                route_turn_accum[rid] = 0.0
                route_last_yaw[rid] = None
                route_success[rid] = False
                wave_verified[rid] = False

        t = step_count * dt



        forward = 0.0

        batched_actions = np.empty((n, 29), dtype=np.float32)

        for i, policy in enumerate(policies):
            height = float(root_positions[i, 2])
            tilt = math.degrees(
                math.acos(float(np.clip(-grav[i, 2], -1.0, 1.0)))
            )

            if not math.isfinite(height) or not math.isfinite(tilt):
                raise RuntimeError(f"G1_{i+1}: Nonfinite robot state")

            if height < 0.55 or tilt > 30.0:
                raise RuntimeError(
                    f"G1_{i+1}_FALL_STOP "
                    f"t={t:.2f}s height={height:.3f}m "
                    f"tilt={tilt:.1f}deg"
                )




            yaw = yaw_from_quat(root_quaternions[i])
            xy = root_positions[i, :2].copy()

            if route_origin[i] is None:
                route_origin[i] = xy.copy()
                route_yaw0[i] = yaw
                route_last_yaw[i] = yaw

            origin = route_origin[i]
            yaw0 = route_yaw0[i]

            dx = float(origin[0] - xy[0])
            dy = float(origin[1] - xy[1])

            forward_distance = float(xy[1] - origin[1])
            lateral_error = float(xy[0] - origin[0])
            home_error = math.hypot(dx, dy)

            vx = 0.0
            wz = 0.0
            phase = route_phase[i]

            # The verified outbound walking direction is world +Y.
            # Relate heading feedback to that measured direction.
            heading_offset = wrap_angle(yaw0 - math.pi / 2.0)

            if phase == "OUT":
                remaining = 11.4 - forward_distance

                if remaining <= 0.20:
                    route_phase[i] = "PAUSE_BEFORE_TURN"
                    route_pause_start[i] = t

                    vx = 0.0
                    wz = 0.0

                    print(
                        f"ROUTE_EVENT G1_{i+1} OUT_COMPLETE "
                        f"t={t:.2f}s "
                        f"distance={forward_distance:.3f}m "
                        f"lateral={lateral_error:+.3f}m",
                        flush=True
                    )

                else:
                    # Gradual acceleration, then destination slowdown.
                    if t < 3.0:
                        vx = 0.0
                    elif t < 6.0:
                        vx = 0.40 * (t - 3.0) / 3.0
                    else:
                        vx = min(
                            0.40,
                            max(0.18, 0.60 * remaining)
                        )

                    # Correct sideways drift and heading error.
                    # Initial forward direction is world +Y.
                    heading_error = wrap_angle(yaw - yaw0)

                    wz = float(max(
                        -0.16,
                        min(
                            0.16,
                            0.65 * lateral_error
                            - 1.10 * heading_error
                        )
                    ))

            elif phase == "PAUSE_BEFORE_TURN":
                vx = 0.0
                wz = 0.0

                if route_pause_start[i] is None:
                    route_pause_start[i] = t

                if t - route_pause_start[i] >= 1.0:
                    route_phase[i] = "TURN"
                    route_turn_accum[i] = 0.0
                    route_last_yaw[i] = yaw
                    route_turn_xy[i] = xy.copy()

                    print(
                        f"ROUTE_EVENT G1_{i+1} TURN_START "
                        f"t={t:.2f}s",
                        flush=True
                    )

            elif phase == "TURN":
                delta = wrap_angle(yaw - route_last_yaw[i])
                route_turn_accum[i] += delta
                route_last_yaw[i] = yaw

                turn_remaining = math.pi - route_turn_accum[i]

                if turn_remaining <= math.radians(3):
                    route_phase[i] = "WAVE"
                    route_wave_start[i] = t

                    displacement = float(
                        np.linalg.norm(xy - route_turn_xy[i])
                    )

                    vx = 0.0
                    wz = 0.0

                    print(
                        f"ROUTE_EVENT G1_{i+1} TURN_COMPLETE "
                        f"t={t:.2f}s "
                        f"angle={math.degrees(route_turn_accum[i]):.1f}deg "
                        f"turn_displacement={displacement:.3f}m",
                        flush=True
                    )
                    print(
                        f"ROUTE_EVENT G1_{i+1} WAVE_START "
                        f"t={t:.2f}s",
                        flush=True
                    )

                else:
                    # Experimental compact turn.
                    # Existing ONNX policy may not support this.
                    vx = 0.18

                    wz = min(
                        0.32,
                        max(0.10, 0.55 * turn_remaining)
                    )

            elif phase == "WAVE":
                vx = 0.0
                wz = 0.0

                if route_wave_start[i] is None:
                    route_wave_start[i] = t

                if t - route_wave_start[i] >= 5.46:
                    route_phase[i] = "WAVE_HOLD"
                    print(
                        f"ROUTE_EVENT G1_{i+1} WAVE_COMPLETE "
                        f"t={t:.2f}s",
                        flush=True
                    )

            elif phase == "WAVE_HOLD":
                vx = 0.0
                wz = 0.0

                if t - route_wave_start[i] >= 6.3:
                    if not wave_verified[i]:
                        wave_verified[i] = True
                        print(
                            f"WAVE_STABLE_SUCCESS G1_{i+1} "
                            f"t={t:.2f}s "
                            f"height={height:.3f}m "
                            f"tilt={tilt:.2f}deg",
                            flush=True,
                        )

                if all(wave_verified):
                    print(
                        "FOUR_G1_WAVE_VALIDATION_COMPLETE",
                        flush=True,
                    )
                    for rid in range(n):
                        route_phase[rid] = "RETURN_OMNI"
                        omni_return_started[rid] = t
                    print(
                        "FOUR_G1_RETURN_START "
                        f"t={t:.2f}s",
                        flush=True,
                    )

            elif phase == "RETURN_OMNI":
                dx = float(route_origin[i][0] - xy[0])
                dy = float(route_origin[i][1] - xy[1])

                distance = math.hypot(dx, dy)

                # Route heading convention already used by v3.
                desired_yaw = wrap_angle(
                    math.atan2(dy, dx) + heading_offset
                )
                heading_error = wrap_angle(desired_yaw - yaw)

                if distance <= 0.20:
                    route_phase[i] = "DONE_OMNI"
                    route_success[i] = True
                    vx = 0.0
                    wz = 0.0
                    print(
                        f"OMNI_HOME_SUCCESS G1_{i+1} "
                        f"t={t:.2f}s "
                        f"distance={distance:.3f}m "
                        f"tilt={tilt:.2f}deg",
                        flush=True,
                    )
                else:
                    # Reduce forward motion if heading error grows.
                    speed_limit = min(0.30, 0.8 * distance)
                    heading_factor = max(
                        0.0, math.cos(heading_error)
                    )

                    vx = speed_limit * heading_factor
                    wz = float(
                        np.clip(
                            1.5 * heading_error,
                            -0.35,
                            0.35,
                        )
                    )

                    if step_count % 50 == 0:
                        print(
                            f"OMNI_RETURN_PROGRESS G1_{i+1} "
                            f"t={t:.2f}s "
                            f"home_error={distance:.3f}m "
                            f"heading_error={math.degrees(heading_error):+.1f}deg "
                            f"vx={vx:.3f} wz={wz:+.3f} "
                            f"height={height:.3f}m "
                            f"tilt={tilt:.2f}deg",
                            flush=True,
                        )

                if (
                    omni_return_started[i] is not None
                    and t - omni_return_started[i] > 100.0
                    and not route_success[i]
                ):
                    raise RuntimeError(
                        f"OMNI_RETURN_TIMEOUT G1_{i+1} "
                        f"home_error={distance:.3f}m"
                    )

            elif phase == "DONE_OMNI":
                vx = 0.0
                wz = 0.0

                if all(route_success):
                    print(
                        "FOUR_G1_ROUTE_COMPLETE "
                        "All robots reached their recorded home coordinates.",
                        flush=True,
                    )
                    raise SystemExit(0)

            elif phase in ("ALIGN", "BACK"):
                # Bearing from current position to the EXACT
                # recorded starting point.
                home_bearing = math.atan2(dy, dx)

                # Convert desired movement bearing to the
                # simulator's body-heading convention.
                target_yaw = wrap_angle(
                    home_bearing + heading_offset
                )

                yaw_error = wrap_angle(target_yaw - yaw)

                # Re-align if the robot turns too far away
                # from the direction of home.
                if abs(yaw_error) > 0.35:
                    route_phase[i] = "ALIGN"
                    vx = 0.30
                    wz = float(
                        max(-0.20, min(0.20, yaw_error))
                    )

                elif home_error <= 0.20:
                    # Position-based stop, including lateral error.
                    route_phase[i] = "DONE"
                    route_success[i] = True

                    print(
                        f"ROUTE_EVENT G1_{i+1} RETURN_COMPLETE "
                        f"t={t:.2f}s "
                        f"home_error={home_error:.3f}m "
                        f"dx={dx:+.3f}m dy={dy:+.3f}m",
                        flush=True
                    )

                else:
                    route_phase[i] = "BACK"

                    # Move forward after the turn, toward home.
                    vx = min(
                        0.30,
                        max(0.04, 0.45 * home_error)
                    )

                    wz = float(
                        max(-0.20, min(0.20, yaw_error))
                    )

            # DONE: zero velocity command; keep balance policy.
            command = np.array(
                [vx, 0.0, wz],
                dtype=np.float32
            )

            if step_count == 0 or step_count % 50 == 0:
                print(
                    f"HOME_ROUTE G1_{i+1} "
                    f"t={t:.2f}s "
                    f"phase={route_phase[i]} "
                    f"out={forward_distance:.3f}m "
                    f"lateral={lateral_error:+.3f}m "
                    f"home_error={home_error:.3f}m "
                    f"yaw={math.degrees(yaw):+.1f}deg "
                    f"vx={vx:.3f} wz={wz:+.3f}",
                    flush=True
                )

            obs, raw, targets = policy.step(
                angular_velocity=ang[i],
                projected_gravity=grav[i],
                velocity_command=command,
                joint_position=positions[i],
                joint_velocity=velocities[i],
            )

            # The original ONNX walker continues to produce
            # correct walking observations and joint targets.
            # Substitute the second policy only during TURN.
            if route_phase[i] in ("TURN", "TURN_SETTLE", "WAVE", "WAVE_HOLD", "RETURN_OMNI", "DONE_OMNI"):

                if omni_turn_started[i] is None:
                    omni_turn_started[i] = t
                    omni_last_actions[i].fill(0.0)
                    print(
                        f"OMNI_TURN_START G1_{i+1} "
                        f"t={t:.2f}s",
                        flush=True,
                    )

                elapsed = t - omni_turn_started[i]

                tilt_deg = math.degrees(
                    math.acos(float(np.clip(-grav[i, 2], -1.0, 1.0)))
                )

                height = float(
                    state.root_link_pos_w.torch[i, 2].item()
                )

                # Stop the trial before severe loss of balance.
                if tilt_deg > 15.0 or height < 0.68:
                    raise RuntimeError(
                        f"OMNI_TURN_SAFETY_STOP G1_{i+1} "
                        f"tilt={tilt_deg:.2f}deg "
                        f"height={height:.3f}m"
                    )

                if route_phase[i] == "TURN" and elapsed >= 20.0:
                    print(
                        f"OMNI_TURN_TIMEOUT G1_{i+1} "
                        f"elapsed={elapsed:.2f}s "
                        f"yaw={math.degrees(route_turn_accum[i]):+.2f}deg",
                        flush=True,
                    )
                    raise RuntimeError("OMNI_TURN_TIMEOUT_STOP")

                # v3 accumulates measured yaw before policy.step.
                if route_phase[i] == "TURN" and abs(math.degrees(route_turn_accum[i])) >= 175.0:
                    print(
                        f"OMNI_TURN_SUCCESS G1_{i+1} "
                        f"yaw={math.degrees(route_turn_accum[i]):+.2f}deg "
                        f"elapsed={elapsed:.2f}s",
                        flush=True,
                    )
                    route_phase[i] = "TURN_SETTLE"
                    omni_settle_started[i] = t
                    print(
                        f"OMNI_TURN_SETTLE_START G1_{i+1} "
                        f"t={t:.2f}s",
                        flush=True,
                    )

                linear_body = (
                    state.root_link_lin_vel_b.torch[i]
                    .detach().cpu().numpy()
                    .astype(np.float32)
                )

                # Training policy expects:
                # lin_vel(3), ang_vel(3), gravity(3),
                # command(3), joint_pos_rel(29),
                # joint_vel(29), last_action(29), scan(143).
                # No additional observation scaling.
                if route_phase[i] == "TURN_SETTLE":
                    omni_command = np.array(
                        [0.0, 0.0, 0.0], dtype=np.float32
                    )
                    if t - omni_settle_started[i] >= 0.8:
                        route_phase[i] = "WAVE"
                        route_wave_start[i] = t
                        print(
                            f"ROUTE_EVENT G1_{i+1} WAVE_START "
                            f"t={t:.2f}s",
                            flush=True,
                        )
                else:
                    omni_command = np.array(
                        [0.0, 0.0, 0.60 if route_phase[i] == "TURN" else 0.0], dtype=np.float32
                    )


                if route_phase[i] == "RETURN_OMNI":
                    omni_command = np.array(
                        [vx, 0.0, wz], dtype=np.float32
                    )

                elif route_phase[i] == "DONE_OMNI":
                    omni_command = np.zeros(
                        3, dtype=np.float32
                    )

                omni_obs = np.concatenate([
                    linear_body,
                    ang[i].astype(np.float32),
                    grav[i].astype(np.float32),
                    omni_command,
                    (
                        positions[i].astype(np.float32)
                        - _omni_remap.DEFAULT_SIM
                    ),
                    velocities[i].astype(np.float32),
                    omni_last_actions[i],
                    _omni_scan,
                ]).astype(np.float32)

                if omni_obs.shape != (242,):
                    raise RuntimeError("OMNI_OBSERVATION_SHAPE_ERROR")

                if not np.isfinite(omni_obs).all():
                    raise RuntimeError("OMNI_NONFINITE_OBSERVATION")

                omni_raw = _omni_session.run(
                    None,
                    {"obs": omni_obs.reshape(1, 242)},
                )[0][0].astype(np.float32)

                if not np.isfinite(omni_raw).all():
                    raise RuntimeError("OMNI_NONFINITE_ACTION")

                maximum = float(np.abs(omni_raw).max())

                if maximum > 3.0:
                    raise RuntimeError(
                        f"OMNI_ACTION_SAFETY_STOP "
                        f"G1_{i+1} raw={maximum:.3f}"
                    )

                # Source policy uses a 0.5 action scale
                # and its own default reference pose.
                omni_last_actions[i] = omni_raw.copy()

                raw = omni_raw.copy()
                targets = (
                    _omni_remap.DEFAULT_SIM
                    + 0.5 * omni_raw
                ).astype(np.float32)

                policy.last_action = (
                    (targets - policy.offset) / policy.scale
                ).astype(np.float32)

                if step_count % 50 == 0:
                    print(
                        f"OMNI_TURN_PROGRESS G1_{i+1} "
                        f"t={t:.2f}s "
                        f"elapsed={elapsed:.2f}s "
                        f"yaw={math.degrees(route_turn_accum[i]):+.2f}deg "
                        f"tilt={tilt_deg:.2f}deg "
                        f"height={height:.3f}m "
                        f"raw_max={maximum:.3f}",
                        flush=True,
                    )


            if obs.shape != (1, 480) or raw.shape != (29,):
                raise RuntimeError(f"G1_{i+1}: Invalid ONNX dimensions")

            if route_phase[i] == "WAVE" and tilt > 8.0:
                raise RuntimeError(
                    f"WAVE_BALANCE_GUARD G1_{i+1} "
                    f"t={t:.2f}s tilt={tilt:.2f}deg"
                )

            ordered_targets = targets[action_indices]
            if route_phase[i] == "WAVE":
                _v16_elapsed = max(
                    0.0, t - route_wave_start[i]
                )
                _v16_elapsed = min(
                    _v16_elapsed, _v16_duration
                )

                _v16_requested = np.array(
                    [
                        np.interp(
                            _v16_elapsed,
                            _v16_times,
                            _v16_q[:, j],
                        )
                        for j in range(7)
                    ],
                    dtype=np.float32,
                )

                # Gradual entry and exit from the pretrained
                # balance controller's arm posture.
                _v16_entry = min(
                    1.0, _v16_elapsed / 0.8
                )
                _v16_exit = min(
                    1.0,
                    max(
                        0.0,
                        (_v16_duration - _v16_elapsed) / 0.8
                    ),
                )

                _v16_mix = min(_v16_entry, _v16_exit)
                _v16_mix = (
                    _v16_mix * _v16_mix *
                    (3.0 - 2.0 * _v16_mix)
                )

                # Diagnostic: test only G1_1 with 20% of the recorded arm motion.
                # Other robots retain the pretrained standing-arm targets.
                _v16_mix *= 0.0
                if step_count % 25 == 0:
                    print(f"V17_BALANCE_PROBE G1_{i+1} mix={_v16_mix:.3f} tilt={tilt:.2f}deg", flush=True)

                for _j, _aid in enumerate(_v16_action_ids):
                    _original = float(ordered_targets[_aid])
                    _requested = float(_v16_requested[_j])

                    ordered_targets[_aid] = (
                        (1.0 - _v16_mix) * _original
                        + _v16_mix * _requested
                    )

                if step_count % 25 == 0:
                    print(
                        f"V16_WAVE_TARGET G1_{i+1} "
                        f"t={t:.2f}s "
                        f"elapsed={_v16_elapsed:.2f}s "
                        f"mix={_v16_mix:.3f} "
                        f"elbow={ordered_targets[_v16_action_ids[3]]:+.3f}rad "
                        f"tilt={tilt:.2f}deg",
                        flush=True,
                    )

            normalized = (
                ordered_targets - action_defaults[i]
            ) / action_scale

            if not np.isfinite(normalized).all():
                raise RuntimeError(f"G1_{i+1}: Nonfinite actions")

            _v16_commands = np.clip(
                normalized, -4.0, 4.0
            ).astype(np.float32)

            # The recorded elbow can require an action
            # magnitude above four. Do not silently clip
            # the seven wave joints after checking soft limits.
            if route_phase[i] == "WAVE":
                for _aid in _v16_action_ids:
                    _v16_commands[_aid] = float(
                        normalized[_aid]
                    )

            batched_actions[i] = _v16_commands

        if t >= 60.0:
            if step_count % 500 == 0:
                print(
                    f"FOUR_G1_VIEWER_RUNNING "
                    f"t={t:.2f}s: maintaining standing command",
                    flush=True,
                )

        action_tensor = torch.as_tensor(
            batched_actions,
            device=sim_env.device,
            dtype=torch.float32,
        )


        # Dex3 joint targets are independent of the 29 body actions.
        # Finish hand-position interpolation during initial standing.
        hand_blend = min(1.0, max(0.0, t / 3.0))

        hand_targets[:] = (
            hand_targets_base
            + hand_blend * (hand_targets_goal - hand_targets_base)
        )

        robot.set_joint_position_target(
            hand_targets,
            joint_ids=hand_indices,
        )

        if t >= 320.0 and not all(route_success):
            raise RuntimeError(
                f"ROUTE_TIMEOUT phases={route_phase}"
            )


        # V23: measured lower-body reference.
        # Applies only to G1_1 during the stationary phase.
        # The other robots use their original controllers.
        if (
            route_phase[0] in ("WAVE", "WAVE_HOLD")
            and route_wave_start[0] is not None
        ):
            _elapsed23 = max(0.0, t - route_wave_start[0])

            # Smoothly enter and leave the posture trial.
            _enter23 = min(1.0, _elapsed23 / 2.0)
            _leave23 = min(
                1.0,
                max(0.0, (5.46 - _elapsed23) / 1.5)
            )
            _blend23 = min(_enter23, _leave23)
            _blend23 = _blend23 * _blend23 * (3.0 - 2.0 * _blend23)
            _blend23 *= _v23_amount

            for _name, _target23 in _v23_targets.items():
                _aid23 = action_names.index(_name)
                _jid23 = runtime_names.index(_name)

                _lo23 = float(limits[0, _jid23, 0])
                _hi23 = float(limits[0, _jid23, 1])

                if not _lo23 + 0.02 <= _target23 <= _hi23 - 0.02:
                    raise RuntimeError(
                        f"V23_REFERENCE_OUT_OF_LIMIT {_name}"
                    )

                _current23 = (
                    float(action_defaults[0, _aid23])
                    + float(action_scale)
                    * float(action_tensor[0, _aid23])
                )

                _requested23 = (
                    (1.0 - _blend23) * _current23
                    + _blend23 * _target23
                )

                _normalized23 = (
                    _requested23
                    - float(action_defaults[0, _aid23])
                ) / float(action_scale)

                if abs(_normalized23) > 4.0:
                    raise RuntimeError(
                        f"V23_ACTION_LIMIT {_name}"
                    )

                action_tensor[0, _aid23] = _normalized23

            if step_count % 25 == 0:
                _state23 = robot.data
                _knees23 = _state23.joint_pos.torch
                _gravity23 = _state23.projected_gravity_b.torch
                _root23 = _state23.root_link_pos_w.torch

                _tilt23 = math.degrees(math.acos(
                    float(np.clip(
                        -_gravity23[0, 2].item(),
                        -1.0, 1.0
                    ))
                ))

                _lk23 = _knees23[
                    0, runtime_names.index("left_knee_joint")
                ].item()
                _rk23 = _knees23[
                    0, runtime_names.index("right_knee_joint")
                ].item()

                print(
                    f"V23_POSTURE t={t:.2f}s "
                    f"blend={_blend23:.3f} "
                    f"left_knee={math.degrees(_lk23):.2f}deg "
                    f"right_knee={math.degrees(_rk23):.2f}deg "
                    f"height={_root23[0,2].item():.3f}m "
                    f"tilt={_tilt23:.2f}deg",
                    flush=True,
                )

        result = original_step(action_tensor)
        import time
        time.sleep(max(0.0, dt))

        step_count += 1

        if step_count == 1 or step_count % 50 == 0:
            measured_hands = (
                robot.data.joint_pos.torch[:, hand_indices]
                .detach()
            )

            errors = torch.abs(measured_hands - hand_targets)

            for rid in range(n):
                print(
                    f"WALK_DEX3_TRACKING "
                    f"t={step_count * dt:.2f}s "
                    f"G1_{rid+1}="
                    f"{float(errors[rid].max()):.4f}rad",
                    flush=True,
                )



        if step_count == 1 or step_count % 50 == 0:
            current = robot.data.root_link_pos_w.torch
            xyz = current.detach().cpu().numpy()
            print(
                f"FOUR_G1_PROGRESS t={step_count*dt:.2f}s "
                f"route_phases={route_phase}",
                flush=True,
            )
            for i in range(n):
                print(
                    f"G1_{i+1} x={xyz[i,0]:+.3f} "
                    f"y={xyz[i,1]:+.3f} "
                    f"z={xyz[i,2]:+.3f}",
                    flush=True,
                )

        return result

    env.reset = reset_four_robots
    env.step = step_four_robots

    print(
        "FOUR_G1_ONNX_READY: "
        "4 independent ONNX histories, "
        "29 actions per robot, "
        "480 observations per robot",
        flush=True,
    )

    return env



# ===== OMNIDIRECTIONAL POLICY: SIMULATION ONLY =====

import importlib.util as _importlib_util
import onnxruntime as _omni_ort

_OMNI_ROOT = Path(
    "/workspace/local/sources/g1_omnidirectional_turning"
)

_spec = _importlib_util.spec_from_file_location(
    "g1_omni_remap",
    _OMNI_ROOT / "deploy/joint_remap.py",
)
_omni_remap = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(_omni_remap)

_omni_scan = np.load(
    _OMNI_ROOT / "weights/g1_deploy_scan_mean.npy",
    allow_pickle=False,
).astype(np.float32).reshape(-1)

if _omni_scan.shape != (143,):
    raise RuntimeError("OMNI: Wrong flat-ground scan shape")

if not np.isfinite(_omni_scan).all():
    raise RuntimeError("OMNI: Invalid scan values")

_omni_session = _omni_ort.InferenceSession(
    str(_OMNI_ROOT / "weights/g1_deploy_walk_policy.onnx"),
    providers=["CPUExecutionProvider"],
)

if _omni_session.get_inputs()[0].shape[-1] != 242:
    raise RuntimeError("OMNI: Wrong observation width")

if _omni_session.get_outputs()[0].shape[-1] != 29:
    raise RuntimeError("OMNI: Wrong action width")

print(
    "OMNI_POLICY_LOADED obs=242 actions=29 "
    "action_scale=0.5",
    flush=True,
)

# ===== END OMNIDIRECTIONAL POLICY =====

gym.make = make_four_robot_environment

sys.argv = [
    "isaaclab",
    "zero_agent",
    "--task",
    TASK_ID,
    "--num_envs",
    "4",
    "--viz",
    "viser",
    "presets=newton_mjwarp",
]

print("Starting four-G1 ONNX walking simulation", flush=True)
print("Viser internal port: 8080", flush=True)
print("Each robot has its own policy history", flush=True)

if __name__ == "__main__":
    cli()
