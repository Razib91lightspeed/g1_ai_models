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

    def wrap_angle(a):
        return math.atan2(math.sin(a), math.cos(a))

    def yaw_from_quat(q):
        x, y, z, w = [float(v) for v in q]
        return math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z)
        )

    original_step = env.step

    original_reset = env.reset
    step_count = 0

    def reset_four_robots(*args, **kwargs):
        nonlocal step_count
        step_count = 0
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
                route_turn_accum[rid] = 0.0
                route_last_yaw[rid] = None
                route_success[rid] = False

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

                if remaining <= 0.30:
                    route_phase[i] = "TURN"
                    route_turn_accum[i] = 0.0
                    route_last_yaw[i] = yaw

                    print(
                        f"ROUTE_EVENT G1_{i+1} OUT_COMPLETE "
                        f"t={t:.2f}s "
                        f"distance={forward_distance:.3f}m "
                        f"lateral={lateral_error:+.3f}m",
                        flush=True
                    )

                elif t < 3.0:
                    vx = 0.0

                elif t < 5.0:
                    vx = 0.32 * (t - 3.0) / 2.0

                else:
                    vx = min(
                        0.32,
                        max(0.16, 0.65 * remaining)
                    )

                # Preserve the straight-outbound test behavior.
                wz = 0.0

            elif phase == "TURN":
                delta = wrap_angle(yaw - route_last_yaw[i])
                route_turn_accum[i] += delta
                route_last_yaw[i] = yaw

                turn_remaining = math.pi - route_turn_accum[i]

                if turn_remaining <= math.radians(5):
                    route_phase[i] = "ALIGN"

                    print(
                        f"ROUTE_EVENT G1_{i+1} TURN_COMPLETE "
                        f"t={t:.2f}s "
                        f"angle={math.degrees(route_turn_accum[i]):.1f}deg",
                        flush=True
                    )

                else:
                    # Walking turn using the verified angular command range.
                    # No root pose manipulation or teleportation.
                    vx = 0.30
                    wz = min(
                        0.20,
                        max(0.08, 0.65 * turn_remaining)
                    )

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

            if obs.shape != (1, 480) or raw.shape != (29,):
                raise RuntimeError(f"G1_{i+1}: Invalid ONNX dimensions")

            ordered_targets = targets[action_indices]
            normalized = (
                ordered_targets - action_defaults[i]
            ) / action_scale

            if not np.isfinite(normalized).all():
                raise RuntimeError(f"G1_{i+1}: Nonfinite actions")

            batched_actions[i] = np.clip(
                normalized, -4.0, 4.0
            ).astype(np.float32)

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
