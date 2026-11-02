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
    "g1_isaaclab_forward_walk_030ms_60s_viser_verified"
)

CONFIG = BASE / "policy/params/deploy.yaml"
MODEL = BASE / "policy/exported/policy.onnx"

sys.path.insert(0, str(BASE))
sys.path.insert(0, str(PROJECT / "simulations/unitree_g1/parallel_rl/configs"))

from g1_onnx_adapter import G1PolicyAdapter
from g1_4env_relaxed_hands_v2_cfg import G14EnvRelaxedHandsV2Cfg

TASK_ID = "Project-G1-Four-Robot-Dex3-Thumb-V4"

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
kwargs["env_cfg_entry_point"] = G14EnvRelaxedHandsV2Cfg

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

    # Independent Dex3 hand-position targets.
    # Keep the 29-action ONNX body controller unchanged.
    hand_names = [
        name for name in runtime_names
        if "_hand_" in name
    ]

    if len(hand_names) != 14:
        raise RuntimeError(
            f"Expected 14 hand joints, found {len(hand_names)}: {hand_names}"
        )

    hand_indices = [runtime_names.index(name) for name in hand_names]

    hand_targets = (
        robot.data.default_joint_pos.torch[:, hand_indices]
        .detach().clone()
    )

    if hand_targets.shape != (n, 14):
        raise RuntimeError("Hand target shape mismatch")

    print(
        "DEX3_HAND_CONTROL_READY: 14 hand joints",
        flush=True,
    )

    for name, value in zip(
        hand_names,
        hand_targets[0].detach().cpu().tolist(),
    ):
        print(f"DEX3_TARGET {name}={value:+.4f}rad", flush=True)

    # Read the simulator's actual soft joint limits.
    limits_data = robot.data.soft_joint_pos_limits
    limits = limits_data.torch if hasattr(limits_data, "torch") else limits_data

    if tuple(limits.shape) != (n, len(runtime_names), 2):
        raise RuntimeError(
            f"Unexpected joint limit shape: {tuple(limits.shape)}"
        )

    hand_targets_base = hand_targets.clone()
    hand_targets_goal = hand_targets.clone()

    for name in hand_names:
        if not (
            name.endswith("thumb_0_joint")
            or name.endswith("thumb_1_joint")
        ):
            continue

        index = runtime_names.index(name)
        hand_index = hand_names.index(name)

        low = float(limits[0, index, 0])
        high = float(limits[0, index, 1])

        # Mirror the preferred test direction between hands.
        preferred = 0.20 if name.startswith("left_") else -0.20

        # Fall back to the opposite direction if required by limits.
        margin = 0.02
        if not (low + margin <= preferred <= high - margin):
            preferred = -preferred

        if not (low + margin <= preferred <= high - margin):
            raise RuntimeError(
                f"No valid conservative thumb target: "
                f"{name} limits=[{low:.3f}, {high:.3f}]"
            )

        hand_targets_goal[:, hand_index] = preferred

        print(
            f"DEX3_V4_LIMIT name={name} "
            f"lower={low:+.3f} upper={high:+.3f} "
            f"test_target={preferred:+.3f}",
            flush=True,
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

        if step_count == 0:
            for policy in policies:
                policy.reset()

        t = step_count * dt

        # Same velocity profile as the verified 60-second walker.
        if t < 3.0:
            forward = 0.0
        elif t < 5.0:
            forward = 0.30 * (t - 3.0) / 2.0
        elif t < 55.0:
            forward = 0.30
        elif t < 57.0:
            forward = 0.30 * (57.0 - t) / 2.0
        else:
            forward = 0.0

        command = np.array([forward, 0.0, 0.0], dtype=np.float32)
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

        # Explicitly hold relaxed Dex3 finger positions.
        # This does not alter the ONNX body action tensor.
        # Gradually move from the original hand pose to V4 targets.
        thumb_blend = min(1.0, max(0.0, t / 3.0))
        hand_targets[:] = (
            hand_targets_base
            + thumb_blend * (hand_targets_goal - hand_targets_base)
        )

        robot.set_joint_position_target(
            hand_targets,
            joint_ids=hand_indices,
        )

        result = original_step(action_tensor)
        step_count += 1

        if step_count == 1 or step_count % 50 == 0:
            actual = robot.data.joint_pos.torch.detach().cpu().numpy()
            grav = robot.data.projected_gravity_b.torch.detach().cpu().numpy()
            heights = robot.data.root_link_pos_w.torch[:, 2].detach().cpu().numpy()

            check_joints = [
                "left_shoulder_pitch_joint",
                "right_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "right_shoulder_roll_joint",
                "left_elbow_joint",
                "right_elbow_joint",
                "left_hand_thumb_0_joint",
                "left_hand_thumb_1_joint",
                "left_hand_thumb_2_joint",
                "right_hand_thumb_0_joint",
                "right_hand_thumb_1_joint",
                "right_hand_thumb_2_joint",
            ]

            for robot_id in range(n):
                tilt = math.degrees(math.acos(
                    float(np.clip(-grav[robot_id, 2], -1.0, 1.0))
                ))
                values = " ".join(
                    f"{name}={actual[robot_id, runtime_names.index(name)]:+.3f}"
                    for name in check_joints
                )
                print(
                    f"G1_POSTURE id={robot_id+1} "
                    f"t={step_count*dt:.2f}s "
                    f"height={heights[robot_id]:.3f}m "
                    f"tilt={tilt:.2f}deg "
                    + values,
                    flush=True,
                )


        if step_count == 1 or step_count % 250 == 0:
            measured_hands = (
                robot.data.joint_pos.torch[:, hand_indices]
                .detach().cpu()
            )
            error = (
                measured_hands - hand_targets.detach().cpu()
            ).abs().max(dim=1).values

            print(
                "DEX3_TRACKING_ERROR "
                + " ".join(
                    f"G1_{i+1}={float(error[i]):.4f}rad"
                    for i in range(n)
                ),
                flush=True,
            )

        if step_count == 1 or step_count % 50 == 0:
            current = robot.data.root_link_pos_w.torch
            xyz = current.detach().cpu().numpy()
            print(
                f"FOUR_G1_PROGRESS t={step_count*dt:.2f}s "
                f"command={forward:.3f}m/s",
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
