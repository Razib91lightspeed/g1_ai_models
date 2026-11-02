"""Unitree G1 29-DoF ONNX controller in Isaac Lab 3.0 / Newton.

Simulation only.
Reuses the working Isaac Lab zero-agent launcher for lifecycle
management, replacing its action source with the existing ONNX policy.

Does not connect to DDS, Unitree SDK, or the physical robot.
"""

import os
import sys
import math
import numpy as np
import torch
import gymnasium as gym

sys.path.insert(0, "/workspace/local/isaac_lab/onnxdeps")
sys.path.insert(0, "/workspace/g1_tasks")

import isaaclab_tasks
from isaaclab.cli import cli

from g1_29dof_env_cfg import G129DoFEnvCfg
from g1_onnx_adapter import G1PolicyAdapter


TASK_ID = "Project-G1-29DoF-ONNX"

CONFIG = "/workspace/g1_policy/params/deploy.yaml"
MODEL = "/workspace/g1_policy/exported/policy.onnx"

# Original MuJoCo actuator order.
# The deploy.yaml joint_ids_map indexes this list.
MOTOR_NAMES = [
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]


class ONNXEnvCfg(G129DoFEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Disable original locomotion reward/termination terms
        # that refer to a different G1 embodiment.
        self.terminations.base_contact = None

        for name, term in vars(self.rewards).items():
            if term is not None and hasattr(term, "func"):
                setattr(self.rewards, name, None)

        # Static standing command for the initial experiment.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


original = gym.spec("Isaac-Velocity-Flat-G1")
kwargs = dict(original.kwargs)
kwargs["env_cfg_entry_point"] = ONNXEnvCfg

gym.register(
    id=TASK_ID,
    entry_point=original.entry_point,
    disable_env_checker=True,
    kwargs=kwargs,
)

original_gym_make = gym.make


def make_onnx_environment(*args, **kwargs):
    env = original_gym_make(*args, **kwargs)
    sim_env = env.unwrapped
    robot = sim_env.scene["robot"]

    policy = G1PolicyAdapter(CONFIG, MODEL)

    # Policy-order names: use the original MuJoCo motor mapping.
    policy_names = [
        MOTOR_NAMES[int(i)] + "_joint"
        for i in policy.ids
    ]

    runtime_names = list(robot.joint_names)
    action_names = list(
        sim_env.cfg.actions.joint_pos.joint_names
    )

    if len(runtime_names) != 43:
        raise RuntimeError(
            f"Expected 43 Newton joints; got {len(runtime_names)}"
        )

    if len(action_names) != 29:
        raise RuntimeError(
            f"Expected 29 action joints; got {len(action_names)}"
        )

    if len(set(policy_names)) != 29:
        raise RuntimeError("Duplicate policy joint name")

    missing = set(policy_names) - set(runtime_names)
    if missing:
        raise RuntimeError(
            f"Newton model missing policy joints: {sorted(missing)}"
        )

    missing_actions = set(policy_names) - set(action_names)
    if missing_actions:
        raise RuntimeError(
            f"Action mapping missing joints: {sorted(missing_actions)}"
        )

    state_indices = [
        runtime_names.index(name)
        for name in policy_names
    ]

    # Convert action-vector order into policy order.
    action_indices = [
        policy_names.index(name)
        for name in action_names
    ]

    # The Isaac Lab action is a normalized offset from its
    # default joint position, with scale configured in env_cfg.
    newton_defaults = (
        robot.data.default_joint_pos.torch[0]
        .detach().cpu().numpy()
    )

    action_defaults = newton_defaults[
        [runtime_names.index(name) for name in action_names]
    ].copy()

    action_scale = float(
        sim_env.cfg.actions.joint_pos.scale
    )

    if action_scale <= 0:
        raise RuntimeError("Invalid Isaac Lab action scale")

    print("\n=== G1 NEWTON ONNX JOINT MAPPING ===", flush=True)

    for i, name in enumerate(policy_names):
        print(
            f"Policy {i:02d} -> "
            f"Newton {state_indices[i]:02d}: {name}",
            flush=True,
        )

    print(
        "ONNX control interval:",
        policy.cfg["step_dt"],
        flush=True,
    )

    print(
        "Isaac Lab action scale:",
        action_scale,
        flush=True,
    )

    print(
        "Joint action count:",
        len(action_indices),
        flush=True,
    )

    # This diagnostic deliberately preserves the Newton
    # actuator gains. Their compatibility is not yet validated.
    print(
        "Actuator gains: Isaac Lab defaults (not yet matched)",
        flush=True,
    )

    original_step = env.step
    original_reset = env.reset

    step_count = 0
    time_step = float(
        sim_env.cfg.sim.dt * sim_env.cfg.decimation
    )

    if abs(time_step - float(policy.cfg["step_dt"])) > 1e-6:
        raise RuntimeError(
            f"Control dt mismatch: Newton={time_step}, "
            f"ONNX={policy.cfg['step_dt']}"
        )

    def reset_with_policy(*reset_args, **reset_kwargs):
        nonlocal step_count

        policy.reset()
        step_count = 0

        return original_reset(*reset_args, **reset_kwargs)

    def step_with_policy(_ignored_zero_action):
        nonlocal step_count

        state = robot.data

        joint_pos = (
            state.joint_pos.torch[0, state_indices]
            .detach().cpu().numpy()
        )

        joint_vel = (
            state.joint_vel.torch[0, state_indices]
            .detach().cpu().numpy()
        )

        angular_velocity = (
            state.root_link_ang_vel_b.torch[0]
            .detach().cpu().numpy()
        )

        gravity = (
            state.projected_gravity_b.torch[0]
            .detach().cpu().numpy()
        )

        # First experiment: standing, zero velocity command.
        command = np.zeros(3, dtype=np.float32)

        obs, raw, targets = policy.step(
            angular_velocity=angular_velocity,
            projected_gravity=gravity,
            velocity_command=command,
            joint_position=joint_pos,
            joint_velocity=joint_vel,
        )

        # Adapter returns joint targets in policy order.
        # Reorder into Isaac Lab action term order.
        ordered_targets = targets[action_indices]

        normalized_actions = (
            ordered_targets - action_defaults
        ) / action_scale

        # Conservative test limit on normalized offsets.
        # This is a diagnostic bound, not a validated policy limit.
        normalized_actions = np.clip(
            normalized_actions, -4.0, 4.0
        ).astype(np.float32)

        if not np.isfinite(normalized_actions).all():
            raise RuntimeError("Nonfinite normalized joint actions")

        action_tensor = torch.as_tensor(
            normalized_actions,
            device=sim_env.device,
        ).reshape(1, 29)

        output = original_step(action_tensor)
        step_count += 1

        if step_count == 1 or step_count % 50 == 0:
            pos = (
                state.root_link_pos_w.torch[0]
                .detach().cpu().numpy()
            )

            gravity_now = (
                state.projected_gravity_b.torch[0]
                .detach().cpu().numpy()
            )

            tilt_deg = math.degrees(
                math.acos(
                    float(np.clip(-gravity_now[2], -1, 1))
                )
            )

            height = float(pos[2])

            # Diagnostic threshold only.
            fallen = height < 0.45 or tilt_deg > 45.0

            print(
                f"G1_MONITOR "
                f"t={step_count * time_step:.2f}s "
                f"height={height:.3f}m "
                f"tilt={tilt_deg:.1f}deg "
                f"max_action={np.max(np.abs(raw)):.2f} "
                f"fallen={fallen}",
                flush=True,
            )

        return output

    env.reset = reset_with_policy
    env.step = step_with_policy

    return env


gym.make = make_onnx_environment

sys.argv = [
    "isaaclab",
    "zero_agent",
    "--task",
    TASK_ID,
    "--num_envs",
    "1",
    "--max_steps",
    "750",
    "--viz",
    "viser",
    "presets=newton_mjwarp",
]

print("Starting G1 Newton ONNX experiment", flush=True)
print("Viewer: container 8080 / Spark 8081", flush=True)
print("Policy: existing Unitree G1 29-DoF ONNX", flush=True)
print("Initial command: stand still", flush=True)

cli()
