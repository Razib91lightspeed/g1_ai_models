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

CONFIG = "/workspace/local/standing_wave_test/isaaclab_simulation/policy/params/deploy.yaml"
MODEL = "/workspace/local/standing_wave_test/isaaclab_simulation/policy/exported/policy.onnx"

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

        # Match the working MuJoCo controller's joint PD gains.
        # Preserve original actuator types and motor limits.
        import re
        import yaml

        with open(CONFIG, encoding="utf-8") as f:
            deploy = yaml.safe_load(f)

        ids = deploy["joint_ids_map"]
        kp = deploy["stiffness"]
        kd = deploy["damping"]

        if not (
            len(ids) == len(kp) == len(kd) == 29
            and sorted(ids) == list(range(29))
        ):
            raise RuntimeError("Invalid deploy.yaml joint mapping")

        policy_names = [
            MOTOR_NAMES[i] + "_joint" for i in ids
        ]

        expected = set(policy_names)
        assigned = set()

        for group_name, actuator in self.scene.robot.actuators.items():
            group_joints = [
                name for name in policy_names
                if any(
                    re.fullmatch(expr, name)
                    for expr in actuator.joint_names_expr
                )
            ]

            if not group_joints:
                continue

            # Never silently configure only part of a motor group.
            # A mixed body/hand group must be reviewed separately.
            all_matching_body = set(group_joints)

            group_kp = {}
            group_kd = {}

            for i, name in enumerate(policy_names):
                if name not in all_matching_body:
                    continue

                if name in assigned:
                    raise RuntimeError(
                        f"Joint belongs to multiple groups: {name}"
                    )

                assigned.add(name)
                group_kp["^" + re.escape(name) + "$"] = float(kp[ids[i]])
                group_kd["^" + re.escape(name) + "$"] = float(kd[ids[i]])

            actuator.stiffness = group_kp
            actuator.damping = group_kd

            print(
                f"GAIN_MATCH group={group_name} "
                f"joints={len(group_joints)}",
                flush=True,
            )

        if assigned != expected:
            raise RuntimeError(
                f"Actuator mapping incomplete: {sorted(expected - assigned)}"
            )

        print("GAIN_MATCH PASS: 29 body joints", flush=True)

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
    action_term = sim_env.action_manager.get_term("joint_pos")
    action_names = list(action_term._joint_names)

    configured_names = list(
        sim_env.cfg.actions.joint_pos.joint_names
    )

    print("=== ACTION ORDER VALIDATION ===", flush=True)

    for i, name in enumerate(action_names):
        print(
            f"Action slot {i:02d}: {name}",
            flush=True,
        )

    if action_names != configured_names:
        raise RuntimeError(
            "ACTION_ORDER_MISMATCH: resolved action order "
            "differs from configured order"
        )

    print(
        "PASS: Resolved Newton action order matches configuration",
        flush=True,
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

    # Validate the actual policy-order reference pose.
    newton_policy_defaults = newton_defaults[state_indices]
    pose_error = np.abs(
        newton_policy_defaults - policy.offset
    )

    print("=== INITIAL POSE COMPATIBILITY ===", flush=True)
    print(
        "Maximum reference pose difference:",
        f"{float(np.max(pose_error)):.4f} rad",
        flush=True,
    )

    for i, error in enumerate(pose_error):
        if error > 0.10:
            print(
                f"POSE_DIFFERENCE {policy_names[i]} "
                f"Newton={newton_policy_defaults[i]:+.4f} "
                f"ONNX={policy.offset[i]:+.4f} "
                f"Difference={error:.4f}",
                flush=True,
            )

    if np.max(pose_error) > 0.35:
        raise RuntimeError(
            "POSE_MISMATCH: Reference positions differ by "
            "more than 0.35 rad. Controller not started."
        )

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


    if os.environ.get("G1_DRY_RUN") == "1":
        print(
            "=== NEWTON ONNX FIRST-ACTION DRY RUN ===",
            flush=True,
        )

        # Initialize the environment before reading joint states.
        # Do not execute any ONNX-generated action.
        env.reset()
        policy.reset()

        state = robot.data
        positions = (
            state.joint_pos.torch[0, state_indices]
            .detach().cpu().numpy()
        )
        velocities = (
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
        root = (
            state.root_link_pos_w.torch[0]
            .detach().cpu().numpy()
        )

        print("Root position:", root.tolist(), flush=True)
        print("Angular velocity:", angular_velocity.tolist(), flush=True)
        print("Projected gravity:", gravity.tolist(), flush=True)

        initial_error = np.abs(positions - policy.offset)
        print(
            "Maximum initial joint position error:",
            float(initial_error.max()),
            flush=True,
        )

        policy.reset()

        obs, raw, targets = policy.step(
            angular_velocity=angular_velocity,
            projected_gravity=gravity,
            velocity_command=np.zeros(3, dtype=np.float32),
            joint_position=positions,
            joint_velocity=velocities,
        )

        print("Observation shape:", obs.shape, flush=True)
        print(
            "Observation maximum magnitude:",
            float(np.max(np.abs(obs))),
            flush=True,
        )
        print(
            "Raw action maximum magnitude:",
            float(np.max(np.abs(raw))),
            flush=True,
        )

        for i, name in enumerate(policy_names):
            print(
                f"{i:02d} {name:32s} "
                f"actual={positions[i]:+.4f} "
                f"reference={policy.offset[i]:+.4f} "
                f"action={raw[i]:+.4f} "
                f"target={targets[i]:+.4f}",
                flush=True,
            )

        print("PASS: First ONNX prediction completed", flush=True)
        print("No environment steps or actions applied.", flush=True)
        raise SystemExit(0)

    original_step = env.step
    original_reset = env.reset

    step_count = 0
    initial_xy = None
    time_step = float(
        sim_env.cfg.sim.dt * sim_env.cfg.decimation
    )

    if abs(time_step - float(policy.cfg["step_dt"])) > 1e-6:
        raise RuntimeError(
            f"Control dt mismatch: Newton={time_step}, "
            f"ONNX={policy.cfg['step_dt']}"
        )

    def reset_with_policy(*reset_args, **reset_kwargs):
        nonlocal step_count, initial_xy

        initial_xy = None
        policy.reset()
        step_count = 0

        return original_reset(*reset_args, **reset_kwargs)

    def step_with_policy(_ignored_zero_action):
        nonlocal step_count, initial_xy

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

        # Store the initial horizontal position.
        current_xy = (
            state.root_link_pos_w.torch[0, :2]
            .detach().cpu().numpy().copy()
        )

        if initial_xy is None:
            initial_xy = current_xy.copy()

        # Fail closed if the robot has already fallen.
        position = (
            state.root_link_pos_w.torch[0]
            .detach().cpu().numpy()
        )

        tilt = math.degrees(
            math.acos(float(np.clip(-gravity[2], -1.0, 1.0)))
        )

        if float(position[2]) < 0.55 or tilt > 30.0:
            raise RuntimeError(
                f"G1_FALL_STOP: t={step_count * time_step:.3f}s "
                f"height={float(position[2]):.3f}m "
                f"tilt={tilt:.1f}deg"
            )

        # G1 FORWARD WALKING EXPERIMENT
        # ONNX velocity command: forward, lateral, yaw.
        # Smooth acceleration and deceleration.
        t = step_count * time_step

        if t < 3.0:
            forward_speed = 0.0
        elif t < 5.0:
            forward_speed = 0.30 * (t - 3.0) / 2.0
        elif t < 15.0:
            forward_speed = 0.30
        elif t < 17.0:
            forward_speed = 0.30 * (17.0 - t) / 2.0
        else:
            forward_speed = 0.0

        command = np.array(
            [forward_speed, 0.0, 0.0],
            dtype=np.float32,
        )

        if step_count % 50 == 0:
            print(
                f"G1_WALK_COMMAND t={t:.2f}s "
                f"forward={forward_speed:.3f}m/s "
                f"lateral=0.000m/s yaw=0.000rad/s",
                flush=True,
            )

        if step_count < 10:
            print(
                "G1_FEEDBACK "
                f"t={step_count * time_step:.3f}s "
                f"angular_velocity={np.round(angular_velocity, 4).tolist()} "
                f"gravity={np.round(gravity, 4).tolist()} "
                f"max_joint_velocity={float(np.max(np.abs(joint_vel))):.3f}",
                flush=True,
            )

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
        unclipped_actions = normalized_actions.copy()

        normalized_actions = np.clip(
            normalized_actions, -4.0, 4.0
        ).astype(np.float32)

        if step_count % 50 == 0:
            clipped_count = int(np.sum(
                np.abs(unclipped_actions) > 4.0
            ))

            print(
                f"G1_ACTION_CHECK "
                f"t={step_count * time_step:.2f}s "
                f"raw_max={float(np.max(np.abs(raw))):.4f} "
                f"normalized_max={float(np.max(np.abs(unclipped_actions))):.4f} "
                f"clipped_joints={clipped_count}",
                flush=True,
            )

        if not np.isfinite(normalized_actions).all():
            raise RuntimeError("Nonfinite normalized joint actions")

        action_tensor = torch.as_tensor(
            normalized_actions,
            device=sim_env.device,
        ).reshape(1, 29)


        # Diagnostic A/B test: hold the configured joint defaults.
        # ONNX inference may still run for observation diagnostics,
        # but none of its output reaches the robot.
        if os.environ.get("G1_HOLD_DEFAULT") == "1":
            action_tensor = torch.zeros_like(action_tensor)

            if step_count == 0:
                print(
                    "G1_HOLD_DEFAULT ACTIVE: "
                    "ONNX actions are NOT applied",
                    flush=True,
                )




        # Standing-only test: no arm overrides.
        test_t = step_count * time_step

        if test_t >= 20.0:
            print(
                "G1_WALK_TEST_COMPLETE "
                "duration=20.0s "
                "result=simulation_reached_end",
                flush=True,
            )
            raise SystemExit(0)

        output = original_step(action_tensor)

        if step_count % 50 == 0:
            current_pos = (
                robot.data.root_link_pos_w.torch[0]
                .detach().cpu().numpy()
            )

            velocity = (
                robot.data.root_link_lin_vel_w.torch[0]
                .detach().cpu().numpy()
            )

            displacement = np.linalg.norm(
                current_pos[:2] - initial_xy
            )

            speed = np.linalg.norm(velocity[:2])

            print(
                f"G1_POSITION t={test_t:.2f}s "
                f"x={current_pos[0]:+.3f} "
                f"y={current_pos[1]:+.3f} "
                f"drift={displacement:.3f}m "
                f"horizontal_speed={speed:.3f}m/s",
                flush=True,
            )

        if step_count % 5 == 0:
            gravity = robot.data.projected_gravity_b
            if hasattr(gravity, "torch"):
                gravity = gravity.torch

            g = gravity[0].detach().cpu().numpy()
            tilt = math.degrees(
                math.acos(max(-1.0, min(1.0, float(-g[2]))))
            )

            position = robot.data.root_link_pos_w
            if hasattr(position, "torch"):
                position = position.torch

            height = float(position[0, 2].item())

            if test_t > 1.0 and (tilt > 10.0 or height < 0.74):
                raise RuntimeError(
                    f"G1_ARM_SAFETY_STOP t={test_t:.2f} "
                    f"tilt={tilt:.1f} height={height:.3f}"
                )

        # Measure actual arm joints, not just the requested action.
        if step_count % 50 == 0:
            joints = (
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_elbow_joint",
            )
            actual = robot.data.joint_pos.torch[0]
            measured = []

            for name in joints:
                idx = runtime_names.index(name)
                measured.append(
                    f"{name}={float(actual[idx].item()):+.3f}"
                )

            print(
                f"G1_ARM_ACTUAL t={test_t:.2f}s "
                + " ".join(measured),
                flush=True,
            )



        if step_count < 10 or step_count % 50 == 0:
            sensor = sim_env.scene["contact_forces"]
            force_values = sensor.data.net_forces_w.torch[0]
            magnitudes = torch.linalg.norm(force_values, dim=-1)

            total_force = torch.sum(force_values, dim=0)
            nonzero = torch.nonzero(
                magnitudes > 0.01
            ).flatten().tolist()

            print(
                f"G1_CONTACT_SUMMARY "
                f"t={(step_count + 1) * time_step:.3f}s "
                f"active_bodies={len(nonzero)} "
                f"force_sum={total_force.detach().cpu().tolist()}",
                flush=True,
            )

            for index in nonzero:
                print(
                    f"G1_CONTACT_ACTIVE "
                    f"body={sensor.body_names[index]} "
                    f"force={force_values[index].detach().cpu().tolist()}",
                    flush=True,
                )

            for index, name in enumerate(robot.body_names):
                if "ankle_roll_link" in name:
                    height = float(
                        robot.data.body_pos_w.torch[0, index, 2]
                        .detach().cpu()
                    )
                    print(
                        f"G1_FOOT_HEIGHT "
                        f"t={(step_count + 1) * time_step:.3f}s "
                        f"body={name} z={height:.4f}m",
                        flush=True,
                    )


        # G1_WALK_ANKLE_TORQUE_DIAGNOSTIC
        if step_count % 50 == 0:
            computed = robot.data.computed_torque.torch[0].detach().cpu().numpy()
            applied = robot.data.applied_torque.torch[0].detach().cpu().numpy()
            limits = robot.data.joint_effort_limits.torch[0].detach().cpu().numpy()

            for name in (
                "left_ankle_pitch_joint",
                "right_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_ankle_roll_joint",
            ):
                idx = runtime_names.index(name)
                print(
                    f"G1_WALK_TORQUE t={(step_count + 1) * time_step:.3f}s "
                    f"joint={name} "
                    f"computed={computed[idx]:+.3f}Nm "
                    f"applied={applied[idx]:+.3f}Nm "
                    f"limit={limits[idx]:.3f}Nm",
                    flush=True,
                )

        # Read the existing Isaac Lab contact sensor.
        # This is diagnostic only; it does not alter physics.
        if step_count < 10:
            sensor = sim_env.scene["contact_forces"]
            forces = sensor.data.net_forces_w.torch[0]

            body_names = list(sensor.body_names)

            for index, name in enumerate(body_names):
                if "ankle_roll_link" not in name:
                    continue

                force = (
                    forces[index]
                    .detach().cpu().numpy()
                )

                magnitude = float(
                    np.linalg.norm(force)
                )

                print(
                    f"G1_CONTACT "
                    f"t={(step_count + 1) * time_step:.3f}s "
                    f"body={name} "
                    f"force={np.round(force, 3).tolist()} "
                    f"magnitude={magnitude:.3f}N",
                    flush=True,
                )


        # Compare commanded and measured positions AFTER physics.
        if step_count < 10 or step_count % 50 == 0:
            measured = (
                robot.data.joint_pos.torch[0, state_indices]
                .detach().cpu().numpy()
            )

            joints_to_track = (
                "left_hip_pitch_joint",
                "right_hip_pitch_joint",
                "left_knee_joint",
                "right_knee_joint",
                "left_ankle_pitch_joint",
                "right_ankle_pitch_joint",
            )

            for name in joints_to_track:
                index = policy_names.index(name)

                print(
                    f"G1_JOINT t={(step_count + 1) * time_step:.3f}s "
                    f"name={name} "
                    f"target={targets[index]:+.4f} "
                    f"actual={measured[index]:+.4f} "
                    f"error={targets[index] - measured[index]:+.4f}",
                    flush=True,
                )

        step_count += 1

        if step_count <= 10 or step_count % 10 == 0:
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
    "--viz",
    "viser",
    "presets=newton_mjwarp",
]

print("Starting G1 Newton ONNX experiment", flush=True)
print("Viewer: Viser on port 8080", flush=True)
print("Policy: existing Unitree G1 29-DoF ONNX", flush=True)
print("Initial command: stand still", flush=True)


if os.environ.get("G1_PREFLIGHT") == "1":
    import re
    import yaml

    cfg = ONNXEnvCfg()
    cfg.validate()

    with open(CONFIG, encoding="utf-8") as f:
        deploy = yaml.safe_load(f)

    ids = deploy["joint_ids_map"]
    kp = deploy["stiffness"]
    kd = deploy["damping"]

    print("=== EFFECTIVE ONNX ACTUATOR PREFLIGHT ===", flush=True)

    for index, motor in enumerate(ids):
        name = MOTOR_NAMES[motor] + "_joint"

        matches = []

        for group_name, actuator in cfg.scene.robot.actuators.items():
            if any(
                re.fullmatch(pattern, name)
                for pattern in actuator.joint_names_expr
            ):
                matches.append((group_name, actuator))

        if len(matches) != 1:
            raise RuntimeError(
                f"{name}: expected one actuator, got {len(matches)}"
            )

        group_name, actuator = matches[0]

        def gain_value(values):
            if isinstance(values, dict):
                found = [
                    value for pattern, value in values.items()
                    if re.fullmatch(pattern, name)
                ]
                if len(found) != 1:
                    raise RuntimeError(
                        f"{name}: ambiguous gain mapping {found}"
                    )
                return float(found[0])
            return float(values)

        actual_kp = gain_value(actuator.stiffness)
        actual_kd = gain_value(actuator.damping)

        expected_kp = float(kp[motor])
        expected_kd = float(kd[motor])

        if abs(actual_kp - expected_kp) > 0.001:
            raise RuntimeError(f"{name}: stiffness mismatch")

        if abs(actual_kd - expected_kd) > 0.001:
            raise RuntimeError(f"{name}: damping mismatch")

        print(
            f"{index:02d} {name:32s} "
            f"{group_name:10s} "
            f"Kp={actual_kp:.1f} "
            f"Kd={actual_kd:.1f}",
            flush=True,
        )

    print("PASS: 29 ONNX actuator gains match deploy.yaml")
    print("No physics started.")
    sys.exit(0)


cli()
