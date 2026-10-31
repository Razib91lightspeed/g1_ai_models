"""Unitree G1 29-DOF parallel PPO environment.

Experimental training configuration for Isaac Lab 3.0/Newton.

Uses the same 29-body-joint asset and initial upper-body
pose as the verified 60-second ONNX walking configuration.

Does not modify or control the physical Unitree G1.
"""

from isaaclab.utils import configclass, replace
from isaaclab_assets import G1_29DOF_CFG
from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg


BODY_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


@configclass
class G129DoFParallelFlatEnvCfg(G1FlatEnvCfg):
    """Parallel training environment for G1 body joints."""

    def __post_init__(self):
        super().__post_init__()

        # Preserve the verified 29-DOF robot asset.
        self.scene.robot = replace(
            G1_29DOF_CFG,
            prim_path="{ENV_REGEX_NS}/Robot",
        )

        # Initial parallel experiment.
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5

        # Newton contact capacity.
        self.sim.physics.newton_mjwarp.solver_cfg.nconmax = 64

        # Verified walking controller's initial body height.
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.80)

        # Match the verified upper-body resting pose.
        self.scene.robot.init_state.joint_pos.update({
            "left_shoulder_pitch_joint": 0.30,
            "right_shoulder_pitch_joint": 0.30,
            "left_shoulder_roll_joint": 0.25,
            "right_shoulder_roll_joint": -0.25,
            "left_elbow_joint": 0.97,
            "right_elbow_joint": 0.97,
            "left_wrist_roll_joint": 0.15,
            "right_wrist_roll_joint": -0.15,
        })

        # Explicit 29-joint action selection.
        self.actions.joint_pos.joint_names = BODY_JOINTS
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.use_default_offset = True
        self.actions.joint_pos.preserve_order = True

        # Training commands: begin with moderate velocities.
        self.commands.base_velocity.ranges.lin_vel_x = (
            0.0, 0.30
        )
        self.commands.base_velocity.ranges.lin_vel_y = (
            -0.10, 0.10
        )
        self.commands.base_velocity.ranges.ang_vel_z = (
            -0.20, 0.20
        )

        # Remove rewards associated with other G1 embodiments.
        self.rewards.joint_deviation_fingers = None
        self.rewards.joint_deviation_torso = None

        # Deterministic seed for initial comparisons.
        self.seed = 42


import re
from pathlib import Path
import yaml


@configclass
class G129DoFParallelCalibratedEnvCfg(G129DoFParallelFlatEnvCfg):
    """29-body-joint PPO configuration with verified motor gains."""

    def __post_init__(self):
        super().__post_init__()

        deployment_file = Path(
            "/workspace/local/working_versions/"
            "g1_isaaclab_forward_walk_030ms_60s_viser_verified/"
            "policy/params/deploy.yaml"
        )

        if not deployment_file.is_file():
            raise FileNotFoundError(deployment_file)

        with deployment_file.open(encoding="utf-8") as file:
            deploy = yaml.safe_load(file)

        ids = deploy["joint_ids_map"]
        kp = deploy["stiffness"]
        kd = deploy["damping"]

        if not (
            len(ids) == len(kp) == len(kd) == 29
            and sorted(ids) == list(range(29))
        ):
            raise RuntimeError("Invalid 29-joint deployment mapping")

        # BODY_JOINTS follows the original motor order.
        # Reproduce the verified controller's gain indexing.
        policy_names = [BODY_JOINTS[index] for index in ids]

        expected = set(policy_names)
        assigned = set()

        for group_name, actuator in self.scene.robot.actuators.items():
            group_joints = [
                name for name in policy_names
                if any(
                    re.fullmatch(pattern, name)
                    for pattern in actuator.joint_names_expr
                )
            ]

            if not group_joints:
                continue

            group_kp = {}
            group_kd = {}

            for index, name in enumerate(policy_names):
                if name not in group_joints:
                    continue

                if name in assigned:
                    raise RuntimeError(
                        f"Duplicate actuator assignment: {name}"
                    )

                assigned.add(name)
                group_kp["^" + re.escape(name) + "$"] = float(
                    kp[ids[index]]
                )
                group_kd["^" + re.escape(name) + "$"] = float(
                    kd[ids[index]]
                )

            actuator.stiffness = group_kp
            actuator.damping = group_kd

            print(
                f"GAIN_MATCH group={group_name} "
                f"joints={len(group_joints)}",
                flush=True,
            )

        if assigned != expected:
            raise RuntimeError(
                "Incomplete motor gain mapping: "
                f"{sorted(expected - assigned)}"
            )

        # The built-in arm reward expects elbow_pitch/elbow_roll.
        # The 29-DOF G1 has left_elbow_joint/right_elbow_joint.
        # Disable this incompatible term for the smoke test.
        self.rewards.joint_deviation_arms = None

        print("GAIN_MATCH PASS: 29 body joints", flush=True)
