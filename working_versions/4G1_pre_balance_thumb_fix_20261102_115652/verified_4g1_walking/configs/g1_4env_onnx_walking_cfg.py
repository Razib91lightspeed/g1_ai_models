"""Four-G1 ONNX walking simulation configuration.

Uses the verified 29-DOF G1 asset and calibrated motor gains.
Four independent robots share pretrained ONNX policy weights.

Simulation only. No physical G1 communication.
"""

from isaaclab.utils import configclass

from g1_29dof_parallel_flat_env_calibrated_cfg import (
    G129DoFParallelCalibratedEnvCfg,
)


@configclass
class G14EnvONNXWalkingCfg(G129DoFParallelCalibratedEnvCfg):

    def __post_init__(self):
        super().__post_init__()

        # Four independent G1 simulation environments.
        self.scene.num_envs = 4
        self.scene.env_spacing = 3.0

        # Match the verified walking configuration.
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.80)

        self.episode_length_s = 120.0
        self.seed = 42

        # This is a pretrained-policy diagnostic, not PPO training.
        # Use repeatable initial conditions.
        self.events.add_base_mass = None
        self.events.push_robot = None
        self.events.base_external_force_torque = None

        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

        self.events.reset_robot_joints.params[
            "position_range"
        ] = (1.0, 1.0)

        self.events.reset_robot_joints.params[
            "velocity_range"
        ] = (0.0, 0.0)

        self.observations.policy.enable_corruption = False

        # Match the original ONNX diagnostic.
        self.terminations.base_contact = None

        for name, term in vars(self.rewards).items():
            if term is not None and hasattr(term, "func"):
                setattr(self.rewards, name, None)

        # Motion commands come from the ONNX controller,
        # not from randomly sampled PPO commands.
        self.commands.base_velocity.ranges.lin_vel_x = (
            0.0, 0.0
        )
        self.commands.base_velocity.ranges.lin_vel_y = (
            0.0, 0.0
        )
        self.commands.base_velocity.ranges.ang_vel_z = (
            0.0, 0.0
        )

        assert len(self.actions.joint_pos.joint_names) == 29

        print(
            "FOUR_G1_CONFIG PASS: 4 robots, 29 actions",
            flush=True,
        )
