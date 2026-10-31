"""Project-local G1 standing-test configuration.

Uses the 29 body joints of the Unitree G1 USD asset.
The extra hand joints are not included in the action space.
No pretrained locomotion policy is loaded.
"""

from isaaclab.utils import configclass, replace
from isaaclab_assets import G1_29DOF_CFG
from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import (
    G1FlatEnvCfg,
)

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
class G129DoFEnvCfg(G1FlatEnvCfg):
    """Standing-test configuration for G1 body-joint control."""

    def __post_init__(self):
        super().__post_init__()

        # Use Isaac Lab's 29-body-joint G1 asset.
        self.scene.robot = replace(
            G1_29DOF_CFG,
            prim_path="{ENV_REGEX_NS}/Robot",
        )

        # Newton contact capacity.
        self.sim.physics.newton_mjwarp.solver_cfg.nconmax = 64

        # One robot; same 50 Hz control rate.
        self.scene.num_envs = 1
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.80)
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.80)
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.80)

        # Match the upper-body resting pose from the
        # working Unitree G1 29-DoF locomotion policy.
        #
        # Leg defaults already match the verified policy
        # reference values and remain unchanged.
        #
        # This updates the robot's configuration defaults,
        # which are used by the joint-position action offset.
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


        # Deterministic standing diagnostic.
        self.episode_length_s = 120.0
        self.seed = 42

        # Disable disturbances and random mass changes.
        self.events.add_base_mass = None
        self.events.push_robot = None
        self.events.base_external_force_torque = None

        # Reset at the configured position and orientation,
        # with zero linear and angular velocity.
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

        # Use exact default joint positions at reset.
        self.events.reset_robot_joints.params["position_range"] = (
            1.0, 1.0
        )
        self.events.reset_robot_joints.params["velocity_range"] = (
            0.0, 0.0
        )

        # Disable observation noise for this diagnostic.
        self.observations.policy.enable_corruption = False


        # Only the 29 body joints receive actions.
        self.actions.joint_pos.joint_names = BODY_JOINTS
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.use_default_offset = True
        self.actions.joint_pos.preserve_order = True

        # Zero commanded translational/rotational velocity.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # Disable reward terms tied to the other G1 embodiment.
        self.rewards.joint_deviation_fingers = None
        self.rewards.joint_deviation_torso = None

        # No reward evaluation is needed for a standing diagnostic.
        # Remaining reward terms will be inspected before running.
