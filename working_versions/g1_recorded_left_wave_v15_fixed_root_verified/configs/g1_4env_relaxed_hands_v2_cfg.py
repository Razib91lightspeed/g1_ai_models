"""Four G1 robots with experimental relaxed Dex3 hands.

Preserves the verified 29-joint ONNX walking configuration.
Finger angles are preliminary and require visual/runtime validation.
"""

from isaaclab.utils import configclass
from g1_4env_onnx_walking_cfg import G14EnvONNXWalkingCfg


@configclass
class G14EnvRelaxedHandsV2Cfg(G14EnvONNXWalkingCfg):

    def __post_init__(self):
        super().__post_init__()

        hand_pose = {}

        # Mirrored Dex3 joints have different rotation signs.
        hand_pose = {
            "left_hand_index_0_joint": -0.15,
            "left_hand_index_1_joint": -0.10,
            "left_hand_middle_0_joint": -0.15,
            "left_hand_middle_1_joint": -0.10,
            "left_hand_thumb_0_joint": 0.0,
            "left_hand_thumb_1_joint": 0.0,
            "left_hand_thumb_2_joint": 0.10,

            "right_hand_index_0_joint": 0.15,
            "right_hand_index_1_joint": 0.10,
            "right_hand_middle_0_joint": 0.15,
            "right_hand_middle_1_joint": 0.10,
            "right_hand_thumb_0_joint": 0.0,
            "right_hand_thumb_1_joint": 0.0,
            "right_hand_thumb_2_joint": -0.10,
        }

        self.scene.robot.init_state.joint_pos.update(hand_pose)

        assert len(hand_pose) == 14
        assert self.scene.num_envs == 4

        print(
            "RELAXED_HANDS_CFG: "
            "14 finger defaults configured for four robots",
            flush=True,
        )
        # Fixed-base Dex3 visual calibration only.
        self.scene.robot.spawn.fix_root_link = True
