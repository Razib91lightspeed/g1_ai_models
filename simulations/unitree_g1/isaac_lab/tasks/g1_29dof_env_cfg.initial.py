"""Project-local Unitree G1 29-DoF configuration.

This is an experimental configuration, not yet a validated
locomotion environment. It does not modify upstream Isaac Lab.
"""

from isaaclab.utils import configclass, replace
from isaaclab_assets import G1_29DOF_CFG
from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import (
    G1FlatEnvCfg,
)


@configclass
class G129DoFEnvCfg(G1FlatEnvCfg):
    """G1 configuration with a 29-DoF asset and larger contact limit."""

    def __post_init__(self):
        super().__post_init__()

        # Use Isaac Lab's separately defined 29-DoF G1 asset.
        self.scene.robot = replace(
            G1_29DOF_CFG,
            prim_path="{ENV_REGEX_NS}/Robot",
        )

        # Increase Newton contact capacity.
        self.sim.physics.newton_mjwarp.solver_cfg.nconmax = 64

        # Preserve the established 50 Hz control rate.
        # The inherited flat environment already uses dt=0.005,
        # with four physics steps per control step.
