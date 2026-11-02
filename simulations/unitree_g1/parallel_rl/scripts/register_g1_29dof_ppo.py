"""Register the project-local Unitree G1 29-DOF PPO task.

Used by Isaac Lab's --external_callback option.

Simulation only. Does not connect to the physical G1.
"""

import sys
from pathlib import Path

import gymnasium as gym

TASK_ID = "Project-G1-29DoF-Parallel-PPO"


def register():
    """Register the calibrated 29-DOF environment."""

    config_dir = Path(__file__).resolve().parents[1] / "configs"
    sys.path.insert(0, str(config_dir))

    from g1_29dof_parallel_flat_env_calibrated_cfg import (
        G129DoFParallelCalibratedEnvCfg,
    )

    from isaaclab_tasks.core.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
        G1FlatPPORunnerCfg,
    )

    original = gym.spec("Isaac-Velocity-Flat-G1")

    kwargs = dict(original.kwargs)

    kwargs["env_cfg_entry_point"] = (
        G129DoFParallelCalibratedEnvCfg
    )

    kwargs["rsl_rl_cfg_entry_point"] = (
        G1FlatPPORunnerCfg
    )

    if TASK_ID not in gym.registry:
        gym.register(
            id=TASK_ID,
            entry_point=original.entry_point,
            disable_env_checker=True,
            kwargs=kwargs,
        )

    print(
        "REGISTERED:", TASK_ID,
        flush=True,
    )

    return []


if __name__ == "__main__":
    register()
