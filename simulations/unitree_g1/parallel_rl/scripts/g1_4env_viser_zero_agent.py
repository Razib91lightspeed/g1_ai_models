"""Four Unitree G1 robots in Isaac Lab Newton with Viser.

Visual diagnostic only: zero-action controller.
Does not train PPO or connect to the physical G1.
"""

import sys
from pathlib import Path

import gymnasium as gym
import isaaclab_tasks
from isaaclab.cli import cli

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
sys.path.insert(0, str(CONFIG_DIR))

from g1_29dof_parallel_flat_env_calibrated_cfg import (
    G129DoFParallelCalibratedEnvCfg,
)

TASK_ID = "Project-G1-29DoF-Four-Robot-Viser"

original = gym.spec("Isaac-Velocity-Flat-G1")
kwargs = dict(original.kwargs)
kwargs["env_cfg_entry_point"] = G129DoFParallelCalibratedEnvCfg

if TASK_ID in gym.registry:
    raise RuntimeError(f"Task already registered: {TASK_ID}")

gym.register(
    id=TASK_ID,
    entry_point=original.entry_point,
    disable_env_checker=True,
    kwargs=kwargs,
)

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

print("========== FOUR G1 ROBOT VISUAL TEST ==========", flush=True)
print("Task:", TASK_ID, flush=True)
print("Robots: 4", flush=True)
print("Controller: zero action / default joint targets", flush=True)
print("Physics: Newton", flush=True)
print("Visualization: Viser, container port 8080", flush=True)
print("No PPO training in this experiment", flush=True)

if __name__ == "__main__":
    cli()
