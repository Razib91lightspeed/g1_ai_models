"""Launch the project-local G1 standing diagnostic.

Uses zero joint-action offsets, Newton physics, and Viser.
Does not load an RL or ONNX policy.
"""

import sys
import gymnasium as gym
import isaaclab_tasks

from isaaclab.cli import cli
from g1_29dof_env_cfg import G129DoFEnvCfg

TASK_ID = "Project-G1-29DoF-Standing"

# Use our own class, leaving the original Isaac Lab task unchanged.
class StandingEnvCfg(G129DoFEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Separate browser viewer.


        # The original termination checks for a link called "base".
        # Disable that task-specific termination in this diagnostic.
        self.terminations.base_contact = None

        # Standing diagnostic: disable inherited locomotion rewards.
        # Those rewards refer to joints from the 37-action G1 model.
        for name, term in vars(self.rewards).items():
            if term is not None and hasattr(term, "func"):
                setattr(self.rewards, name, None)


        # This is a static standing test, not a velocity task.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


# Reuse the upstream task's registered environment implementation.
original = gym.spec("Isaac-Velocity-Flat-G1")

kwargs = dict(original.kwargs)
kwargs["env_cfg_entry_point"] = StandingEnvCfg

gym.register(
    id=TASK_ID,
    entry_point=original.entry_point,
    disable_env_checker=True,
    kwargs=kwargs,
)

sys.argv = [
    "isaaclab",
    "zero_agent",
    "--viz",
    "viser",
    "--task",
    TASK_ID,
    "--num_envs",
    "1",
    "presets=newton_mjwarp",
]

print("Starting:", TASK_ID, flush=True)
print("Viewer port: 8081", flush=True)

cli()
