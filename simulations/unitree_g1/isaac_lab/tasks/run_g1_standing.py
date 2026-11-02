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


# Inspect the actual articulation created by Isaac Lab.
# This is local to this diagnostic process.
original_gym_make = gym.make

def inspected_gym_make(*args, **kwargs):
    env = original_gym_make(*args, **kwargs)
    robot = env.unwrapped.scene["robot"]

    names = list(robot.joint_names)
    pos = robot.data.joint_pos.torch[0].detach().cpu().tolist()
    defaults = robot.data.default_joint_pos.torch[0].detach().cpu().tolist()

    print("\\n=== NEWTON G1 RUNTIME JOINT INSPECTION ===", flush=True)
    print("Runtime joint count:", len(names), flush=True)

    for i, name in enumerate(names):
        print(
            f"{i:02d} | {name:32s} "
            f"| position={pos[i]:+.4f} "
            f"| default={defaults[i]:+.4f}",
            flush=True,
        )

    for field in ("joint_stiffness", "joint_damping"):
        try:
            values = getattr(robot.data, field).torch[0].detach().cpu().tolist()
            print(f"=== {field} ===", flush=True)
            for i, name in enumerate(names):
                print(f"{name}: {values[i]:.3f}", flush=True)
        except Exception as exc:
            print(f"{field} unavailable: {exc}", flush=True)

    return env

gym.make = inspected_gym_make

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
