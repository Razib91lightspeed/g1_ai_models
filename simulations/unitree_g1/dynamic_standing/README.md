# Unitree G1 — Dynamic Standing and Walking

## Overview

This directory contains simulation-only experiments for
the Unitree G1 29-DOF humanoid robot.

Compute: NVIDIA DGX Spark
Physics: MuJoCo
Model: Unitree G1 29-DOF, floating base
Locomotion: Unitree RL Lab pretrained velocity policy
Inference: ONNX Runtime (CPU)
Physical robot connection: None

## Verified configuration

| Parameter | Value |
|---|---|
| Robot model | scene_29dof.xml |
| Actuators | 29 torque actuators |
| MuJoCo timestep | 0.002 s (500 Hz) |
| ONNX policy timestep | 0.02 s (50 Hz) |
| Policy input | 480 float32 observations |
| Policy output | 29 float32 actions |
| Action conversion | raw action * scale + offset |
| Velocity control | Joint-position PD control |
| Policy joint mapping | deploy.yaml: joint_ids_map |
| ONNX Runtime | 1.31.1 |
| Docker image | g1-locomotion-sim:0.1 |
| Browser viewer port | 8768 |

### Critical correction: PD gain ordering

The Unitree deployment YAML stores stiffness and damping
in motor order.

The policy uses a different joint order, specified by
joint_ids_map.

The gains must therefore be rearranged:

```python
ids = np.asarray(cfg["joint_ids_map"], dtype=int)
kp = np.asarray(cfg["stiffness"], dtype=float)[ids]
kd = np.asarray(cfg["damping"], dtype=float)[ids]
```

Without this correction, standing worked but the robot
failed to produce continuous forward walking.

Do not revert this mapping.

### Observation history

Six observation terms, five frames each:

- base_ang_vel: 3 values
- projected_gravity: 3 values
- velocity_commands: 3 values
- joint_pos_rel: 29 values
- joint_vel_rel: 29 values
- last_action: 29 values

Total: (3 + 3 + 3 + 29 + 29 + 29) * 5 = 480.

The deployment uses term-major history grouping,
with older frames before newer frames.

### Supported initialization

Simulation-only virtual support:

- 0–2 s: full virtual support
- 2–4 s: support gradually reduced
- After 4 s: no external support

The simulator starts from the policy's configured
default standing joint positions.

The virtual support in our Python experiment is a
custom stabilization harness, not an exact copy of
Unitree's C++ elastic band.

## Experimental results

### Unsupported standing test

Simulation duration: 20 seconds.
Support fully removed after 4 seconds.

Results:
- 16 seconds of unsupported standing
- Final pelvis height: approximately 0.788 m
- Final body tilt: approximately 0.8 degrees
- No fall during the tested interval

### Forward walking test

Test file: test_walk_corrected.py

Command sequence:
- 0–4 s: supported initialization
- 4–6 s: unsupported standing
- 6–8 s: ramp forward velocity to 0.30 m/s
- 8–20 s: maintain 0.30 m/s command

Results:
- Final forward displacement: 3.2328 m
- Final lateral displacement: approximately 0.050 m
- Final body tilt: approximately 1.0 degrees
- Maximum leg-joint displacement: 0.3247 rad
- Stable pelvis height near 0.78–0.79 m

Forward motion was also visually confirmed in the
browser viewer. Further gait and disturbance tests
are needed before claiming robust locomotion.

## Running the walking viewer

From the repository root:

```bash
cd ~/Desktop/razib/ai_models

docker run --rm -it \
  --name g1-walk-viewer \
  --shm-size=4g \
  -p 127.0.0.1:8768:8768 \
  -e MUJOCO_GL=osmesa \
  -e PYOPENGL_PLATFORM=osmesa \
  -e LIBGL_ALWAYS_SOFTWARE=1 \
  --mount type=bind,source="$(pwd)/simulations/unitree_g1/unitree_mujoco",target=/workspace/unitree_mujoco,readonly \
  --mount type=bind,source="$(pwd)/simulations/unitree_g1/dynamic_standing/unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/v0",target=/workspace/policy,readonly \
  --mount type=bind,source="$(pwd)/simulations/unitree_g1/dynamic_standing/walk_view.py",target=/workspace/walk_view.py,readonly \
  --entrypoint python \
  g1-locomotion-sim:0.1 \
  -u /workspace/walk_view.py
```

On a Mac:

```bash
ssh -N -L 8768:127.0.0.1:8768 rhasan@10.11.0.50
```

Open: http://localhost:8768

The separate waving viewer remains on port 8767.

## Upstream references

Unitree MuJoCo:
https://github.com/unitreerobotics/unitree_mujoco

Unitree RL Lab:
https://github.com/unitreerobotics/unitree_rl_lab

The pretrained policy is from Unitree RL Lab, not
NVIDIA Isaac GR00T.

## Safety and scope

These scripts are for simulation only.

Do not deploy the experimental Python PD controller,
virtual support, or ONNX action pipeline directly
to the physical G1.

Physical robot locomotion must use compatible,
validated controllers and the robot's safety systems.
