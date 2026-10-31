
# Unitree G1 — Verified Isaac Lab Forward Walking

## Overview

This folder contains a preserved Unitree G1 29-DOF walking simulation using NVIDIA Isaac Lab 3.0, the Newton physics backend, and a pretrained ONNX locomotion policy.

The configuration was visually verified on October 31, 2026, using the Viser browser interface on port 8080.

The robot successfully walked forward with a commanded velocity of 0.30 m/s. The 60-second version extends the walking demonstration and retains the existing tilt, height, and fall-related safety checks.

**Status:** Visually verified walking simulation
**Robot:** Unitree G1, 29 DOF
**Platform:** NVIDIA DGX Spark (ARM64, NVIDIA GB10)
**Simulator:** NVIDIA Isaac Lab 3.0 / Newton
**Policy:** Pretrained ONNX locomotion policy
**Commanded walking speed:** 0.30 m/s
**Simulation duration:** 60 seconds
**Viewer:** Viser, port 8080

This is a policy-inference demonstration, not a reinforcement-learning training run.

## Project location

The configuration is stored at:

```text
/home/rhasan/Desktop/razib/ai_models/
└── working_versions/
    └── g1_isaaclab_forward_walk_030ms_60s_viser_verified/
        ├── README.md
        ├── manifest.json
        ├── walk_60s_visual_isaaclab.py
        ├── g1_29dof_env_cfg.py
        ├── g1_onnx_adapter.py
        ├── policy/
        │   ├── exported/
        │   │   └── policy.onnx
        │   └── params/
        │       └── deploy.yaml
        └── logs/
            └── g1_walk_60s_visual.log
```

The manifest records the preserved configuration and file checksums. Check the actual files against the manifest before reproducing the experiment.

## Components

### Walking controller

`walk_60s_visual_isaaclab.py`

Runs the ONNX walking policy, provides velocity commands, records simulation measurements, and checks the robot's safety conditions.

### Robot configuration

`g1_29dof_env_cfg.py`

Defines the G1 simulation environment, articulation configuration, actuators, and related Isaac Lab settings.

### ONNX policy adapter

`g1_onnx_adapter.py`

Connects simulator observations to the pretrained policy and converts policy actions into robot joint-position targets.

### Pretrained policy

`policy/exported/policy.onnx`

Contains the neural-network weights used for locomotion inference.

### Deployment parameters

`policy/params/deploy.yaml`

Contains deployment information such as joint ordering, stiffness, damping, and related policy parameters.

### Experiment log

`logs/g1_walk_60s_visual.log`

Preserved output from the visual walking experiment.

## Walking schedule

| Simulation time | Command |
|---|---|
| 0–3 s | Initial standing |
| 3–5 s | Accelerate towards 0.30 m/s |
| 5–55 s | Command 0.30 m/s |
| 55–57 s | Decelerate to zero |
| 57–60 s | Remain stopped |

The schedule is time-based. It does not guarantee a fixed travel distance.

## Requirements

- NVIDIA DGX Spark with a working NVIDIA container runtime
- Docker with NVIDIA GPU access
- NVIDIA Isaac Lab container image:
  `nvcr.io/nvidia/isaac-lab:3.0.0-kitless`
- Compatible Isaac Lab and Newton environment
- ONNX Runtime dependencies available in the existing project-local runtime
- Network access to the viewer through port 8080

The previously working simulation used project-local runtime dependencies, including ONNX Runtime under `temp/isaac_lab/onnxdeps`.

The commands below preserve that environment instead of installing dependencies globally.

## Running the verified simulation

Run on DGX Spark.

First navigate to the main project folder:

```bash
cd /home/rhasan/Desktop/razib/ai_models
```

Ensure the project runtime directories exist:

```bash
mkdir -p \
  temp/isaac_lab \
  temp/isaac_lab-home \
  temp/xdg-cache \
  temp/torch-cache \
  logs/isaac_lab
```

Start the preserved walking controller:

```bash
ROOT="$PWD"

docker run --rm \
  --name g1-walk-verified \
  --user 0:0 \
  --device nvidia.com/gpu=all \
  -p 127.0.0.1:8080:8080 \
  -e HOME=/workspace/local/isaac_lab-home \
  -e TMPDIR=/workspace/local/isaac_lab \
  -e XDG_CACHE_HOME=/workspace/local/xdg-cache \
  -e TORCHINDUCTOR_CACHE_DIR=/workspace/local/torch-cache \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e G1_HOLD_DEFAULT=0 \
  -e G1_PREFLIGHT=0 \
  -v "$ROOT/temp:/workspace/local" \
  -v "$ROOT/working_versions/g1_isaaclab_forward_walk_030ms_60s_viser_verified:/workspace/verified:ro" \
  -v "$ROOT/logs/isaac_lab:/workspace/isaaclab/logs" \
  -w /workspace/verified \
  --entrypoint python3 \
  nvcr.io/nvidia/isaac-lab:3.0.0-kitless \
  -u walk_60s_visual_isaaclab.py
```

The controller is mounted read-only to protect the verified files.

The simulation runs until its configured completion time or an earlier safety stop.

The preserved source has been validated in the original writable simulation workspace. If the read-only archive run encounters path assumptions, run an experimental copy from the original workspace and correct those paths there. Do not modify the verified archive.

## Opening the visual viewer

On your Mac, start an SSH tunnel:

```bash
ssh -N -L 8080:127.0.0.1:8080 rhasan@10.11.0.50
```

Keep the SSH terminal open.

Open the following address in your browser:

http://localhost:8080

The viewer is only accessible while its simulation server is running.

If the SSH terminal reports:

```text
channel 2: open failed: connect failed: Connection refused
```

check that the simulation container is still running and listening on port 8080.

## Verified walking results

The earlier 20-second test at 0.30 m/s produced:

| Simulation time | Robot Y position | Horizontal speed |
|---|---:|---:|
| 8 s | 0.716 m | 0.285 m/s |
| 10 s | 1.272 m | 0.282 m/s |
| 12 s | 1.827 m | 0.277 m/s |
| 15 s | 2.657 m | 0.283 m/s |
| 17 s | 2.985 m | 0.001 m/s |

The simulation completed without a reported fall or arm safety stop.

Foot-height measurements also showed evidence of alternating stepping.

A separate extended 60-second walking simulation was subsequently confirmed visually by the operator. Its detailed numerical results should be taken from the corresponding saved log, not inferred from the shorter experiment.

## Previous diagnostic findings

An earlier experiment with a commanded speed of 0.15 m/s did not produce sustained walking.

At 0.30 m/s, the robot initiated forward locomotion.

Ankle actuator measurements during the stationary experiment showed approximately 9–10 Nm of applied pitch torque against a reported 50 Nm limit. Computed and applied torques matched in the sampled measurements, with no evidence of saturation.

These results do not establish a general minimum walking velocity for the policy. They describe the tested configurations.

## Safety and limitations

- This configuration has been tested in simulation.
- It has not been validated for physical G1 walking.
- The robot's safety checks must remain enabled in further experiments.
- The controller does not provide obstacle avoidance.
- The 60-second demonstration does not stop at an exact distance.
- Terrain adaptation and robustness have not been evaluated systematically.
- The ONNX policy is pretrained; running it does not update its weights.

Do not deploy the controller directly on the physical robot without a separate hardware safety and control review.

## Parallel reinforcement learning

A planned extension is to train or fine-tune G1 locomotion using multiple Isaac Lab environments simultaneously.

The proposed development sequence is:

1. Preserve the verified single-robot ONNX inference configuration.
2. Create a separate parallel simulation experiment.
3. Confirm that 4, 16, then 64 simulated G1 robots can step and reset independently.
4. Integrate a compatible reinforcement-learning training environment.
5. Define observations, actions, rewards, termination conditions, and training configurations.
6. Begin controlled PPO training using RSL-RL.
7. Compare trained policy checkpoints against the verified walking baseline.

Parallel simulation by itself does not train the ONNX model. Training requires a learner, rollout collection, reward definitions, and optimizer updates.

Avoid treating the pretrained ONNX file as a trainable checkpoint unless a compatible training architecture and weights are available.

Reference:

https://docs.nvidia.com/isaaclab/3.0/concepts/reinforcement-learning

## Project file policy

All generated project files, simulation outputs, checkpoints, caches, and temporary files must stay inside:

```text
/home/rhasan/Desktop/razib/ai_models
```

Do not create project files under host `/tmp`, outside this workspace, or in unrelated users' directories.

Keep verified versions unchanged.

Use separate experimental directories for training and testing.

## Version control

The verified scripts, configuration, README, and manifest should be committed to Git.

Do not commit temporary caches, generated training logs, or large checkpoint histories.

ONNX models may require Git LFS depending on their size and the repository's existing policies.

The manifest and saved log provide evidence of the verified simulation, but the log should be retained locally or archived separately if excluded from Git.

## Next development targets

- 11-metre target-based forward walking
- Controlled deceleration and stopping
- Parallel simulation environments
- Reinforcement-learning training experiments
- Terrain and obstacle handling
- Evaluation of locomotion stability and tracking
- Separate physical G1 safety validation

## Verification status

**Visual verification:** Passed
**Forward walking at 0.30 m/s:** Passed
**Extended 60-second demonstration:** Visually confirmed
**Parallel reinforcement learning:** Not yet implemented
**Physical G1 deployment:** Not validated
