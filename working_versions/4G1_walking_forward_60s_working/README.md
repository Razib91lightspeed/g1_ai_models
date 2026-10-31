# Working G1: Four parallel ONNX walkers

## Status

Four Unitree G1 robots were observed walking stably in the
Isaac Lab 3.0 / Newton simulation using Viser.

This saved copy has the controller, configuration files,
ONNX adapter, and pretrained ONNX policy. The saved copy
must be tested independently before marking it reproduced.

## Configuration

- Four independent simulation environments
- 29 controlled joints per G1
- Pretrained ONNX locomotion policy
- Independent observation history per robot
- ONNX observations: 480 values per robot
- ONNX actions: 29 values per robot
- Forward walking reference: 0.30 m/s
- Newton physics, Viser browser visualization
- Simulation only; no physical robot commands

## Files

- `scripts/g1_4env_onnx_walking_viser.py`: simulation entry point
- `configs/g1_4env_onnx_walking_cfg.py`: four-robot configuration
- `configs/g1_29dof_parallel_flat_env_calibrated_cfg.py`: motor gains
- `configs/g1_29dof_parallel_flat_env_cfg.py`: 29-joint environment
- `g1_onnx_adapter.py`: ONNX observations and action conversion
- `policy/params/deploy.yaml`: policy deployment settings
- `policy/exported/policy.onnx`: pretrained weights
- `manifest.json`: experiment record

## Running on DGX Spark

From `/home/rhasan/Desktop/razib/ai_models`:

```bash
ROOT="$PWD"
BASE="$ROOT/simulations/unitree_g1/parallel_rl"

docker run --rm -it \
  --name g1-four-walking-saved-test \
  --user 0:0 \
  --device nvidia.com/gpu=all \
  --ipc=host \
  -p 127.0.0.1:8083:8080 \
  --mount "type=bind,src=$ROOT,dst=/workspace/local" \
  --workdir /workspace/local \
  -e HOME=/workspace/local/simulations/unitree_g1/parallel_rl/cache/home \
  -e XDG_CACHE_HOME=/workspace/local/simulations/unitree_g1/parallel_rl/cache/xdg \
  -e TMPDIR=/workspace/local/tmp \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/workspace/local/temp/isaac_lab/onnxdeps \
  nvcr.io/nvidia/isaac-lab:3.0.0-kitless \
  /opt/isaaclab-venv/bin/python -B \
  /workspace/local/working_versions/4G1_walking_forward_60s_working/scripts/g1_4env_onnx_walking_viser.py
```

On your Mac:

```bash
ssh -N -L 8083:127.0.0.1:8083 rhasan@10.11.0.50
```

Open http://localhost:8083.

The project-local ONNX Runtime dependency must already be installed
at `temp/isaac_lab/onnxdeps`.

## Preservation

The original one-robot verified configuration is unchanged.
The PPO training configurations are unchanged.
