
# Physical AI Model Workspace — NVIDIA DGX Spark

Central GPU inference workspace for Vision-Language Models (VLMs), Vision-Language-Action (VLA) models, and future Physical AI experiments with the Unitree G1 EDU.

**Host:** `ailab-spark-01`  
**User:** `rhasan`  
**Workspace:** `/home/rhasan/Desktop/razib/ai_models`  
**Platform:** NVIDIA DGX Spark, Ubuntu Linux ARM64  
**Last documented:** 29 October 2026

---

# Part I — Project README

## 1. Project overview

This workspace stores and executes AI models used in robotics experiments.

The goal is to support a modular architecture in which a central AI manager receives requests from a robot, selects an appropriate AI model, and returns a structured response.

Different models have different responsibilities:

- **VLM:** Image understanding, visual reasoning, object recognition and scene interpretation.
- **LLM:** Language understanding, task interpretation and planning.
- **VLA:** Predicts robot actions from visual observations, instructions and robot state.
- **Response composer:** Converts model results into natural-language responses suitable for robot speech or API clients.
- **AI manager:** Coordinates requests, model selection and response handling.

The manager and response composer are planned primarily for a separate DEVBOX. Spark is reserved for GPU-dependent inference and model experiments.

### Proposed architecture

```text
                  Unitree G1 EDU
                        |
           Camera / Sensors / Requests
                        |
                        v
                DEVBOX AI Manager
                        |
             Request classification
                        |
          +-------------+-------------+
          |                           |
          v                           v
     Qwen3-VL-4B                UnifoLM-VLA-0
   Visual understanding         Action prediction
          |                           |
          +-------------+-------------+
                        |
                        v
              Response coordination
                        |
                 Safety validation
                        |
                        v
                  Unitree G1 EDU
              Speech / Robot actions
```

The DEVBOX manager and robot gateway are not yet implemented.

Physical actions must be validated by a robot-specific controller. Emergency stopping and low-level motion safety must remain local to the robot.

## 2. Hardware environment

| Component | Configuration |
|---|---|
| Computer | NVIDIA DGX Spark |
| GPU | NVIDIA GB10 |
| CPU architecture | ARM64 / aarch64 |
| NVIDIA driver | 580.178.04 |
| Driver CUDA support | 13.0 |
| GPU compute capability | 12.1 |
| System memory | 121 GiB |
| Storage | Approximately 3.7 TB total |
| Docker | Rootless Docker 29.6.2 |
| Host sudo | Not required for current workflow |

The NVIDIA PyTorch Docker image provides the CUDA runtime used for AI inference.

## 3. Installed AI models

| Model | Type | Parameters | Status |
|---|---|---|---|
| Qwen3-VL-4B-Instruct | VLM | 4B | API running at last verification |
| UnifoLM-VLM-Base | VLM backbone | 8B | Downloaded and loaded |
| UnifoLM-VLA-Base | VLA | Full count unverified | Offline inference passed |
| UnifoLM-WLA-1.0 | WLA | 6B model family | Source only |
| NVIDIA Isaac GR00T | VLA | Checkpoint not selected | Placeholder directory |

### Qwen3-VL-4B-Instruct

Developer: Qwen / Alibaba

Purpose:
- Camera image understanding
- Object and scene interpretation
- Visual question answering
- Support for higher-level robot task decisions

Model directory:

`models/Qwen3-VL-4B-Instruct_VLM_4B`

The directory is a symbolic link to the existing Hugging Face model cache.

Existing inference container:

`g1-qwen-api`

Existing Docker image:

`g1-qwen-api:0.1`

Last verified host endpoint:

`127.0.0.1:8000`

The endpoint is bound to Spark's localhost, not automatically accessible from DEVBOX or other computers.

### UnifoLM-VLA-0

Developer: Unitree Robotics

Official repository:

https://github.com/unitreerobotics/unifolm-vla

Purpose:
- Vision-language-action inference
- Humanoid manipulation
- Prediction of action sequences from images, instructions and robot state

Components:

**Vision-language backbone**

`models/UnifoLM-VLM-Base_VLM_8B`

Approximately 16 GB downloaded, stored in four SafeTensors shards.

**VLA policy checkpoint**

`models/UnifoLM-VLA-0_VLA_params-unverified/checkpoints/pytorch_model.pt`

Approximately 19 GB downloaded.

Additional files:
- `config.yaml`
- `dataset_statistics.json`
- `LICENSE`
- `README.md`

Source code:

`sources/UnifoLM-VLA-0`

Working inference image:

`g1-unifolm-vla:0.2`

Status: Full checkpoint loading and one offline action-prediction test passed on the NVIDIA GB10.

This does not establish safe or successful manipulation on the physical robot.

### UnifoLM-WLA-1.0

Official repository:

https://github.com/unitreerobotics/unifolm-wla

Source directory:

`sources/UnifoLM-WLA-1.0`

Prepared model directory:

`models/UnifoLM-WLA-1.0_WLA_6B`

Status: Repository cloned; model weights and inference environment not installed.

### NVIDIA Isaac GR00T

Prepared directory:

`models/Isaac-GR00T_VLA_params-unverified`

Status: No checkpoint installed. Reserved for future comparison.

## 4. Repository organization

```text
ai_models/
├── README.md
│
├── models/
│   ├── Qwen3-VL-4B-Instruct_VLM_4B/
│   ├── UnifoLM-VLM-Base_VLM_8B/
│   ├── UnifoLM-VLA-0_VLA_params-unverified/
│   ├── UnifoLM-WLA-1.0_WLA_6B/
│   └── Isaac-GR00T_VLA_params-unverified/
│
├── huggingface/
│   ├── hub/
│   ├── xet/
│   └── unifolm-download-cache/
│
├── sources/
│   ├── UnifoLM-VLA-0/
│   └── UnifoLM-WLA-1.0/
│
├── docker/
│   ├── unifolm_vla/
│   │   ├── Dockerfile
│   │   └── Dockerfile.v0.2
│   ├── unifolm_wla/
│   ├── manager/
│   └── simulation/
│
├── services/
│   ├── Qwen3-VL-4B-Instruct_VLM_4B/
│   └── qwen3_vl -> Qwen3-VL-4B-Instruct_VLM_4B
│
├── orchestrator/
│   ├── manager/
│   ├── router/
│   ├── registry/
│   └── tasks/
│
├── response/
│   ├── composer/
│   ├── schemas/
│   └── speech/
│
├── robot_gateway/
│   ├── api/
│   ├── commands/
│   ├── perception/
│   └── telemetry/
│
├── safety/
│   ├── policies/
│   └── validation/
│
├── configs/
│   └── unifolm_vla/
│       └── environment.txt
│
├── images/
│   └── test.jpg
│
├── logs/
└── tests/
```

Some folders are reserved for future development and are currently empty.

### Folder responsibilities

| Directory | Responsibility |
|---|---|
| `models/` | Downloaded pretrained model weights |
| `huggingface/` | Hugging Face caches |
| `sources/` | Original upstream repositories |
| `docker/` | Docker image definitions |
| `services/` | Model-serving applications |
| `orchestrator/` | Proposed central manager components |
| `response/` | Proposed response-composition components |
| `robot_gateway/` | Proposed robot communication interfaces |
| `safety/` | Planned validation and safety policies |
| `configs/` | Environment and configuration records |
| `images/` | Test images |
| `logs/` | Future application logs |
| `tests/` | Future automated test scripts |

Do not store training datasets, model weights or credentials in a public Git repository.

## 5. Verified UnifoLM-VLA experiment

**Date:** 28–29 October 2026  
**Hardware:** NVIDIA DGX Spark, GB10  
**Container:** `g1-unifolm-vla:0.2`

### Software environment

| Component | Version |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.10.0a0+b558c986e8.nv25.11 |
| CUDA runtime | 13.0 |
| TensorFlow | 2.20.0 |
| Transformers | 4.52.3 |
| Accelerate | 1.5.2 |
| Diffusers | 0.35.1 |

The original Unitree project specifies an older PyTorch/CUDA environment. The configuration above is an experimental adaptation validated for checkpoint loading and one offline prediction, not an officially certified Unitree configuration.

### Completed tests

1. ARM64 Docker execution.
2. NVIDIA GB10 CUDA detection.
3. PyTorch GPU matrix multiplication.
4. TensorFlow CPU image resizing.
5. Unitree Python module imports.
6. VLM and VLA checkpoint loading.
7. BF16 model transfer to GPU.
8. G1 normalization statistics inspection.
9. Synthetic-observation action prediction.

### Offline inference results

| Metric | Result |
|---|---|
| Inference | Successful |
| Task normalization key | `g1_stack_block` |
| Output shape | `(25, 23)` |
| Action sequence length | 25 |
| Values per action | 23 |
| Numerical validity | All finite |
| Prediction time | 2.456 seconds |
| CUDA memory allocated | 16.58 GiB |
| Robot connected | No |
| Physical movement | None |

Input consisted of three synthetic black camera images, a synthetic 23-value robot state and the instruction "Stack the blocks."

The 2.456-second measurement is a single execution result, not a latency benchmark. Synthetic inputs cannot establish actual manipulation accuracy.

The 588,135,424 reported DiT parameters refer to the action-model component, not the complete VLA model.

### G1 action representation

Source constants:

- `NUM_ACTIONS_CHUNK = 25`
- `ACTION_DIM = 23`
- `PROPRIO_DIM = 23`
- `ACTION_PROPRIO_NORMALIZATION_TYPE = BOUNDS_Q99`

The checkpoint normalization file provides task-specific statistics.

The original `config.yaml` also contains generic action dimensions, so use the actual loaded model and task-specific action mapping when evaluating input/output shapes.

The 23-value action representation must not be interpreted as a direct motor-command interface without verifying Unitree's G1 deployment mappings.

## 6. Development and deployment boundaries

### NVIDIA DGX Spark

Used for:
- GPU inference
- Model checkpoint storage
- CUDA compatibility testing
- GPU-based simulation and evaluation
- Model-serving Docker containers

### DEVBOX

Planned for:
- AI manager development
- Task routing
- Request and response schemas
- API authentication
- Response composer
- General Python testing
- Git development and deployment coordination

### Unitree G1

Used for:
- Camera and sensor observations
- Local robot telemetry
- Speech interaction
- Robot-specific execution
- Local motion safety

The system is currently at the offline inference stage for UnifoLM-VLA.

---

# Part II — Operation and maintenance guide

This section is for returning to the workspace later and understanding how to operate, inspect, rebuild and troubleshoot the current setup.

## 7. Accessing Spark

Connect using your configured SSH host or address:

```bash
ssh rhasan@ailab-spark-01
```

Open the workspace:

```bash
cd ~/Desktop/razib/ai_models
pwd
ls
```

Expected location:

```text
/home/rhasan/Desktop/razib/ai_models
```

Check directory organization:

```bash
tree -L 3 -d
```

## 8. Check NVIDIA hardware

Check GPU information:

```bash
nvidia-smi
```

Check system architecture:

```bash
uname -m
```

Expected architecture:

```text
aarch64
```

Check available memory:

```bash
free -h
```

Check disk space:

```bash
df -h .
```

## 9. Check Docker containers and images

List running containers:

```bash
docker ps
```

Include stopped containers:

```bash
docker ps -a
```

List available images:

```bash
docker images
```

Inspect Docker storage:

```bash
docker system df
```

The currently configured AI Docker images include:

```text
g1-qwen-api:0.1
g1-qwen-vlm:0.1
g1-unifolm-vla:dev
g1-unifolm-vla:0.1
g1-unifolm-vla:0.2
nvcr.io/nvidia/pytorch:25.11-py3
```

The images share Docker layers, so their displayed sizes should not simply be added together to calculate total disk use.

## 10. Operating the Qwen service

The existing Qwen service runs in a Docker container named:

```text
g1-qwen-api
```

Check whether it is running:

```bash
docker ps --filter name=g1-qwen-api
```

Inspect logs:

```bash
docker logs --tail 50 g1-qwen-api
```

Follow new logs:

```bash
docker logs -f g1-qwen-api
```

Inspect port mappings:

```bash
docker port g1-qwen-api
```

At the last verification, port 8000 was bound to:

```text
127.0.0.1:8000
```

This is accessible from Spark itself. Remote access requires an explicitly configured network route, tunnel or authenticated API gateway.

Start an existing stopped Qwen container:

```bash
docker start g1-qwen-api
```

Do not run this if it is already running.

Avoid stopping, deleting or rebuilding the Qwen container while other robot experiments depend on it.

Qwen application source:

```text
services/Qwen3-VL-4B-Instruct_VLM_4B/server.py
```

Inspect the source:

```bash
less services/Qwen3-VL-4B-Instruct_VLM_4B/server.py
```

## 11. Using the UnifoLM-VLA Docker environment

Current working image:

```text
g1-unifolm-vla:0.2
```

This is a saved Docker image, not a persistent running inference service.

Start a disposable interactive container:

```bash
cd ~/Desktop/razib/ai_models

docker run --rm -it \
  --device nvidia.com/gpu=all \
  --network none \
  --mount type=bind,source="$(pwd)/sources/UnifoLM-VLA-0",target=/workspace/unifolm-vla,readonly \
  --mount type=bind,source="$(pwd)/models",target=/models,readonly \
  --entrypoint /bin/bash \
  g1-unifolm-vla:0.2
```

Inside the container:

```bash
pwd
python --version
```

Check CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Check the GPU:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

Exit:

```bash
exit
```

Because the container uses `--rm`, it is deleted after exiting. The Docker image and host-mounted files remain.

The `--network none` setting is suitable for offline testing. It must be changed intentionally when an authenticated inference service is deployed.

## 12. Finding the Unitree model files

VLM backbone:

```bash
ls -lh models/UnifoLM-VLM-Base_VLM_8B/
```

VLA checkpoint:

```bash
ls -lh \
  models/UnifoLM-VLA-0_VLA_params-unverified/checkpoints/
```

Inspect model configuration:

```bash
cat models/UnifoLM-VLA-0_VLA_params-unverified/config.yaml
```

Inspect the normalization task keys:

```bash
python3 - <<'PY'
import json

path = (
    "models/UnifoLM-VLA-0_VLA_params-unverified/"
    "dataset_statistics.json"
)

with open(path) as f:
    data = json.load(f)

for task in data:
    print(task)
PY
```

The G1 task statistics include stacking blocks, folding towels, cleaning tables and other manipulation tasks.

Do not manually change normalization dimensions to match generic YAML settings without checking the model's task-specific action space.

## 13. Rebuilding the inference image

Two Dockerfile versions exist:

```text
docker/unifolm_vla/Dockerfile
docker/unifolm_vla/Dockerfile.v0.2
```

Build the original experimental image:

```bash
docker build \
  -t g1-unifolm-vla:0.1 \
  -f docker/unifolm_vla/Dockerfile \
  docker/unifolm_vla
```

Build the working TensorFlow-enabled image:

```bash
docker build \
  -t g1-unifolm-vla:0.2 \
  -f docker/unifolm_vla/Dockerfile.v0.2 \
  docker/unifolm_vla
```

The v0.2 Dockerfile depends on `g1-unifolm-vla:0.1`, so that base image must already exist.

Do not replace NVIDIA's working PyTorch build with Unitree's original pinned version without a separate compatibility test.

After rebuilding, verify:

```bash
docker run --rm \
  --device nvidia.com/gpu=all \
  --entrypoint python \
  g1-unifolm-vla:0.2 \
  -c '
import torch
import tensorflow as tf

print("PyTorch:", torch.__version__)
print("TensorFlow:", tf.__version__)
print("CUDA:", torch.cuda.is_available())

assert torch.cuda.is_available()
print("GPU TEST PASSED")
'
```

This only verifies GPU access and package imports; it does not repeat the complete VLA prediction test.

## 14. Reviewing the successful environment

The environment record is saved at:

```text
configs/unifolm_vla/environment.txt
```

View it:

```bash
cat configs/unifolm_vla/environment.txt
```

Or open it with:

```bash
less configs/unifolm_vla/environment.txt
```

This file documents the Docker image, hardware, upstream source revision and Python environment captured after the successful offline prediction.

For future reproducibility, also preserve the exact checkpoint revision and checksum before comparing experimental results across machines.

## 15. Troubleshooting

### Docker container immediately disappears

An interactive container started with `--rm` is removed after its main process exits. This is expected.

Use:

```bash
docker ps
```

to inspect running containers, and:

```bash
docker images
```

to verify that its image still exists.

### GPU unavailable inside Docker

Check that the container includes:

```text
--device nvidia.com/gpu=all
```

Then inspect:

```bash
nvidia-ctk cdi list
nvidia-smi
```

### Permission denied while downloading models

Rootless Docker can map host users differently when an explicit `--user` option is supplied.

The working download approach used the container's default user with writable mounts for the intended model directories.

Do not solve permission problems using recursive `chmod 777`.

### CUDA unavailable after installing TensorFlow

Confirm that the Docker container actually has GPU access.

A previous TensorFlow test reported `CUDA available: False` because the test container was launched without a GPU device.

### Model import fails

Inspect the installed versions:

```bash
docker run --rm \
  --entrypoint python \
  g1-unifolm-vla:0.2 \
  -m pip list
```

Compare against:

```text
configs/unifolm_vla/environment.txt
```

Do not install packages directly into the Spark host Python environment.

### Model produces invalid actions

Check:
- Camera image keys and shapes
- Robot-state dimensionality
- Task normalization key
- Model and checkpoint compatibility
- Action decoding and postprocessing
- Numerical validity of outputs

Do not send unverified action sequences to the real robot.

## 16. Project status and next tasks

### Completed

- [x] Organized the model workspace.
- [x] Preserved the existing Qwen model and API.
- [x] Cloned Unitree UnifoLM-VLA and WLA repositories.
- [x] Created the ARM64 GPU Docker environment.
- [x] Downloaded the VLM backbone and VLA checkpoint.
- [x] Installed compatible inference dependencies.
- [x] Verified TensorFlow preprocessing.
- [x] Loaded the UnifoLM-VLA checkpoint onto GB10.
- [x] Produced an offline `(25, 23)` action sequence.
- [x] Saved the environment record.

### Pending

- [ ] Create a reusable offline inference test script.
- [ ] Run repeated inference and latency measurements.
- [ ] Download and evaluate the separate LIBERO checkpoint.
- [ ] Prepare a reproducible simulation environment.
- [ ] Verify action mapping for the actual G1 EDU configuration.
- [ ] Build an internal persistent UnifoLM inference service.
- [ ] Develop the central AI manager on DEVBOX.
- [ ] Implement API authentication and robot-specific authorization.
- [ ] Test real G1 observations without motor execution.
- [ ] Perform supervised real-robot validation after safety checks.

## 17. Safety and maintenance rules

1. Keep GPU workloads and model inference on Spark where practical.
2. Develop general orchestration and API software on DEVBOX.
3. Keep each AI model in its own dependency environment.
4. Preserve upstream Unitree source code separately from custom modifications.
5. Store model weights outside disposable Docker containers.
6. Do not expose inference endpoints publicly without authentication.
7. Do not send raw VLA predictions directly to robot actuators.
8. Keep low-level motion safety and emergency stopping local to the G1.
9. Record model revisions, environment changes and test results.
10. Avoid deleting images or caches before checking whether active services depend on them.

---

## References

- [Unitree UnifoLM-VLA-0](https://github.com/unitreerobotics/unifolm-vla)
- [Unitree UnifoLM-WLA-1.0](https://github.com/unitreerobotics/unifolm-wla)
- [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- [UnifoLM-VLM-Base](https://huggingface.co/unitreerobotics/UnifoLM-VLM-Base)
- [UnifoLM-VLA-Base](https://huggingface.co/unitreerobotics/UnifoLM-VLA-Base)
- [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
- [Docker documentation](https://docs.docker.com/)

**Project status:** Qwen service operational at last inspection; UnifoLM-VLA offline inference verified. Simulation and real-robot action validation remain pending.
