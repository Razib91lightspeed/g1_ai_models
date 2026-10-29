# Unitree G1 MuJoCo Simulation — NVIDIA DGX Spark

## Purpose

This folder contains **simulation-only** experiments for the Unitree G1 29-actuator humanoid. Physics and camera rendering run inside an isolated Docker container on NVIDIA DGX Spark; the live MJPEG stream is viewed on a Mac through an SSH tunnel. **No code here connects to or commands the physical robot.** The existing DEVBOX → FIWARE telemetry pipeline is independent and must remain unchanged.

## Workspace

Spark root: `/home/rhasan/Desktop/razib/ai_models`

```text
simulations/unitree_g1/
├── README.md                    # This guide
├── unitree_mujoco/              # Upstream Unitree repository (separate Git checkout)
├── preview/
│   ├── create_scene.py          # Makes an isolated fixed-base G1 XML
│   └── g1_fixed_base.xml        # Generated model
├── live_view.py                 # Basic G1 physics/standing preview, port 8766
└── controllers/
    └── wave_view.py             # Simulation-only right-hand wave, port 8767
```

The upstream repository is [unitreerobotics/unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco). Do not modify or commit a copy of upstream sources into the Spark project repository. Record the upstream revision with `git -C simulations/unitree_g1/unitree_mujoco rev-parse HEAD`.

## Confirmed simulation environment

- Docker image: `g1-libero-sim:0.2` (already installed on Spark; contains MuJoCo and OSMesa).
- G1 original scene: `unitree_mujoco/unitree_robots/g1/scene.xml`.
- G1 original robot XML: `unitree_mujoco/unitree_robots/g1/g1_29dof.xml`.
- Original floating-base model: 31 bodies, 30 joints (including one free joint), 29 motor actuators; timestep `0.002 s`.
- Fixed-base preview: 31 bodies, 29 joints, 29 motor actuators, `nq=29`.
- Original initial pelvis height: `0.793 m`; original base quaternion: `[1, 0, 0, 0]`; all 29 initial joint angles were zero.
- No predefined keyframes were found in the original scene.

**Important:** Fixed-base preview removes the pelvis free joint from a *copy* of the model. It holds the pelvis in the world; it does **not** demonstrate standing balance. The free-base model falls without an appropriate whole-body controller.

## Verified working right-hand wave

The script `controllers/wave_view.py` uses torque-limited PD feedback to track joint targets. The best measured candidate from a discrete forward-kinematics search was:

| Right-arm actuator | Target during arm raise (radians) |
| --- | ---: |
| `right_shoulder_pitch` | `-1.50` |
| `right_shoulder_roll` | `-1.20` |
| `right_shoulder_yaw` | `0.00` |
| `right_elbow` | `0.00` before wave oscillation |
| `right_wrist_pitch` | `0.00` |

The search measured the right wrist about **0.237 m above the shoulder reference body** with ~**0.195 m horizontal offset** for that shoulder/elbow configuration. This is a geometric result, not a dynamically validated grasp/wave trajectory.

During the wave phase, the controller currently adds approximately:

```python
phase = 2.0 * math.pi * 1.2 * (cycle - 2.5)
right_wrist_yaw = 0.45 * math.sin(phase)
right_elbow = 0.35 * math.sin(phase)
```

The motion repeats on an approximately **11-second simulated-time cycle**: rest → raise → wave → lower → rest. The raised-arm targets are multiplied by a smooth interpolated `amount`. All 29 actuators are commanded using feedback and clipped to the actuator `ctrlrange`.

**Joint sign warning:** These signs are verified only for the checked-out **MuJoCo G1 29-actuator model**. Do not reuse them as real G1 motor commands or assume identical mappings for other G1 variants.

## Waist stabilization — essential fix

The initial generic PD gains produced severe forward lean. During the wave, the torso tilted about **23–35 degrees**, and waist pitch reached its `+0.52 rad` joint limit.

The successful simulation-only adjustment was to use the **waist-specific** gains:

```python
elif name.startswith("waist_"):
    kp, kd = 180.0, 12.0
```

The other configured gains were preserved:

| Joint group | Kp | Kd |
| --- | ---: | ---: |
| Shoulder | 24.0 | 2.5 |
| Elbow | 20.0 | 2.0 |
| Wrist | 6.0 | 0.7 |
| Waist | **180.0** | **12.0** |
| Other joints | 35.0 | 3.0 |

**Observed after the change** (while the wave ran):

| Simulation time | Torso tilt | Waist pitch | Waist roll |
| ---: | ---: | ---: | ---: |
| 1.00 s | 1.60° | 0.028 rad | ~0 rad |
| 2.00 s | 2.79° | 0.047 rad | 0.012 rad |
| 3.05 s | 3.21° | 0.049 rad | 0.027 rad |
| 4.05 s | 2.24° | 0.037 rad | 0.012 rad |
| 5.05 s | 1.75° | 0.030 rad | 0.007 rad |
| 7.05 s | 3.18° | 0.049 rad | 0.027 rad |

These readings show a significant improvement in this **fixed-pelvis** test. They are not evidence of free-standing stability, and the gains are **not certified safe for hardware**. Do not further raise gains without checking actuator saturation, contacts, and simulation stability.

## Camera angle and view

The user preferred a three-quarter view because it shows the chest and right arm more clearly:

```python
camera.azimuth = 135
camera.elevation = -12
camera.lookat[:] = [0.0, 0.0, 0.8]
camera.distance = 2.8
```

The above parameters are the reference camera configuration for `controllers/wave_view.py` (check that current file agrees before running). Azimuth `0` previously showed the robot's **back**, and `90` showed its **side**. Camera orientation is not the same thing as torso tilt; torso tilt was checked from the world-space `torso_link` rotation matrix.

## Running the wave on Spark

First make sure the generated preview exists:

```bash
cd ~/Desktop/razib/ai_models
ls -lh simulations/unitree_g1/preview/g1_fixed_base.xml
```

If the XML needs regeneration, use the existing `preview/create_scene.py` within the same two read/write mounts as in the original setup. It must retain the compiler mesh path pointing to `/workspace/unitree_mujoco/unitree_robots/g1/meshes`.

Start the waving simulation:

```bash
cd ~/Desktop/razib/ai_models

docker run --rm -it \
  --name g1-wave-sim \
  --shm-size=4g \
  -p 127.0.0.1:8767:8767 \
  -e MUJOCO_GL=osmesa \
  -e PYOPENGL_PLATFORM=osmesa \
  -e LIBGL_ALWAYS_SOFTWARE=1 \
  --mount type=bind,source="$(pwd)/simulations/unitree_g1/unitree_mujoco",target=/workspace/unitree_mujoco,readonly \
  --mount type=bind,source="$(pwd)/simulations/unitree_g1/preview",target=/workspace/preview,readonly \
  --mount type=bind,source="$(pwd)/simulations/unitree_g1/controllers/wave_view.py",target=/app/wave_view.py,readonly \
  --entrypoint python \
  g1-libero-sim:0.2 \
  -u /app/wave_view.py
```

Expected startup:

```text
G1 fixed-base model loaded
29 torque actuators available
Simulation-only right-arm wave enabled
G1 WAVE VIEWER READY on port 8767
```



On the **Mac**, create the private SSH tunnel in a separate terminal:

```bash
ssh -N -L 8767:127.0.0.1:8767 rhasan@10.11.0.50
```

Then open `http://localhost:8767` in the browser. Keep the Spark container and Mac tunnel running. Stop only this simulation using `docker stop g1-wave-sim` on Spark or `Ctrl+C` in its terminal.

## Troubleshooting notes

- **Robot falls forward:** Check whether the viewer loaded the *free-base* `scene.xml` instead of the fixed-base preview.
- **Upper torso leans:** Read the printed `Torso tilt` and `waist_pitch_joint` values; the initial low waist gains produced ~23–35° lean. Confirm waist Kp/Kd `180/12` in `wave_view.py`.
- **Arm moves downward:** The original hand-crafted arm angles gave the wrong motion; use the tested negative shoulder pitch and roll targets above as the reference.
- **Robot looks sideways/backwards:** Check camera azimuth. The user preferred ~135° three-quarter view.
- **Missing `pelvis.STL`:** The copied preview XML references original mesh assets using an absolute path. The container must mount **both** `/workspace/preview` and `/workspace/unitree_mujoco`.
- **Docker name conflict:** Run `docker ps --filter name=g1-wave-sim`; stop the old instance before restarting the same container name.
- **No model training occurs here:** This script follows an explicitly defined wave trajectory; it does not invoke UnifoLM-VLA or a learned balancing policy.

## Next milestones

1. Save this proven fixed-base wave as a repeatable regression test (joint tracking errors, waist tilt, torque saturation and video output).
2. Provide a browser camera selector without changing the controller.
3. Develop a separate floating-base G1 standing experiment using a documented balance controller or compatible policy.
4. Integrate model-generated manipulation actions only after simulation-specific mapping, limits, watchdogs and safety validation.
5. Keep the robot hardware and DEVBOX → FIWARE telemetry pipeline untouched throughout simulation work.
