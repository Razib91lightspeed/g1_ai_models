
# Unitree G1 EDU — Dex3 Resting Hand Pose

**Status:** Visually verified in Isaac Lab  
**Date:** 2 November 2026  
**Platform:** Unitree G1 EDU with Dex3 hands  
**Simulation:** Isaac Lab 3.0 / Newton on NVIDIA DGX Spark  
**Viewer:** Viser, port 8082

## 1. Purpose

This directory preserves the Dex3 hand configuration that was visually accepted in Isaac Lab after reading actual motor positions from the physical Unitree G1 EDU.

The objective is to reproduce the natural curled-finger resting position observed on the physical robot while it was standing. This pose is intended for later integration with the verified G1 walking controller.

The configuration was developed by:

1. Testing individual Dex3 finger motors in Isaac Lab.
2. Comparing different finger-closing percentages.
3. Reading actual Dex3 motor positions from the physical G1.
4. Mapping the measured motor values to Isaac Lab joints.
5. Applying the values while respecting existing simulation soft limits.
6. Visually inspecting and accepting the resulting hand configuration.

**Verification scope:** The hand pose is visually confirmed in a four-robot fixed-base simulation. Walking stability with these hand positions has not yet been verified.

Do not change this saved configuration while experimenting with walking or further finger adjustments.

## 2. Saved working version

Project root:

    /home/rhasan/Desktop/razib/ai_models

Saved directory:

    working_versions/g1_dex3_physical_hand_rest_pose_visual_verified_20261102/

Directory structure:

    g1_dex3_physical_hand_rest_pose_visual_verified_20261102/
    ├── README.md
    ├── manifest.sha256
    ├── physical_dex3_angles.json
    ├── configs/
    │   └── g1_4env_relaxed_hands_v2_cfg.py
    ├── scripts/
    │   └── g1_4env_physical_dex3_pose_visual.py
    └── logs/
        └── verification.log

### File descriptions

| File | Description |
|---|---|
| `README.md` | Configuration, motor mapping, measurements, and future development reference |
| `physical_dex3_angles.json` | Actual physical G1 measurements for all 14 Dex3 motors |
| `scripts/g1_4env_physical_dex3_pose_visual.py` | Isaac Lab experiment applying the physical hand pose |
| `configs/g1_4env_relaxed_hands_v2_cfg.py` | Fixed-base simulation environment configuration |
| `logs/verification.log` | Saved configuration and joint-tracking output |
| `manifest.sha256` | File integrity checksums |

The entire version was saved successfully and the original checksum verification passed for all five listed files.

If any saved file is intentionally updated, its checksum will change. Preserve this version unchanged and create a new experiment directory instead.

## 3. Physical G1 Dex3 measurements

The motor positions were obtained directly from the physical robot using Unitree SDK2 hand-state feedback.

Measurement field:

    HandState_.motor_state()[motor_index].q()

The raw feedback values are in radians. The following tables show degrees.

The readings were collected while the physical G1 was standing with its hands in the desired resting configuration.

### 3.1 Left hand

| Finger | Motor part | SDK index | Isaac Lab joint name | Physical angle |
|---|---|---:|---|---:|
| Thumb | Base | 0 | `left_hand_thumb_0_joint` | +30.6766° |
| Thumb | Middle | 1 | `left_hand_thumb_1_joint` | +60.3186° |
| Thumb | Tip | 2 | `left_hand_thumb_2_joint` | +85.4347° |
| Middle | Base | 3 | `left_hand_middle_0_joint` | -91.6749° |
| Middle | Tip | 4 | `left_hand_middle_1_joint` | -98.5590° |
| Index | Base | 5 | `left_hand_index_0_joint` | -91.9567° |
| Index | Tip | 6 | `left_hand_index_1_joint` | -100.0102° |

### 3.2 Right hand

| Finger | Motor part | SDK index | Isaac Lab joint name | Physical angle |
|---|---|---:|---|---:|
| Thumb | Base | 0 | `right_hand_thumb_0_joint` | -2.4500° |
| Thumb | Middle | 1 | `right_hand_thumb_1_joint` | -55.8133° |
| Thumb | Tip | 2 | `right_hand_thumb_2_joint` | -94.3222° |
| Index | Base | 3 | `right_hand_index_0_joint` | +87.8579° |
| Index | Tip | 4 | `right_hand_index_1_joint` | +97.9258° |
| Middle | Base | 5 | `right_hand_middle_0_joint` | +88.3624° |
| Middle | Tip | 6 | `right_hand_middle_1_joint` | +98.5291° |

### Important motor-index distinction

The left and right Dex3 hands use different SDK indices for the index and middle fingers.

For example:

- Left index base: SDK motor index 5.
- Right index base: SDK motor index 3.
- Left middle base: SDK motor index 3.
- Right middle base: SDK motor index 5.

**Never assume the SDK motor index is the same for corresponding left and right fingers.**

In Isaac Lab, identify motors using their full joint names instead of SDK indices.

## 4. Dex3 finger motor structure

Each Dex3 hand has seven controlled motors:

- Three thumb motors.
- Two index finger motors.
- Two middle finger motors.

Therefore, both hands contain 14 controlled Dex3 joints.

### 4.1 Index finger

During calibration, the two index motors were described as follows:

**Motor 2 — Base motor (palm side)**

    left_hand_index_0_joint
    right_hand_index_0_joint

**Motor 1 — Tip motor**

    left_hand_index_1_joint
    right_hand_index_1_joint

These labels refer to the base/tip terminology used in the experiments. They are not SDK motor indices.

### 4.2 Middle finger

Base motor:

    left_hand_middle_0_joint
    right_hand_middle_0_joint

Tip motor:

    left_hand_middle_1_joint
    right_hand_middle_1_joint

### 4.3 Thumb

Each thumb has three controlled joints.

Base:

    left_hand_thumb_0_joint
    right_hand_thumb_0_joint

Middle:

    left_hand_thumb_1_joint
    right_hand_thumb_1_joint

Tip:

    left_hand_thumb_2_joint
    right_hand_thumb_2_joint

The thumb is mechanically different from the index and middle fingers. Its three joints must be calibrated independently.

## 5. Joint angles versus visible finger orientation

A joint angle is a rotation about the joint's local axis.

It is not the same as the finger's visible angle relative to the floor or the camera.

For example, the physical G1 reported:

    left_hand_index_0_joint = -91.9567 degrees

This does not necessarily mean the index finger points 91.9567 degrees downward.

The visible finger orientation depends on:

- Palm orientation.
- Joint rotation axis.
- Finger-link geometry.
- Other finger joint positions.
- Camera perspective.

This distinction is important because earlier experiments attempted to infer the visual finger direction directly from joint-angle signs.

For future work, always separate the following:

1. Requested motor target.
2. Actual joint-position feedback.
3. Visible finger-link orientation.

A command being accepted does not automatically prove that the rendered finger pose matches the physical hand.

## 6. Transferring physical measurements into Isaac Lab

The physical motor positions were stored in the simulation script as degrees.

They are converted to radians using:

    requested = math.radians(physical_deg)

The simulation already defines soft joint-position limits.

Some physical G1 measurements exceed those limits.

The accepted visual experiment applies the following limiting calculation:

    applied = min(
        high - 0.02,
        max(low + 0.02, requested)
    )

Where:

- `requested` is the measured physical angle converted to radians.
- `low` is the simulation's lower soft joint limit.
- `high` is the simulation's upper soft joint limit.
- `applied` is the final target sent to Isaac Lab.

The 0.02 rad margin keeps targets slightly inside the reported soft limits.

**The measured physical angle and applied simulation angle are not necessarily identical.**

For every joint, the script prints:

    PHYSICAL_DEX3
    measured=...
    applied=...
    difference=...
    status=EXACT or LIMITED

The final pose was visually accepted with this existing limit-handling behavior.

Do not remove the limit handling or expand the model's limits without validating the articulation model and intended motor behavior.

## 7. Why the left and right thumbs differ

The real G1 reported asymmetric thumb positions.

For example:

    Left thumb base:  +30.6766 degrees
    Right thumb base:  -2.4500 degrees

The left and right thumb values should not be automatically mirrored or replaced with zeros.

The preserved configuration uses the values measured independently from each physical hand.

Any further thumb adjustment should be tested separately without changing the accepted index and middle finger configuration.

## 8. Isaac Lab simulation configuration

Simulation host:

    NVIDIA DGX Spark

Project location:

    /home/rhasan/Desktop/razib/ai_models

Simulation framework:

    Isaac Lab 3.0
    Newton physics

Robot:

    Unitree G1 EDU
    Dex3 hands

Experimental container:

    g1-visual-experiment

Docker image:

    nvcr.io/nvidia/isaac-lab:3.0.0-kitless

Viewer:

    Viser

Host viewer port:

    8082

Python environment inside container:

    /opt/isaaclab-venv/bin/python

Saved simulation script:

    working_versions/g1_dex3_physical_hand_rest_pose_visual_verified_20261102/scripts/g1_4env_physical_dex3_pose_visual.py

Saved configuration file:

    working_versions/g1_dex3_physical_hand_rest_pose_visual_verified_20261102/configs/g1_4env_relaxed_hands_v2_cfg.py

### 8.1 Four-robot experiment

The experiment uses four fixed-base G1 robots.

Each robot receives the same set of 14 hand-joint targets, preserving the distinct physical left-hand and right-hand measurements.

The base is fixed during this calibration. This is not proof that the complete pose is stable during walking.

### 8.2 How hand targets are applied

The experiment interpolates from the initial hand positions toward the stored target positions.

The script then sends the updated targets using the Isaac Lab articulation interface:

    robot.set_joint_position_target(
        hand_targets,
        joint_ids=hand_indices,
    )

This controls the 14 Dex3 hand joints.

The original fixed-base experiment does not command the physical G1.

## 9. Viser visualization

The host publishes the experimental viewer on:

    127.0.0.1:8082

From the Mac, establish the SSH tunnel:

    ssh -N -o ExitOnForwardFailure=yes \
      -L 8082:127.0.0.1:8082 \
      rhasan@10.11.0.50

Open the viewer in the Mac browser:

    http://localhost:8082/

### Reproducing the experiment

Use the saved hand-pose script in the dedicated Isaac Lab experiment container, with:

- NVIDIA GPU access enabled.
- The project root mounted at `/workspace/local`.
- The saved simulation configuration directory included in `PYTHONPATH`.
- Isaac Lab's existing Python environment.
- Project-local cache and temporary directories.
- Viser's container port 8080 published as host port 8082.

The saved version contains the hand-pose configuration. Reproducing it requires the same compatible Isaac Lab environment.

Do not modify other containers, including the inference service or navigation simulation.

## 10. Physical G1 read-only hand-state measurement

Physical robot account:

    unitree@ubuntu

Existing reader directory:

    /home/unitree/dex3_readonly

Reader source:

    read_all_fingers.cpp

Reader executable:

    read_all_fingers

SDK:

    /home/unitree/Desktop/razib/unitree_sdk2_official

Physical network interface:

    eth0

The successful all-finger reader subscribed to:

    rt/dex3/left/state
    rt/dex3/right/state

The reader used the `HandState_` message and retrieved:

    motor_state()[i].q()

The raw values are in radians.

The program displayed all seven motor positions for each hand and collected more than 8,000 samples per hand.

The measurements were stable between sample 1 and sample 100.

### Physical robot safety

The hand-state reader is read-only.

It does not publish motor commands.

Do not force fingers into another position while the motors are actively resisting movement.

Do not assume standby mode means the finger actuators are unpowered.

Future physical robot experiments must use a separately reviewed and controlled movement procedure.

## 11. Earlier finger-closing comparison

Before the physical measurements were available, a four-robot experiment tested different finger-closing percentages.

| Robot | Configuration |
|---|---|
| G1_1 | Reference finger position |
| G1_2 | 50% finger closure |
| G1_3 | 75% finger closure |
| G1_4 | 90% finger closure |

That comparison demonstrated visible finger curling in Isaac Lab.

The earlier experiment is preserved in:

    working_versions/g1_dex3_finger_closing_visual_confirmed_20261102/

The later physical measurement experiment provided a stronger reference because it used the actual positions of all 14 Dex3 motors.

The two experiments serve different purposes:

- Finger-closing comparison: identify and observe joint movement.
- Physical resting-pose experiment: reproduce the real G1's accepted hand posture.

Neither is proof of walking stability.

## 12. Protected working versions

Do not modify or delete the following directories:

    working_versions/g1_isaaclab_forward_walk_030ms_60s_viser_verified/

    working_versions/4G1_walking_forward_60s_working/

    working_versions/g1_dex3_finger_closing_visual_confirmed_20261102/

    working_versions/4G1_pre_balance_thumb_fix_20261102_115652/

    working_versions/g1_dex3_physical_hand_rest_pose_visual_verified_20261102/

The verified walking configurations are separate from the hand-calibration experiments.

All future experiments should use new filenames and separate folders.

## 13. Project filesystem rules

All DGX Spark project files must remain under:

    /home/rhasan/Desktop/razib/ai_models

Disposable working files belong under:

    /home/rhasan/Desktop/razib/ai_models/tmp

Do not create project files, temporary files, caches, checkpoints, or logs outside the project root.

Preserve verified configurations and avoid modifying unrelated projects or Docker containers.

## 14. Verify this saved version

Enter the saved directory:

    cd /home/rhasan/Desktop/razib/ai_models/working_versions/g1_dex3_physical_hand_rest_pose_visual_verified_20261102

Check file integrity:

    sha256sum -c manifest.sha256

Inspect the physical measurements:

    cat physical_dex3_angles.json

Inspect the saved simulation targets:

    grep 'PHYSICAL_DEX3 G1_1' logs/verification.log

Inspect configuration confirmation when present:

    grep 'PHYSICAL_DEX3_CONFIG' logs/verification.log

If the active Docker logs were unavailable during saving, the saved verification log may not contain every runtime message. The user's visual confirmation remains the evidence for the accepted hand shape.

The checksum manifest applies to the saved files. If README.md is replaced with this revised documentation, regenerate its checksum in a new manifest or update the README.md entry intentionally.

## 15. Future integration with walking

The next development stage is to integrate the accepted Dex3 hand pose into a **copy** of the existing verified walking controller.

Recommended process:

1. Copy the verified walking experiment into a new working directory.
2. Load the 14 hand targets from `physical_dex3_angles.json`.
3. Apply the same validated simulation limit handling.
4. Preserve the existing 29-DOF locomotion policy and action mapping.
5. Run the new experiment in Isaac Lab before attempting physical deployment.
6. Check finger tracking and body stability during walking.
7. Preserve a new working version only after the walking test succeeds.

Do not overwrite the existing verified walking baseline.

## 16. Final verification status

| Component | Status |
|---|---|
| Physical Dex3 feedback | Measured |
| Left hand: 7 motors | Recorded |
| Right hand: 7 motors | Recorded |
| Physical motor-to-joint mapping | Documented |
| Isaac Lab hand target configuration | Implemented |
| Simulation soft-limit handling | Implemented |
| Four-robot fixed-base experiment | Executed |
| Desired hand shape | Visually confirmed |
| Saved configuration and checksums | Verified |
| Walking integration | Not yet tested |
| Physical robot control | Unchanged |

**This directory is the reference version for the visually accepted Dex3 resting-hand pose. Preserve it unchanged and build future experiments from copies.**
