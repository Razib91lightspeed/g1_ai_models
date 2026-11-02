#!/usr/bin/env python3
"""G1 reverse-recovery screening. MuJoCo only."""

import argparse
import math
import numpy as np
import mujoco
import live_demo_v8 as nav

parser = argparse.ArgumentParser()
parser.add_argument("--trial", type=int, required=True)
args = parser.parse_args()

nav.reset_sim()

model = nav.model
data = nav.data

# Disable navigation obstacles for the isolated locomotion test.
# Ground contacts and robot contacts remain enabled.
for gid in nav.obstacle_ids:
    model.geom_contype[gid] = 0
    model.geom_conaffinity[gid] = 0

SPEED = -0.15
TARGET_M = 0.10
BRAKE_AT_M = 0.07
HARD_LIMIT_M = 0.25

STABILIZATION_S = 4.0
MAX_REVERSE_S = 5.0
STOP_OBSERVATION_S = 3.0

MAX_TILT_DEG = 15.0
MIN_HEIGHT_M = 0.65

start_xy = data.qpos[:2].copy()
start_yaw = nav.yaw_angle()

forward_axis = np.array([
    math.cos(start_yaw),
    math.sin(start_yaw)
])

phase = "stabilizing"
reverse_start = None
brake_start = None
brake_position = None
stop_reason = None

max_reverse = 0.0
max_tilt = 0.0
min_height = float(data.qpos[2])
last_print = -1.0
unsafe = False


def measurements():
    mujoco.mj_forward(model, data)

    rotation = data.xmat[nav.pelvis].reshape(3, 3)

    tilt = math.degrees(
        math.acos(float(np.clip(rotation[2, 2], -1.0, 1.0)))
    )

    delta = data.qpos[:2] - start_xy
    signed = float(np.dot(delta, forward_axis))

    return -signed, tilt, float(data.qpos[2])


def apply_control(t, step):
    if step % nav.interval == 0:
        nav.update_policy()

    torque = (
        nav.kp * (nav.target - data.qpos[nav.qids])
        - nav.kd * data.qvel[nav.vids]
    )

    data.ctrl[nav.ids] = np.clip(
        torque,
        model.actuator_ctrlrange[nav.ids, 0],
        model.actuator_ctrlrange[nav.ids, 1]
    )

    # Same startup support as the existing forward test.
    data.xfrc_applied[nav.pelvis, :] = 0.0

    support = nav.clamp(
        (4.0 - t) / 2.0, 0.0, 1.0
    )

    if support > 0:
        mujoco.mj_forward(model, data)

        force = (
            np.array([0.0, 0.0, nav.total_mass * 9.81])
            + 250.0 * (
                np.array([0.0, 0.0, nav.initial_z])
                - data.qpos[:3]
            )
            - 60.0 * data.qvel[:3]
        )

        rotation = data.xmat[nav.pelvis].reshape(3, 3)

        vel = np.zeros(6)

        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            nav.pelvis,
            vel,
            0
        )

        angular = (
            180.0 * np.cross(
                rotation[:, 2],
                np.array([0.0, 0.0, 1.0])
            )
            - 35.0 * vel[:3]
        )

        data.xfrc_applied[nav.pelvis, :3] = (
            support * force
        )

        data.xfrc_applied[nav.pelvis, 3:] = (
            support * angular
        )

    mujoco.mj_step(model, data)


print(f"\n=== REVERSE TRIAL {args.trial}/5 ===", flush=True)
print("Speed command: -0.15 m/s", flush=True)
print("Target reverse distance: 0.10 m", flush=True)
print("Early braking starts at: 0.07 m", flush=True)

max_steps = int(14.0 / nav.dt)

for step in range(max_steps):
    t = float(data.time)

    reverse, tilt, height = measurements()

    if phase == "stabilizing" and t >= STABILIZATION_S:
        phase = "reversing"
        reverse_start = t
        print("EVENT: reverse started", flush=True)

    if phase == "reversing":
        nav.command[:] = [SPEED, 0.0, 0.0]

        if reverse >= BRAKE_AT_M:
            stop_reason = "early_braking_target"

        elif t - reverse_start >= MAX_REVERSE_S:
            stop_reason = "reverse_timeout"

        if stop_reason:
            phase = "braking"
            brake_start = t
            brake_position = reverse
            nav.command[:] = 0.0

            print(
                f"EVENT: braking at {reverse:.3f} m, "
                f"reason={stop_reason}",
                flush=True
            )

    else:
        nav.command[:] = 0.0

    if (
        not np.isfinite(data.qpos).all()
        or not np.isfinite(data.qvel).all()
        or tilt > MAX_TILT_DEG
        or height < MIN_HEIGHT_M
        or reverse >= HARD_LIMIT_M
    ):
        unsafe = True
        nav.command[:] = 0.0
        stop_reason = "safety_limit"
        print("SAFETY: test aborted", flush=True)
        break

    apply_control(t, step)

    if (
        not np.isfinite(data.qpos).all()
        or not np.isfinite(data.qvel).all()
    ):
        unsafe = True
        stop_reason = "nonfinite_state"
        print("SAFETY: invalid simulation state", flush=True)
        break

    reverse, tilt, height = measurements()

    max_reverse = max(max_reverse, reverse)
    max_tilt = max(max_tilt, tilt)
    min_height = min(min_height, height)

    # Guard immediately after each physics step.
    if (
        reverse >= HARD_LIMIT_M
        or tilt > MAX_TILT_DEG
        or height < MIN_HEIGHT_M
    ):
        unsafe = True
        stop_reason = "post_step_safety_limit"
        nav.command[:] = 0.0
        print("SAFETY: post-step limit exceeded", flush=True)
        break

    if t - last_print >= 0.5:
        last_print = t

        print(
            f"t={t:.2f}s "
            f"phase={phase} "
            f"reverse={reverse:.3f}m "
            f"tilt={tilt:.2f}deg "
            f"height={height:.3f}m",
            flush=True
        )

    if (
        phase == "braking"
        and float(data.time) - brake_start >= STOP_OBSERVATION_S
    ):
        break

final_reverse, final_tilt, final_height = measurements()

braking_drift = (
    final_reverse - brake_position
    if brake_position is not None
    else None
)

passed = (
    not unsafe
    and stop_reason == "early_braking_target"
    and 0.05 <= final_reverse <= TARGET_M
    and max_reverse < HARD_LIMIT_M
    and max_tilt <= MAX_TILT_DEG
    and min_height >= MIN_HEIGHT_M
    and braking_drift is not None
    and abs(braking_drift) <= 0.05
)

print("\n=== TRIAL RESULT ===")
print(f"Trial: {args.trial}")
print(f"Maximum reverse displacement: {max_reverse:.3f} m")
print(f"Final reverse displacement: {final_reverse:.3f} m")
print(f"Braking displacement change: {braking_drift}")
print(f"Maximum tilt: {max_tilt:.2f} deg")
print(f"Minimum pelvis height: {min_height:.3f} m")
print(f"Stop reason: {stop_reason}")
print(f"Trial passed: {passed}", flush=True)

if unsafe:
    raise SystemExit(2)

if not passed:
    raise SystemExit(1)
