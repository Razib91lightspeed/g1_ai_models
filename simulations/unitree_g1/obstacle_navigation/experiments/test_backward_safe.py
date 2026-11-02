#!/usr/bin/env python3
"""Isolated G1 backward locomotion screening test. MuJoCo only."""

import argparse
import math

import mujoco
import numpy as np

import live_demo_v8 as nav


def tilt_degrees():
    mujoco.mj_forward(model, data)
    rotation = data.xmat[nav.pelvis].reshape(3, 3)
    return math.degrees(
        math.acos(float(np.clip(rotation[2, 2], -1.0, 1.0)))
    )


parser = argparse.ArgumentParser()
parser.add_argument(
    "--speed", type=float, choices=(-0.15, -0.20), required=True
)
args = parser.parse_args()

nav.reset_sim()
model = nav.model
data = nav.data

# Isolated flat-ground test. Keep robot-ground contacts enabled.
for gid in nav.obstacle_ids:
    model.geom_contype[gid] = 0
    model.geom_conaffinity[gid] = 0

# Match the control and startup support of test_forward_v8.py.
initial_xy = data.qpos[:2].copy()
initial_yaw = nav.yaw_angle()
forward_axis = np.array([
    math.cos(initial_yaw), math.sin(initial_yaw)
])

MAX_REVERSE_M = 0.25
MAX_TILT_DEG = 15.0
MIN_HEIGHT_M = 0.65
STABILIZE_S = 4.0
MAX_ACTIVE_S = 5.0
STOP_OBSERVATION_S = 3.0

phase = "stabilizing"
active_start = None
stop_start = None
stop_reason = None
peak_tilt = 0.0
min_height = float(data.qpos[2])
maximum_reverse = 0.0
last_log_time = -1.0

print("=== G1 BACKWARD LOCOMOTION SCREENING ===", flush=True)
print(f"Requested speed: {args.speed:.2f} m/s", flush=True)
print("Simulation only; original locomotion policy unchanged", flush=True)

while True:
    t = float(data.time)

    if phase == "stabilizing" and t >= STABILIZE_S:
        phase = "reversing"
        active_start = t
        print("EVENT: reverse command started", flush=True)

    displacement = data.qpos[:2] - initial_xy
    signed_forward = float(np.dot(displacement, forward_axis))
    reverse_distance = max(0.0, -signed_forward)

    if phase == "reversing":
        nav.command[:] = [args.speed, 0.0, 0.0]

        tilt = tilt_degrees()
        height = float(data.qpos[2])

        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            stop_reason = "nonfinite_state"
        elif tilt > MAX_TILT_DEG:
            stop_reason = "tilt_limit"
        elif height < MIN_HEIGHT_M:
            stop_reason = "height_limit"
        elif reverse_distance >= MAX_REVERSE_M:
            stop_reason = "distance_limit"
        elif t - active_start >= MAX_ACTIVE_S:
            stop_reason = "duration_limit"

        if stop_reason is not None:
            phase = "stopping"
            stop_start = t
            nav.command[:] = 0.0
            print(f"EVENT: reverse ended: {stop_reason}", flush=True)

    else:
        nav.command[:] = 0.0

    if int(round(t / nav.dt)) % nav.interval == 0:
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

    # Preserve the original test's stabilization support.
    data.xfrc_applied[nav.pelvis, :] = 0
    support = nav.clamp((4.0 - t) / 2.0, 0.0, 1.0)

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
            model, data, mujoco.mjtObj.mjOBJ_BODY,
            nav.pelvis, vel, 0
        )

        angular_support = (
            180.0 * np.cross(
                rotation[:, 2], np.array([0.0, 0.0, 1.0])
            )
            - 35.0 * vel[:3]
        )

        data.xfrc_applied[nav.pelvis, :3] = support * force
        data.xfrc_applied[nav.pelvis, 3:] = (
            support * angular_support
        )

    mujoco.mj_step(model, data)

    # Measure after physics integration.
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        print("FAIL: nonfinite simulation state", flush=True)
        break

    tilt = tilt_degrees()
    height = float(data.qpos[2])
    displacement = data.qpos[:2] - initial_xy
    signed_forward = float(np.dot(displacement, forward_axis))
    reverse_distance = max(0.0, -signed_forward)

    peak_tilt = max(peak_tilt, tilt)
    min_height = min(min_height, height)
    maximum_reverse = max(maximum_reverse, reverse_distance)

    if phase == "reversing":
        if (
            tilt > MAX_TILT_DEG
            or height < MIN_HEIGHT_M
            or reverse_distance >= MAX_REVERSE_M
        ):
            stop_reason = "post_step_safety_limit"
            phase = "stopping"
            stop_start = float(data.time)
            nav.command[:] = 0.0
            print("EVENT: safety limit reached", flush=True)

    if float(data.time) - last_log_time >= 0.5:
        last_log_time = float(data.time)
        print(
            f"t={data.time:.2f}s "
            f"phase={phase} "
            f"reverse={reverse_distance:.3f}m "
            f"height={height:.3f}m "
            f"tilt={tilt:.2f}deg "
            f"command={nav.command[0]:+.2f}",
            flush=True
        )

    if phase == "stopping":
        if (
            tilt > MAX_TILT_DEG
            or height < MIN_HEIGHT_M
        ):
            print("FAIL: stability threshold during stopping", flush=True)
            break

        if float(data.time) - stop_start >= STOP_OBSERVATION_S:
            break

print()
print("=== FINAL RESULT ===")
print(f"Requested reverse command: {args.speed:.2f} m/s")
print(f"Maximum reverse displacement: {maximum_reverse:.3f} m")
print(f"Final body-axis displacement: {signed_forward:+.3f} m")
print(f"Maximum tilt: {peak_tilt:.2f} deg")
print(f"Minimum pelvis height: {min_height:.3f} m")
print(f"Stop reason: {stop_reason or 'safety_abort'}")

if (
    maximum_reverse >= 0.10
    and peak_tilt <= MAX_TILT_DEG
    and min_height >= MIN_HEIGHT_M
    and stop_reason in ("duration_limit", "distance_limit")
):
    print("SCREENING: reverse movement demonstrated")
else:
    print("SCREENING: reverse movement not established or safety abort")
