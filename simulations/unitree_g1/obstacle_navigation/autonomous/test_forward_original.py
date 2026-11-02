#!/usr/bin/env python3
"""Isolated ONNX walking test. No browser or navigation planner."""

import math
import numpy as np
import mujoco
import live_demo as nav

nav.reset_sim()

model = nav.model
data = nav.data

# Disable only navigation-obstacle contacts for this test.
# Ground and robot contacts remain active.
for gid in nav.obstacle_ids:
    model.geom_contype[gid] = 0
    model.geom_conaffinity[gid] = 0

start_x = float(data.qpos[0])
start_y = float(data.qpos[1])

print("=== G1 Forward Locomotion Test ===", flush=True)
print("0–4 s: stabilization")
print("4–7 s: forward command 0.15 m/s")
print("7–9 s: stop")
print()

for step in range(int(9.0 / nav.dt)):
    t = float(data.time)

    if t < 4.0:
        nav.command[:] = [0.0, 0.0, 0.0]
    elif t < 7.0:
        nav.command[:] = [0.15, 0.0, 0.0]
    else:
        nav.command[:] = [0.0, 0.0, 0.0]

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

    # Match the stabilization logic from the live demo.
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
            model, data,
            mujoco.mjtObj.mjOBJ_BODY,
            nav.pelvis, vel, 0
        )

        angular_support = (
            180.0 * np.cross(
                rotation[:, 2],
                np.array([0.0, 0.0, 1.0])
            )
            - 35.0 * vel[:3]
        )

        data.xfrc_applied[nav.pelvis, :3] = support * force
        data.xfrc_applied[nav.pelvis, 3:] = (
            support * angular_support
        )

    mujoco.mj_step(model, data)

    if step % int(round(1.0 / nav.dt)) == 0:
        mujoco.mj_forward(model, data)

        rotation = data.xmat[nav.pelvis].reshape(3, 3)

        tilt = math.degrees(math.acos(
            np.clip(rotation[2, 2], -1.0, 1.0)
        ))

        print(
            f"time={data.time:5.2f}s "
            f"x={data.qpos[0]:+.3f} "
            f"y={data.qpos[1]:+.3f} "
            f"z={data.qpos[2]:.3f} "
            f"tilt={tilt:.1f}deg "
            f"cmd={nav.command[0]:.2f} "
            f"action_rms={np.sqrt(np.mean(nav.last_action**2)):.3f}",
            flush=True
        )

    if not np.isfinite(data.qpos).all():
        print("FAIL: nonfinite simulation state")
        break

print()
print("=== Final result ===")
print(f"X displacement: {data.qpos[0] - start_x:+.3f} m")
print(f"Y displacement: {data.qpos[1] - start_y:+.3f} m")
print(f"Final height: {data.qpos[2]:.3f} m")
