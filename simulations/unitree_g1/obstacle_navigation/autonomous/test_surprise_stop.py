#!/usr/bin/env python3
"""Walk, encounter an unexpected obstacle, stop. Simulation only."""

import math
import numpy as np
import mujoco
import live_demo as nav

nav.reset_sim()
model = nav.model
data = nav.data

# Keep all five obstacles inactive until the surprise event.
for gid in nav.obstacle_ids:
    model.geom_pos[gid] = [100.0, 100.0, 1.0]
    model.geom_contype[gid] = 0
    model.geom_conaffinity[gid] = 0

surprise_id = model.geom("nav_obstacle_01").id
mujoco.mj_forward(model, data)

spawned = False
detected = False
collision = False
stop_time = None
spawn_position = None
first_detection = None

print("=== G1 Surprise Obstacle Stop Test ===", flush=True)
print("0-4 s: stabilize")
print("4-6 s: walk at 0.40 m/s")
print("6 s: obstacle appears ahead")
print("After detection: command STOP")
print()

total_steps = int(12.0 / nav.dt)

for step in range(total_steps):
    t = float(data.time)

    # The simulator controls insertion.
    # Navigation does not receive obstacle coordinates.
    if t >= 6.0 and not spawned:
        heading = nav.yaw_angle()

        forward = np.array([
            math.cos(heading), math.sin(heading)
        ])

        robot_xy = np.asarray(data.qpos[:2]).copy()
        spawn_position = robot_xy + 1.50 * forward

        model.geom_pos[surprise_id] = [
            float(spawn_position[0]),
            float(spawn_position[1]),
            0.50
        ]
        model.geom_size[surprise_id] = [0.22, 0.35, 0.50]
        model.geom_contype[surprise_id] = 1
        model.geom_conaffinity[surprise_id] = 1

        mujoco.mj_forward(model, data)
        spawned = True

        print(
            f"SPAWN: t={t:.2f}s "
            f"obstacle placed 1.50m ahead",
            flush=True
        )

    if t < 4.0:
        nav.command[:] = [0.0, 0.0, 0.0]

    elif not detected:
        nav.command[:] = [0.40, 0.0, 0.0]

    else:
        nav.command[:] = [0.0, 0.0, 0.0]

    # Sensor feedback, every 0.1 seconds.
    if step % max(1, int(round(0.1 / nav.dt))) == 0:
        mujoco.mj_forward(model, data)
        nav.scan_lidar()

        # Only obstacle returns in the forward 30-degree cone.
        front = [
            distance
            for angle, distance, _ in nav.last_scan
            if abs(nav.wrap(math.radians(angle))) <
               math.radians(30)
        ]

        front_distance = min(front) if front else None

        if (
            spawned and not detected
            and front_distance is not None
            and front_distance <= 1.30
        ):
            detected = True
            stop_time = t
            first_detection = front_distance
            nav.command[:] = 0.0

            print(
                f"DETECTED: t={t:.2f}s "
                f"front range={front_distance:.3f}m",
                flush=True
            )
            print("STOP requested", flush=True)

    # Known-working ONNX policy and PD controller.
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

    # Identical stabilization technique to test_forward.py.
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
                rotation[:, 2], [0.0, 0.0, 1.0]
            )
            - 35.0 * vel[:3]
        )

        data.xfrc_applied[nav.pelvis, :3] = support * force
        data.xfrc_applied[nav.pelvis, 3:] = (
            support * angular_support
        )

    mujoco.mj_step(model, data)

    for ci in range(data.ncon):
        contact = data.contact[ci]
        if (
            contact.geom1 == surprise_id
            or contact.geom2 == surprise_id
        ):
            collision = True
            break

    if step % int(round(1.0 / nav.dt)) == 0:
        print(
            f"t={data.time:5.2f}s "
            f"x={data.qpos[0]:+.3f} "
            f"y={data.qpos[1]:+.3f} "
            f"cmd={nav.command[0]:.2f} "
            f"nearest={nav.nearest} "
            f"collision={collision}",
            flush=True
        )

    if collision or not np.isfinite(data.qpos).all():
        print("TEST STOPPED: collision or invalid state")
        break

print("\n=== RESULTS ===")
print("Surprise obstacle spawned:", spawned)
print("Detected via LiDAR:", detected)
print("First detection distance:", first_detection)
print("Stop command at:", stop_time)
print("Collision:", collision)
print("Final position:", data.qpos[:3])

if spawned and detected and not collision:
    print("PASS: obstacle detected and no contact observed")
else:
    print("NOT PASSED: inspect detection and collision results")
