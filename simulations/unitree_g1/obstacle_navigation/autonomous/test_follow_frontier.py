"""Short, controlled A* route-following experiment."""

import math
import numpy as np
import mujoco

import live_demo as nav
from lidar_mapper import LidarMapper
from path_planner import AStarPlanner
from frontier_planner import select_frontier

nav.reset_sim()
model, data = nav.model, nav.data

# Test environment only: hide unused obstacles.
for gid in nav.obstacle_ids:
    model.geom_pos[gid] = [100., 100., 1.]
    model.geom_contype[gid] = 0
    model.geom_conaffinity[gid] = 0

# The planner never reads these coordinates.
obstacle = model.geom("nav_obstacle_01").id
model.geom_pos[obstacle] = [1.50, 0.0, 0.50]
model.geom_size[obstacle] = [0.22, 0.35, 0.50]
model.geom_contype[obstacle] = 1
model.geom_conaffinity[obstacle] = 1

mujoco.mj_forward(model, data)

mapper = LidarMapper(model)
mapper.observe(data)
planner = AStarPlanner(mapper, clearance=0.40)

result = select_frontier(
    mapper, planner,
    np.array([0., 0.]),
    np.array([3.048, 0.])
)

if result["status"] != "frontier_found":
    raise RuntimeError("No verified frontier path")

path = result["path"]
print("Planned waypoints:", len(path))
print("Frontier target:", result["target"])

initial_xy = data.qpos[:2].copy()
max_travel = 0.50
collision = False
stopped_for_obstacle = False

print("Starting short route-following test")

for step in range(int(12.0 / nav.dt)):
    t = float(data.time)
    xy = data.qpos[:2].copy()
    travelled = float(np.linalg.norm(xy - initial_xy))

    if t < 4.0 or travelled >= max_travel:
        nav.command[:] = 0

    else:
        mapper.observe(data)

        # Follow an observed-free lookahead point on the path.
        distances = np.linalg.norm(path - xy, axis=1)
        nearest_idx = int(np.argmin(distances))

        lookahead_idx = nearest_idx
        while lookahead_idx < len(path) - 1:
            if np.linalg.norm(
                path[lookahead_idx] - xy
            ) >= 0.35:
                break
            lookahead_idx += 1

        target = path[lookahead_idx]
        dx, dy = target - xy

        desired_heading = math.atan2(dy, dx)
        heading_error = nav.wrap(
            desired_heading - nav.yaw_angle()
        )

        # A preliminary obstacle proximity stop.
        # Uses sensor measurements, not obstacle coordinates.
        origin, rays = mapper.scan(data)
        forward_hits = [
            r["range"] for r in rays
            if r["hit"] and abs(
                nav.wrap(math.radians(r["angle_deg"]))
            ) <= math.radians(25)
        ]

        if forward_hits and min(forward_hits) < 0.70:
            nav.command[:] = 0
            stopped_for_obstacle = True
        elif stopped_for_obstacle:
            nav.command[:] = 0
        else:
            yaw = float(np.clip(
                1.0 * heading_error, -0.30, 0.30
            ))

            # Do not force forward movement when facing
            # substantially away from the planned route.
            # This ONNX policy has demonstrated reliable
            # turning while walking, but not turning in place.
            # Maintain forward locomotion while steering.
            nav.command[:] = [0.40, 0.0, yaw]

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

    # Reuse the previously tested startup support.
    data.xfrc_applied[nav.pelvis, :] = 0
    support = nav.clamp((4.0 - t) / 2.0, 0.0, 1.0)

    if support > 0:
        mujoco.mj_forward(model, data)

        force = (
            np.array([0., 0., nav.total_mass * 9.81])
            + 250.0 * (
                np.array([0., 0., nav.initial_z])
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

        data.xfrc_applied[nav.pelvis, :3] = support * force
        data.xfrc_applied[nav.pelvis, 3:] = support * (
            180.0 * np.cross(
                rotation[:, 2], [0., 0., 1.]
            ) - 35.0 * vel[:3]
        )

    mujoco.mj_step(model, data)

    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 == obstacle or c.geom2 == obstacle:
            collision = True
            break

    if step % 500 == 0:
        print(
            f"t={data.time:.1f} "
            f"x={data.qpos[0]:.3f} "
            f"y={data.qpos[1]:.3f} "
            f"heading={math.degrees(nav.yaw_angle()):+.1f}deg "
            f"cmd={nav.command.round(2)}",
            flush=True
        )

    if collision or not np.isfinite(data.qpos).all():
        break

print("\n=== TEST RESULT ===")
print("Final position:", data.qpos[:2])
print("Travel:", np.linalg.norm(data.qpos[:2] - initial_xy))
print("Collision:", collision)
print("Proximity stop:", stopped_for_obstacle)
print("Final height:", data.qpos[2])
