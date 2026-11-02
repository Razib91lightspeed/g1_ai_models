#!/usr/bin/env python3
"""Simulation-only LiDAR occupancy mapping test.

Uses MuJoCo ray casting against obstacle geometry group 3.
Does not connect to the physical Unitree G1.
"""

from pathlib import Path
import math

import mujoco
import numpy as np
from PIL import Image

SCENE = (
    "/workspace/unitree_mujoco/unitree_robots/g1/"
    "scene_nav_lidar_29dof.xml"
)

OUTPUT = Path("/tmp/g1_nav_map")

RESOLUTION = 0.10
X_MIN, X_MAX = -1.0, 8.0
Y_MIN, Y_MAX = -3.0, 3.0

MAX_RANGE = 8.0
MIN_RANGE = 0.20

WIDTH = int(round((X_MAX - X_MIN) / RESOLUTION))
HEIGHT = int(round((Y_MAX - Y_MIN) / RESOLUTION))

# Unknown = -1, Free = 0, Occupied = 100
grid = np.full((HEIGHT, WIDTH), -1, dtype=np.int16)

model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

torso_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "torso_link"
)

rotation = data.xmat[torso_id].reshape(3, 3)
origin = data.xpos[torso_id] + rotation @ np.array(
    [0.0, 0.0, 0.15]
)

geomgroup = np.zeros(6, dtype=np.uint8)
geomgroup[3] = 1


def world_to_grid(x, y):
    gx = int(math.floor((x - X_MIN) / RESOLUTION))
    gy = int(math.floor((y - Y_MIN) / RESOLUTION))

    if 0 <= gx < WIDTH and 0 <= gy < HEIGHT:
        return gx, gy

    return None


def trace_cells(start, end):
    """Return all grid cells crossed by a 2D line."""

    distance = float(np.linalg.norm(end[:2] - start[:2]))
    steps = max(1, int(math.ceil(distance / (RESOLUTION * 0.5))))

    cells = []
    previous = None

    for i in range(steps + 1):
        fraction = i / steps
        p = start + fraction * (end - start)

        cell = world_to_grid(float(p[0]), float(p[1]))

        if cell is not None and cell != previous:
            cells.append(cell)
            previous = cell

    return cells


def scan_lidar():
    returns = []
    free_cells = set()
    occupied_cells = set()

    vertical_angles = [-15, -7.5, 0, 7.5, 15]

    for elevation in vertical_angles:
        e = math.radians(elevation)

        for angle in range(360):
            a = math.radians(angle)

            direction_local = np.array([
                math.cos(e) * math.cos(a),
                math.cos(e) * math.sin(a),
                math.sin(e)
            ])

            direction = rotation @ direction_local
            direction /= np.linalg.norm(direction)

            hit_id = np.array([-1], dtype=np.int32)

            distance = mujoco.mj_ray(
                model,
                data,
                origin,
                direction,
                geomgroup,
                1,
                -1,
                hit_id
            )

            valid_hit = (
                hit_id[0] >= 0
                and MIN_RANGE <= distance <= MAX_RANGE
            )

            # Use maximum sensing range for missed rays.
            ray_distance = float(distance) if valid_hit else MAX_RANGE
            end = origin + ray_distance * direction

            # Limit mapping to rays near the obstacle-height band.
            # This avoids treating overhead or ground-directed rays
            # as evidence of free space throughout the 2D map.
            if abs(float(direction[2])) > 0.14:
                continue

            cells = trace_cells(origin, end)

            if not cells:
                continue

            if valid_hit:
                free_cells.update(cells[:-1])
                occupied_cells.add(cells[-1])
                returns.append(end.copy())
            else:
                free_cells.update(cells)

    # Apply free observations first, then occupied observations.
    # Occupied cells take precedence when rays disagree.
    for gx, gy in free_cells:
        grid[gy, gx] = 0

    for gx, gy in occupied_cells:
        grid[gy, gx] = 100

    return np.asarray(returns).reshape(-1, 3)


def save_results(points):
    OUTPUT.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT / "occupancy_grid.npy", grid)
    np.save(OUTPUT / "lidar_points.npy", points)

    # White = free, gray = unknown, orange = occupied
    image = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[:] = [100, 105, 115]
    image[grid == 0] = [235, 240, 245]
    image[grid == 100] = [240, 125, 45]

    # Mark robot position in blue
    robot_cell = world_to_grid(float(origin[0]), float(origin[1]))

    if robot_cell:
        gx, gy = robot_cell
        image[gy, gx] = [45, 125, 255]

    image = np.flipud(image)

    # Resize with nearest neighbor so cell boundaries remain clear
    Image.fromarray(image).resize(
        (WIDTH * 6, HEIGHT * 6),
        Image.Resampling.NEAREST
    ).save(OUTPUT / "occupancy_map.png")


points = scan_lidar()
save_results(points)

print("=== G1 LiDAR occupancy map ===")
print("Map dimensions:", WIDTH, "x", HEIGHT)
print("Resolution:", RESOLUTION, "m/cell")
print("Map area:", X_MAX - X_MIN, "x", Y_MAX - Y_MIN, "m")
print("Valid mapped returns:", len(points))
print("Occupied cells:", int(np.sum(grid == 100)))
print("Free cells:", int(np.sum(grid == 0)))
print("Unknown cells:", int(np.sum(grid == -1)))
print("Output:", OUTPUT)
print("Mapping test completed.")
