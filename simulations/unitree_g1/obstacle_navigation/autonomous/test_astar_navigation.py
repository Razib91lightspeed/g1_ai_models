"""Stationary sensor-map and A* test with an unknown obstacle."""

import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

import live_demo as nav
from lidar_mapper import LidarMapper
from path_planner import AStarPlanner


nav.reset_sim()
model = nav.model
data = nav.data

# Scenario setup belongs ONLY to the simulator.
# The mapper and A* planner never receive these positions.
for gid in nav.obstacle_ids:
    model.geom_pos[gid] = [100.0, 100.0, 1.0]
    model.geom_contype[gid] = 0
    model.geom_conaffinity[gid] = 0

mujoco.mj_forward(model, data)

mapper = LidarMapper(
    model,
    xlim=(-1.0, 4.5),
    ylim=(-2.0, 2.0),
    resolution=0.10,
    max_range=5.0,
)

planner = AStarPlanner(
    mapper,
    clearance=0.40,
    allow_unknown=False
)

start = np.array([0.0, 0.0])
goal = np.array([3.048, 0.0])

print("=== Before obstacle ===", flush=True)

origin, measurements = mapper.observe(data)
print("LiDAR hits:", sum(m["hit"] for m in measurements))

initial_path, initial_status = planner.plan(start, goal)
print("A*:", initial_status)

# The test environment introduces a new obstacle.
gid = model.geom("nav_obstacle_01").id
model.geom_pos[gid] = [1.50, 0.0, 0.50]
model.geom_size[gid] = [0.22, 0.35, 0.50]
model.geom_contype[gid] = 1
model.geom_conaffinity[gid] = 1

mujoco.mj_forward(model, data)

# Reconstruct the map using the new sensor readings.
mapper.reset()
origin, measurements = mapper.observe(data)

print("\n=== After surprise obstacle ===")
print("LiDAR hits:", sum(m["hit"] for m in measurements))
print("Occupied cells:", int(np.sum(
    mapper.grid == mapper.OCCUPIED
)))
print("Observed free cells:", int(np.sum(
    mapper.grid == mapper.FREE
)))
print("Unknown cells:", int(np.sum(
    mapper.grid == mapper.UNKNOWN
)))

path, status = planner.plan(start, goal)
print("A*:", status)

if path is not None:
    length = float(np.sum(
        np.linalg.norm(np.diff(path, axis=0), axis=1)
    ))
    print("Path points:", len(path))
    print("Path length:", round(length, 3), "m")
else:
    print("No fully observed safe route found.")
    print("Navigation must STOP and gather more observations.")

# Render a 2D diagnostic map.
blocked = planner.build_blocked_map()

image = np.zeros(
    (mapper.height, mapper.width, 3),
    dtype=np.uint8
)

image[:] = (95, 101, 112)
image[mapper.grid == mapper.FREE] = (235, 240, 245)
image[blocked] = (245, 185, 92)
image[mapper.grid == mapper.OCCUPIED] = (220, 65, 45)

# Flip vertical axis: world +Y points up in the drawing.
scale = 9
canvas = Image.fromarray(np.flipud(image)).resize(
    (mapper.width*scale, mapper.height*scale),
    Image.Resampling.NEAREST
)

draw = ImageDraw.Draw(canvas)

def pixel(xy):
    cell = mapper.world_to_cell(*xy)
    if cell is None:
        return None

    row, col = cell
    return (
        int((col+0.5)*scale),
        int((mapper.height-row-0.5)*scale)
    )

if path is not None:
    points = [pixel(p) for p in path]
    points = [p for p in points if p is not None]

    if len(points) > 1:
        draw.line(points, fill=(25, 100, 240), width=4)

for point, color in (
    (start, (15, 95, 255)),
    (goal, (30, 180, 75))
):
    p = pixel(point)
    if p:
        x, y = p
        draw.ellipse(
            (x-8, y-8, x+8, y+8),
            fill=color,
            outline=(255, 255, 255),
            width=2
        )

out = Path("/tmp/g1_astar_navigation.png")
canvas.save(out)

print("\nMap image:", out)
print("Test completed.")
