import mujoco
import numpy as np
import live_demo as nav

nav.reset_sim()

model = nav.model
data = nav.data

# Disable all navigation obstacles initially.
for gid in nav.obstacle_ids:
    model.geom_contype[gid] = 0
    model.geom_conaffinity[gid] = 0
    model.geom_rgba[gid, 3] = 0

# LiDAR only considers geometries in group 3.
# Move unused obstacles far outside the test area.
for gid in nav.obstacle_ids:
    model.geom_pos[gid] = [100.0, 100.0, 1.0]

mujoco.mj_forward(model, data)

def report(label):
    nav.scan_lidar()

    print(f"\n=== {label} ===")
    print("Robot position:", data.qpos[:3])
    print("Nearest LiDAR return:", nav.nearest)
    print("Obstacle returns:", len(nav.last_scan))

report("Empty corridor")

# Introduce one unexpected obstacle in front of G1.
gid = model.geom("nav_obstacle_01").id

model.geom_pos[gid] = [1.2, 0.0, 0.5]
model.geom_size[gid] = [0.20, 0.25, 0.5]

model.geom_contype[gid] = 1
model.geom_conaffinity[gid] = 1
model.geom_rgba[gid, 3] = 1

mujoco.mj_forward(model, data)

report("Surprise obstacle inserted")

print("\nTest complete.")
