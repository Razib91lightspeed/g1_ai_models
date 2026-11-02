import mujoco
import numpy as np

import live_demo as nav
from lidar_mapper import LidarMapper
from path_planner import AStarPlanner
from frontier_planner import select_frontier

nav.reset_sim()

model = nav.model
data = nav.data

# Environment-only setup: clear original obstacles.
for gid in nav.obstacle_ids:
    model.geom_pos[gid] = [100.0, 100.0, 1.0]
    model.geom_contype[gid] = 0
    model.geom_conaffinity[gid] = 0

# Environment-only setup: insert an unknown obstacle.
gid = model.geom("nav_obstacle_01").id

model.geom_pos[gid] = [1.50, 0.0, 0.50]
model.geom_size[gid] = [0.22, 0.35, 0.50]
model.geom_contype[gid] = 1
model.geom_conaffinity[gid] = 1

mujoco.mj_forward(model, data)

mapper = LidarMapper(model)
mapper.observe(data)

planner = AStarPlanner(
    mapper,
    clearance=0.40,
    allow_unknown=False
)

start = np.array([0.0, 0.0])
goal = np.array([3.048, 0.0])

direct_path, direct_status = planner.plan(start, goal)

print("=== DIRECT GOAL PLANNING ===")
print("Result:", direct_status)

print("\n=== FRONTIER EXPLORATION ===")
result = select_frontier(
    mapper, planner, start, goal
)

print("Result:", result["status"])

if result["target"] is not None:
    print(
        "Exploration target:",
        np.round(result["target"], 3)
    )
    print("A* path points:", len(result["path"]))

    length = np.sum(np.linalg.norm(
        np.diff(result["path"], axis=0),
        axis=1
    ))
    print("Path length:", round(float(length), 3), "m")
    print("Next action: navigate to the viewpoint and rescan")
else:
    print("Next action: remain stopped")
    print("Reason: no verified reachable viewpoint")

print("\nTest complete.")
