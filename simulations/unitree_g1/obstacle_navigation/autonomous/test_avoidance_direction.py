import math
import mujoco
import live_demo as nav

nav.reset_sim()
model = nav.model
data = nav.data

# Start with an empty corridor.
for gid in nav.obstacle_ids:
    model.geom_pos[gid] = [100, 100, 1]
    model.geom_contype[gid] = 0
    model.geom_conaffinity[gid] = 0

# Test environment inserts a previously unknown obstacle.
gid = model.geom("nav_obstacle_01").id
model.geom_pos[gid] = [1.5, 0.0, 0.5]
model.geom_size[gid] = [0.22, 0.35, 0.5]
model.geom_contype[gid] = 1
model.geom_conaffinity[gid] = 1

mujoco.mj_forward(model, data)
nav.scan_lidar()

left = []
right = []

for angle_deg, distance, point in nav.last_scan:
    angle = nav.wrap(math.radians(angle_deg))

    if 0 < angle < math.pi / 2:
        left.append(distance)
    elif -math.pi / 2 < angle < 0:
        right.append(distance)

print("=== UNKNOWN OBSTACLE ANALYSIS ===")
print("Total LiDAR returns:", len(nav.last_scan))
print("Nearest obstacle:", nav.nearest)
print("Left-side returns:", len(left))
print("Right-side returns:", len(right))

# Diagnostic only: fewer nearby returns suggests
# a less obstructed side, not a validated safe path.
left_score = sum(1 / max(d, 0.1) for d in left)
right_score = sum(1 / max(d, 0.1) for d in right)

print("Left obstruction score:", round(left_score, 3))
print("Right obstruction score:", round(right_score, 3))

if left_score < right_score:
    print("Candidate avoidance direction: LEFT")
elif right_score < left_score:
    print("Candidate avoidance direction: RIGHT")
else:
    print("Candidate avoidance direction: EITHER / UNDECIDED")

print("Path clearance not yet validated.")
