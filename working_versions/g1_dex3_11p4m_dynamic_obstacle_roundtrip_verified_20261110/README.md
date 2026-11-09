# G1 Dex3 — verified dynamic obstacle round trip

Status: Successful complete simulation reported by operator,
10 November 2026 (Finland local time).

## Verified demonstration

- Unitree G1 43-actuator Dex3 MuJoCo simulation
- Nominal HOME-to-GOAL separation: 11.4 m
- Closed hands with -8 degree left thumb base target offset
- Dynamic randomized obstacles during outbound walking
- LiDAR observations and A* waypoint navigation
- 180-degree turn at destination
- Outbound obstacles removed before return
- New obstacles spawned during return
- Returned to HOME, as visually confirmed by operator
- Green START and GOAL markers
- Live interactive Viser viewer
- Physical robot not involved

The completed journey has not yet been benchmarked for
shortest total distance, travel time, or collision count.
Simulated depth-camera navigation is not implemented.

## Entry points

Mission: dex3_astar_mission_v24.py
Viewer: dex3_live_viser_v14.py

## Reproduction

Run from the Spark repository root. The snapshot is mounted
at the same container path expected by the existing viewer.

```bash
cd /home/rhasan/Desktop/razib/ai_models

docker run --rm -it \
  --name g1-dex3-verified-roundtrip \
  --network bridge \
  -p 127.0.0.1:8102:8080 \
  --volumes-from g1-nav-sim:ro \
  -v "$PWD/working_versions/g1_dex3_11p4m_dynamic_obstacle_roundtrip_verified_20261110:/workspace/obstacle_navigation/experiments/dex3_random_navigation_v1:ro" \
  -v "$PWD/sources/g1_dex3_mujoco_reference:/workspace/dex3:ro" \
  -v "$PWD/sources/g1_omnidirectional_turning:/workspace/omni:ro" \
  -v "$PWD/working_versions/g1_dex3_physical_hand_rest_pose_visual_verified_20261102:/workspace/hand_reference:ro" \
  -e DEX3_OBSTACLE_COUNT=5 \
  -e DEX3_OBSTACLE_SEED=42 \
  -e DEX3_THUMB0_OFFSET_DEG=-8 \
  -e DEX3_AUTO_SPAWN=1 \
  -e DEX3_MISSION_DISTANCE=11.4 \
  --workdir /workspace/obstacle_navigation/experiments/dex3_random_navigation_v1 \
  --entrypoint python3 \
  g1-mujoco-viser:0.1 \
  -B dex3_live_viser_v14.py
```

On Mac, create an SSH tunnel:

```bash
ssh -N -L 8102:127.0.0.1:8102 rhasan@10.11.0.50
```

Open http://127.0.0.1:8102/

## Preservation

This is a snapshot. Do not modify the verified files in
place. Create separate versions for future experiments.
