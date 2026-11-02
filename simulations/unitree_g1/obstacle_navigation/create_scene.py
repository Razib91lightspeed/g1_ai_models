#!/usr/bin/env python3
"""Create a simulation-only obstacle course for Unitree G1."""

from pathlib import Path
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parents[1]

scene_dir = (
    root / "unitree_mujoco" / "unitree_robots" / "g1"
)

source = scene_dir / "scene_29dof.xml"
destination = scene_dir / "scene_obstacles_29dof.xml"

tree = ET.parse(source)
mujoco = tree.getroot()
worldbody = mujoco.find("worldbody")

if worldbody is None:
    raise RuntimeError("MuJoCo worldbody missing")

# One static obstacle directly ahead.
ET.SubElement(
    worldbody,
    "geom",
    {
        "name": "nav_obstacle_1",
        "type": "box",
        "pos": "1.5 0 0.45",
        "size": "0.25 0.35 0.45",
        "rgba": "0.95 0.35 0.12 1",
        "contype": "1",
        "conaffinity": "1",
    },
)

# A green goal marker on the ground.
# Visualization only: no physical collision.
ET.SubElement(
    worldbody,
    "geom",
    {
        "name": "nav_goal",
        "type": "cylinder",
        "pos": "3.0 0 0.015",
        "size": "0.20 0.015",
        "rgba": "0.1 0.85 0.3 0.75",
        "contype": "0",
        "conaffinity": "0",
    },
)

ET.indent(tree, space="  ")
tree.write(destination, encoding="unicode")

print("Created:", destination)
print("Obstacle: x=1.5m, y=0m")
print("Goal: x=3.0m, y=0m")
print("Original scene: unchanged")
