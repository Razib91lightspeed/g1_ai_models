#!/usr/bin/env python3
"""Generate an isolated fixed-base G1 preview scene."""

from pathlib import Path
import xml.etree.ElementTree as ET

source = Path(
    "/workspace/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"
)
output = Path("/workspace/preview/g1_fixed_base.xml")

tree = ET.parse(source)
root = tree.getroot()

worldbody = root.find("worldbody")
pelvis = worldbody.find(".//body[@name='pelvis']")

if pelvis is None:
    raise RuntimeError("G1 pelvis not found")

joint = pelvis.find("joint[@name='floating_base_joint']")

if joint is None:
    raise RuntimeError("Floating-base joint not found")

# Fix the pelvis in the world for visualization.
pelvis.remove(joint)

# Preserve Unitree's original mesh directory.
compiler = root.find("compiler")
if compiler is None:
    compiler = ET.Element("compiler")
    root.insert(0, compiler)

compiler.set(
    "meshdir",
    "/workspace/unitree_mujoco/unitree_robots/g1/meshes"
)

tree.write(output, encoding="unicode")

print("Created:", output)
print("Original model unchanged")
