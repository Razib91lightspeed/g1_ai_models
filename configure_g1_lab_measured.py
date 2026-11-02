#!/usr/bin/env python3
"""Create a separate, simulation-only G1 measured-lab configuration.

Source: V12 returnfix. No physical robot code. Does not alter source.
The floor dimensions are user estimates and are not validated safety limits.
"""
from pathlib import Path
import ast
import re
import sys

root = Path.cwd() / 'simulations/unitree_g1/obstacle_navigation/autonomous'
src = root / 'g1_dynamic_browser_v12_returnfix.py'
dst = root / 'g1_dynamic_browser_v12_measured_lab.py'
if not src.is_file():
    sys.exit('Missing input controller: ' + str(src))
if dst.exists():
    sys.exit('Already exists, preserving file: ' + str(dst))
s = src.read_text()

def set_assignment(name, value):
    global s
    pat = re.compile(r'(?m)^' + re.escape(name) + r'\s*=\s*[^\n]+$')
    matches = list(pat.finditer(s))
    if len(matches) != 1:
        sys.exit(f'Expected one {name} assignment, found {len(matches)}; nothing written')
    s = pat.sub(lambda m: f'{name} = {value}', s, count=1)

# Place the 7.34 m dimension along world X and 7.00 m along world Y.
# Shift X origin so HOME=(0,0) is INSIDE the protected operating zone.
set_assignment('LAB_WIDTH_M', '7.00')
set_assignment('LAB_LENGTH_M', '7.34')
set_assignment('WALL_MARGIN_M', '0.70  # provisional, not a verified G1 whole-body margin')
set_assignment('LAB_X_MIN', '-2.00')
set_assignment('GOAL', 'np.array([2.00, -1.60])  # trial point, NOT guaranteed U-turn feasibility')

# Lab geofence min/max computed from these source constants automatically.
# Do not enlarge the safety envelope to make the turn pass.
ast.parse(s, filename=str(dst))
compile(s, str(dst), 'exec')
dst.write_text(s)
print('Created:', dst)
print('Physical room (estimate): X length 7.34 m, Y width 7.00 m')
print('Allowed reference-point area: X [-1.30, 4.64], Y [-2.80, 2.80] m')
print('Experimental goal: (2.00, -1.60) m; HOME (0,0) m')
print('Kept collision masks, dynamic mocap obstacles, and all existing stop conditions.')
print('Important: full-body boundary and reliable return-turn maneuver NOT validated.')
