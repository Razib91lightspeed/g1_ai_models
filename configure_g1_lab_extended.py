#!/usr/bin/env python3
"""Create a conservative simulation-only extended-lab variant from V12.

Only updates map and lab parameters. Does not claim to repair turning or return planning.
Does not overwrite an existing generated controller.
"""
from pathlib import Path
import ast
import re
import sys

root = Path.cwd() / 'simulations/unitree_g1/obstacle_navigation/autonomous'
source = root / 'g1_dynamic_browser_v12_measured_lab.py'
target = root / 'g1_dynamic_browser_v12_extended_lab.py'
if not source.is_file():
    sys.exit(f'Missing source: {source}')
if target.exists():
    sys.exit(f'Output exists; preserving: {target}')
s = source.read_text()

def replace_assignment(name, expression):
    global s
    pattern = re.compile(r'(?m)^' + re.escape(name) + r'\s*=\s*[^\n]+$')
    matches = list(pattern.finditer(s))
    if len(matches) != 1:
        sys.exit(f'Expected exactly one {name} assignment, got {len(matches)}; no files written')
    s = pattern.sub(lambda _: f'{name} = {expression}', s, count=1)

replace_assignment('LAB_LENGTH_M', '14.68')
replace_assignment('LAB_WIDTH_M', '7.00')
replace_assignment('WALL_MARGIN_M', '0.70  # provisional; not full-body safety validation')
replace_assignment('GOAL', 'np.array([5.00, -1.00])  # experimental; turning not validated')
# The occupancy grid should cover the entire lab, but actual permission is enforced by the geofence.
pattern = re.compile(r'mapper\s*=\s*LidarMapper\(model,\s*xlim=\(\s*-2(?:\.0)?\s*,\s*36(?:\.0)?\s*\),\s*ylim=\(\s*-4(?:\.0)?\s*,\s*4(?:\.0)?\s*\),\s*resolution=0\.1\)')
if len(pattern.findall(s)) != 1:
    sys.exit('Unexpected LidarMapper initialization; no files written')
s = pattern.sub('mapper = LidarMapper(model, xlim=(-2.0, 13.0), ylim=(-4.0, 4.0), resolution=0.1)', s, count=1)
ast.parse(s)
compile(s, str(target), 'exec')
target.write_text(s)
print('Created:', target)
print('Estimated room: X [-2.00, 12.68], Y [-3.50, 3.50] m')
print('Geofence centre limits: X [-1.30, 11.98], Y [-2.80, 2.80] m')
print('Goal: (5.00, -1.00) m; HOME: (0, 0) m')
print('WARNING: Config only. U-turn and return controller are unchanged and unvalidated.')
