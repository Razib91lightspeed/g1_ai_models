#!/usr/bin/env python3
"""Create separate *simulation-only* V11 normal-lab turnaround trial.

The original V11 normal/extended controllers are not changed.
This is a hypothesis test, not a validated motion planner or real-robot program.
"""
from pathlib import Path
import ast
import sys

base = Path.home() / 'Desktop/razib/ai_models/simulations/unitree_g1/obstacle_navigation/autonomous'
if not base.is_dir():
    base = Path.cwd() / 'simulations/unitree_g1/obstacle_navigation/autonomous'
source = base / 'g1_dynamic_browser_v11_normal.py'
target = base / 'g1_dynamic_browser_v11_turntest.py'
if not source.is_file():
    sys.exit(f'Missing source: {source}')
if target.exists():
    sys.exit(f'Turn test already exists; original preserved: {target}')
s = source.read_text()
replacements = [
    ('GOAL = np.array([8.0, 0.0])', 'GOAL = np.array([8.0, -1.0])'),
    ("turn_direction = 1.0 if (min(left, default=mapper.max_range) >= min(right, default=mapper.max_range)) else -1.0",
     "turn_direction = 1.0  # Turn from the lower-y approach lane toward room centre"),
    ("print('MISSION: outbound goal reached; beginning moving turnaround', flush=True)",
     "print('MISSION: outbound goal reached; testing LEFT turnaround from offset lane', flush=True)"),
]
for old, new in replacements:
    n = s.count(old)
    if n != 1:
        sys.exit(f'Source mismatch for {old!r}: found {n}. No file written.')
    s = s.replace(old, new, 1)
ast.parse(s, filename=str(target))
compile(s, str(target), 'exec')
target.write_text(s)
print('Created:', target)
print('Test destination: (8.0, -1.0) m; initial U-turn direction: LEFT')
print('Geofence, collision detection, route memory and stop conditions unchanged.')
print('SIMULATION ONLY. First test: no obstacles. A boundary stop remains a failed test.')
