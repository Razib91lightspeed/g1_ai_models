#!/usr/bin/env python3
"""Simulation-only conservative lab U-turn trial. Does not alter existing versions.

Creates a separate browser server from g1_dynamic_browser_v11_turntest.py.
Uses the already-verified gait; no real robot interface.
"""
from pathlib import Path
import ast
import sys

root=Path.home()/ 'Desktop/razib/ai_models/simulations/unitree_g1/obstacle_navigation/autonomous'
if not root.is_dir():
    root=Path.cwd()/ 'simulations/unitree_g1/obstacle_navigation/autonomous'
src=root/'g1_dynamic_browser_v11_turntest.py'
dst=root/'g1_dynamic_browser_v11_safe_uturn.py'
if not src.is_file(): sys.exit(f'Missing source: {src}')
if dst.exists(): sys.exit(f'Output exists; refusing overwrite: {dst}')
s=src.read_text()

def sub(old,new,label):
    global s
    n=s.count(old)
    if n != 1: sys.exit(f'{label}: expected 1 occurrence, got {n}. No file written.')
    s=s.replace(old,new,1)

# Leave extra longitudinal room beyond the goal for a moving 180-degree turn.
sub('GOAL = np.array([8.0, -1.0])','GOAL = np.array([7.0, -1.0])','goal offset')

helper='''def select_lab_turn_direction(pos, yaw):
    """Precheck both approximate 180-degree arcs against the lab geofence.

    Based on commanded v/yaw, NOT a certified or calibrated robot turning radius.
    LiDAR and geofence are checked again while actually moving.
    """
    radius = CRUISE / YAW_LIMIT
    scored = []
    for direction in (1.0, -1.0):
        points=[]
        feasible=True
        for phi in np.linspace(0.0, math.pi, 41):
            theta = yaw + direction*phi
            x = pos[0] + radius/direction * (math.sin(theta)-math.sin(yaw))
            y = pos[1] - radius/direction * (math.cos(theta)-math.cos(yaw))
            xy=np.array([x,y])
            points.append(xy)
            if not in_lab(xy,extra=0.20):
                feasible=False
                break
        if feasible:
            smallest = min(min(p[0]-GEOFENCE_X_MIN,GEOFENCE_X_MAX-p[0],
                               p[1]-GEOFENCE_Y_MIN,GEOFENCE_Y_MAX-p[1]) for p in points)
            scored.append((smallest,direction))
    return max(scored)[1] if scored else None


'''
sub('def command_for_path():',helper+'def command_for_path():','insertion point')
sub('''            turn_direction = 1.0  # Turn from the lower-y approach lane toward room centre
            print('MISSION: outbound goal reached; testing LEFT turnaround from offset lane', flush=True)''', '''            direction = select_lab_turn_direction(pos, nav.yaw_angle())
            if direction is None:
                status = 'turn_blocked'
                nav.command[:] = 0.0
                print('SAFETY: neither model-predicted U-turn fits lab; assistance required',flush=True)
                return
            turn_direction = direction
            print(f'MISSION: U-turn precheck fits estimated arc; direction={direction:+.0f}; '
                  'starting obstacle-monitored turnaround',flush=True)''','turn start block')

# Preserve all measured trace and return logic exactly as in source.
ast.parse(s,filename=str(dst))
compile(s,str(dst),'exec')
dst.write_text(s)
print('Created:',dst)
print('Goal: (7.0, -1.0)m; turn direction selected from two approximate lab-bounded arcs.')
print('Return uses the pre-existing recorded route follower. Stop checks unchanged.')
print('SIMULATION ONLY. First run without spawned obstacles; no real-robot claims.')
