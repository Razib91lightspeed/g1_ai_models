#!/usr/bin/env python3
"""Create a separate, simulation-only G1 V12 lab trial from V11 safe_uturn.

The real robot MUST NOT use this controller. The turning envelope is a provisional
measurement from an incomplete simulation trace, not a certified safety bound.
"""
from pathlib import Path
import ast, sys

root=Path.home()/ 'Desktop/razib/ai_models/simulations/unitree_g1/obstacle_navigation/autonomous'
if not root.is_dir(): root=Path.cwd()/'simulations/unitree_g1/obstacle_navigation/autonomous'
src=root/'g1_dynamic_browser_v11_safe_uturn.py'
dst=root/'g1_dynamic_browser_v12_lab_trial.py'
if not src.is_file(): sys.exit(f'Missing V11 source: {src}')
if dst.exists(): sys.exit(f'Output exists; not overwritten: {dst}')
s=src.read_text()

def replace(old,new,reason):
    global s
    n=s.count(old)
    if n!=1: sys.exit(f'Source mismatch for {reason}: {n} matches. No file created.')
    s=s.replace(old,new,1)

# Move turnaround point upstream. This is a test goal, not the wall.
replace('GOAL = np.array([7.0, -1.0])', 'GOAL = np.array([5.2, -1.25])', 'test goal')

# Measured (partial turn) radius and safety buffer. Precheck both turn directions
# over entire semicircle. Turning cannot start unless the whole assumed arc fits.
start=s.index('def select_lab_turn_direction(pos, yaw):')
end=s.index('\ndef command_for_path():', start)
new_helper='''# Experimental only: radius estimated from the partial 2026-10-29 trace.
# Realized curvature changes with gait/speed and terrain; never infer safety
# merely because a geometric prediction fits.
MEASURED_TURN_RADIUS_M = 1.80
TURN_BODY_BUFFER_M = 0.55  # provisional; not a validated swept body model
SPEED_CLEAR_M = 0.40       # validated gait command
SPEED_APPROACH_M = 0.32    # experimental, validate balance before obstacle trials
SPEED_NEAR_M = 0.25        # experimental, validate balance before obstacle trials

def predicted_turn_arc(pos, yaw, direction, radius=MEASURED_TURN_RADIUS_M):
    points=[]
    for phi in np.linspace(0.0,math.pi,91):
        theta=yaw+direction*phi
        x=pos[0]+radius/direction*(math.sin(theta)-math.sin(yaw))
        y=pos[1]-radius/direction*(math.cos(theta)-math.cos(yaw))
        points.append(np.array([x,y],dtype=float))
    return points

def select_lab_turn_direction(pos,yaw):
    scored=[]
    for direction in (1.0,-1.0):
        points=predicted_turn_arc(pos,yaw,direction)
        if not all(in_lab(p,extra=TURN_BODY_BUFFER_M) for p in points):
            continue
        # Require measured front scan clearance to begin a moving turn.
        forward_ranges=[r['range'] for r in cached_rays if r['hit'] and
            abs(nav.wrap(math.radians(r['angle_deg'])))<math.radians(35)]
        if min(forward_ranges,default=mapper.max_range)<TURN_STOP_RANGE:
            continue
        buffer=min(min(p[0]-GEOFENCE_X_MIN,GEOFENCE_X_MAX-p[0],
                       p[1]-GEOFENCE_Y_MIN,GEOFENCE_Y_MAX-p[1]) for p in points)
        scored.append((buffer,direction))
    return max(scored)[1] if scored else None

def variable_cruise(heading_error,near):
    # Never use a low commanded forward velocity as an assumed pivot gait.
    # Zero is used only for holding a safe stop; yaw is also set to zero.
    if near is not None and near < STOP_RANGE:
        return 0.0
    if near is not None and near < 1.6:
        return SPEED_NEAR_M
    if abs(heading_error)>math.radians(25):
        return SPEED_APPROACH_M
    if near is not None and near < 2.4:
        return SPEED_APPROACH_M
    return SPEED_CLEAR_M

'''
s=s[:start]+new_helper+s[end:]
replace('''    if not swept_lab_check(pos,nav.yaw_angle(),distance=1.0,yaw_rate=yaw_cmd):''', '''    # Short-horizon geofence precheck; physics is still independently checked.
    if not swept_lab_check(pos,nav.yaw_angle(),distance=1.0,yaw_rate=yaw_cmd):''', 'geofence comment')
replace('''    nav.command[:] = [CRUISE, 0.0, yaw_cmd]
    status = 'walking_to_' + path_kind''', '''    v=variable_cruise(error,near)
    if v <= 0.0:
        nav.command[:]=0.0
        status='blocked'
        return
    nav.command[:] = [v, 0.0, yaw_cmd]
    status = 'walking_to_' + path_kind''', 'adaptive travel speed')
# Turning uses the same already-tested 0.40 m/s gait command.
replace("nav.command[:] = [CRUISE, 0.0, turn_direction * YAW_LIMIT]",
        "nav.command[:] = [SPEED_CLEAR_M, 0.0, turn_direction * YAW_LIMIT]",
        'turning speed')
# Dynamic speed never loosens stop constraints; curvature is uncalibrated at new speeds.
ast.parse(s,filename=str(dst))
compile(s,str(dst),'exec')
dst.write_text(s)
print('Created:',dst)
print('Test goal (5.2,-1.25)m; measured-radius full-U-turn precheck with provisional body buffer')
print('Adaptive straight/path speeds: 0.40, 0.32, 0.25 m/s (lower two are UNVALIDATED)')
print('No in-place pivot; stop and geofence checks retained. SIMULATION ONLY.')
