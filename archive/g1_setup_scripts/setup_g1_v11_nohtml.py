#!/usr/bin/env python3
"""Build a SIMULATION-ONLY V11 lab-geofence browser from the existing V10.

Run from ~/Desktop/razib/ai_models on Spark.
Does not modify earlier versions or physical robot software.
No validated physical safety guarantee. Measurements are provisional.
"""
from pathlib import Path
import ast
import sys
import re

MODE = sys.argv[1] if len(sys.argv)>1 else 'normal'
if MODE not in ('normal','extended'):
    raise SystemExit('Usage: python3 setup_g1_v11.py [normal|extended]')
base=Path.cwd()/'simulations/unitree_g1/obstacle_navigation/autonomous'
src=base/'g1_dynamic_browser_v10.py'
dst=base/f'g1_dynamic_browser_v11_{MODE}.py'
if not src.exists(): raise SystemExit(f'Missing V10: {src}')
if dst.exists(): raise SystemExit(f'Already exists, refusing overwrite: {dst}')
s=src.read_text()
def replace(old,new):
    global s
    if s.count(old)!=1:
        raise SystemExit(f'V10 layout mismatch: expected one match, got {s.count(old)} for {old[:95]!r}. Nothing written.')
    s=s.replace(old,new,1)

replace('GOAL = np.array([10.0, 0.0])',f'''# Survey these dimensions before any real-robot use.
LAB_MODE = '{MODE}'
LAB_WIDTH_M = 6.0
LAB_LENGTH_M = {12.0 if MODE=='normal' else 36.0}
WALL_MARGIN_M = 0.50  # provisional centre-position guard, NOT body envelope
LAB_X_MIN = -1.0
LAB_X_MAX = LAB_X_MIN + LAB_LENGTH_M
LAB_Y_MIN, LAB_Y_MAX = -LAB_WIDTH_M/2, LAB_WIDTH_M/2
GEOFENCE_X_MIN = LAB_X_MIN + WALL_MARGIN_M
GEOFENCE_X_MAX = LAB_X_MAX - WALL_MARGIN_M
GEOFENCE_Y_MIN = LAB_Y_MIN + WALL_MARGIN_M
GEOFENCE_Y_MAX = LAB_Y_MAX - WALL_MARGIN_M
GOAL = np.array([{8.0 if MODE=='normal' else 30.0}, 0.0])''')
replace('xlim=(-2, 16.0), ylim=(-6.0, 6.0)',
        'xlim=(-2, 36.0), ylim=(-4.0, 4.0)')
replace('planner = AStarPlanner(mapper, clearance=CLEARANCE, allow_unknown=False)', '''planner = AStarPlanner(mapper, clearance=CLEARANCE, allow_unknown=False)

# Enforce the surveyed operating zone in the A* and frontier blocked maps.
# This uses the map's actual cell_to_world conversion to avoid x/y axis errors.
_original_build_blocked_map = planner.build_blocked_map
_grid_x = np.empty(mapper.grid.shape,dtype=float)
_grid_y = np.empty(mapper.grid.shape,dtype=float)
for _i in range(mapper.grid.shape[0]):
    for _j in range(mapper.grid.shape[1]):
        try:
            _p = mapper.cell_to_world((_i,_j))
        except TypeError:
            _p = mapper.cell_to_world(_i,_j)
        _grid_x[_i,_j] = float(_p[0])
        _grid_y[_i,_j] = float(_p[1])
_lab_forbidden = ((_grid_x < GEOFENCE_X_MIN) | (_grid_x > GEOFENCE_X_MAX) |
                  (_grid_y < GEOFENCE_Y_MIN) | (_grid_y > GEOFENCE_Y_MAX))
def lab_blocked_map():
    blocked = _original_build_blocked_map().copy()
    blocked |= _lab_forbidden
    return blocked
planner.build_blocked_map = lab_blocked_map

def in_lab(xy, extra=0.0):
    return (GEOFENCE_X_MIN+extra <= float(xy[0]) <= GEOFENCE_X_MAX-extra
            and GEOFENCE_Y_MIN+extra <= float(xy[1]) <= GEOFENCE_Y_MAX-extra)

def swept_lab_check(pos, yaw, distance=1.2, yaw_rate=0.0):
    # Conservative forward arc check; not a certified whole-body swept volume.
    # Policy commanded yaw is not an exact physical yaw rate.
    speed = CRUISE
    for fraction in np.linspace(0,1,13):
        t=distance/max(speed,0.01)*float(fraction)
        theta=yaw + yaw_rate*t
        if abs(yaw_rate) < 1e-6:
            xy=pos + distance*fraction*np.array([math.cos(yaw),math.sin(yaw)])
        else:
            radius=speed/yaw_rate
            xy=pos + np.array([radius*(math.sin(theta)-math.sin(yaw)),
                               -radius*(math.cos(theta)-math.cos(yaw))])
        if not in_lab(xy, extra=0.15):
            return False
    return True
''')
replace("plan_target = None\n\nHTML =", '''plan_target = None
recovery_attempts = 0
MAX_RECOVERY_ATTEMPTS = 2
RECOVERY_SCAN_DELAY = 0.7

HTML =''')
# Add geofence to the actual V10 candidate loop without assuming its numeric threshold.
match = re.search(r"(?m)^(?P<indent>[ \t]*)if\s+np\.linalg\.norm\(xy\s*-\s*robot_xy\)\s*<\s*[^\n:]+:", s)
if match is None:
    raise SystemExit('Cannot locate distance validation inside spawn_obstacle; nothing written')
indent = match.group('indent')
s = s[:match.start()] + f"{indent}if not in_lab(xy, extra=math.hypot(*model.geom_size[gid,:2])+0.25):\n{indent}    continue\n" + s[match.start():]
replace('''def path_cells_clear(points):''','''def path_cells_clear(points):''') if False else None
replace('''    pos = data.qpos[:2].copy()
    destination = GOAL if phase == 'outbound' else HOME''','''    pos = data.qpos[:2].copy()
    if not in_lab(pos):
        nav.command[:]=0.0
        status='geofence_violation'
        print('SAFETY: robot outside permitted lab position; stopped',flush=True)
        return
    destination = GOAL if phase == 'outbound' else HOME''')
replace('''    nav.command[:] = [CRUISE, 0.0, float(np.clip(1.1*error, -YAW_LIMIT, YAW_LIMIT))]
    status = 'walking_to_' + path_kind''','''    yaw_cmd=float(np.clip(1.1*error,-YAW_LIMIT,YAW_LIMIT))
    if not swept_lab_check(pos,nav.yaw_angle(),distance=1.0,yaw_rate=yaw_cmd):
        nav.command[:]=0.0
        status='boundary_blocked'
        print('SAFETY: projected walking arc crosses lab boundary',flush=True)
        return
    nav.command[:] = [CRUISE, 0.0, yaw_cmd]
    status = 'walking_to_' + path_kind''')
replace('''    next_cell = mapper.world_to_cell(*projected)
    if nearest_forward < TURN_STOP_RANGE or next_cell is None:''','''    next_cell = mapper.world_to_cell(*projected)
    if (nearest_forward < TURN_STOP_RANGE or next_cell is None
            or not swept_lab_check(pos,heading,distance=1.2,
                                   yaw_rate=turn_direction*YAW_LIMIT)):
'''.rstrip('\n'))
# A bounded stationary recovery scan is attempted twice. There is NO blind
# backing up or rotation: unresolved unknown cells require operator assistance.
replace('''    global last_progress, last_progress_time
    reset()
''','''    global last_progress, last_progress_time, recovery_attempts
    reset()
''')
replace('''                    reset()
                    next_scan = next_plan = next_view = 0.0
''','''                    reset()
                    recovery_attempts=0
                    next_scan = next_plan = next_view = 0.0
''')
replace('''                    make_plan()
                    next_plan=t+PLAN_PERIOD
                    if path is not None:
                        status='walking_to_'+path_kind''','''                    make_plan()
                    next_plan=t+PLAN_PERIOD
                    if path is not None:
                        recovery_attempts=0
                        status='walking_to_'+path_kind
                    elif status == 'blocked':
                        recovery_attempts+=1
                        if recovery_attempts<=MAX_RECOVERY_ATTEMPTS:
                            nav.command[:]=0.0
                            status='recovery_scan'
                            pause_until=t+RECOVERY_SCAN_DELAY
                            print(f'RECOVERY: stationary rescan {recovery_attempts}/{MAX_RECOVERY_ATTEMPTS}',flush=True)
                        else:
                            status='assistance_required'
                            print('SAFETY: no verified path after rescans; assistance required',flush=True)
            elif status=='recovery_scan':
                nav.command[:]=0.0
                if t>=pause_until:
                    scan()
                    status='replanning'
                    next_plan=t
                    pause_until=t''')
replace("'turn_required','home_heading_unverified','stalled','turn_blocked','turn_timeout'):","'turn_required','home_heading_unverified','stalled','turn_blocked','turn_timeout',\n                          'boundary_blocked','geofence_violation','assistance_required'):")
replace("'turn_timeout','stalled'):","'turn_timeout','stalled','boundary_blocked','geofence_violation','assistance_required'):")
replace("        if path == '/status':", "        if path == '/status':") if False else None
# Dashboard wording varies across V10 revisions; this is purely cosmetic.
s=re.sub(r'10 m outbound \+ return to HOME',
         f"{'8' if MODE=='normal' else '30'} m outbound + return to HOME",
         s, count=1)
replace("G1 MuJoCo browser experiment V10", "G1 MuJoCo browser experiment V11") if "G1 MuJoCo browser experiment V10" in s else None
# Visual marker on the occupancy map, not a fabricated physical partition wall.
replace('''    arr[mapper.grid == mapper.OCCUPIED] = (216, 80, 43)''','''    arr[mapper.grid == mapper.OCCUPIED] = (216, 80, 43)
    arr[_lab_forbidden] = (66, 70, 84)  # unavailable room / margin
''')
# Check entire planned path is inside lab: reject paths from any planner implementation.
replace('''    replans += 1
    print(f'PLAN #{replans}: {path_kind or "blocked"}''','''    if path is not None and any(not in_lab(p) for p in path):
        path,path_kind,plan_target=None,None,None
        status='boundary_blocked'
        nav.command[:]=0
    replans += 1
    print(f'PLAN #{replans}: {path_kind or "blocked"}''')
replace('''    idx,target,candidate,mode=selected
    trace_target_index=idx''','''    idx,target,candidate,mode=selected
    if any(not in_lab(p) for p in candidate):
        path,path_kind,plan_target,trace_target_index=None,None,None,None
        status='boundary_blocked'
        nav.command[:]=0
        print('SAFETY: return route leaves the lab boundary',flush=True)
        return
    trace_target_index=idx''')
# Report context with telemetry, without modifying telemetry serialization.
replace('''        return_mode=return_mode,''','''        return_mode=return_mode,
        lab_mode=LAB_MODE,
        lab_width_m=LAB_WIDTH_M,
        lab_length_m=LAB_LENGTH_M,
        wall_margin_m=WALL_MARGIN_M,
        geofence_ok=in_lab(data.qpos[:2]),''')
# Validate all assumptions and never write partially patched source.
ast.parse(s,filename=str(dst)); compile(s,str(dst),'exec')
dst.write_text(s)
print('Created:',dst)
print(f'LAB: {(12 if MODE=="normal" else 36)}m × 6m, margin 0.5m, goal {8 if MODE=="normal" else 30}m')
print('Old versions unchanged. Simulation-only geofence (NOT real-world safety-rated).')
