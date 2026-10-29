#!/usr/bin/env python3
"""G1 MuJoCo browser experiment V11: unknown obstacles and sensor-based replanning.

Simulation only. Reuses the validated ONNX locomotion from live_demo.py.
The navigation controller never reads obstacle geom positions; only the separate
simulation event handler can place an obstacle. Not a physical-robot safety system.
"""
import io
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mujoco
import numpy as np
from PIL import Image, ImageDraw

import live_demo_v8 as nav
from lidar_mapper import LidarMapper
from path_planner import AStarPlanner
from frontier_planner import select_frontier

PORT = 8771
# Survey these dimensions before any real-robot use.
LAB_MODE = 'normal'
LAB_WIDTH_M = 6.0
LAB_LENGTH_M = 12.0
WALL_MARGIN_M = 0.50  # provisional centre-position guard, NOT body envelope
LAB_X_MIN = -1.0
LAB_X_MAX = LAB_X_MIN + LAB_LENGTH_M
LAB_Y_MIN, LAB_Y_MAX = -LAB_WIDTH_M/2, LAB_WIDTH_M/2
GEOFENCE_X_MIN = LAB_X_MIN + WALL_MARGIN_M
GEOFENCE_X_MAX = LAB_X_MAX - WALL_MARGIN_M
GEOFENCE_Y_MIN = LAB_Y_MIN + WALL_MARGIN_M
GEOFENCE_Y_MAX = LAB_Y_MAX - WALL_MARGIN_M
GOAL = np.array([8.0, 0.0])
HOME = np.array([0.0, 0.0])
CRUISE = 0.40  # validated at this speed; not a physical safety value
YAW_LIMIT = 0.30
STOP_RANGE = 0.80  # emergency stop range along commanded travel
SLOW_RANGE = 1.20  # stop conservatively when route is not verified
CLEARANCE = 0.50  # experimental inflation radius
SCAN_PERIOD = 0.15
PLAN_PERIOD = 1.25
FRONTIER_TOLERANCE = 0.30
MIN_SPAWN_CLEARANCE = 1.15
TURN_STOP_RANGE = 1.35
TURN_TIMEOUT = 25.0
VIEW_PERIOD = 0.35

model, data = nav.model, nav.data
mapper = LidarMapper(model, xlim=(-2, 36.0), ylim=(-4.0, 4.0), resolution=0.1)
planner = AStarPlanner(mapper, clearance=CLEARANCE, allow_unknown=False)

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

obstacle_slots = sorted(nav.obstacle_ids)
MAX_OBSTACLES = len(obstacle_slots)

# In-RAM path memory. At 10 cm spacing, a 10 m leg is ~100 poses.
# No cloud, disk writes, or AI-model memory is required.
TRACE_SPACING_M = 0.10
TRACE_AHEAD_M = 0.75
TRACE_REACHED_M = 0.30
REJOIN_MAX_HEADING_DEG = 75.0
REJOIN_LOOKAHEAD = 35  # recorded points searched for a feasible join
RETRACE_CHUNK_POINTS = 7  # ~0.7 m along the actual recorded polyline
RETRACE_GRID_STEP_M = 0.05
return_mode = 'none'  # 'recorded_corridor' or 'astar_reconnection'

MAX_TRACE_POINTS = 3000  # Fail safe for inadvertently long experiments.
outbound_trace = []    # (x, y, yaw) in the fixed simulation world frame
trace_cursor = None   # index of the next return checkpoint
trace_target_index = None
trace_detours = 0
trace_replans = 0

# Mocap bodies represent obstacles; do not edit model.geom_pos at runtime.
obstacle_mocap = {}
for _gid in obstacle_slots:
    _bid = int(model.geom_bodyid[_gid])
    _mid = int(model.body_mocapid[_bid])
    if _mid < 0:
        raise RuntimeError(f'Obstacle {_gid} is not attached to a mocap body')
    obstacle_mocap[_gid] = _mid

def obstacle_center(gid):
    return data.mocap_pos[obstacle_mocap[gid]]

rng = np.random.default_rng()


lock = threading.Lock()
requests = []
images = {'scene': b'', 'depth': b'', 'map': b''}
telemetry = {}
status = 'idle'
phase = 'outbound'
path = None
path_kind = None
spawn_count = 0
detected_count = 0
last_detected = set()
collision = False
replans = 0
last_front_range = None
pause_until = 0.0
last_event = 'Ready'
last_spawn_result = 'No spawn requested'
spawn_attempts = 0
turn_start = 0.0
turn_direction = 1.0
last_route_stop = None
cached_rays = []
last_progress = None
last_progress_time = 0.0
plan_target = None
recovery_attempts = 0
MAX_RECOVERY_ATTEMPTS = 2
RECOVERY_SCAN_DELAY = 0.7

HTML = '''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>G1 Dynamic Navigation</title><style>
body{background:#101820;color:#eff5fa;font:16px system-ui;max-width:1250px;margin:auto;padding:20px}
button{border:0;border-radius:8px;padding:12px 20px;margin:4px;font-size:15px;font-weight:650;cursor:pointer}
.start{background:#39c991}.stop{background:#eda94e}.restart{background:#88acf9}.spawn{background:#f58259}
.panel{background:#213141;padding:15px;border-radius:12px;min-width:0}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.wide{grid-column:1/-1}
img{width:100%;display:block;background:#101820;border-radius:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere}
small{color:#bbc9d5}@media(max-width:750px){.grid{grid-template-columns:1fr}.wide{grid-column:auto}}
</style></head><body><h2>Unitree G1 — Unknown Obstacle Experiment</h2>
<p>8 m outbound + return to HOME · 30 precompiled randomized obstacle slots · LiDAR map · A* / frontier replanning · simulation only</p>
<button class="start" onclick="cmd('start')">▶ Start</button>
<button class="stop" onclick="cmd('stop')">■ Stop</button>
<button class="restart" onclick="cmd('restart')">↻ Restart</button>
<button class="spawn" onclick="cmd('spawn')">＋ Spawn Obstacle</button>
<p id="feedback">Ready. The Spawn button is one-shot; see the response below.</p>
<div class="panel" style="margin:12px 0">🏠 <b>HOME / START: (0.00, 0.00) m</b> &nbsp; | &nbsp; <span id="distance-home">Distance from home: 0.00 m</span> &nbsp; | &nbsp; <span id="spawn-feedback">No obstacle requested</span></div><div class="grid">
<div class="panel wide"><h3>Live MuJoCo</h3><img id="scene" src="/scene.jpg"></div>
<div class="panel"><h3>LiDAR occupancy map</h3><img id="map" src="/map.jpg"><small>Gray unknown · light free · orange detected obstacle / buffer · blue route</small></div>
<div class="panel"><h3>Depth camera</h3><img id="depth" src="/depth.jpg"><small>Depth visualisation only; not fused into the controller yet</small></div>
<div class="panel wide"><h3>Measured state</h3><pre id="status"></pre></div></div>
<script>
async function cmd(action){try{let r=await fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action})});let d=await r.json();document.getElementById('feedback').textContent=d.message||d.error;}catch(e){document.getElementById('feedback').textContent=String(e)}}
async function refresh(){try{let r=await fetch('/status');let d=await r.json();document.getElementById('status').textContent=JSON.stringify(d,null,2);document.getElementById('distance-home').textContent='Distance from home: '+d.home_distance_m+' m';document.getElementById('spawn-feedback').textContent=d.last_spawn_result||'';let t=Date.now();for(let v of ['scene','map','depth'])document.getElementById(v).src='/'+v+'.jpg?t='+t;}catch(e){}}
setInterval(refresh,750);refresh();</script></body></html>'''


def jpg(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format='JPEG', quality=77)
    return buf.getvalue()


def reset():
    global status, phase, path, path_kind, spawn_count, collision, replans
    global last_front_range, pause_until, detected_count, last_detected, last_event, last_route_stop, cached_rays
    global last_progress, last_progress_time, plan_target, turn_start, turn_direction
    global last_spawn_result, spawn_attempts
    global outbound_trace, trace_cursor, trace_target_index, trace_detours, trace_replans, return_mode
    nav.reset_sim()
    outbound_trace = [(float(data.qpos[0]), float(data.qpos[1]), float(nav.yaw_angle()))]
    trace_cursor = None
    trace_target_index = None
    trace_detours = 0
    trace_replans = 0
    return_mode = 'none'
    # Simulator-owned course setup: hide all obstacles until Spawn is clicked.
    for gid in nav.obstacle_ids:
        # Move the mocap body offstage. Keep physical collision masks enabled.
        half_height = float(model.geom_size[gid, 2])
        data.mocap_pos[obstacle_mocap[gid]] = [100.0 + obstacle_slots.index(gid), 100.0, half_height]
        model.geom_rgba[gid, 3] = 1.0
    # Relocate visual-only goal marker to the 10 m mission endpoint.
    marker = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'nav_finish')
    if marker >= 0:
        model.geom_pos[marker, :2] = GOAL
    mujoco.mj_forward(model, data)
    mapper.reset()
    status, phase = 'idle', 'outbound'
    path, path_kind = None, None
    spawn_count, collision, replans = 0, False, 0
    detected_count, last_detected = 0, set()
    last_event, last_route_stop, cached_rays = 'Ready', None, []
    last_spawn_result, spawn_attempts = 'No spawn requested', 0
    last_front_range, pause_until = None, 0.0
    last_progress, last_progress_time, plan_target = None, 0.0, None
    turn_start, turn_direction = 0.0, 1.0


def spawn_obstacle():
    """Insert one precompiled random-size box, moving its mocap body.

    This environment routine can read geometry; the navigation routines cannot.
    """
    global spawn_count, last_event, last_spawn_result, spawn_attempts
    spawn_attempts += 1

    def reject(reason):
        global last_event, last_spawn_result
        last_spawn_result = 'Spawn rejected: ' + reason
        last_event = last_spawn_result
        print(last_event, flush=True)
        return last_spawn_result

    if spawn_count >= MAX_OBSTACLES:
        return reject('all obstacle slots used; Restart to reset')
    if not (status.startswith('walking') or status == 'replanning'):
        return reject('robot not walking (' + status + ')')

    gid = obstacle_slots[spawn_count]
    sx, sy, sz = [float(v) for v in model.geom_size[gid]]
    radius = math.hypot(sx, sy)
    heading = nav.yaw_angle()
    forward = np.array([math.cos(heading), math.sin(heading)])
    lateral = np.array([-forward[1], forward[0]])
    robot_xy = data.qpos[:2].copy()
    chosen = None

    for _ in range(80):
        distance = float(rng.uniform(2.4, 3.4))
        offset = float(rng.uniform(-0.65, 0.65))
        xy = robot_xy + distance*forward + offset*lateral
        if not in_lab(xy, extra=math.hypot(*model.geom_size[gid,:2])+0.25):
            continue
        if np.linalg.norm(xy - robot_xy) < 1.8 + radius:
            continue
        if np.linalg.norm(xy - HOME) < 1.1 + radius:
            continue
        if np.linalg.norm(xy - GOAL) < 1.1 + radius:
            continue
        if not (mapper.xmin+0.9+radius < xy[0] < mapper.xmax-0.9-radius and
                mapper.ymin+0.9+radius < xy[1] < mapper.ymax-0.9-radius):
            continue
        # Prevent spawning a box on an existing one or close to the robot.
        overlaps = False
        for old_gid in obstacle_slots[:spawn_count]:
            old = obstacle_center(old_gid)
            old_size = model.geom_size[old_gid]
            if (abs(xy[0]-old[0]) < sx+old_size[0]+0.65 and
                abs(xy[1]-old[1]) < sy+old_size[1]+0.65):
                overlaps = True
                break
        if overlaps:
            continue
        cell = mapper.world_to_cell(float(xy[0]),float(xy[1]))
        if cell is None or mapper.grid[cell] == mapper.OCCUPIED:
            continue
        chosen = xy
        break

    if chosen is None:
        return reject('no non-overlapping spawn position ahead')

    data.mocap_pos[obstacle_mocap[gid]] = [float(chosen[0]),float(chosen[1]),sz]
    mujoco.mj_forward(model,data)
    spawn_count += 1
    last_spawn_result = (f'Obstacle {spawn_count}/{MAX_OBSTACLES} placed (mocap physics); '
                         f'size {2*sx:.2f} x {2*sy:.2f} x {2*sz:.2f} m')
    last_event = last_spawn_result
    print('EVENT:',last_event,flush=True)
    return last_spawn_result

def scan():
    """Update map using LiDAR only; no geom coordinates enter navigation."""
    global last_front_range, detected_count, last_detected, cached_rays
    origin, rays = mapper.observe(data)
    cached_rays = rays
    front = [r['range'] for r in rays if r['hit'] and
             abs(nav.wrap(math.radians(r['angle_deg']))) < math.radians(35)]
    last_front_range = min(front) if front else None
    # This is return count, not physical object identity or ground-truth ID.
    detected_count = sum(1 for r in rays if r['hit'])
    return origin, rays


def route_is_clear():
    """Check upcoming planned path against the CURRENT inflated grid.

    Always replan if a new occupied/unknown cell touches the route.
    Do not reject a safe lateral detour just because an object is forward.
    """
    if path is None or len(path) < 2:
        return False
    blocked = planner.build_blocked_map()
    pos = data.qpos[:2].copy()
    distances = np.linalg.norm(path - pos, axis=1)
    idx = int(np.argmin(distances))
    # Check 1.5 m ahead along the currently selected path.
    traveled = 0.0
    previous = pos
    for point in path[idx:]:
        d = float(np.linalg.norm(point - previous))
        traveled += d
        cell = mapper.world_to_cell(float(point[0]), float(point[1]))
        if cell is None or blocked[cell]:
            return False
        if traveled >= 1.5:
            break
        previous = point
    return True


def motion_clearance():
    """Check LiDAR in the direction of the next path segment.

    This is a conservative heuristic, not full-body collision certification.
    """
    if path is None or len(path) < 2:
        return None
    pos = data.qpos[:2].copy()
    idx = int(np.argmin(np.linalg.norm(path - pos, axis=1)))
    target = path[min(idx + 3, len(path) - 1)]
    angle = nav.wrap(math.atan2(target[1]-pos[1], target[0]-pos[0]) - nav.yaw_angle())
    # If goal is behind the robot, do not use the forward sensor cone.
    if abs(angle) > math.radians(75):
        return None
    values = [r['range'] for r in cached_rays if r['hit'] and
              abs(nav.wrap(math.radians(r['angle_deg']) - angle)) < math.radians(22)]
    return min(values) if values else None


def make_plan_original():
    global path, path_kind, replans, status, plan_target
    pos = data.qpos[:2].copy()
    goal = GOAL if phase == 'outbound' else HOME
    direct, reason = planner.plan(pos, goal)
    if direct is not None:
        path, path_kind = direct, 'goal'
        plan_target = goal.copy()
    else:
        frontier = select_frontier(mapper, planner, pos, goal)
        if frontier['status'] == 'frontier_found':
            path, path_kind = frontier['path'], 'frontier'
            plan_target = np.asarray(frontier['target']).copy()
        else:
            path, path_kind, plan_target = None, None, None
            status = 'blocked'
    if path is not None and any(not in_lab(p) for p in path):
        path,path_kind,plan_target=None,None,None
        status='boundary_blocked'
        nav.command[:]=0
    replans += 1
    print(f'PLAN #{replans}: {path_kind or "blocked"} ({reason}), target={plan_target}', flush=True)
    if path is None:
        print('SAFETY: no verified path; G1 stopped. Do not override unknown/blocked cells.', flush=True)


def remember_outbound_pose():
    """Store measured path in RAM; never infer path from control commands."""
    if phase != 'outbound' or status not in ('walking_to_goal', 'walking_to_frontier'):
        return
    if len(outbound_trace) >= MAX_TRACE_POINTS:
        return
    xy = data.qpos[:2]
    previous = outbound_trace[-1]
    if math.hypot(float(xy[0])-previous[0], float(xy[1])-previous[1]) >= TRACE_SPACING_M:
        outbound_trace.append((float(xy[0]), float(xy[1]), float(nav.yaw_angle())))


def begin_return_trace():
    """Freeze outbound path and choose a checkpoint behind the destination."""
    global trace_cursor, trace_target_index
    point = (float(data.qpos[0]), float(data.qpos[1]), float(nav.yaw_angle()))
    if not outbound_trace or math.hypot(point[0]-outbound_trace[-1][0],
                                        point[1]-outbound_trace[-1][1]) > 0.02:
        outbound_trace.append(point)
    trace_cursor = len(outbound_trace)-1
    trace_target_index = None
    print(f'MEMORY: captured {len(outbound_trace)} outbound poses in RAM; returning by checkpoints',flush=True)


def path_cells_clear(points):
    """Validate each polyline segment against current unknown/occupied inflated cells.

    This is 2D collision inflation, NOT full swept-body or motion feasibility.
    """
    blocked = planner.build_blocked_map()
    for a, b in zip(points[:-1], points[1:]):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        length = float(np.linalg.norm(b-a))
        count = max(1, int(math.ceil(length / RETRACE_GRID_STEP_M)))
        for t in np.linspace(0.0, 1.0, count + 1):
            p = a + t * (b-a)
            cell = mapper.world_to_cell(float(p[0]), float(p[1]))
            if cell is None or blocked[cell]:
                return False
    return True


def initial_heading_error(points, pos):
    """Heading required to begin a proposed connection, avoiding in-place reversal."""
    for point in points[1:]:
        delta = np.asarray(point) - pos
        if np.linalg.norm(delta) >= 0.20:
            desired = math.atan2(float(delta[1]), float(delta[0]))
            return abs(nav.wrap(desired - nav.yaw_angle()))
    return 0.0


def make_plan():
    """Return on the ACTUAL saved polyline; A* only when rejoining or obstructed."""
    global path, path_kind, replans, status, plan_target, return_mode
    global trace_cursor, trace_target_index, trace_detours, trace_replans
    if phase != 'returning' or trace_cursor is None:
        return make_plan_original()

    pos = data.qpos[:2].copy()
    if float(np.linalg.norm(pos-HOME)) < 0.25:
        path = np.asarray([pos, HOME])
        path_kind, plan_target, trace_target_index = 'home_trace', HOME.copy(), 0
        return_mode = 'recorded_corridor'
        return
    if len(outbound_trace) < 3:
        print('MEMORY: too few poses; original home planner',flush=True)
        return make_plan_original()

    trace_replans += 1
    # Only advance return-index after physically reaching a saved checkpoint.
    # Never jump cursor to a globally nearest point: loops can be close in XY.
    cursor = min(int(trace_cursor), len(outbound_trace)-1)
    if trace_target_index is None:
        # First return join after U-turn. The robot might be several metres
        # away from the recorded path. Try checkpoints in a local window.
        lower = max(0, cursor-REJOIN_LOOKAHEAD)
        candidate_indices = list(range(cursor-1,lower-1,-3))
        if lower not in candidate_indices:
            candidate_indices.append(lower)
    else:
        # Continue backward in small sequential portions of the saved path.
        candidate_indices = [max(0,cursor-RETRACE_CHUNK_POINTS)]

    selected = None
    limit = math.radians(REJOIN_MAX_HEADING_DEG)
    for idx in candidate_indices:
        target = np.asarray(outbound_trace[idx][:2],dtype=float)
        recorded = np.asarray(
            [pos] + [list(p[:2]) for p in outbound_trace[idx:cursor+1][::-1]],
            dtype=float
        )
        # A direct join to the saved track is allowed only when the whole
        # segment, not just its endpoint, is known free and heading is feasible.
        # Avoid accidentally skipping the saved corridor with a direct shortcut.
        if (initial_heading_error(recorded,pos) <= limit
                and path_cells_clear(recorded)):
            selected = (idx,target,recorded,'recorded_corridor')
            break
        # A* is a reconnection fallback, not the default path generator.
        candidate,reason=planner.plan(pos,target)
        if candidate is None or len(candidate)<2:
            continue
        candidate=np.asarray(candidate,dtype=float)
        if (initial_heading_error(candidate,pos) <= limit
                and path_cells_clear(candidate)):
            selected=(idx,target,candidate,'astar_reconnection')
            break

    replans += 1
    if selected is None:
        path,path_kind,plan_target,trace_target_index=None,None,None,None
        return_mode='none'
        status='blocked'
        nav.command[:]=0
        print('SAFETY: no collision-checked, heading-feasible return join; stopped',flush=True)
        return

    idx,target,candidate,mode=selected
    if any(not in_lab(p) for p in candidate):
        path,path_kind,plan_target,trace_target_index=None,None,None,None
        status='boundary_blocked'
        nav.command[:]=0
        print('SAFETY: return route leaves the lab boundary',flush=True)
        return
    trace_target_index=idx
    path=np.asarray(candidate,dtype=float)
    path_kind='home_trace'
    plan_target=target.copy()
    return_mode=mode
    if mode=='astar_reconnection':
        trace_detours+=1
    print(f'PLAN #{replans}: return {mode} checkpoint #{idx}/{len(outbound_trace)-1}, '
          f'target={target.round(2)}',flush=True)


def command_for_path():
    """Closed-loop waypoint following with measured front-sector stop."""
    global status, phase, path, pause_until, plan_target, turn_start, turn_direction
    pos = data.qpos[:2].copy()
    if not in_lab(pos):
        nav.command[:]=0.0
        status='geofence_violation'
        print('SAFETY: robot outside permitted lab position; stopped',flush=True)
        return
    destination = GOAL if phase == 'outbound' else HOME
    if np.linalg.norm(pos - destination) < 0.20:
        nav.command[:] = 0.0
        if phase == 'outbound':
            begin_return_trace()
            phase = 'returning'
            path = None
            status = 'turnaround'
            turn_start = float(data.time)
            # Select left or right turn from relative obstacle ranges.
            left = [r['range'] for r in cached_rays if r['hit'] and 0 < r['angle_deg'] < 100]
            right = [r['range'] for r in cached_rays if r['hit'] and 260 < r['angle_deg'] < 360]
            turn_direction = 1.0 if (min(left, default=mapper.max_range) >= min(right, default=mapper.max_range)) else -1.0
            print('MISSION: outbound goal reached; beginning moving turnaround', flush=True)
        else:
            heading_error = nav.wrap(nav.start_yaw - nav.yaw_angle())
            if abs(heading_error) < math.radians(8):
                status = 'finished'
            else:
                # In-place turning was NOT verified. Do not claim posture recovery.
                status = 'home_heading_unverified'
        return

    # Frontier is a temporary observation target, not the final goal.
    # On arrival we must scan again and choose a NEW route.
    if path_kind == 'frontier' and plan_target is not None:
        if np.linalg.norm(pos - plan_target) < FRONTIER_TOLERANCE:
            nav.command[:] = 0.0
            path = None
            status = 'replanning'
            pause_until = float(data.time) + 0.15
            print('FRONTIER: viewpoint reached; rescanning/replanning', flush=True)
            return

    if (phase == 'returning' and path_kind == 'home_trace'
            and plan_target is not None
            and float(np.linalg.norm(pos-plan_target)) < TRACE_REACHED_M
            and float(np.linalg.norm(pos-HOME)) >= 0.20):
        global trace_cursor, trace_target_index
        if trace_target_index is not None:
            trace_cursor = min(trace_cursor, trace_target_index)
        nav.command[:] = 0.0
        path = None
        status = 'replanning'
        pause_until = float(data.time) + 0.12
        print(f'MEMORY: return checkpoint #{trace_cursor} reached; selecting next',flush=True)
        return

    if path is None or len(path) == 0:
        nav.command[:] = 0.0
        status = 'blocked'
        return

    distances = np.linalg.norm(path - pos, axis=1)
    idx = int(np.argmin(distances))
    look = idx
    while look < len(path)-1 and np.linalg.norm(path[look]-pos) < 0.42:
        look += 1
    target = path[look]
    desired_heading = math.atan2(target[1]-pos[1], target[0]-pos[0])
    error = nav.wrap(desired_heading - nav.yaw_angle())

    # Only stop for an obstruction along the actual planned direction.
    # A side obstacle must not repeatedly cancel a valid detour.
    near = motion_clearance()
    if near is not None and near < STOP_RANGE:
        nav.command[:] = 0.0
        status = 'blocked'
        nav.command[:] = 0.0
        print('SAFETY: insufficient clearance along path; stopped', flush=True)
        return

    if abs(error) > math.radians(105):
        # A tight reverse turn is outside the validated moving-turn regime.
        nav.command[:] = 0.0
        status = 'turn_required'
        print(f'SAFETY: route requires {math.degrees(error):+.1f} deg reversal; no verified maneuver', flush=True)
        return

    if abs(error) > math.radians(55):
        # The old threshold (65.9 deg) rejected valid bends. G1 has demonstrated
        # steering while moving forward at 0.40 m/s, not a pivot turn.
        # Before a larger arc, conservatively require visible, obstacle-free
        # forward travel. A claimed safe path is not a guarantee of full-body clearance.
        forward_hits = [r['range'] for r in cached_rays if r['hit'] and
                        abs(nav.wrap(math.radians(r['angle_deg']))) < math.radians(40)]
        if min(forward_hits, default=mapper.max_range) < 1.35:
            nav.command[:] = 0.0
            status = 'turn_blocked'
            print('SAFETY: obstacle limits turning arc; safe stop (detour controller needs improvement)', flush=True)
            return
        heading = nav.yaw_angle()
        fwd = np.array([math.cos(heading), math.sin(heading)])
        inflated = planner.build_blocked_map()
        for distance_ahead in (0.35, 0.65, 0.90):
            check = pos + distance_ahead * fwd
            cell = mapper.world_to_cell(float(check[0]), float(check[1]))
            if cell is None or inflated[cell]:
                nav.command[:] = 0.0
                status = 'turn_blocked'
                print('SAFETY: unverified space ahead; moving turn refused', flush=True)
                return

    yaw_cmd=float(np.clip(1.1*error,-YAW_LIMIT,YAW_LIMIT))
    if not swept_lab_check(pos,nav.yaw_angle(),distance=1.0,yaw_rate=yaw_cmd):
        nav.command[:]=0.0
        status='boundary_blocked'
        print('SAFETY: projected walking arc crosses lab boundary',flush=True)
        return
    nav.command[:] = [CRUISE, 0.0, yaw_cmd]
    status = 'walking_to_' + path_kind



def command_turnaround():
    """Moving U-turn, because in-place yaw was not supported by the gait.

    Simulator-only trial: stop if forward clearance or mapped boundary is
    uncertain. Does not assert full-body collision safety.
    """
    global status, pause_until, path, last_event
    pos = data.qpos[:2].copy()
    if float(data.time) - turn_start > TURN_TIMEOUT:
        nav.command[:] = 0.0
        status = 'turn_timeout'
        print('SAFETY: turnaround timeout', flush=True)
        return
    # Follow the tangent of the saved return corridor, not the direct HOME
    # bearing. This reduces the risk of aligning one way then facing a
    # remembered checkpoint at >105 degrees in the opposite direction.
    if len(outbound_trace) >= 6:
        start_idx = max(0, len(outbound_trace)-1-16)
        to_join = np.asarray(outbound_trace[start_idx][:2]) - pos
        if float(np.linalg.norm(to_join)) < 0.4:
            to_join = HOME - pos
    else:
        to_join = HOME - pos
    desired = math.atan2(to_join[1], to_join[0])
    error = nav.wrap(desired - nav.yaw_angle())
    # A moving turn must complete before attempting reverse-facing A* route.
    if abs(error) < math.radians(25):
        nav.command[:] = 0.0
        path = None
        status = 'replanning'
        pause_until = float(data.time) + 0.25
        print(f'MISSION: turnaround aligned ({math.degrees(error):+.1f} deg); planning home route', flush=True)
        return
    # Obstacle scan is from actual LiDAR, not the simulator's object coordinates.
    forward_ranges = [r['range'] for r in cached_rays if r['hit'] and
                      abs(nav.wrap(math.radians(r['angle_deg']))) < math.radians(35)]
    nearest_forward = min(forward_ranges, default=mapper.max_range)
    heading = nav.yaw_angle()
    projected = pos + 1.1 * np.array([math.cos(heading), math.sin(heading)])
    next_cell = mapper.world_to_cell(*projected)
    if (nearest_forward < TURN_STOP_RANGE or next_cell is None
            or not swept_lab_check(pos,heading,distance=1.2,
                                   yaw_rate=turn_direction*YAW_LIMIT)):
        nav.command[:] = 0.0
        status = 'turn_blocked'
        print(f'SAFETY: turnaround blocked; front={nearest_forward:.2f}m; projected={projected.round(2)}', flush=True)
        return
    nav.command[:] = [CRUISE, 0.0, turn_direction * YAW_LIMIT]
    status = 'turnaround'


def draw_map():
    arr = np.empty((*mapper.grid.shape, 3), dtype=np.uint8)
    arr[:] = (94, 101, 112)
    arr[mapper.grid == mapper.FREE] = (230, 237, 242)
    inflated = planner.build_blocked_map()
    arr[inflated & (mapper.grid != mapper.UNKNOWN)] = (249, 185, 100)
    arr[mapper.grid == mapper.OCCUPIED] = (216, 80, 43)
    arr[_lab_forbidden] = (66, 70, 84)  # unavailable room / margin

    s = 7
    pic = Image.fromarray(np.flipud(arr)).resize(
        (mapper.width*s, mapper.height*s), Image.Resampling.NEAREST)
    pen = ImageDraw.Draw(pic)

    def pix(p):
        cell = mapper.world_to_cell(float(p[0]), float(p[1]))
        if cell is None:
            return None
        r, c = cell
        return (int((c+0.5)*s), int((mapper.height-r-0.5)*s))

    if len(outbound_trace) >= 2:
        remembered_pixels = [pix(p[:2]) for p in outbound_trace]
        remembered_pixels = [p for p in remembered_pixels if p is not None]
        if len(remembered_pixels) > 1:
            pen.line(remembered_pixels,fill=(0,185,180),width=3)
    if path is not None:
        pixels = [pix(p) for p in path]
        pixels = [p for p in pixels if p is not None]
        if len(pixels)>1:
            pen.line(pixels, fill=(27, 95, 255), width=4)
    for xy, color in ((HOME,'#2b83ff'), (GOAL,'#2bc96b'), (data.qpos[:2],'#171d26')):
        p = pix(xy)
        if p:
            x,y=p
            if np.linalg.norm(np.asarray(xy) - HOME) < 1e-9:
                pen.ellipse((x-13,y-13,x+13,y+13), outline='#00e0ff', width=4)
                pen.ellipse((x-7,y-7,x+7,y+7), fill=color, outline='white', width=2)
                pen.text((x+17,y-17), 'HOME / START (0,0)', fill='#003fa0', stroke_width=2, stroke_fill='white')
            else:
                pen.ellipse((x-7,y-7,x+7,y+7), fill=color, outline='white', width=2)
    return jpg(np.array(pic))


def render(scene_renderer, depth_renderer):
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [4.7, 0.0, 0.55]
    camera.distance = 14.0
    camera.azimuth = 115
    camera.elevation = -40
    options = mujoco.MjvOption()
    options.geomgroup[:] = 1
    scene_renderer.update_scene(data, camera=camera, scene_option=options)
    scene = jpg(scene_renderer.render().copy())
    depth_renderer.update_scene(data, camera='g1_depth_camera', scene_option=options)
    depth = depth_renderer.render().copy()
    grayscale = ((1.0-np.clip(depth,0,6)/6.0)*255).astype(np.uint8)
    depth_image = jpg(np.repeat(grayscale[:,:,None],3,axis=2))

    rotation = data.xmat[nav.pelvis].reshape(3,3)
    tilt = math.degrees(math.acos(np.clip(rotation[2,2],-1,1)))
    new_state = dict(
        status=status, mission=phase, simulation_time_s=round(float(data.time),2),
        x_m=round(float(data.qpos[0]),3), y_m=round(float(data.qpos[1]),3),
        heading_deg=round(math.degrees(nav.yaw_angle()),1),
        pelvis_height_m=round(float(data.qpos[2]),3), tilt_deg=round(tilt,1),
        front_lidar_m=round(last_front_range,3) if last_front_range is not None else None,
        obstacles_spawned=spawn_count, obstacle_slots=MAX_OBSTACLES,
        lidar_hit_rays=detected_count, contact_with_obstacle=collision,
        replans=replans, plan_kind=path_kind, last_event=last_event,
        recorded_outbound_poses=len(outbound_trace),
        estimated_trace_memory_bytes=len(outbound_trace)*12,
        return_checkpoint_index=trace_target_index,
        return_cursor_index=trace_cursor,
        return_reconnections=trace_detours,
        return_mode=return_mode,
        lab_mode=LAB_MODE,
        lab_width_m=LAB_WIDTH_M,
        lab_length_m=LAB_LENGTH_M,
        wall_margin_m=WALL_MARGIN_M,
        geofence_ok=in_lab(data.qpos[:2]),
        return_checkpoint_plans=trace_replans,
        last_spawn_result=last_spawn_result, spawn_attempts=spawn_attempts,
        home_x_m=0.0, home_y_m=0.0,
        home_distance_m=round(float(np.linalg.norm(data.qpos[:2]-HOME)),3),
        home_heading_error_deg=round(math.degrees(abs(nav.wrap(nav.start_yaw-nav.yaw_angle()))),1),
        planned_waypoints=len(path) if path is not None else 0,
        current_plan_target=[round(float(v),2) for v in plan_target] if plan_target is not None else None,
        velocity_command=[round(float(v),3) for v in nav.command],
        goal_distance_m=round(float(np.linalg.norm((GOAL if phase=='outbound' else HOME)-data.qpos[:2])),3),
        note='V9: RAM route trace + LiDAR-verified return checkpoints; mocap contacts; experimental safety')
    with lock:
        images.update(scene=scene, depth=depth_image, map=draw_map())
        telemetry.clear()
        telemetry.update(new_state)


class Handler(BaseHTTPRequestHandler):
    def send_payload(self, code, body, kind):
        self.send_response(code)
        self.send_header('Content-Type',kind)
        self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length',str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split('?')[0]
        if path == '/':
            return self.send_payload(200, HTML.encode(), 'text/html; charset=utf-8')
        with lock:
            state = dict(telemetry)
            payload = {'/scene.jpg':images['scene'], '/depth.jpg':images['depth'],
                       '/map.jpg':images['map']}.get(path)
        if path == '/status':
            return self.send_payload(200,json.dumps(state).encode(),'application/json')
        if payload:
            return self.send_payload(200,payload,'image/jpeg')
        self.send_error(404)

    def do_POST(self):
        if self.path != '/control':
            return self.send_error(404)
        try:
            size = int(self.headers.get('Content-Length','0'))
            if size < 1 or size > 256:
                raise ValueError('Invalid request length')
            value = json.loads(self.rfile.read(size))
            action = value.get('action')
            if action not in ('start','stop','restart','spawn'):
                raise ValueError('Unknown action')
            with lock:
                requests.append(action)
            answer = {'message':action+' queued'}
            self.send_payload(202,json.dumps(answer).encode(),'application/json')
        except (ValueError,TypeError) as exc:
            self.send_payload(400,json.dumps({'error':str(exc)}).encode(),'application/json')

    def log_message(self,*args):
        pass


def main():
    global status, phase, path, collision, pause_until, last_event
    global last_progress, last_progress_time, recovery_attempts
    reset()
    scene_renderer = mujoco.Renderer(model, height=480, width=640)
    depth_renderer = mujoco.Renderer(model, height=240, width=320)
    depth_renderer.enable_depth_rendering()
    render(scene_renderer, depth_renderer)
    server = ThreadingHTTPServer(('0.0.0.0',PORT), Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    print(f'G1 dynamic browser ready on port {PORT}',flush=True)
    next_scan = next_plan = next_view = 0.0
    physics_steps = 0
    wall_start = time.monotonic()
    try:
        while True:
            with lock:
                incoming = list(requests)
                requests.clear()
            for action in incoming:
                if action == 'restart':
                    reset()
                    recovery_attempts=0
                    next_scan = next_plan = next_view = 0.0
                    physics_steps=0
                    wall_start=time.monotonic()
                elif action == 'start' and status not in ('walking_to_goal','walking_to_frontier','replanning','turnaround'):
                    if status in ('collision','fallen','finished','home_heading_unverified','blocked','turn_blocked','turn_timeout','stalled','boundary_blocked','geofence_violation','assistance_required'):
                        reset()
                        physics_steps=0
                        wall_start=time.monotonic()
                    status='stabilizing' if data.time<4 else 'replanning'
                    pause_until=float(data.time)+0.2
                    next_plan=0.0
                elif action == 'stop':
                    status='stopped'
                    nav.command[:]=0.0
                elif action == 'spawn':
                    message=spawn_obstacle()
                    if message.startswith('Obstacle ') and (status.startswith('walking') or status in ('replanning', 'stabilizing')):
                        status='replanning'
                        nav.command[:]=0
                        pause_until=float(data.time)+0.5
                        next_scan=next_plan=0.0

            if status in ('idle','stopped','finished','blocked','collision','fallen',
                          'turn_required','home_heading_unverified','stalled','turn_blocked','turn_timeout',
                          'boundary_blocked','geofence_violation','assistance_required'):
                # Paused simulation: freeze physics and keep the page responsive.
                if next_view <= time.monotonic():
                    render(scene_renderer,depth_renderer)
                    next_view=time.monotonic()+VIEW_PERIOD
                time.sleep(0.05)
                continue

            t=float(data.time)
            if t >= next_scan and t>=4.0:
                scan()
                next_scan=t+SCAN_PERIOD
                # Replan only if the current *path* becomes obstructed.
                # Seeing the same object in the front sensor is not enough.
                if status.startswith('walking') and not route_is_clear():
                    status='replanning'
                    nav.command[:]=0
                    pause_until=t+0.4
                    next_plan=t+0.4
                    print('SENSOR: planned route obstructed; replanning', flush=True)

            if t < 4.0:
                nav.command[:]=0.0
                status='stabilizing'
            elif status == 'stabilizing':
                nav.command[:] = 0.0
                status='replanning'
                pause_until=t+0.25
                next_plan=t+0.25
            elif status=='replanning':
                nav.command[:]=0.0
                if t>=pause_until and t>=next_plan:
                    make_plan()
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
                    pause_until=t
            elif status == 'turnaround':
                command_turnaround()
            elif status.startswith('walking'):
                # Keep an already safe frontier path. Replanning every 1.25 s
                # previously made the robot change its target repeatedly.
                # The LiDAR route-clearance check above triggers replanning
                # when a newly observed obstacle actually blocks the route.
                command_for_path()

            # The exact control loop already validated in the walking tests.
            if physics_steps % nav.interval == 0:
                nav.update_policy()
            torque=(nav.kp*(nav.target-data.qpos[nav.qids])
                    -nav.kd*data.qvel[nav.vids])
            data.ctrl[nav.ids]=np.clip(torque, model.actuator_ctrlrange[nav.ids,0],
                                      model.actuator_ctrlrange[nav.ids,1])
            data.xfrc_applied[nav.pelvis,:]=0
            support=nav.clamp((4.0-t)/2.0,0.0,1.0)
            if support>0:
                mujoco.mj_forward(model,data)
                force=(np.array([0.,0.,nav.total_mass*9.81])
                       +250*(np.array([0.,0.,nav.initial_z])-data.qpos[:3])
                       -60*data.qvel[:3])
                rotation=data.xmat[nav.pelvis].reshape(3,3)
                velocity=np.zeros(6)
                mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,
                                         nav.pelvis,velocity,0)
                angular=180*np.cross(rotation[:,2],np.array([0.,0.,1.]))-35*velocity[:3]
                data.xfrc_applied[nav.pelvis,:3]=support*force
                data.xfrc_applied[nav.pelvis,3:]=support*angular
            mujoco.mj_step(model,data)
            physics_steps+=1
            remember_outbound_pose()
            for ci in range(data.ncon):
                contact=data.contact[ci]
                if contact.geom1 in obstacle_slots[:spawn_count] or contact.geom2 in obstacle_slots[:spawn_count]:
                    collision=True
                    status='collision'
                    nav.command[:]=0
                    print('FAIL: obstacle contact',flush=True)
                    break
            rotation=data.xmat[nav.pelvis].reshape(3,3)
            tilt=math.degrees(math.acos(np.clip(rotation[2,2],-1,1)))
            if (tilt>40 or data.qpos[2]<0.50 or not np.isfinite(data.qpos).all()):
                status='fallen'
                nav.command[:]=0
                print('FAIL: unstable robot',flush=True)
            if t >= 4.0 and status.startswith('walking'):
                if last_progress is None:
                    last_progress = data.qpos[:2].copy()
                    last_progress_time = t
                elif t - last_progress_time >= 4.0:
                    moved = float(np.linalg.norm(data.qpos[:2] - last_progress))
                    print(f'PROGRESS: {moved:.2f} m in 4 s, position=({data.qpos[0]:.2f},{data.qpos[1]:.2f}), status={status}', flush=True)
                    last_progress = data.qpos[:2].copy()
                    last_progress_time = t
                    if moved < 0.12:
                        nav.command[:] = 0.0
                        status = 'stalled'
                        print('FAIL: no useful progress; stopped instead of endless planning', flush=True)
            else:
                last_progress = None
            if time.monotonic()>=next_view:
                render(scene_renderer,depth_renderer)
                next_view=time.monotonic()+VIEW_PERIOD
            delay=wall_start+float(data.time)-time.monotonic()
            if 0<delay<0.05:
                time.sleep(delay)
    finally:
        server.shutdown()
        scene_renderer.close()
        depth_renderer.close()

if __name__=='__main__':
    main()
