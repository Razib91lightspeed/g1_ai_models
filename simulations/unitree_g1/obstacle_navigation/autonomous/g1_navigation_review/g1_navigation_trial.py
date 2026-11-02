#!/usr/bin/env python3
"""G1 navigation review trial derived from the uploaded V12 source.

Simulation only. Reuses the validated ONNX locomotion from live_demo.py.
The navigation controller never reads obstacle geom positions; only the separate
simulation event handler can place an obstacle. Not a physical-robot safety system.
"""
import io
import hashlib
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mujoco
import numpy as np
from PIL import Image, ImageDraw

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from g1_motion_checks import adaptive_speed, slew, arc_points, arc_margin, reverse_segment, PolylineTracker

import live_demo_v8 as nav
from lidar_mapper import LidarMapper
from path_planner import AStarPlanner
from frontier_planner import select_frontier

PORT = 8771
# Survey these dimensions before any real-robot use.
LAB_MODE = 'normal'
LAB_WIDTH_M = 7.00
LAB_LENGTH_M = 14.68
WALL_MARGIN_M = 0.70  # provisional; not full-body safety validation
LAB_X_MIN = -2.00
LAB_X_MAX = LAB_X_MIN + LAB_LENGTH_M
LAB_Y_MIN, LAB_Y_MAX = -LAB_WIDTH_M/2, LAB_WIDTH_M/2
GEOFENCE_X_MIN = LAB_X_MIN + WALL_MARGIN_M
GEOFENCE_X_MAX = LAB_X_MAX - WALL_MARGIN_M
GEOFENCE_Y_MIN = LAB_Y_MIN + WALL_MARGIN_M
GEOFENCE_Y_MAX = LAB_Y_MAX - WALL_MARGIN_M
GOAL = np.array([5.00, -1.00])  # experimental; turning not validated
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
TURN_TIMEOUT = 45.0
VIEW_PERIOD = 0.35
SLOW_SPEED_TRIALS = os.environ.get('G1_ENABLE_SLOW_TRIALS', '1') == '1'
ALLOW_SPAWN = os.environ.get('G1_ENABLE_OBSTACLES', '0') == '1'
return_join_index = None
return_joined = False
turn_entry_yaw = 0.0
turn_previous_yaw = 0.0
turn_angle = 0.0
turn_progress_time = 0.0
turn_progress_angle = 0.0
turn_next_check = 0.0
applied_command = np.zeros(3)
path_tracker = PolylineTracker()
measured_speed = 0.0
measured_yaw_rate = 0.0

model, data = nav.model, nav.data
mapper = LidarMapper(model, xlim=(-2.0, 13.0), ylim=(-4.0, 4.0), resolution=0.1)
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

def swept_lab_check(pos, yaw, distance=1.2, yaw_rate=0.0, speed=None):
    # A geometric screening model, not an assertion that commanded yaw is realized.
    speed = max(0.05, float(abs(nav.command[0]) if speed is None else speed))
    if abs(yaw_rate) < 1e-6:
        points = [pos + d*np.array([math.cos(yaw), math.sin(yaw)])
                  for d in np.linspace(0, distance, 25)]
    else:
        points = arc_points(pos, yaw, 1 if yaw_rate > 0 else -1,
                            speed/abs(yaw_rate), angle=abs(yaw_rate)*distance/speed, samples=25)
    return all(in_lab(p, extra=0.15) for p in points)


obstacle_slots = sorted(nav.obstacle_ids)
MAX_OBSTACLES = len(obstacle_slots)

# In-RAM path memory. At 10 cm spacing, a 10 m leg is ~100 poses.
# No cloud, disk writes, or AI-model memory is required.
TRACE_SPACING_M = 0.10
TRACE_AHEAD_M = 0.75
TRACE_REACHED_M = 0.12
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
button:disabled{opacity:.45;cursor:not-allowed}
.start{background:#39c991}.stop{background:#eda94e}.restart{background:#88acf9}.spawn{background:#f58259}
.panel{background:#213141;padding:15px;border-radius:12px;min-width:0}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.wide{grid-column:1/-1}
img{width:100%;display:block;background:#101820;border-radius:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere}
small{color:#bbc9d5}@media(max-width:750px){.grid{grid-template-columns:1fr}.wide{grid-column:auto}}
</style></head><body><h2>Unitree G1 — Unknown Obstacle Experiment</h2>
<p>Navigation review trial · goal (5.00, -1.00) m · recorded return route · simulation only</p>
<button class="start" onclick="cmd('start')">▶ Start</button>
<button class="stop" onclick="cmd('stop')">■ Stop</button>
<button class="restart" onclick="cmd('restart')">↻ Restart</button>
<button class="spawn" id="spawn-button" disabled onclick="cmd('spawn')">＋ Spawn Obstacle</button>
<p id="spawn-policy">Checking whether obstacle spawning is enabled…</p>
<p id="feedback">Ready. The Spawn button is one-shot; see the response below.</p>
<div class="panel" style="margin:12px 0">🏠 <b>HOME / START: (0.00, 0.00) m</b> &nbsp; | &nbsp; <span id="distance-home">Distance from home: 0.00 m</span> &nbsp; | &nbsp; <span id="spawn-feedback">No obstacle requested</span></div><div class="grid">
<div class="panel wide"><h3>Live MuJoCo</h3><img id="scene" src="/scene.jpg"></div>
<div class="panel"><h3>LiDAR occupancy map</h3><img id="map" src="/map.jpg"><small>Gray unknown · light free · orange detected obstacle / buffer · blue route</small></div>
<div class="panel"><h3>Depth camera</h3><img id="depth" src="/depth.jpg"><small>Depth visualisation only; not fused into the controller yet</small></div>
<div class="panel wide"><h3>Measured state</h3><pre id="status"></pre></div></div>
<script>
async function cmd(action){try{let r=await fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action})});let d=await r.json();document.getElementById('feedback').textContent=d.message||d.error;}catch(e){document.getElementById('feedback').textContent=String(e)}}
async function refresh(){try{let r=await fetch('/status');let d=await r.json();document.getElementById('status').textContent=JSON.stringify(d,null,2);document.getElementById('distance-home').textContent='Distance from home: '+d.home_distance_m+' m';document.getElementById('spawn-feedback').textContent=d.last_spawn_result||'';document.getElementById('spawn-button').disabled=!d.obstacle_spawning_enabled||!String(d.status).startsWith('walking');document.getElementById('spawn-policy').textContent=d.obstacle_spawning_enabled?'Obstacle spawning enabled; available while walking.':'Obstacle spawning disabled for this validation run.';let t=Date.now();for(let v of ['scene','map','depth'])document.getElementById(v).src='/'+v+'.jpg?t='+t;}catch(e){}}
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
    global HOME, return_join_index, return_joined, turn_entry_yaw, recovery_attempts
    global turn_previous_yaw, turn_angle, turn_progress_time, turn_progress_angle, turn_next_check
    nav.reset_sim()
    HOME = data.qpos[:2].copy()
    return_join_index, return_joined = None, False
    turn_entry_yaw = nav.yaw_angle()
    turn_previous_yaw = turn_entry_yaw
    turn_angle = turn_progress_time = turn_progress_angle = turn_next_check = 0.0
    applied_command[:] = 0
    path_tracker.reset()
    recovery_attempts = 0
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

    if not ALLOW_SPAWN:
        return reject('obstacle-free validation; obstacle spawning is disabled in this trial')

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
        offset = float(rng.uniform(-0.10, 0.10))
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
    if path is None or len(path) < 2:
        return False
    pos = data.qpos[:2].copy()
    path_tracker.advance(path, pos)
    return path_cells_clear(np.asarray([pos, *path_tracker.remaining(1.5)]))



def motion_clearance():
    """Check LiDAR in the direction of the next path segment.

    This is a conservative heuristic, not full-body collision certification.
    """
    if path is None or len(path) < 2:
        return None
    pos = data.qpos[:2].copy()
    target = path_tracker.advance(path, pos)
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
    global trace_cursor, trace_target_index, return_join_index, return_joined
    point = (float(data.qpos[0]), float(data.qpos[1]), float(nav.yaw_angle()))
    if not outbound_trace or math.hypot(point[0]-outbound_trace[-1][0],
                                        point[1]-outbound_trace[-1][1]) > 0.02:
        outbound_trace.append(point)
    trace_cursor = len(outbound_trace)-1
    trace_target_index = None
    return_join_index, return_joined = None, False
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
    """Preserve the selected return target across planner calls and obstruction retries."""
    global path, path_kind, replans, status, plan_target, return_mode
    global trace_cursor, trace_target_index, trace_detours, trace_replans
    if phase != 'returning':
        return make_plan_original()
    if trace_cursor is None or len(outbound_trace) < 3:
        path = None
        status = 'blocked'
        print('RETURN: missing frozen outbound route', flush=True)
        return
    pos = data.qpos[:2].copy()
    cursor = min(int(trace_cursor), len(outbound_trace)-1)
    if not return_joined:
        if return_join_index is None:
            path = None
            status = 'blocked'
            print('RETURN: no checkpoint handed over by turnaround', flush=True)
            return
        idx = return_join_index
    else:
        idx = trace_target_index if trace_target_index is not None else max(0, cursor-RETRACE_CHUNK_POINTS)
    trace_target_index = idx
    target_xy = np.asarray(outbound_trace[idx][:2], dtype=float)
    recorded = np.asarray(reverse_segment(outbound_trace, cursor, idx, pos,
                                         joining=not return_joined), dtype=float)
    if len(recorded) < 2:
        recorded = np.asarray([pos, target_xy])
    trace_replans += 1
    replans += 1
    clear = path_cells_clear(recorded)
    feasible_heading = initial_heading_error(recorded, pos) <= math.radians(REJOIN_MAX_HEADING_DEG)
    candidate, mode = None, 'none'
    if clear and feasible_heading:
        candidate, mode = recorded, 'recorded_corridor'
    elif not clear:
        # Reconnect only when the remembered segment is blocked, to the SAME target.
        proposed, reason = planner.plan(pos, target_xy)
        if proposed is not None:
            proposed = np.asarray([pos, *proposed, target_xy], dtype=float)
            if (path_cells_clear(proposed) and
                    initial_heading_error(proposed, pos) <= math.radians(REJOIN_MAX_HEADING_DEG)):
                candidate, mode = proposed, 'astar_reconnection'
    if candidate is None or any(not in_lab(p) for p in candidate):
        path, path_kind, plan_target = None, None, target_xy.copy()
        status = 'blocked'
        nav.command[:] = 0
        print(f'RETURN: checkpoint #{idx} retained; '
              f'corridor_clear={clear}, initial_heading_feasible={feasible_heading}', flush=True)
        return
    path, path_kind, plan_target = candidate, 'home_trace', target_xy.copy()
    return_mode = mode
    if mode == 'astar_reconnection':
        trace_detours += 1
    print(f'PLAN #{replans}: return {mode}; pinned checkpoint #{idx}; cursor={cursor}', flush=True)



# Full-turn screening results at 0.32 m/s and +/-0.30 rad/s yaw command.
# The added 0.15 m is an experimental allowance, NOT a measured confidence bound.
# Actual trajectories, contact, standing and geofence checks remain necessary.
TURN_RADII = {1: 1.398361142713348+0.15, -1: 1.5135019872916735+0.15}
TURN_SPEED_MPS = 0.32
TURN_MAX_ANGLE = math.radians(230)
TURN_BODY_BUFFER_M = 0.55  # provisional; not a validated swept body model
SPEED_CLEAR_M = 0.40
SPEED_APPROACH_M = SPEED_NEAR_M = 0.32

def predicted_turn_arc(pos, yaw, direction, radius=None):
    radius = TURN_RADII[int(direction)] if radius is None else radius
    return [np.asarray(p, dtype=float) for p in
            arc_points(pos, yaw, direction, radius, angle=TURN_MAX_ANGLE, samples=116)]


def select_lab_turn_direction(pos, yaw):
    """Screen a direction-specific 0.32 m/s arc, including space to rejoin."""
    candidates = []
    bounds = (GEOFENCE_X_MIN, GEOFENCE_X_MAX, GEOFENCE_Y_MIN, GEOFENCE_Y_MAX)
    forward = [r['range'] for r in cached_rays if r['hit'] and
               abs(nav.wrap(math.radians(r['angle_deg']))) < math.radians(35)]
    for direction in (1, -1):
        points = predicted_turn_arc(pos, yaw, direction)
        margin = arc_margin(points, bounds, TURN_BODY_BUFFER_M)
        map_clear = path_cells_clear(points)
        front_clear = bool(cached_rays) and min(forward, default=mapper.max_range) >= TURN_STOP_RANGE
        print(f'TURN CHECK: direction={direction:+d}, screening_radius={TURN_RADII[direction]:.2f}m, '
              f'min_boundary_margin={margin:+.3f}m, mapped_arc_clear={map_clear}, '
              f'front_clear={front_clear}; speed=0.32m/s; rejoin window<=230deg. '
              'One screening trial per direction; continuation beyond 180deg remains experimental.', flush=True)
        if margin >= 0 and map_clear and front_clear:
            candidates.append((margin, direction))
    return max(candidates)[1] if candidates else None


def variable_cruise(heading_error, near):
    destination = GOAL if phase == 'outbound' else HOME
    remaining = float(np.linalg.norm(data.qpos[:2]-destination))
    return adaptive_speed(heading_error, near, remaining, SLOW_SPEED_TRIALS)



def command_for_path():
    global status, phase, path, pause_until, plan_target, trace_cursor, trace_target_index
    global return_join_index, return_joined
    pos = data.qpos[:2].copy()
    if not in_lab(pos):
        nav.command[:] = 0
        status = 'geofence_violation'
        return
    destination = GOAL if phase == 'outbound' else HOME
    if np.linalg.norm(pos-destination) < 0.20:
        nav.command[:] = 0
        path = None
        if phase == 'outbound':
            begin_return_trace()
            phase = 'returning'
            status = 'turn_preparing'
        else:
            status = 'home_settling'
        return
    if path_kind == 'frontier' and plan_target is not None and np.linalg.norm(pos-plan_target) < FRONTIER_TOLERANCE:
        nav.command[:] = 0
        path = None
        status = 'replanning'
        pause_until = float(data.time)+0.15
        return
    if (phase == 'returning' and path_kind == 'home_trace' and plan_target is not None
            and float(np.linalg.norm(pos-plan_target)) < TRACE_REACHED_M):
        if trace_target_index is None:
            status = 'blocked'
            nav.command[:] = 0
            return
        trace_cursor = min(trace_cursor, trace_target_index)
        if not return_joined:
            return_joined = True
            return_join_index = None
        trace_target_index = None
        # Continue into the next remembered section without a stop at every checkpoint.
        make_plan()
        if path is None:
            return
        print(f'MEMORY: reached #{trace_cursor}; continuing backward along saved route', flush=True)
    if path is None or len(path) < 2:
        nav.command[:] = 0
        status = 'blocked'
        return
    target = np.asarray(path_tracker.advance(path, pos))
    delta = target-pos
    if not path_cells_clear(np.asarray([pos, target])):
        nav.command[:] = 0
        path = None
        status = 'replanning'
        pause_until = float(data.time)+0.5
        return
    error = nav.wrap(math.atan2(delta[1], delta[0])-nav.yaw_angle())
    near = motion_clearance()
    if near is not None and near <= STOP_RANGE:
        nav.command[:] = 0
        path = None
        status = 'replanning'
        pause_until = float(data.time)+0.5
        return
    if abs(error) > math.radians(105):
        nav.command[:] = 0
        status = 'turn_required'
        print(f'RETURN/PATH: required reversal {math.degrees(error):+.1f}deg', flush=True)
        return
    if abs(error) > math.radians(55):
        forward_hits = [r['range'] for r in cached_rays if r['hit'] and
                        abs(nav.wrap(math.radians(r['angle_deg']))) < math.radians(40)]
        heading = nav.yaw_angle()
        forward_path = np.asarray([pos+d*np.array([math.cos(heading), math.sin(heading)])
                                   for d in np.linspace(0, 0.90, 19)])
        if min(forward_hits, default=mapper.max_range) < 1.35 or not path_cells_clear(forward_path):
            nav.command[:] = 0
            status = 'turn_blocked'
            return
    yaw_cmd = float(np.clip(1.1*error, -YAW_LIMIT, YAW_LIMIT))
    speed = variable_cruise(error, near)
    # Screen current and requested speeds, because acceleration takes time.
    if not all(swept_lab_check(pos, nav.yaw_angle(), distance=1.0,
                              yaw_rate=yaw_cmd, speed=v)
               for v in (max(float(nav.command[0]), 0.05), max(speed, 0.05))):
        nav.command[:] = 0
        status = 'boundary_blocked'
        return
    nav.command[:] = [speed, 0, yaw_cmd]
    status = 'walking_to_'+path_kind




def command_turnaround():
    global status, pause_until, path, return_join_index, trace_target_index, return_joined
    global turn_previous_yaw, turn_angle, turn_progress_time, turn_progress_angle, turn_next_check
    pos, heading = data.qpos[:2].copy(), nav.yaw_angle()
    turn_angle += turn_direction*nav.wrap(heading-turn_previous_yaw)
    turn_previous_yaw = heading
    if float(data.time)-turn_start > TURN_TIMEOUT:
        nav.command[:] = 0
        status = 'turn_timeout'
        return
    reversal = turn_angle
    if reversal > TURN_MAX_ANGLE:
        nav.command[:] = 0
        status = 'turn_blocked'
        print('TURN: reached the configured search limit without a feasible join; stopping', flush=True)
        return
    if float(data.time)-turn_progress_time >= 4.0:
        if reversal-turn_progress_angle < math.radians(8):
            nav.command[:] = 0
            status = 'stalled'
            print('TURN: insufficient measured yaw progress', flush=True)
            return
        turn_progress_time, turn_progress_angle = float(data.time), reversal
    if len(outbound_trace) >= 3 and trace_cursor is not None and reversal >= math.radians(140):
        cursor = min(int(trace_cursor), len(outbound_trace)-1)
        # Examine every point, in reverse order; preserve the first feasible selection.
        for idx in range(cursor-1, max(-1, cursor-REJOIN_LOOKAHEAD-1), -1):
            target_xy = np.asarray(outbound_trace[idx][:2])
            vec = target_xy-pos
            if np.linalg.norm(vec) < 0.40:
                continue
            error = nav.wrap(math.atan2(vec[1], vec[0])-heading)
            tangent = np.asarray(outbound_trace[max(0, idx-2)][:2])-target_xy
            if np.linalg.norm(tangent) < 0.05:
                continue
            tangent_error = nav.wrap(math.atan2(tangent[1], tangent[0])-heading)
            if abs(error) > math.radians(20) or abs(tangent_error) > math.radians(45):
                continue
            if not path_cells_clear(np.asarray([pos, target_xy])):
                continue
            return_join_index = trace_target_index = idx
            return_joined = False
            nav.command[:] = 0
            path = None
            status = 'replanning'
            pause_until = float(data.time)+0.25
            print(f'TURN HANDOFF: checkpoint #{idx} pinned; '
                  f'reversal from turn entry={math.degrees(reversal):.1f}deg', flush=True)
            return
    if not in_lab(pos, extra=TURN_BODY_BUFFER_M):
        nav.command[:] = 0
        status = 'turn_blocked'
        return
    if float(data.time) >= turn_next_check:
        front = [r['range'] for r in cached_rays if r['hit'] and
                 abs(nav.wrap(math.radians(r['angle_deg']))) < math.radians(35)]
        # Screen the remaining rejoin window, not only the first 180 degrees.
        remaining = max(0.15, TURN_MAX_ANGLE-reversal)
        points = np.asarray(arc_points(pos, heading, int(turn_direction),
                                      TURN_RADII[int(turn_direction)], angle=remaining))
        if (not cached_rays or min(front, default=mapper.max_range) < TURN_STOP_RANGE
                or not all(in_lab(p, extra=TURN_BODY_BUFFER_M) for p in points)
                or not path_cells_clear(points)):
            nav.command[:] = 0
            status = 'turn_blocked'
            print('TURN: remaining arc is no longer verified; stopping', flush=True)
            return
        turn_next_check = float(data.time)+SCAN_PERIOD
    nav.command[:] = [TURN_SPEED_MPS, 0, turn_direction*YAW_LIMIT]
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
        obstacle_spawning_enabled=ALLOW_SPAWN,
        lidar_hit_rays=detected_count, contact_with_obstacle=collision,
        replans=replans, plan_kind=path_kind, last_event=last_event,
        recorded_outbound_poses=len(outbound_trace),
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
        home_x_m=round(float(HOME[0]), 3), home_y_m=round(float(HOME[1]), 3),
        home_distance_m=round(float(np.linalg.norm(data.qpos[:2]-HOME)),3),
        home_heading_error_deg=round(math.degrees(abs(nav.wrap(nav.start_yaw-nav.yaw_angle()))),1),
        planned_waypoints=len(path) if path is not None else 0,
        current_plan_target=[round(float(v),2) for v in plan_target] if plan_target is not None else None,
        velocity_command=[round(float(v),3) for v in nav.command],
        measured_planar_speed_mps=round(measured_speed, 3),
        measured_world_yaw_rate_rps=round(measured_yaw_rate, 3),
        return_join_index=return_join_index,
        return_joined=return_joined,
        slow_speed_trials_enabled=SLOW_SPEED_TRIALS,
        path_progress_m=round(path_tracker.progress, 3),
        goal_distance_m=round(float(np.linalg.norm((GOAL if phase=='outbound' else HOME)-data.qpos[:2])),3),
        note='Review trial; full round trip and final HOME heading remain unvalidated')
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


def verify_motion_evidence():
    """Refuse to reuse these measurements with different primary model inputs."""
    evidence = json.loads(Path(__file__).with_name('motion_screening_summary.json').read_text())
    paths = {'policy_sha256': nav.POLICY, 'config_sha256': nav.CONFIG,
             'scene_xml_sha256': nav.SCENE, 'locomotion_source_sha256': nav.__file__}
    for key, filename in paths.items():
        digest = hashlib.sha256()
        with open(filename, 'rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
        if digest.hexdigest() != evidence[key]:
            raise RuntimeError(f'Motion evidence does not match {key}; retain the working files and review the mismatch')
    if mujoco.__version__ != evidence['mujoco_version']:
        raise RuntimeError('MuJoCo version differs from the motion report')
    for direction, name in ((1, 'v0.32_left_r1'), (-1, 'v0.32_right_r1')):
        case = next(c for c in evidence['cases'] if c['name'] == name)
        if (not case['trial_passed'] or case['speed_command_mps'] != TURN_SPEED_MPS
                or case['yaw_command_rps'] != direction*YAW_LIMIT):
            raise RuntimeError('The required 0.32 m/s turn screening trial did not pass')
        TURN_RADII[direction] = float(case['circle_fit_diagnostic']['radius_m'])+0.15


def write_trial_report(rows):
    def finite(value):
        if isinstance(value, dict):
            return {k: finite(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite(v) for v in value]
        if isinstance(value, (float, np.floating)):
            return float(value) if math.isfinite(value) else None
        return value
    report = dict(simulation_only=True, status=status, phase=phase,
                  completed_all_criteria=status == 'finished', collision=collision,
                  simulation_time_s=float(data.time),
                  home=HOME.tolist(), home_yaw_rad=float(nav.start_yaw), goal=GOAL.tolist(),
                  home_distance_m=float(np.linalg.norm(data.qpos[:2]-HOME)),
                  home_heading_error_deg=math.degrees(abs(nav.wrap(nav.start_yaw-nav.yaw_angle()))),
                  trace_cursor=trace_cursor, trace_target_index=trace_target_index,
                  return_join_index=return_join_index, return_joined=return_joined,
                  turn_progress_deg=math.degrees(turn_angle),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  outbound_trace=outbound_trace, rows=rows,
                  last_path=path.tolist() if path is not None else None)
    output = Path('/tmp/g1_roundtrip_report_latest.json')
    temporary = output.with_suffix('.pending')
    temporary.write_text(json.dumps(finite(report), indent=2, allow_nan=False)+'\n')
    temporary.replace(output)
    print(f'ROUND-TRIP REPORT: {output}; status={status}; '
          f'distance={report["home_distance_m"]:.3f}m; '
          f'heading_error={report["home_heading_error_deg"]:.1f}deg', flush=True)


def main():
    global status, phase, path, collision, pause_until, last_event
    global turn_start, turn_direction, turn_entry_yaw, measured_speed, measured_yaw_rate
    global turn_previous_yaw, turn_angle, turn_progress_time, turn_progress_angle, turn_next_check
    global last_progress, last_progress_time
    verify_motion_evidence()
    reset()
    scene_renderer = mujoco.Renderer(model, height=480, width=640)
    depth_renderer = mujoco.Renderer(model, height=240, width=320)
    depth_renderer.enable_depth_rendering()
    server = ThreadingHTTPServer(('0.0.0.0', PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f'G1 review trial on port {PORT}; obstacles={ALLOW_SPAWN}; '
          f'slow-speed trials={SLOW_SPEED_TRIALS}', flush=True)
    print('Using direction-specific 0.32 m/s screening results; obstacle-free round trip is still a trial.', flush=True)
    next_scan = next_view = next_log = 0.0
    steps = 0
    rows = []
    next_record = 0.0
    reported = False
    stable_since = None
    wait_state = None
    wait_started = 0.0
    stop_destination = 'stopped'
    bodies = {nav.pelvis}
    for body in range(1, model.nbody):
        if int(model.body_parentid[body]) in bodies:
            bodies.add(body)
    robot_geoms = {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies}
    frozen = {'idle', 'stopped', 'finished', 'home_heading_unverified', 'stop_unverified',
              'collision', 'fallen', 'geofence_violation', 'blocked', 'turn_required',
              'turn_blocked', 'turn_timeout', 'boundary_blocked', 'assistance_required', 'stalled'}
    controlled_failures = frozen-{'idle', 'stopped', 'finished', 'home_heading_unverified',
                                  'stop_unverified', 'collision', 'fallen', 'geofence_violation'}
    try:
        while True:
            wall_step = time.monotonic()
            with lock:
                incoming = list(requests)
                requests.clear()
            for action in incoming:
                if action == 'restart' or (action == 'start' and status in frozen):
                    reset()
                    steps = 0
                    next_scan = next_view = next_log = 0.0
                    stable_since = wait_state = None
                    rows, next_record, reported = [], 0.0, False
                    if action == 'start':
                        status = 'stabilizing'
                elif action == 'stop' and status not in frozen:
                    status, stop_destination = 'stopping', 'stopped'
                elif action == 'spawn':
                    # Placement is an environment event, never planner ground truth.
                    # The LiDAR route check, not the Spawn button, triggers replanning.
                    spawn_obstacle()
            if status in frozen:
                if status != 'idle' and not reported:
                    write_trial_report(rows)
                    reported = True
                if time.monotonic() >= next_view:
                    render(scene_renderer, depth_renderer)
                    next_view = time.monotonic()+VIEW_PERIOD
                time.sleep(0.05)
                continue
            t = float(data.time)
            if t >= 4 and t >= next_scan:
                scan()
                next_scan = t+SCAN_PERIOD
                if status.startswith('walking') and not route_is_clear():
                    status = 'replanning'
                    path = None
                    pause_until = t+0.4
                    print('SENSOR: route blocked; decelerating before replanning', flush=True)
            if t < 4:
                nav.command[:] = 0
                status = 'stabilizing'
            elif status == 'stabilizing':
                nav.command[:] = 0
                status = 'replanning'
                pause_until = t+0.25
            elif status in ('replanning', 'turn_preparing', 'home_settling', 'stopping'):
                nav.command[:] = 0
                if wait_state != status:
                    wait_state, wait_started, stable_since = status, t, None
                stable = (measured_speed < 0.08 and abs(measured_yaw_rate) < 0.10
                          and np.linalg.norm(applied_command) < 0.01)
                rotation = data.xmat[nav.pelvis].reshape(3, 3)
                tilt = math.degrees(math.acos(np.clip(rotation[2, 2], -1, 1)))
                stable = stable and tilt < 12 and data.qpos[2] > max(0.50, nav.initial_z-0.12)
                stable_since = (stable_since if stable_since is not None else t) if stable else None
                hold = 2.0 if status in ('home_settling', 'stopping') else 0.75
                if stable_since is not None and t-stable_since >= hold and t >= pause_until:
                    if status == 'replanning':
                        scan()
                        make_plan()
                        if path is not None:
                            status = 'walking_to_'+path_kind
                    elif status == 'turn_preparing':
                        direction = select_lab_turn_direction(data.qpos[:2].copy(), nav.yaw_angle())
                        if direction is None:
                            status = 'turn_blocked'
                        else:
                            turn_direction = direction
                            turn_start, turn_entry_yaw = t, nav.yaw_angle()
                            turn_previous_yaw, turn_angle = turn_entry_yaw, 0.0
                            turn_progress_time, turn_progress_angle, turn_next_check = t, 0.0, t
                            status = 'turnaround'
                    elif status == 'home_settling':
                        distance = float(np.linalg.norm(data.qpos[:2]-HOME))
                        error = abs(nav.wrap(nav.start_yaw-nav.yaw_angle()))
                        status = 'finished' if distance <= 0.20 and error < math.radians(8) else 'home_heading_unverified'
                        print(f'HOME: {status}; distance={distance:.3f}m, heading={math.degrees(error):.2f}deg', flush=True)
                    else:
                        status = stop_destination
                    wait_state = None
                    if status in frozen:
                        continue
                elif t-wait_started > 10:
                    status = 'stop_unverified'
                    nav.command[:] = applied_command[:] = 0
                    print('STOP: could not establish standing stability; trial frozen, not passed', flush=True)
            elif status == 'turnaround':
                wait_state = None
                command_turnaround()
            elif status.startswith('walking'):
                wait_state = None
                command_for_path()
            if status in controlled_failures:
                stop_destination, status = status, 'stopping'
                nav.command[:] = 0
                wait_state = None
            if status in frozen:
                continue
            requested_command = nav.command.copy()
            applied_command[0] = slew(float(applied_command[0]), float(requested_command[0]), nav.dt)
            applied_command[1] = 0
            applied_command[2] += np.clip(requested_command[2]-applied_command[2], -0.30*nav.dt, 0.30*nav.dt)
            nav.command[:] = applied_command
            if steps % nav.interval == 0:
                nav.update_policy()
            torque = nav.kp*(nav.target-data.qpos[nav.qids])-nav.kd*data.qvel[nav.vids]
            data.ctrl[nav.ids] = np.clip(torque, model.actuator_ctrlrange[nav.ids, 0],
                                       model.actuator_ctrlrange[nav.ids, 1])
            data.xfrc_applied[nav.pelvis, :] = 0
            support = nav.clamp((4.0-t)/2.0, 0.0, 1.0)
            if support > 0:
                mujoco.mj_forward(model, data)
                origin = np.array([HOME[0], HOME[1], nav.initial_z])
                force = np.array([0., 0., nav.total_mass*9.81])+250*(origin-data.qpos[:3])-60*data.qvel[:3]
                rotation = data.xmat[nav.pelvis].reshape(3, 3)
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, nav.pelvis, velocity, 0)
                angular = 180*np.cross(rotation[:, 2], [0., 0., 1.])-35*velocity[:3]
                data.xfrc_applied[nav.pelvis, :3] = support*force
                data.xfrc_applied[nav.pelvis, 3:] = support*angular
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            steps += 1
            remember_outbound_pose()
            measured_speed = float(np.linalg.norm(data.qvel[:2]))
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, nav.pelvis, velocity, 0)
            measured_yaw_rate = float(velocity[2])
            for ci in range(data.ncon):
                contact = data.contact[ci]
                a, b = int(contact.geom1), int(contact.geom2)
                if ((a in robot_geoms and b in obstacle_slots[:spawn_count]) or
                        (b in robot_geoms and a in obstacle_slots[:spawn_count])):
                    collision, status = True, 'collision'
                    print('FAIL: robot-obstacle contact', flush=True)
                    break
            rotation = data.xmat[nav.pelvis].reshape(3, 3)
            tilt = math.degrees(math.acos(np.clip(rotation[2, 2], -1, 1)))
            if (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
                    or tilt > 40 or data.qpos[2] < 0.50):
                status = 'fallen'
            if not in_lab(data.qpos[:2]):
                status = 'geofence_violation'
            if status in ('fallen', 'collision', 'geofence_violation'):
                nav.command[:] = applied_command[:] = 0
                print('FAIL: '+status+'; simulation frozen for inspection', flush=True)
            if float(data.time) >= next_record:
                rows.append(dict(t=float(data.time), x=float(data.qpos[0]), y=float(data.qpos[1]),
                                 yaw=float(nav.yaw_angle()), height=float(data.qpos[2]), tilt_deg=tilt,
                                 status=status, phase=phase, command=applied_command.tolist(),
                                 measured_speed=measured_speed, measured_yaw_rate=measured_yaw_rate,
                                 cursor=trace_cursor, target=trace_target_index,
                                 turn_angle_deg=math.degrees(turn_angle)))
                next_record = float(data.time)+0.10
            if t >= 4 and status.startswith('walking'):
                if last_progress is None:
                    last_progress, last_progress_time = data.qpos[:2].copy(), t
                elif t-last_progress_time >= 4:
                    moved = float(np.linalg.norm(data.qpos[:2]-last_progress))
                    if moved < 0.12:
                        status, stop_destination = 'stopping', 'stalled'
                    last_progress, last_progress_time = data.qpos[:2].copy(), t
            else:
                last_progress = None
            if t >= next_log:
                print(f'MOTION t={t:.2f} status={status} command={applied_command[0]:.3f}m/s '
                      f'measured={measured_speed:.3f}m/s yaw_command={applied_command[2]:+.3f}rad/s '
                      f'yaw_measured={measured_yaw_rate:+.3f}rad/s tilt={tilt:.1f}deg', flush=True)
                next_log = t+1
            if time.monotonic() >= next_view:
                render(scene_renderer, depth_renderer)
                next_view = time.monotonic()+VIEW_PERIOD
            delay = nav.dt-(time.monotonic()-wall_step)
            if delay > 0:
                time.sleep(delay)
    finally:
        server.shutdown()
        server.server_close()
        scene_renderer.close()
        depth_renderer.close()


if __name__=='__main__':
    main()
