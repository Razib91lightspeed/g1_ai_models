#!/usr/bin/env python3
"""Create simulation-only V9 route-memory browser from tested V8 on Spark.

Run from ~/Desktop/razib/ai_models. Does NOT replace V8, XML or policy.
V8's mocap physics and corrected robot-only support mass are reused.
"""
from pathlib import Path
import ast
import shutil

root = Path.cwd()
auto = root/'simulations/unitree_g1/obstacle_navigation/autonomous'
src = auto/'g1_dynamic_browser_v8.py'
dst = auto/'g1_dynamic_browser_v9.py'
nav_src = auto/'live_demo_v8.py'
if not src.is_file() or not nav_src.is_file():
    raise SystemExit('Missing V8 browser or locomotion module')
if dst.exists():
    raise SystemExit('V9 browser already exists; refusing to overwrite it')
if 'body_mocapid' not in nav_src.read_text() or 'scene_nav_mocap_v8_29dof.xml' not in nav_src.read_text():
    raise SystemExit('V8 robot-only mass fix or mocap scene reference missing; do not proceed')
s = src.read_text()

def replace_one(old,new):
    global s
    n=s.count(old)
    if n != 1:
        raise SystemExit(f'Expected 1 matching anchor, found {n}: {old[:100]!r}; no file written')
    s=s.replace(old,new,1)

replace_one('import live_demo_v8 as nav', 'import live_demo_v8 as nav') if False else None
replace_one('MAX_OBSTACLES = len(obstacle_slots)', '''MAX_OBSTACLES = len(obstacle_slots)

# In-RAM path memory. At 10 cm spacing, a 10 m leg is ~100 poses.
# No cloud, disk writes, or AI-model memory is required.
TRACE_SPACING_M = 0.10
TRACE_AHEAD_M = 0.75
TRACE_REACHED_M = 0.30
MAX_TRACE_POINTS = 3000  # Fail safe for inadvertently long experiments.
outbound_trace = []    # (x, y, yaw) in the fixed simulation world frame
trace_cursor = None   # index of the next return checkpoint
trace_target_index = None
trace_detours = 0
trace_replans = 0
''')

# Route recordings initialize with true robot initial pose and heading.
anchor = "    nav.reset_sim()"
replace_one(anchor, '''    global outbound_trace, trace_cursor, trace_target_index, trace_detours, trace_replans
'''+anchor+'''
    outbound_trace = [(float(data.qpos[0]), float(data.qpos[1]), float(nav.yaw_angle()))]
    trace_cursor = None
    trace_target_index = None
    trace_detours = 0
    trace_replans = 0''')

# Reuse original outbound A*/frontier planner untouched, V9 return planner below.
replace_one('def make_plan():', 'def make_plan_original():')
insert_before='def command_for_path():'
trace_impl='''def remember_outbound_pose():
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


def make_plan():
    """Use A* for outward exploration; prefer remembered path for return."""
    global path, path_kind, replans, status, plan_target
    global trace_cursor, trace_target_index, trace_detours, trace_replans
    if phase != 'returning' or trace_cursor is None:
        return make_plan_original()

    pos = data.qpos[:2].copy()
    if float(np.linalg.norm(pos-HOME)) < 0.25:
        # Delegate final pose / heading verification to command_for_path().
        path = np.asarray([pos, HOME])
        path_kind = 'home_trace'
        plan_target = HOME.copy()
        trace_target_index = 0
        return

    if len(outbound_trace) < 3:
        print('MEMORY: too few outbound poses; using original planner',flush=True)
        return make_plan_original()

    # Find progress along the original path without moving the cursor back
    # toward the destination. Check only a local window to avoid looping
    # into a visually nearby but topologically unrelated section.
    search_start = max(0, trace_cursor-18)
    near_idx = min(range(search_start, trace_cursor+1),
                   key=lambda i: float(np.linalg.norm(pos-np.asarray(outbound_trace[i][:2]))))
    trace_cursor = min(trace_cursor, near_idx)
    trace_replans += 1

    # Prefer the next ~0.75 m of recorded track. Check the remembered pose
    # with current LiDAR occupancy; A* handles a local detour if necessary.
    # All targets are previously visited, but can be newly obstructed.
    target_index = trace_cursor
    walked = 0.0
    while target_index > 0 and walked < TRACE_AHEAD_M:
        a = outbound_trace[target_index]
        b = outbound_trace[target_index-1]
        walked += math.hypot(a[0]-b[0],a[1]-b[1])
        target_index -= 1

    candidates = [target_index]
    # If a new obstacle blocks this segment, reconnect to another visited
    # waypoint farther toward HOME, never to an unverified straight line.
    for offset in (5, 10, 16, 24, 36):
        i = max(0, target_index-offset)
        if i not in candidates:
            candidates.append(i)

    selected = None
    for idx in candidates:
        target = np.asarray(outbound_trace[idx][:2],dtype=float)
        candidate, reason = planner.plan(pos,target)
        if candidate is not None and len(candidate) >= 2:
            selected = (idx,target,candidate,reason)
            break

    replans += 1
    if selected is None:
        path, path_kind, plan_target, trace_target_index = None,None,None,None
        status = 'blocked'
        nav.command[:] = 0.0
        print('SAFETY: recorded return corridor blocked; no LiDAR-verified A* reconnection',flush=True)
        return

    idx,target,candidate,reason=selected
    if idx < target_index:
        trace_detours += 1
        print(f'MEMORY: original checkpoint blocked; rerouting to earlier recorded pose #{idx}',flush=True)
    trace_target_index = idx
    path = np.asarray(candidate,dtype=float)
    path_kind = 'home_trace'
    plan_target = target.copy()
    print(f'PLAN #{replans}: remembered return checkpoint #{idx}/{len(outbound_trace)-1} '
          f'({reason}), target={target.round(2)}',flush=True)


'''
replace_one(insert_before,trace_impl+insert_before)

# When outbound mission completes, freeze the trace before turnaround.
needle = "            phase = 'returning'"
replace_one(needle, '            begin_return_trace()\n'+needle)

# When a return checkpoint is reached, move on to next recorded segment.
needle = '    if path is None or len(path) == 0:'
if s.count(needle) != 1:
    raise SystemExit('Return waypoint hook location missing')
hook='''    if (phase == 'returning' and path_kind == 'home_trace'
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

'''
replace_one(needle,hook+needle)

# At next physics tick after dynamics, record only real measured outbound motion.
replace_one('            physics_steps+=1', '            physics_steps+=1\n            remember_outbound_pose()')

# Render path-memory overlay as distinct turquoise trace on map (no physics changes).
needle = '    if path is not None:\n        pixels = [pix(p) for p in path]'
replace_one(needle,'''    if len(outbound_trace) >= 2:
        remembered_pixels = [pix(p[:2]) for p in outbound_trace]
        remembered_pixels = [p for p in remembered_pixels if p is not None]
        if len(remembered_pixels) > 1:
            pen.line(remembered_pixels,fill=(0,185,180),width=3)
    if path is not None:
        pixels = [pix(p) for p in path]''')

# Add explicit telemetry under existing plan info.
replace_one('        replans=replans, plan_kind=path_kind, last_event=last_event,', '''        replans=replans, plan_kind=path_kind, last_event=last_event,
        recorded_outbound_poses=len(outbound_trace),
        estimated_trace_memory_bytes=len(outbound_trace)*12,
        return_checkpoint_index=trace_target_index,
        return_cursor_index=trace_cursor,
        return_reconnections=trace_detours,
        return_checkpoint_plans=trace_replans,''')

s=s.replace('G1 MuJoCo browser experiment V8','G1 MuJoCo browser experiment V9')
s=s.replace('V8: mocap collision bodies; real contacts verified only in isolated test; navigation unvalidated',
            'V9: RAM route trace + LiDAR-verified return checkpoints; mocap contacts; experimental safety')

# AST check both top-level syntax and global rules; no files written until success.
ast.parse(s,filename=str(dst))
compile(s,str(dst),'exec')
dst.write_text(s)
print('Created:',dst)
print('V8 locomotion module and scene reused unchanged.')
print('Outbound route sampled each 0.10 m in RAM, bounded to 3000 poses.')
print('Return: reversed recorded checkpoints, each planned with live occupancy A*.')
print('WARNING: code syntax checked; real simulation & turn safety NOT verified.')
