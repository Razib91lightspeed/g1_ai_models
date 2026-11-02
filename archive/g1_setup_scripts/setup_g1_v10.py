#!/usr/bin/env python3
"""Create V10 route-retrace experiment from V9, leaving V8/V9 unchanged.

Run from ~/Desktop/razib/ai_models on Spark, after V9 exists.
Simulation-only. No physical robot use. No claim of verified obstacle avoidance.
"""
from pathlib import Path
import ast
import shutil

folder = Path.cwd()/'simulations/unitree_g1/obstacle_navigation/autonomous'
src = folder/'g1_dynamic_browser_v9.py'
dst = folder/'g1_dynamic_browser_v10.py'
if not src.is_file(): raise SystemExit('Missing V9 browser; run setup_g1_v9.py first')
if dst.exists(): raise SystemExit('V10 already exists; refusing to overwrite')
s = src.read_text()

def once(old,new):
    global s
    n=s.count(old)
    if n!=1: raise SystemExit(f'Expected one anchor, got {n}: {old[:100]!r}; nothing written')
    s=s.replace(old,new,1)

# Ensure the successful map extension in V8 is inherited, not a new silent regression.
if 'xlim=(-2, 12.0), ylim=(-4.0, 4.0)' in s:
    once('xlim=(-2, 12.0), ylim=(-4.0, 4.0)', 'xlim=(-2, 16.0), ylim=(-6.0, 6.0)')

once('TRACE_REACHED_M = 0.30', '''TRACE_REACHED_M = 0.30
REJOIN_MAX_HEADING_DEG = 75.0
REJOIN_LOOKAHEAD = 35  # recorded points searched for a feasible join
RETRACE_CHUNK_POINTS = 7  # ~0.7 m along the actual recorded polyline
RETRACE_GRID_STEP_M = 0.05
return_mode = 'none'  # 'recorded_corridor' or 'astar_reconnection'
''')
once('    global outbound_trace, trace_cursor, trace_target_index, trace_detours, trace_replans\n', '''    global outbound_trace, trace_cursor, trace_target_index, trace_detours, trace_replans, return_mode
''')
once('    trace_replans = 0\n', '''    trace_replans = 0
    return_mode = 'none'
''')

start=s.index('def make_plan():')
end=s.index('\ndef command_for_path():',start)
replacement='''def path_cells_clear(points):
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
    trace_target_index=idx
    path=np.asarray(candidate,dtype=float)
    path_kind='home_trace'
    plan_target=target.copy()
    return_mode=mode
    if mode=='astar_reconnection':
        trace_detours+=1
    print(f'PLAN #{replans}: return {mode} checkpoint #{idx}/{len(outbound_trace)-1}, '
          f'target={target.round(2)}',flush=True)

'''
s=s[:start]+replacement+s[end:]

# Turn toward a feasible reverse-path approach, not just the home coordinate.
# The final rejoin choice is still verified against LiDAR by make_plan().
once('''    to_home = HOME - pos
    desired = math.atan2(to_home[1], to_home[0])
    error = nav.wrap(desired - nav.yaw_angle())''','''    # Follow the tangent of the saved return corridor, not the direct HOME
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
    error = nav.wrap(desired - nav.yaw_angle())''')

once('        return_reconnections=trace_detours,','''        return_reconnections=trace_detours,
        return_mode=return_mode,''')
s=s.replace('G1 MuJoCo browser experiment V9','G1 MuJoCo browser experiment V10')
s=s.replace('5 obstacle slots · LiDAR map','30 obstacle slots · LiDAR map')

ast.parse(s,filename=str(dst))
compile(s,str(dst),'exec')
dst.write_text(s)
print('Created:',dst)
print('V8/V9, mocap physics, ONNX controller, and scene are unchanged.')
print('V10 uses the recorded reverse polyline when known clear; A* only to reconnect.')
print('WARNING: heading and 2D occupancy checks do not prove feasible full-body turns.')
