#!/usr/bin/env python3
"""Simulation-only surgical V12 return-join bug fix. Creates V12 returnfix copy.
Does not modify locomotion, obstacles, geofence or installed V12.
"""
from pathlib import Path
import ast
import sys

root = Path.home()/'Desktop/razib/ai_models/simulations/unitree_g1/obstacle_navigation/autonomous'
if not root.is_dir():
    root = Path.cwd()/'simulations/unitree_g1/obstacle_navigation/autonomous'
src = root/'g1_dynamic_browser_v12_lab_trial.py'
dst = root/'g1_dynamic_browser_v12_returnfix.py'
if not src.is_file(): sys.exit('Missing V12 source: '+str(src))
if dst.exists(): sys.exit('Output exists; not overwritten: '+str(dst))
s = src.read_text()
tree=ast.parse(s)
functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
for name in ('make_plan','command_turnaround'):
    if name not in functions: sys.exit('Missing function '+name)

def patch_make_plan(txt):
    old = '''        recorded = np.asarray(
            [pos] + [list(p[:2]) for p in outbound_trace[idx:cursor+1][::-1]],
            dtype=float
        )'''
    new = '''        if trace_target_index is None:
            # First rejoin: aim at the chosen checkpoint itself.
            # DO NOT first jump toward the final outbound point (cursor),
            # which is frequently BEHIND the robot after its U-turn.
            recorded = np.asarray([pos, target], dtype=float)
        else:
            # Once physically on the route, retrace saved coordinates
            # sequentially in reverse without taking a direct shortcut.
            recorded = np.asarray(
                [pos] + [list(p[:2]) for p in outbound_trace[idx:cursor+1][::-1]],
                dtype=float
            )'''
    if txt.count(old)!=1: raise RuntimeError('V12 make_plan recorded-polyline block differs; no file created')
    return txt.replace(old,new,1)

def patch_turn(txt):
    start=txt.index('    # Follow the tangent of the saved return corridor')
    end=txt.index('    # Obstacle scan is from actual LiDAR',start)
    new='''    # Do not accept "turnaround aligned" just because a fixed tangent is
    # nearby. A feasible actual rejoin heading and substantial yaw reversal
    # must BOTH be observed. The ONNX gait cannot pivot in place.
    heading = nav.yaw_angle()
    reversal = abs(nav.wrap(heading - nav.start_yaw))
    chosen_idx = None
    candidate_error = None
    if len(outbound_trace) >= 3 and trace_cursor is not None:
        cursor = min(int(trace_cursor), len(outbound_trace)-1)
        lower = max(0, cursor-REJOIN_LOOKAHEAD)
        indices = list(range(cursor-1, lower-1, -3))
        if lower not in indices:
            indices.append(lower)
        # Prefer targets giving an actual forward path after turning.
        for idx in indices:
            target = np.asarray(outbound_trace[idx][:2],dtype=float)
            vec = target - pos
            if float(np.linalg.norm(vec)) < 0.40:
                continue
            desired = math.atan2(float(vec[1]),float(vec[0]))
            err = nav.wrap(desired-heading)
            if abs(err) > math.radians(20):
                continue
            if not path_cells_clear(np.asarray([pos,target],dtype=float)):
                continue
            chosen_idx, candidate_error = idx, err
            break
    # A geometrically feasible candidate alone is insufficient: avoid the
    # prior false 25-degree alignment while still facing outbound direction.
    if reversal >= math.radians(140) and chosen_idx is not None:
        nav.command[:] = 0.0
        path = None
        status = 'replanning'
        pause_until = float(data.time) + 0.25
        print(f'MISSION: turn genuinely reversed {math.degrees(reversal):.1f} deg; '
              f'forward-reachable saved checkpoint #{chosen_idx}; '
              f'heading error {math.degrees(candidate_error):+.1f} deg', flush=True)
        return
'''
    return txt[:start]+new+txt[end:]

repls=[]
for name,fn in ((name,functions[name]) for name in ('make_plan','command_turnaround')):
    segment=''.join(s.splitlines(keepends=True)[fn.lineno-1:fn.end_lineno])
    # AST line offsets are used only to select named functions, not brittle text
    if name=='make_plan':
        new=patch_make_plan(segment)
    else:
        new=patch_turn(segment)
    repls.append((fn.lineno,fn.end_lineno,new))
lines=s.splitlines(keepends=True)
for a,b,new in sorted(repls,reverse=True):
    if not new.endswith('\n'):new+='\n'
    lines[a-1:b]=[new]
output=''.join(lines)
ast.parse(output,filename=str(dst))
compile(output,str(dst),'exec')
dst.write_text(output)
print('Created:',dst)
print('Changed: initial return connector + verified U-turn completion only')
print('Unchanged: lab boundaries, collision model, obstacle spawning, gait and adaptive speed')
print('SIMULATION-ONLY, syntax validated; NOT tested in MuJoCo')
