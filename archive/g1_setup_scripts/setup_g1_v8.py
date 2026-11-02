#!/usr/bin/env python3
"""Run on Spark from ~/Desktop/razib/ai_models. Build separate physics-correct V8.
Does not overwrite V7, the locomotion policy, or the original XML.
"""
from pathlib import Path
import ast
import re
import random
import xml.etree.ElementTree as ET

root = Path.cwd()
auto = root/'simulations/unitree_g1/obstacle_navigation/autonomous'
scenes = root/'simulations/unitree_g1/unitree_mujoco/unitree_robots/g1'
scene_source = scenes/'scene_nav_mocap_29dof.xml'
scene_dest = scenes/'scene_nav_mocap_v8_29dof.xml'
nav_source = auto/'live_demo_v7.py'
nav_dest = auto/'live_demo_v8.py'
source = auto/'g1_dynamic_browser_v7.py'
dest = auto/'g1_dynamic_browser_v8.py'
for p in (scene_source, nav_source, source):
    if not p.is_file(): raise SystemExit(f'Missing file: {p}')
for p in (scene_dest, nav_dest, dest):
    if p.exists(): raise SystemExit(f'Will not overwrite {p}')

# Compile every randomized shape once. Move bodies at runtime, never reshape a geom.
scene = ET.parse(scene_source)
world = scene.getroot().find('worldbody')
rng = random.Random(20261029)
n = 0
for body in world.findall('body'):
    if not body.get('name','').startswith('dynamic_nav_obstacle_'): continue
    geom = body.find('geom')
    if geom is None: raise SystemExit(f'No geom in {body.get("name")}')
    sx = rng.uniform(.12,.27)
    sy = rng.uniform(.15,.31)
    sz = rng.uniform(.25,.58)
    geom.set('size',f'{sx:.6f} {sy:.6f} {sz:.6f}')
    geom.set('pos','0 0 0')
    geom.set('contype','1')
    geom.set('conaffinity','1')
    geom.set('rgba','1 .4 .1 1')
    body.set('pos',f'{100+n} 100 {sz:.6f}')
    n += 1
if n != 30: raise SystemExit(f'Expected 30 mocap bodies, got {n}')
# Put the visual-only goal at its final compiled location (10 m).
for geom in world.findall('geom'):
    if geom.get('name') == 'nav_finish':
        old_pos = [float(v) for v in geom.get('pos', '3.048 0 0.015').split()]
        geom.set('pos', f'10 0 {old_pos[2]}')
scene.write(scene_dest,encoding='unicode')

nav = nav_source.read_text()
if 'scene_nav_many_29dof.xml' not in nav: raise SystemExit('V7 nav scene reference not found')
nav = nav.replace('scene_nav_many_29dof.xml','scene_nav_mocap_v8_29dof.xml',1)
compile(nav,str(nav_dest),'exec')

s = source.read_text()
if 'import live_demo_v7 as nav' not in s: raise SystemExit('V7 browser nav import not found')
s = s.replace('import live_demo_v7 as nav','import live_demo_v8 as nav',1)
# Add mocap mapping and helper. The only owner of simulator obstacle positions is the event handler.
anchor = 'MAX_OBSTACLES = len(obstacle_slots)'
if anchor not in s: raise SystemExit('Obstacles anchor missing')
helper = '''MAX_OBSTACLES = len(obstacle_slots)
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
'''
s=s.replace(anchor,helper,1)
# Modify only reset's obstacle teardown; keep collisions enabled and move offstage.
old='''        model.geom_pos[gid] = [100.0, 100.0, 1.0]
        model.geom_contype[gid] = 0
        model.geom_conaffinity[gid] = 0
        model.geom_rgba[gid, 3] = 0'''
new='''        # Move the mocap body offstage. Keep physical collision masks enabled.
        half_height = float(model.geom_size[gid, 2])
        data.mocap_pos[obstacle_mocap[gid]] = [100.0 + obstacle_slots.index(gid), 100.0, half_height]
        model.geom_rgba[gid, 3] = 1.0'''
if old not in s: raise SystemExit('Reset obstacle block not found')
s=s.replace(old,new,1)
# Change only the spawn function while retaining simulator-only ground truth handling.
a=s.index('def spawn_obstacle():')
b=s.index('\ndef scan():',a)
spawn='''def spawn_obstacle():
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
'''
s=s[:a]+spawn+s[b:]
# Remove runtime goal-marker mutation? It is visual-only static geom: relocate in XML instead.
# Prevent false collision flags from robot-floor or unused geometry; only spawned slots count.
old='if contact.geom1 in obstacle_slots or contact.geom2 in obstacle_slots:'
new='if contact.geom1 in obstacle_slots[:spawn_count] or contact.geom2 in obstacle_slots[:spawn_count]:'
if old not in s: raise SystemExit('Contact check not found')
s=s.replace(old,new,1)
s=s.replace('G1 MuJoCo browser experiment V7','G1 MuJoCo browser experiment V8')
s=s.replace('V7: 30 random-size obstacle slots; unvalidated navigation safety; stopped states require review',
            'V8: mocap collision bodies; real contacts verified only in isolated test; navigation unvalidated')
s=s.replace('30 randomized obstacle slots','30 precompiled randomized obstacle slots')
compile(s,str(dest),'exec')
nav_dest.write_text(nav)
dest.write_text(s)
print('Created:',scene_dest)
print('Created:',nav_dest)
print('Created:',dest)
print('Obstacle shapes: 30 precompiled randomized dimensions; mocap movement')
print('V7 unchanged. V8 syntax checked. Runtime navigation has NOT been validated.')
