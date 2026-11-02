#!/usr/bin/env python3
"""Run on Spark from ~/Desktop/razib/ai_models. Creates isolated V7 demo files.
No external dependencies; do not run inside the read-only Docker mount.
"""
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

root = Path.cwd()
auto = root / 'simulations/unitree_g1/obstacle_navigation/autonomous'
scenes = root / 'simulations/unitree_g1/unitree_mujoco/unitree_robots/g1'
base_scene = scenes / 'scene_nav_roundtrip_29dof.xml'
new_scene = scenes / 'scene_nav_many_29dof.xml'
base_nav = auto / 'live_demo.py'
base_browser = auto / 'g1_dynamic_browser_v6.py'
new_nav = auto / 'live_demo_v7.py'
new_browser = auto / 'g1_dynamic_browser_v7.py'
for p in (base_scene, base_nav, base_browser):
    if not p.is_file():
        raise SystemExit(f'Missing source: {p}')
for p in (new_scene, new_nav, new_browser):
    if p.exists():
        raise SystemExit(f'Will not overwrite existing V7 file: {p}')

scene_tree = ET.parse(base_scene)
world = scene_tree.getroot().find('worldbody')
if world is None:
    raise SystemExit('Scene lacks worldbody')
existing = {g.get('name'):g for g in scene_tree.getroot().iter('geom') if g.get('name')}
for i in range(1, 31):
    name = f'nav_obstacle_{i:02d}'
    if name not in existing:
        ET.SubElement(world, 'geom', name=name, type='box', pos=f'{100+i} 100 0.5',
                      size='0.2 0.2 0.5', group='3', contype='0', conaffinity='0',
                      rgba='1 0.38 0.12 0')
scene_tree.write(new_scene, encoding='unicode')

nav = base_nav.read_text()
assert 'scene_nav_roundtrip_29dof.xml' in nav
nav = nav.replace('scene_nav_roundtrip_29dof.xml', 'scene_nav_many_29dof.xml', 1)
new_nav.write_text(nav)

source = base_browser.read_text()
assert 'import live_demo as nav' in source
source = source.replace('import live_demo as nav', 'import live_demo_v7 as nav', 1)
source = source.replace('G1 MuJoCo browser experiment V6', 'G1 MuJoCo browser experiment V7', 1)
source = source.replace('5 obstacle slots', '30 randomized obstacle slots')
source = source.replace('5 obstacle slots ·', '30 randomized obstacle slots ·')
source = source.replace("note='V6: home marker + spawn feedback; 10m round trip; simulated LiDAR only; unvalidated navigation safety'", "note='V7: 30 random-size obstacle slots; unvalidated navigation safety; stopped states require review'")

# Distinct experiment event counter and collision geometry are kept out of planner.
source = source.replace('MAX_OBSTACLES = len(obstacle_slots)',
    'MAX_OBSTACLES = len(obstacle_slots)\nrng = np.random.default_rng()\n')

start = source.index('def spawn_obstacle():')
end = source.index('\ndef scan():', start)
spawn_new = '''def spawn_obstacle():
    """Spawn one random-size obstacle via simulator only.

    The planner never receives coordinates or dimensions; ray casting is the
    only observation pathway. Rejections are explicit, not silently queued.
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
        return reject(f'all {MAX_OBSTACLES} slots are occupied; Restart to clear')
    if not (status.startswith('walking') or status == 'replanning'):
        return reject(f'robot is {status}, not walking')

    heading = nav.yaw_angle()
    forward = np.array([math.cos(heading), math.sin(heading)])
    lateral = np.array([-forward[1], forward[0]])
    robot_xy = data.qpos[:2].copy()

    # Each click generates a different obstacle footprint and height.
    # Half-sizes in metres. Keep them small enough for the test corridor.
    sx = float(rng.uniform(0.12, 0.27))
    sy = float(rng.uniform(0.15, 0.31))
    sz = float(rng.uniform(0.25, 0.58))
    radius = math.hypot(sx, sy)
    chosen = None

    # Try multiple randomized locations on one click, 2.4-3.4m ahead.
    # This is environment setup; navigation sees only LiDAR returns.
    for _ in range(70):
        distance = float(rng.uniform(2.4, 3.4))
        offset = float(rng.uniform(-0.65, 0.65))
        xy = robot_xy + distance * forward + offset * lateral

        if np.linalg.norm(xy - robot_xy) < 1.8 + radius:
            continue
        if np.linalg.norm(xy - HOME) < 1.1 + radius:
            continue
        if np.linalg.norm(xy - GOAL) < 1.1 + radius:
            continue
        if not (mapper.xmin + 0.9 + radius < xy[0] < mapper.xmax - 0.9 - radius
                and mapper.ymin + 0.9 + radius < xy[1] < mapper.ymax - 0.9 - radius):
            continue
        # Never create overlap with previously spawned physical geometry.
        overlaps = False
        for gid in obstacle_slots[:spawn_count]:
            other_size = model.geom_size[gid]
            if (abs(xy[0] - model.geom_pos[gid, 0]) < sx + other_size[0] + 0.65
                and abs(xy[1] - model.geom_pos[gid, 1]) < sy + other_size[1] + 0.65):
                overlaps = True
                break
        if overlaps:
            continue
        cell = mapper.world_to_cell(float(xy[0]), float(xy[1]))
        if cell is None or mapper.grid[cell] == mapper.OCCUPIED:
            continue
        chosen = xy
        break

    if chosen is None:
        return reject('no safe insertion location in front; move or Restart')

    gid = obstacle_slots[spawn_count]
    model.geom_pos[gid] = [float(chosen[0]), float(chosen[1]), sz]
    model.geom_size[gid] = [sx, sy, sz]
    model.geom_contype[gid] = 1
    model.geom_conaffinity[gid] = 1
    model.geom_rgba[gid] = [float(rng.uniform(0.75, 1.0)),
                            float(rng.uniform(0.25, 0.58)), 0.08, 1.0]
    mujoco.mj_forward(model, data)
    spawn_count += 1
    last_spawn_result = (f'Obstacle {spawn_count}/{MAX_OBSTACLES} placed; '
                         f'size {2*sx:.2f} x {2*sy:.2f} x {2*sz:.2f} m')
    last_event = last_spawn_result
    print('EVENT:', last_event, flush=True)
    return last_spawn_result

'''
source = source[:start] + spawn_new + source[end:]
# A turn blocked by an obstacle is still a safe terminal outcome, not permission
# to ignore a LiDAR reading. Report it accurately.
source = source.replace('SAFETY: insufficient forward LiDAR clearance for moving turn',
                        'SAFETY: obstacle limits turning arc; safe stop (detour controller needs improvement)')
compile(nav, str(new_nav), 'exec')
compile(source, str(new_browser), 'exec')
new_browser.write_text(source)
print('Created:', new_scene)
print('Created:', new_nav)
print('Created:', new_browser)
print('30 obstacle slots available; random dimensions on each accepted click.')
print('WARNING: A* and turning are still experimental; a blocked turn must stop.')
