from pathlib import Path
import xml.etree.ElementTree as ET

base = Path('simulations/unitree_g1')
scene_dir = base / 'unitree_mujoco/unitree_robots/g1'
source = scene_dir / 'scene_29dof.xml'
scene = scene_dir / 'scene_nav_demo_29dof.xml'
viewer = base / 'zigzag/zigzag_viewer.py'
assert source.exists() and viewer.exists(), 'Missing existing G1 scene or zigzag viewer'

# Use a separate scene so the other three demos remain unchanged.
tree = ET.parse(source)
world = tree.getroot().find('worldbody')
assert world is not None
for name,x,y in [('nav_box_left',1.55,-0.67), ('nav_box_right',2.65,0.73)]:
    ET.SubElement(world,'geom',{'name':name,'type':'box', 'pos':f'{x} {y} 0.40',
      'size':'0.20 0.22 0.40','rgba':'1.0 0.38 0.10 1',
      'contype':'1','conaffinity':'1'})
ET.SubElement(world,'geom',{'name':'nav_finish','type':'cylinder',
  'pos':'3.5 0.1 0.015','size':'0.22 0.015','rgba':'0.12 0.85 0.30 1',
  'contype':'0','conaffinity':'0'})
ET.indent(tree,space='  ')
tree.write(scene, encoding='unicode')

code = viewer.read_text()
old = '/workspace/unitree_mujoco/unitree_robots/g1/scene_29dof.xml'
new = '/workspace/unitree_mujoco/unitree_robots/g1/scene_nav_demo_29dof.xml'
old_alt = '/workspace/unitree_mujoco/unitree_robots/g1/scene_obstacles_29dof.xml'
assert code.count(old)+code.count(old_alt)==1, 'Unexpected scene definition'
code=code.replace(old,new,1).replace(old_alt,new,1)

# Requires the already installed Replay/Start/Stop patch.
old_block = '''            elif elapsed < 16.0:
                status = 'zigzag'
                ramp = float(np.clip((elapsed - 2.0) / 2.0, 0.0, 1.0))
                turn = 0.05 if elapsed < 7.0 or elapsed >= 11.0 else -0.05
                command[:] = [0.30 * ramp, 0.0, turn * ramp]
            else:
                mode = 'stopping'
                stop_start = t
                stop_initial = float(command[0])'''
new_block = '''            elif elapsed < 45.0:
                # Ground-truth 2D obstacle map. This is a sim-only
                # reactive detour, NOT camera perception or real-robot safety.
                px, py = map(float, data.qpos[:2])
                rot_nav = data.xmat[pelvis].reshape(3, 3)
                yaw_nav = math.atan2(rot_nav[1, 0], rot_nav[0, 0])
                # Geometric clearance between pelvis center and nearest box.
                obstacles = [(1.55, -0.67), (2.65, 0.73)]
                clearance_nav = min(math.hypot(
                    max(abs(px - ox) - 0.20, 0.0),
                    max(abs(py - oy) - 0.22, 0.0))
                    for ox, oy in obstacles)
                # Gently curve around alternating obstacle sides.
                # Time-independent feedback: route phase depends on actual x.
                if px < 1.15:
                    turn_nav = 0.05
                elif px < 2.25:
                    turn_nav = -0.05
                else:
                    turn_nav = 0.05
                ramp = float(np.clip((elapsed - 2.0) / 2.0, 0.0, 1.0))
                status = 'navigating'
                command[:] = [0.30 * ramp, 0.0, turn_nav * ramp]
                # Latch a stop if clearance is inadequate or robot tips.
                # Pelvis-only clearance is NOT sufficient for real safety.
                if clearance_nav < 0.36 or data.qpos[2] < 0.65:
                    print(f'NAV: STOP clearance={clearance_nav:.3f} x={px:.2f} y={py:.2f}', flush=True)
                    mode = 'stopping'
                    status = 'obstacle_stop'
                    stop_start = t
                    stop_initial = float(command[0])
                elif px >= 3.45:
                    print(f'NAV: goal approach x={px:.2f} y={py:.2f}', flush=True)
                    mode = 'stopping'
                    stop_start = t
                    stop_initial = float(command[0])
            else:
                print('NAV: timeout (goal not reached)', flush=True)
                mode = 'stopping'
                stop_start = t
                stop_initial = float(command[0])'''
assert code.count(old_block)==1, 'Replay zigzag motion block not found. Apply the replay patch first.'
code=code.replace(old_block,new_block,1)
# Keep obstacle ahead and render further away.
code=code.replace('camera.distance = 4.5','camera.distance = 5.3',1)
old_camera='camera.lookat[:] = [float(data.qpos[0]), float(data.qpos[1]), 0.8]'
new_camera='camera.lookat[:] = [float(data.qpos[0]) + 0.8, float(data.qpos[1]), 0.75]'
assert old_camera in code
code=code.replace(old_camera,new_camera,1)
code=code.replace('Unitree G1 — Zigzag Replay','Unitree G1 — Obstacle Navigation Test')
code=code.replace('Unitree G1 — Zigzag Walking','Unitree G1 — Obstacle Navigation Test')
# Detect any contact on actual obstacle geoms and log it once per transition.
needle='    mujoco.mj_step(model, data)'
assert code.count(needle)==1
code=code.replace(needle,needle+'''
    for ci in range(data.ncon):
        contact = data.contact[ci]
        for geom_name in ('nav_box_left', 'nav_box_right'):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if gid in (contact.geom1, contact.geom2):
                if mode != 'collision':
                    print('NAV: COLLISION with ' + geom_name, flush=True)
                mode = 'collision'
                status = 'collision'
                command[:] = 0.0
''',1)
# Make contact stop persistent, not a transient status overwrite.
code=code.replace("    if mode == 'fallen':\n", "    if mode == 'collision':\n        status = 'collision'\n        command[:] = 0.0\n    elif mode == 'fallen':\n",1)
backup=viewer.with_suffix('.py.pre_nav.bak')
if not backup.exists(): backup.write_text(viewer.read_text())
viewer.write_text(code)
print('Scene created:',scene)
print('Viewer updated:',viewer)
print('Backup:',backup)
print('Obstacles: x=1.55,y=-0.67 and x=2.65,y=+0.73; goal x=3.5,y=+0.1')
print('Note: tested code syntax only. Navigation is not yet validated.')
