#!/usr/bin/env python3
"""G1 MuJoCo browser experiment: unexpected obstacle and sensor-based replanning.

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

import live_demo as nav
from lidar_mapper import LidarMapper
from path_planner import AStarPlanner
from frontier_planner import select_frontier

PORT = 8771
GOAL = np.array([10.0, 0.0])
HOME = np.array([0.0, 0.0])
CRUISE = 0.40  # validated at this speed; not a physical safety value
YAW_LIMIT = 0.30
STOP_RANGE = 0.80  # emergency stop range along commanded travel
SLOW_RANGE = 1.20  # stop conservatively when route is not verified
CLEARANCE = 0.50  # experimental inflation radius
SCAN_PERIOD = 0.15
PLAN_PERIOD = 1.25
VIEW_PERIOD = 0.35

model, data = nav.model, nav.data
mapper = LidarMapper(model, xlim=(-2, 12.0), ylim=(-4.0, 4.0), resolution=0.1)
planner = AStarPlanner(mapper, clearance=CLEARANCE, allow_unknown=False)
obstacle_slots = sorted(nav.obstacle_ids)
MAX_OBSTACLES = len(obstacle_slots)

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
last_route_stop = None
cached_rays = []

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
<p>10 m goal · up to 5 surprise obstacles · LiDAR map · A* / frontier replanning · simulation only</p>
<button class="start" onclick="cmd('start')">▶ Start</button>
<button class="stop" onclick="cmd('stop')">■ Stop</button>
<button class="restart" onclick="cmd('restart')">↻ Restart</button>
<button class="spawn" onclick="cmd('spawn')">＋ Spawn Obstacle</button>
<p id="feedback">Ready.</p><div class="grid">
<div class="panel wide"><h3>Live MuJoCo</h3><img id="scene" src="/scene.jpg"></div>
<div class="panel"><h3>LiDAR occupancy map</h3><img id="map" src="/map.jpg"><small>Gray unknown · light free · orange detected obstacle / buffer · blue route</small></div>
<div class="panel"><h3>Depth camera</h3><img id="depth" src="/depth.jpg"><small>Depth visualisation only; not fused into the controller yet</small></div>
<div class="panel wide"><h3>Measured state</h3><pre id="status"></pre></div></div>
<script>
async function cmd(action){try{let r=await fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action})});let d=await r.json();document.getElementById('feedback').textContent=d.message||d.error;}catch(e){document.getElementById('feedback').textContent=String(e)}}
async function refresh(){try{let r=await fetch('/status');let d=await r.json();document.getElementById('status').textContent=JSON.stringify(d,null,2);let t=Date.now();for(let v of ['scene','map','depth'])document.getElementById(v).src='/'+v+'.jpg?t='+t;}catch(e){}}
setInterval(refresh,750);refresh();</script></body></html>'''


def jpg(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format='JPEG', quality=77)
    return buf.getvalue()


def reset():
    global status, phase, path, path_kind, spawn_count, collision, replans
    global last_front_range, pause_until, detected_count, last_detected, last_event, last_route_stop, cached_rays
    nav.reset_sim()
    # Simulator-owned course setup: hide all obstacles until Spawn is clicked.
    for gid in nav.obstacle_ids:
        model.geom_pos[gid] = [100.0, 100.0, 1.0]
        model.geom_contype[gid] = 0
        model.geom_conaffinity[gid] = 0
        model.geom_rgba[gid, 3] = 0
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
    last_front_range, pause_until = None, 0.0


def spawn_obstacle():
    """Simulator-only event; planner never reads obstacle placement."""
    global spawn_count, last_event
    if spawn_count >= MAX_OBSTACLES:
        return f'All {MAX_OBSTACLES} obstacle slots used. Restart for a new run.'
    # The obstacle is placed ahead of the actual robot, not in its body.
    # Vary its side and range to make successive encounters different.
    theta = nav.yaw_angle()
    forward = np.array([math.cos(theta), math.sin(theta)])
    side = np.array([-forward[1], forward[0]])
    offsets = [0.0, 0.30, -0.30, 0.15, -0.15]
    distances = [2.0, 2.25, 2.1, 2.35, 2.2]
    k = spawn_count
    xy = data.qpos[:2].copy() + distances[k] * forward + offsets[k] * side
    # Never place an object beyond the mapped course or on the goal marker.
    if not (mapper.xmin + 0.5 < xy[0] < mapper.xmax - 0.5 and
            mapper.ymin + 0.5 < xy[1] < mapper.ymax - 0.5):
        return 'Cannot spawn here: location outside navigation map.'
    gid = obstacle_slots[k]
    model.geom_pos[gid] = [float(xy[0]), float(xy[1]), 0.55]
    model.geom_size[gid] = [0.25, 0.34, 0.55]
    model.geom_contype[gid] = 1
    model.geom_conaffinity[gid] = 1
    model.geom_rgba[gid] = [1.0, 0.38, 0.12, 1.0]
    mujoco.mj_forward(model, data)
    spawn_count += 1
    last_event = f'Obstacle {spawn_count}/{MAX_OBSTACLES} inserted in simulator'
    print('EVENT:', last_event, flush=True)
    return last_event


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


def make_plan():
    global path, path_kind, replans, status
    pos = data.qpos[:2].copy()
    goal = GOAL if phase == 'outbound' else HOME
    direct, reason = planner.plan(pos, goal)
    if direct is not None:
        path, path_kind = direct, 'goal'
    else:
        frontier = select_frontier(mapper, planner, pos, goal)
        if frontier['status'] == 'frontier_found':
            path, path_kind = frontier['path'], 'frontier'
        else:
            path, path_kind = None, None
            status = 'blocked'
    replans += 1
    print(f'PLAN #{replans}: {path_kind or "blocked"} ({reason})', flush=True)


def command_for_path():
    """Closed-loop waypoint following with measured front-sector stop."""
    global status, phase, path, pause_until
    pos = data.qpos[:2].copy()
    destination = GOAL if phase == 'outbound' else HOME
    if np.linalg.norm(pos - destination) < 0.20:
        nav.command[:] = 0.0
        if phase == 'outbound':
            phase = 'returning'
            path = None
            status = 'replanning'
            pause_until = data.time + 0.6
        else:
            heading_error = nav.wrap(nav.start_yaw - nav.yaw_angle())
            if abs(heading_error) < math.radians(8):
                status = 'finished'
            else:
                # In-place turning was NOT verified. Do not claim posture recovery.
                status = 'home_heading_unverified'
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

    if abs(error) > 0.85:
        nav.command[:] = 0.0
        status = 'turn_required'
        # No in-place turning assumed; the next version will use an
        # explicitly validated turning maneuver or cautious arcs.
        return

    nav.command[:] = [CRUISE, 0.0, float(np.clip(1.1*error, -YAW_LIMIT, YAW_LIMIT))]
    status = 'walking_to_' + path_kind


def draw_map():
    arr = np.empty((*mapper.grid.shape, 3), dtype=np.uint8)
    arr[:] = (94, 101, 112)
    arr[mapper.grid == mapper.FREE] = (230, 237, 242)
    inflated = planner.build_blocked_map()
    arr[inflated & (mapper.grid != mapper.UNKNOWN)] = (249, 185, 100)
    arr[mapper.grid == mapper.OCCUPIED] = (216, 80, 43)
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

    if path is not None:
        pixels = [pix(p) for p in path]
        pixels = [p for p in pixels if p is not None]
        if len(pixels)>1:
            pen.line(pixels, fill=(27, 95, 255), width=4)
    for xy, color in ((HOME,'#2b83ff'), (GOAL,'#2bc96b'), (data.qpos[:2],'#171d26')):
        p = pix(xy)
        if p:
            x,y=p
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
        planned_waypoints=len(path) if path is not None else 0,
        velocity_command=[round(float(v),3) for v in nav.command],
        goal_distance_m=round(float(np.linalg.norm((GOAL if phase=='outbound' else HOME)-data.qpos[:2])),3),
        note='10m experimental course; LiDAR-only navigation; no physical-robot safety guarantee')
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
                    next_scan = next_plan = next_view = 0.0
                    physics_steps=0
                    wall_start=time.monotonic()
                elif action == 'start' and status not in ('walking_to_goal','walking_to_frontier','replanning'):
                    if status in ('collision','fallen','finished','home_heading_unverified'):
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
                    print(message,flush=True)
                    if message.startswith('Obstacle ') and (status.startswith('walking') or status in ('replanning', 'stabilizing')):
                        status='replanning'
                        nav.command[:]=0
                        pause_until=float(data.time)+0.5
                        next_scan=next_plan=0.0

            if status in ('idle','stopped','finished','blocked','collision','fallen',
                          'turn_required','home_heading_unverified'):
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
                        status='walking_to_'+path_kind
            elif status.startswith('walking'):
                if t>=next_plan:
                    make_plan()
                    next_plan=t+PLAN_PERIOD
                if status.startswith('walking'):
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
            for ci in range(data.ncon):
                contact=data.contact[ci]
                if contact.geom1 in obstacle_slots or contact.geom2 in obstacle_slots:
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
