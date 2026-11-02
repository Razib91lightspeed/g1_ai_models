#!/usr/bin/env python3
"""Local-only browser control of simulated G1. No robot/DDS connections."""
import io
import json
import math
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mujoco
import numpy as np
import onnxruntime as ort
import yaml
from PIL import Image
sys.path.insert(0, "/workspace/skills")
from skill_executor import validate_command

MODEL = '/workspace/unitree_mujoco/unitree_robots/g1/scene_nav_demo_29dof.xml'
CONFIG = '/workspace/policy/params/deploy.yaml'
POLICY = '/workspace/policy/exported/policy.onnx'

with open(CONFIG, encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
model = mujoco.MjModel.from_xml_path(MODEL)
data = mujoco.MjData(model)
ort_session = ort.InferenceSession(POLICY, providers=['CPUExecutionProvider'])
ids = np.asarray(cfg['joint_ids_map'], dtype=int)
assert sorted(ids.tolist()) == list(range(29)) and model.nu == 29
qids = np.asarray([model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in ids])
vids = np.asarray([model.jnt_dofadr[model.actuator_trnid[i, 0]] for i in ids])
defaults = np.asarray(cfg['default_joint_pos'], dtype=float)
kp = np.asarray(cfg['stiffness'], dtype=float)[ids]  # CRITICAL: policy order
kd = np.asarray(cfg['damping'], dtype=float)[ids]
action_cfg = cfg['actions']['JointPositionAction']
scale = np.asarray(action_cfg['scale'], dtype=float)
offset = np.asarray(action_cfg['offset'], dtype=float)
terms_cfg = cfg['observations']
history = {name: deque(maxlen=int(term['history_length'])) for name, term in terms_cfg.items()}
last_action = np.zeros(29, dtype=float)
target = defaults.copy()
command = np.zeros(3)
data.qpos[qids] = defaults
data.qvel[:] = 0
mujoco.mj_forward(model, data)
pelvis = model.body('pelvis').id
initial_z = float(data.qpos[2])
total_mass = float(np.sum(model.body_mass))
physics_dt = float(model.opt.timestep)
policy_interval = int(round(float(cfg['step_dt']) / physics_dt))
assert abs(policy_interval * physics_dt - cfg['step_dt']) < 1e-8

state_lock = threading.Lock()
frame = None
pending = None
status = 'initializing'
current_task = None
cruise_elapsed = 0.0
v_cmd = 0.0
# ZIGZAG_REPLAY_PATCH_V1
control_request = None
mode = 'idle'
play_start = None
stop_start = None
stop_initial = 0.0
auto_start = False
telemetry = {'status': 'initializing', 'time_s': 0, 'x_m': 0, 'tilt_deg': 0, 'speed_command_mps': 0}

HTML = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>G1 Zigzag Replay</title><style>
body{background:#101419;color:#eef2f5;font:16px system-ui;margin:20px auto;max-width:950px;padding:0 16px}
img{width:100%;border-radius:12px;background:#202730}button{padding:12px 20px;margin:8px 5px 8px 0;cursor:pointer;font:inherit;border:0;border-radius:8px}
pre{background:#202730;padding:15px;border-radius:9px;white-space:pre-wrap}small{color:#acb9c6}
</style></head><body><h2>Unitree G1 — Obstacle Navigation Test</h2>
<p><small>MuJoCo + ONNX · Simulation only · port 8770</small></p>
<img src="/stream" alt="G1 simulation feed">
<p><button onclick="control('start')">Start</button><button onclick="control('replay')">Replay from start</button><button onclick="control('stop')">Stop</button></p>
<p id="message">Press Start to begin. Replay resets the simulation.</p>
<pre id="info">Loading...</pre>
<script>
async function control(action){
 try{const r=await fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action})});
 const d=await r.json();document.getElementById('message').textContent=(r.ok?'OK: ':'ERROR: ')+(d.message||d.error);
 }catch(e){document.getElementById('message').textContent=String(e)}
}
async function refresh(){try{let r=await fetch('/status');let d=await r.json();document.getElementById('info').textContent=JSON.stringify(d,null,2)}catch(e){}}
setInterval(refresh,500);refresh();
</script></body></html>'''

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        global frame
        if self.path == '/':
            self.send_bytes(200, 'text/html; charset=utf-8', HTML.encode())
        elif self.path == '/status':
            with state_lock:
                payload = json.dumps(telemetry).encode()
            self.send_bytes(200, 'application/json', payload)
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            try:
                while True:
                    with state_lock:
                        jpg = frame
                    if jpg:
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(jpg)).encode() + b'\r\n\r\n' + jpg + b'\r\n')
                        self.wfile.flush()
                    time.sleep(0.13)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
        else:
            self.send_error(404)

    def do_POST(self):
        global control_request
        if self.path != '/control':
            self.send_error(404)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 256:
                raise ValueError('Invalid request length')
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict) or set(payload) != {'action'}:
                raise ValueError('Expected {"action": ...}')
            action = payload['action']
            if action not in ('start', 'replay', 'stop'):
                raise ValueError('Unknown control action')
            with state_lock:
                control_request = action
            self.send_bytes(202, 'application/json', json.dumps({'message': action + ' requested'}).encode())
        except (ValueError, TypeError) as exc:
            self.send_bytes(400, 'application/json', json.dumps({'error': str(exc)}).encode())

    def send_bytes(self, code, content_type, body):
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass

server = ThreadingHTTPServer(('0.0.0.0', 8770), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
renderer = mujoco.Renderer(model, height=480, width=640)
camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_FREE
camera.distance = 5.3
camera.azimuth = 135
camera.elevation = -12


def update_policy():
    global target, last_action
    mujoco.mj_forward(model, data)
    body_vel = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, pelvis, body_vel, 1)
    rot = data.xmat[pelvis].reshape(3, 3)
    measurements = {
        'base_ang_vel': body_vel[:3],
        'projected_gravity': rot.T @ np.array([0., 0., -1.]),
        'velocity_commands': command,
        'joint_pos_rel': data.qpos[qids] - defaults,
        'joint_vel_rel': data.qvel[vids],
        'last_action': last_action,
    }
    for name, settings in terms_cfg.items():
        values = np.asarray(measurements[name], dtype=np.float32)
        clip = settings.get('clip')
        if clip is not None:
            values = np.clip(values, clip[0], clip[1])
        values = values * np.asarray(settings['scale'], dtype=np.float32)
        history[name].append(values)
        while len(history[name]) < history[name].maxlen:
            history[name].appendleft(history[name][0].copy())
    obs = np.concatenate([np.concatenate(list(history[name])) for name in terms_cfg]).astype(np.float32)[None, :]
    assert obs.shape == (1, 480)
    raw = ort_session.run(['actions'], {'obs': obs})[0][0]
    if not np.isfinite(raw).all():
        raise RuntimeError('Nonfinite policy action')
    last_action = raw.astype(float)
    target = last_action * scale + offset
    for i, motor in enumerate(ids):
        joint = model.actuator_trnid[motor, 0]
        if model.jnt_limited[joint]:
            target[i] = np.clip(target[i], *model.jnt_range[joint])


print('G1 ZIGZAG VIEWER READY on port 8770', flush=True)
wall_start = time.monotonic()
steps = 0
while True:
    t = float(data.time)
    with state_lock:
        requested = control_request
        control_request = None

    if requested == 'replay':
        # Reset simulation and *all* recurrent observation state, only in the physics thread.
        mujoco.mj_resetData(model, data)
        data.qpos[qids] = defaults
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(model, data)
        for h in history.values():
            h.clear()
        last_action = np.zeros_like(last_action)
        target = defaults.copy()
        command[:] = 0.0
        v_cmd = 0.0
        steps = 0
        mode = 'idle'
        status = 'initializing'
        play_start = None
        stop_start = None
        auto_start = True
        t = float(data.time)
        wall_start = time.monotonic() - t
        print('ZIGZAG: replay reset complete', flush=True)

    if requested == 'start' and mode in ('idle', 'finished'):
        if mode == 'finished':
            # Starting a finished run must reset policy history too.
            with state_lock:
                control_request = 'replay'
        else:
            auto_start = True

    if requested == 'stop':
        auto_start = False
        if mode == 'playing':
            mode = 'stopping'
            stop_start = t
            stop_initial = float(command[0])
            print('ZIGZAG: stopping requested', flush=True)

    if mode == 'collision':
        status = 'collision'
        command[:] = 0.0
    elif mode == 'fallen':
        status = 'fallen'
        command[:] = 0.0
    elif t < 4.0:
        status = 'initializing'
        command[:] = 0.0
    else:
        if auto_start and mode == 'idle':
            mode = 'playing'
            play_start = t
            auto_start = False
            print('ZIGZAG: started', flush=True)
        if mode == 'playing':
            elapsed = t - play_start
            if elapsed < 2.0:
                status = 'standing'
                command[:] = 0.0
            elif elapsed < 45.0:
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
                stop_initial = float(command[0])
        if mode == 'stopping':
            status = 'stopping'
            remaining = max(0.0, stop_initial - 0.15 * (t - stop_start))
            command[:] = [remaining, 0.0, 0.0]
            if remaining == 0.0:
                mode = 'finished'
        if mode == 'finished':
            status = 'standing'
            command[:] = 0.0
        elif mode == 'idle':
            status = 'standing'
            command[:] = 0.0
    v_cmd = float(command[0])

    if steps % policy_interval == 0:
        update_policy()
    torque = kp * (target - data.qpos[qids]) - kd * data.qvel[vids]
    data.ctrl[ids] = np.clip(torque, model.actuator_ctrlrange[ids, 0], model.actuator_ctrlrange[ids, 1])
    data.xfrc_applied[pelvis, :] = 0.0
    support = float(np.clip((4.0 - t) / 2.0, 0.0, 1.0))
    if support > 0:
        mujoco.mj_forward(model, data)
        force = np.array([0., 0., total_mass * 9.81]) + 250.0 * (np.array([0., 0., initial_z]) - data.qpos[:3]) - 60.0 * data.qvel[:3]
        rot = data.xmat[pelvis].reshape(3, 3)
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, pelvis, vel, 0)
        torque_support = 180.0 * np.cross(rot[:, 2], [0., 0., 1.]) - 35.0 * vel[:3]
        data.xfrc_applied[pelvis, :3] = support * force
        data.xfrc_applied[pelvis, 3:] = support * torque_support
    mujoco.mj_step(model, data)
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

    steps += 1
    rot = data.xmat[pelvis].reshape(3, 3)
    tilt = math.degrees(math.acos(float(np.clip(rot[2, 2], -1., 1.))))
    if t >= 4 and (tilt > 40.0 or data.qpos[2] < 0.5 or not np.isfinite(data.qpos).all()):
        status = 'fallen'
        mode = 'fallen'
        current_task = None
        v_cmd = 0.0
        command[:] = 0.0
    if steps % 50 == 0:
        with state_lock:
            telemetry = {'status': status, 'time_s': round(float(data.time), 2), 'x_m': round(float(data.qpos[0]), 3), 'y_m': round(float(data.qpos[1]), 3), 'pelvis_z_m': round(float(data.qpos[2]), 3), 'tilt_deg': round(tilt, 2), 'speed_command_mps': round(v_cmd, 3), 'replay_mode': mode, 'replay_elapsed_s': round(t - play_start, 2) if play_start is not None else None, 'turn_command_rad_s': round(float(command[2]), 3), 'heading_deg': round(math.degrees(math.atan2(rot[1, 0], rot[0, 0])), 2), 'support': round(support, 2), 'task': current_task, 'cruise_elapsed_s': round(cruise_elapsed, 2)}
    if steps % 65 == 0:
        camera.lookat[:] = [float(data.qpos[0]) + 0.8, float(data.qpos[1]), 0.75]
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
        output = io.BytesIO()
        Image.fromarray(pixels).save(output, format='JPEG', quality=76)
        with state_lock:
            frame = output.getvalue()
    delay = wall_start + data.time - time.monotonic()
    if delay > 0:
        time.sleep(delay)
