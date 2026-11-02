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

MODEL = '/workspace/unitree_mujoco/unitree_robots/g1/scene_29dof.xml'
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
telemetry = {'status': 'initializing', 'time_s': 0, 'x_m': 0, 'tilt_deg': 0, 'speed_command_mps': 0}

HTML = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>G1 Simulation Skills</title><style>body{background:#101419;color:#eef2f5;font:16px system-ui;margin:24px auto;max-width:950px;padding:0 16px}
img{width:100%;border-radius:10px;background:#222}button,input{font:inherit;padding:10px;margin:5px}button{cursor:pointer}input{width:90px}
#info{white-space:pre-wrap;background:#202730;border-radius:8px;padding:12px}small{color:#aeb8c5}</style></head><body>
<h2>Unitree G1 — Interactive Simulation</h2><small>DGX Spark · MuJoCo · Unitree RL Lab ONNX · simulation only</small>
<img src="/stream" alt="Live G1 simulation"><p><button onclick="send({skill:'stand'})">Stand</button>
<button onclick="send({skill:'walk_forward',speed_mps:Number(document.getElementById('speed').value),duration_s:Number(document.getElementById('duration').value),stop_after:true})">Walk Forward</button>
<button onclick="send({skill:'stop'})">Stop</button></p>
<label>Speed (m/s) <input id="speed" type="number" value="0.3" min="0.01" max="0.3" step="0.05"></label>
<label>Duration (s) <input id="duration" type="number" value="5" min="0.1" max="10" step="0.5"></label>
<p id="message"></p><pre id="info">Initializing simulator...</pre>
<script>async function send(payload){try{let r=await fetch('/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});let j=await r.json();document.getElementById('message').textContent=(r.ok?'Accepted: ':'Rejected: ')+(j.message||j.error||'');}catch(e){document.getElementById('message').textContent=String(e)}}
async function refresh(){try{let r=await fetch('/status');let d=await r.json();document.getElementById('info').textContent=JSON.stringify(d,null,2)}catch(e){}}setInterval(refresh,500);refresh()</script>
</body></html>'''

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
        global pending
        if self.path != '/command':
            self.send_error(404)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 2048:
                raise ValueError('Invalid request length')
            req = json.loads(self.rfile.read(length))
            validated = validate_command(req)
            with state_lock:
                if status == 'initializing':
                    raise ValueError('G1 is initializing; retry after 4 seconds')
                if status == 'fallen':
                    raise ValueError('Simulation fell; restart viewer to reset')
                pending = validated
            self.send_bytes(202, 'application/json', json.dumps({'message': 'Command queued', 'task': validated}).encode())
        except (ValueError, TypeError, UnicodeDecodeError) as exc:
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

server = ThreadingHTTPServer(('0.0.0.0', 8769), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
renderer = mujoco.Renderer(model, height=480, width=640)
camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_FREE
camera.distance = 2.8
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


print('G1 INTERACTIVE VIEWER READY on port 8769', flush=True)
wall_start = time.monotonic()
steps = 0
while True:
    t = float(data.time)
    with state_lock:
        if pending is not None and status != 'fallen' and t >= 4.0:
            chosen = pending
            pending = None
            if chosen['skill'] == 'walk_forward':
                current_task = chosen
                cruise_elapsed = 0.0
                status = 'walking'
            else:
                current_task = None
                status = 'stopping' if abs(v_cmd) > 0.001 else 'standing'

    # A velocity-command stop is not an emergency stop.
    desired = 0.0
    if current_task is not None and status == 'walking':
        desired = current_task['speed_mps']
        if abs(v_cmd - desired) < 0.005:
            cruise_elapsed += physics_dt
        if cruise_elapsed >= current_task['duration_s']:
            current_task = None
            status = 'stopping'
            desired = 0.0

    # Same two-second acceleration/deceleration for 0.3 m/s as tested.
    dv = 0.15 * physics_dt
    v_cmd += float(np.clip(desired - v_cmd, -dv, dv))
    if abs(v_cmd) < 1e-6:
        v_cmd = 0.0
    if status == 'stopping' and v_cmd == 0.0:
        status = 'standing'
    if t < 4.0:
        status = 'initializing'
        v_cmd = 0.0
    elif status == 'initializing':
        # Initialization finished and support has been released.
        status = 'standing'

    command[:] = [v_cmd, 0.0, 0.0]

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
    steps += 1
    rot = data.xmat[pelvis].reshape(3, 3)
    tilt = math.degrees(math.acos(float(np.clip(rot[2, 2], -1., 1.))))
    if t >= 4 and (tilt > 40.0 or data.qpos[2] < 0.5 or not np.isfinite(data.qpos).all()):
        status = 'fallen'
        current_task = None
        v_cmd = 0.0
        command[:] = 0.0
    if steps % 50 == 0:
        with state_lock:
            telemetry = {'status': status, 'time_s': round(float(data.time), 2), 'x_m': round(float(data.qpos[0]), 3), 'y_m': round(float(data.qpos[1]), 3), 'pelvis_z_m': round(float(data.qpos[2]), 3), 'tilt_deg': round(tilt, 2), 'speed_command_mps': round(v_cmd, 3), 'support': round(support, 2), 'task': current_task, 'cruise_elapsed_s': round(cruise_elapsed, 2)}
    if steps % 65 == 0:
        camera.lookat[:] = [float(data.qpos[0]), float(data.qpos[1]), 0.8]
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
        output = io.BytesIO()
        Image.fromarray(pixels).save(output, format='JPEG', quality=76)
        with state_lock:
            frame = output.getvalue()
    delay = wall_start + data.time - time.monotonic()
    if delay > 0:
        time.sleep(delay)
