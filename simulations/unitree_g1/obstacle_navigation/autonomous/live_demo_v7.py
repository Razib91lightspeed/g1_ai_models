
#!/usr/bin/env python3
"""G1 autonomous round-trip experiment. MuJoCo only."""

import io
import json
import math
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mujoco
import numpy as np
import onnxruntime as ort
import yaml
from PIL import Image, ImageDraw

SCENE = (
    "/workspace/unitree_mujoco/unitree_robots/g1/"
    "scene_nav_many_29dof.xml"
)
CONFIG = "/workspace/policy/params/deploy.yaml"
POLICY = "/workspace/policy/exported/policy.onnx"

PORT = 8771
GOAL = np.array([3.048, 0.0])
HOME = np.array([0.0, 0.0])
MAX_SPEED = 0.40
MAX_TURN = 0.30
STOP_DISTANCE = 0.40
SCAN_RANGE = 5.0
MAP_RES = 0.10

model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)

with open(CONFIG, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

session = ort.InferenceSession(
    POLICY, providers=["CPUExecutionProvider"]
)

ids = np.asarray(cfg["joint_ids_map"], dtype=int)
assert sorted(ids.tolist()) == list(range(29))
assert model.nu == 29

qids = np.asarray([
    model.jnt_qposadr[model.actuator_trnid[i, 0]]
    for i in ids
])
vids = np.asarray([
    model.jnt_dofadr[model.actuator_trnid[i, 0]]
    for i in ids
])

defaults = np.asarray(cfg["default_joint_pos"], dtype=float)
kp = np.asarray(cfg["stiffness"], dtype=float)[ids]
kd = np.asarray(cfg["damping"], dtype=float)[ids]

action_cfg = cfg["actions"]["JointPositionAction"]
scale = np.asarray(action_cfg["scale"], dtype=float)
offset = np.asarray(action_cfg["offset"], dtype=float)

terms = cfg["observations"]
history = {
    name: deque(maxlen=int(term["history_length"]))
    for name, term in terms.items()
}

pelvis = model.body("pelvis").id
torso = model.body("torso_link").id
cam_id = model.camera("g1_depth_camera").id

dt = float(model.opt.timestep)
interval = int(round(float(cfg["step_dt"]) / dt))
assert interval > 0
assert abs(interval * dt - float(cfg["step_dt"])) < 1e-8

total_mass = float(np.sum(model.body_mass))
command = np.zeros(3)
last_action = np.zeros(29)
target = defaults.copy()

# Map: -1 unknown, 0 free, 100 occupied
XMIN, XMAX = -1.5, 4.5
YMIN, YMAX = -2.5, 2.5
MW = int((XMAX - XMIN) / MAP_RES)
MH = int((YMAX - YMIN) / MAP_RES)
grid = np.full((MH, MW), -1, dtype=np.int16)

geomgroup = np.zeros(6, dtype=np.uint8)
geomgroup[3] = 1

obstacle_ids = set()
for gid in range(model.ngeom):
    name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_GEOM, gid
    )
    if name and name.startswith("nav_obstacle_"):
        obstacle_ids.add(gid)

lock = threading.Lock()
requested = None
state = "idle"
phase = "idle"
collision = False
steps = 0
initial_z = 0.0
start_yaw = 0.0
start_pose = np.zeros(3)
run_start = None
last_scan = []
nearest = None

scene_jpg = None
depth_jpg = None
map_jpg = None
info = {}

viewer = None
depth_renderer = None


def clamp(value, low, high):
    return max(low, min(high, value))


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def make_jpg(image, quality=75):
    buffer = io.BytesIO()
    Image.fromarray(image).save(
        buffer, format="JPEG", quality=quality
    )
    return buffer.getvalue()


def position():
    return np.asarray(data.qpos[:2], dtype=float).copy()


def yaw_angle():
    rotation = data.xmat[pelvis].reshape(3, 3)
    return math.atan2(rotation[1, 0], rotation[0, 0])


def cell(x, y):
    ix = int(math.floor((x - XMIN) / MAP_RES))
    iy = int(math.floor((y - YMIN) / MAP_RES))
    if 0 <= ix < MW and 0 <= iy < MH:
        return ix, iy
    return None


def reset_sim():
    global state, phase, collision, steps
    global last_action, target, initial_z
    global start_yaw, start_pose, run_start, last_scan
    global nearest

    mujoco.mj_resetData(model, data)
    data.qpos[qids] = defaults
    data.qvel[:] = 0
    data.ctrl[:] = 0
    data.xfrc_applied[:] = 0
    mujoco.mj_forward(model, data)

    for h in history.values():
        h.clear()

    command[:] = 0
    last_action = np.zeros(29)
    target = defaults.copy()

    grid[:] = -1
    collision = False
    steps = 0
    initial_z = float(data.qpos[2])
    start_yaw = yaw_angle()
    start_pose = np.asarray(data.qpos[:3]).copy()
    run_start = None
    last_scan = []
    nearest = None
    state = "idle"
    phase = "idle"


def update_policy():
    global last_action, target

    mujoco.mj_forward(model, data)

    body_vel = np.zeros(6)
    mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_BODY,
        pelvis, body_vel, 1
    )

    rotation = data.xmat[pelvis].reshape(3, 3)

    measurements = {
        "base_ang_vel": body_vel[:3],
        "projected_gravity":
            rotation.T @ np.array([0., 0., -1.]),
        "velocity_commands": command,
        "joint_pos_rel": data.qpos[qids] - defaults,
        "joint_vel_rel": data.qvel[vids],
        "last_action": last_action
    }

    for name, settings in terms.items():
        values = np.asarray(
            measurements[name], dtype=np.float32
        )

        clip = settings.get("clip")

        if clip is not None:
            values = np.clip(
                values,
                clip[0],
                clip[1]
            )

        values *= np.asarray(
            settings["scale"], dtype=np.float32
        )

        history[name].append(values)

        while len(history[name]) < history[name].maxlen:
            history[name].appendleft(
                history[name][0].copy()
            )

    obs = np.concatenate([
        np.concatenate(list(history[name]))
        for name in terms
    ]).astype(np.float32)[None, :]

    assert obs.shape == (1, 480)

    action = session.run(
        ["actions"], {"obs": obs}
    )[0][0]

    if not np.isfinite(action).all():
        raise RuntimeError("Nonfinite policy output")

    last_action = action.astype(float)
    target = last_action * scale + offset

    for i, motor in enumerate(ids):
        joint = model.actuator_trnid[motor, 0]
        if model.jnt_limited[joint]:
            target[i] = np.clip(
                target[i], *model.jnt_range[joint]
            )


def scan_lidar():
    """Obstacle-only LiDAR diagnostic in robot coordinates."""
    global last_scan, nearest

    rotation = data.xmat[torso].reshape(3, 3)
    origin = (
        data.xpos[torso]
        + rotation @ np.array([0., 0., 0.15])
    )

    results = []
    minimum = float("inf")

    # 72 horizontal directions; 5-degree spacing.
    # Two vertical angles improve obstacle coverage.
    for elevation in (-8.0, 0.0):
        e = math.radians(elevation)

        for angle in range(0, 360, 5):
            a = math.radians(angle)

            local = np.array([
                math.cos(e) * math.cos(a),
                math.cos(e) * math.sin(a),
                math.sin(e)
            ])

            direction = rotation @ local
            hit_id = np.array([-1], dtype=np.int32)

            distance = float(mujoco.mj_ray(
                model, data, origin, direction,
                geomgroup, 1, -1, hit_id
            ))

            hit = (
                hit_id[0] in obstacle_ids
                and 0.2 <= distance <= SCAN_RANGE
            )

            endpoint = origin + (
                distance if hit else SCAN_RANGE
            ) * direction

            # Horizontal samples update the 2D map.
            if elevation == 0.0:
                count = max(
                    1,
                    int(np.linalg.norm(
                        endpoint[:2] - origin[:2]
                    ) / (MAP_RES * 0.5))
                )

                cells = []
                for j in range(count + 1):
                    point = origin + (
                        j / count
                    ) * (endpoint - origin)

                    c = cell(point[0], point[1])
                    if c and (not cells or c != cells[-1]):
                        cells.append(c)

                for c in (cells[:-1] if hit else cells):
                    if grid[c[1], c[0]] != 100:
                        grid[c[1], c[0]] = 0

                if hit and cells:
                    c = cells[-1]
                    grid[c[1], c[0]] = 100

            if hit:
                results.append((
                    angle, distance, endpoint.copy()
                ))
                minimum = min(minimum, distance)

    last_scan = results
    nearest = minimum if np.isfinite(minimum) else None


def navigation_command():
    """Reactive heading selection from measured LiDAR returns."""
    global state, phase

    pos = position()
    heading = yaw_angle()

    destination = GOAL if phase == "outbound" else HOME
    delta = destination - pos
    distance_to_goal = float(np.linalg.norm(delta))

    if distance_to_goal < 0.22:
        command[:] = 0

        if phase == "outbound":
            phase = "returning"
            state = "returning"
            return

        # Finish only when original heading is recovered.
        error = wrap(start_yaw - heading)
        if abs(error) < math.radians(8):
            phase = "finished"
            state = "finished"
        else:
            command[2] = clamp(
                1.0 * error, -MAX_TURN, MAX_TURN
            )
        return

    goal_heading = math.atan2(delta[1], delta[0])
    desired = wrap(goal_heading - heading)

    # Find a collision-aware local steering direction.
    # No obstacle geometry coordinates are used here.
    candidate_angles = np.radians(
        np.arange(-85, 86, 10)
    )

    best = None
    best_cost = float("inf")

    for candidate in candidate_angles:
        heading_cost = abs(wrap(candidate - desired))
        obstacle_cost = 0.0
        blocked = False

        for angle_deg, distance, _ in last_scan:
            ray_angle = math.radians(angle_deg)
            ray_angle = wrap(ray_angle)

            difference = abs(wrap(
                ray_angle - candidate
            ))

            if difference < math.radians(24):
                if distance < 0.55:
                    blocked = True
                    break

                if distance < 1.10:
                    obstacle_cost += (
                        1.10 - distance
                    ) * (1.0 - difference / math.radians(24))

        if blocked:
            continue

        cost = heading_cost + obstacle_cost * 2.0

        if cost < best_cost:
            best_cost = cost
            best = candidate

    if best is None:
        command[:] = 0
        state = "blocked"
        return

    turn = clamp(
        0.9 * best, -MAX_TURN, MAX_TURN
    )

    forward = MAX_SPEED * clamp(
        1.0 - abs(best) / math.radians(90),
        0.15, 1.0
    )

    if nearest is not None and nearest < 0.65:
        forward *= clamp(
            (nearest - STOP_DISTANCE) / 0.25,
            0.0, 1.0
        )

    command[:] = [forward, 0, turn]
    state = phase


def update_map_image():
    image = np.zeros((MH, MW, 3), dtype=np.uint8)
    image[:] = (95, 101, 112)
    image[grid == 0] = (235, 240, 245)
    image[grid == 100] = (245, 125, 45)

    image = Image.fromarray(np.flipud(image))
    image = image.resize(
        (MW * 7, MH * 7),
        Image.Resampling.NEAREST
    )

    draw = ImageDraw.Draw(image)

    def xy(point):
        return (
            int((point[0] - XMIN) / MAP_RES * 7),
            int((YMAX - point[1]) / MAP_RES * 7)
        )

    for point, color in (
        (HOME, "#2585ff"),
        (GOAL, "#18d76b"),
        (position(), "#101820")
    ):
        x, y = xy(point)
        draw.ellipse(
            (x - 7, y - 7, x + 7, y + 7),
            fill=color, outline="white"
        )

    return make_jpg(np.asarray(image))


def render_views():
    global scene_jpg, depth_jpg, map_jpg, info

    viewer_camera = mujoco.MjvCamera()
    viewer_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer_camera.lookat[:] = [1.5, 0, 0.45]
    viewer_camera.distance = 5.5
    viewer_camera.azimuth = 110
    viewer_camera.elevation = -48

    # Display all geometry groups, including obstacles in group 3.
    scene_options = mujoco.MjvOption()
    scene_options.geomgroup[:] = 1

    viewer.update_scene(
        data,
        camera=viewer_camera,
        scene_option=scene_options
    )
    scene = viewer.render().copy()

    depth_renderer.update_scene(
        data, camera="g1_depth_camera"
    )
    depth = depth_renderer.render().copy()

    # Visualization: near surfaces bright, far surfaces dark.
    depth_gray = (
        (1.0 - np.clip(depth, 0, 6) / 6.0) * 255
    ).astype(np.uint8)

    depth_rgb = np.repeat(
        depth_gray[:, :, None], 3, axis=2
    )

    rotation = data.xmat[pelvis].reshape(3, 3)
    tilt = math.degrees(math.acos(
        clamp(float(rotation[2, 2]), -1, 1)
    ))

    telemetry = {
        "state": state,
        "phase": phase,
        "time_s": round(float(data.time), 2),
        "x_m": round(float(data.qpos[0]), 3),
        "y_m": round(float(data.qpos[1]), 3),
        "heading_deg": round(
            math.degrees(yaw_angle()), 2
        ),
        "tilt_deg": round(tilt, 2),
        "pelvis_z_m": round(float(data.qpos[2]), 3),
        "nearest_lidar_m": (
            round(nearest, 3) if nearest is not None else None
        ),
        "lidar_returns": len(last_scan),
        "collision": collision,
        "velocity_command": command.tolist(),
        "goal_distance_m": round(float(np.linalg.norm(
            (GOAL if phase == "outbound" else HOME)
            - position()
        )), 3)
    }

    with lock:
        scene_jpg = make_jpg(scene)
        depth_jpg = make_jpg(depth_rgb)
        map_jpg = update_map_image()
        info = telemetry


HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>G1 Autonomous Navigation</title>
<style>
body{background:#101820;color:#eaf0f6;font:16px system-ui;
max-width:1200px;margin:20px auto;padding:0 18px}
.panel{background:#1d2b38;padding:16px;border-radius:12px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.full{grid-column:1/-1}
img{width:100%;border-radius:8px;background:#080d12}
button{border:0;border-radius:8px;padding:12px 22px;
font:inherit;font-weight:600;margin:5px;cursor:pointer}
.start{background:#27c989}.stop{background:#f0b24d}
.restart{background:#7caaff}
pre{white-space:pre-wrap;font-size:14px}
small{color:#b4c3d0}
@media(max-width:700px){.grid{grid-template-columns:1fr}
.full{grid-column:auto}}
</style>
</head>
<body>
<h2>Unitree G1 — Autonomous Zigzag Round Trip</h2>
<p>10 feet outbound + return home | Simulation only</p>

<div>
<button class="start" onclick="send('start')">Start Demo</button>
<button class="stop" onclick="send('stop')">Stop</button>
<button class="restart" onclick="send('restart')">Restart</button>
</div>

<p id="message">Ready.</p>

<div class="grid">
<div class="panel full">
<h3>Live MuJoCo simulation</h3>
<img id="scene" src="/scene.jpg">
</div>
<div class="panel">
<h3>Live LiDAR occupancy map</h3>
<img id="map" src="/map.jpg">
<small>Orange: occupied | White: free | Gray: unknown</small>
</div>
<div class="panel">
<h3>G1 depth camera</h3>
<img id="depth" src="/depth.jpg">
</div>
<div class="panel full">
<h3>Robot telemetry</h3>
<pre id="status">Connecting...</pre>
</div>
</div>

<script>
async function send(action){
 try{
  const r=await fetch('/control',{
   method:'POST',
   headers:{'Content-Type':'application/json'},
   body:JSON.stringify({action})
  });
  const d=await r.json();
  document.getElementById('message').textContent =
   d.message || d.error || action;
 }catch(e){document.getElementById('message').textContent=String(e);}
}
async function refresh(){
 try{
  const r=await fetch('/status');
  const d=await r.json();
  document.getElementById('status').textContent =
   JSON.stringify(d,null,2);
  const token=Date.now();
  for(const id of ['scene','map','depth']){
   document.getElementById(id).src='/'+id+'.jpg?t='+token;
  }
 }catch(e){}
}
setInterval(refresh,600);
refresh();
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def send_body(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            return self.send_body(
                200, HTML.encode(), "text/html"
            )

        name = self.path.split("?")[0]
        with lock:
            payloads = {
                "/scene.jpg": scene_jpg,
                "/depth.jpg": depth_jpg,
                "/map.jpg": map_jpg
            }
            content = payloads.get(name)
            status_payload = dict(info)

        if name == "/status":
            return self.send_body(
                200, json.dumps(status_payload).encode(),
                "application/json"
            )

        if content is not None:
            return self.send_body(
                200, content, "image/jpeg"
            )

        self.send_error(404)

    def do_POST(self):
        global requested

        if self.path != "/control":
            return self.send_error(404)

        try:
            length = int(
                self.headers.get("Content-Length", "0")
            )
            if not 0 < length < 256:
                raise ValueError("Invalid request size")

            body = json.loads(
                self.rfile.read(length)
            )
            action = body.get("action")
            if action not in ("start", "stop", "restart"):
                raise ValueError("Invalid action")

            with lock:
                requested = action

            self.send_body(
                202,
                json.dumps({
                    "message": action + " requested"
                }).encode(),
                "application/json"
            )
        except (ValueError, TypeError) as e:
            self.send_body(
                400,
                json.dumps({"error": str(e)}).encode(),
                "application/json"
            )

    def log_message(self, *args):
        pass


def main():
    global requested, state, phase, run_start
    global collision, steps, viewer, depth_renderer

    reset_sim()

    viewer = mujoco.Renderer(
        model, height=480, width=640
    )

    depth_renderer = mujoco.Renderer(
        model, height=240, width=320
    )
    depth_renderer.enable_depth_rendering()

    render_views()

    server = ThreadingHTTPServer(
        ("0.0.0.0", PORT), Handler
    )
    threading.Thread(
        target=server.serve_forever, daemon=True
    ).start()

    print(
        f"G1 autonomous demo ready on port {PORT}",
        flush=True
    )

    wall_start = time.monotonic()
    last_scan_step = -999

    try:
        while True:
            with lock:
                action = requested
                requested = None

            if action == "restart":
                reset_sim()
                wall_start = time.monotonic()
                last_scan_step = -999

            if action == "stop":
                state = "stopped"
                command[:] = 0

            if action == "start":
                if state not in ("outbound", "returning"):
                    reset_sim()
                    wall_start = time.monotonic()
                    last_scan_step = -999
                    state = "outbound"
                    phase = "outbound"
                    run_start = float(data.time)
                    print("NAV: START received", flush=True)

            # Pause physics until Start is pressed.
            # Also freeze terminal states for inspection.
            if state in (
                "idle", "stopped", "finished",
                "collision", "fallen", "timeout", "blocked"
            ):
                if steps % 150 == 0:
                    render_views()
                time.sleep(0.05)
                continue

            # Initial stabilization, then autonomous navigation.
            if state in ("outbound", "returning"):
                if data.time < 4.0:
                    command[:] = 0
                elif run_start is not None and (
                    data.time - run_start > 120
                ):
                    state = "timeout"
                    command[:] = 0
                else:
                    if steps - last_scan_step >= 100:
                        scan_lidar()
                        last_scan_step = steps
                    navigation_command()
            else:
                command[:] = 0

            if steps % interval == 0:
                update_policy()

            torque = (
                kp * (target - data.qpos[qids])
                - kd * data.qvel[vids]
            )

            data.ctrl[ids] = np.clip(
                torque,
                model.actuator_ctrlrange[ids, 0],
                model.actuator_ctrlrange[ids, 1]
            )

            # Startup stabilizer retained from zigzag viewer.
            data.xfrc_applied[pelvis, :] = 0
            support = clamp(
                (4.0 - float(data.time)) / 2.0,
                0.0, 1.0
            )

            if support > 0:
                mujoco.mj_forward(model, data)

                force = (
                    np.array([0., 0., total_mass * 9.81])
                    + 250.0 * (
                        np.array([0., 0., initial_z])
                        - data.qpos[:3]
                    )
                    - 60.0 * data.qvel[:3]
                )

                rotation = data.xmat[pelvis].reshape(3, 3)
                vel = np.zeros(6)
                mujoco.mj_objectVelocity(
                    model, data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    pelvis, vel, 0
                )
                angular_support = (
                    180.0 * np.cross(
                        rotation[:, 2], [0., 0., 1.]
                    ) - 35.0 * vel[:3]
                )
                data.xfrc_applied[pelvis, :3] = support * force
                data.xfrc_applied[pelvis, 3:] = (
                    support * angular_support
                )

            mujoco.mj_step(model, data)
            steps += 1

            # Monitor contact with every course obstacle.
            for ci in range(data.ncon):
                contact = data.contact[ci]
                if (
                    contact.geom1 in obstacle_ids
                    or contact.geom2 in obstacle_ids
                ):
                    collision = True
                    state = "collision"
                    command[:] = 0
                    break

            rotation = data.xmat[pelvis].reshape(3, 3)
            tilt = math.degrees(math.acos(
                clamp(float(rotation[2, 2]), -1, 1)
            ))

            if (
                tilt > 40
                or data.qpos[2] < 0.50
                or not np.isfinite(data.qpos).all()
            ):
                if data.time >= 4:
                    state = "fallen"
                    command[:] = 0

            if steps % 150 == 0:
                render_views()

            # Approximate real-time playback.
            delay = (
                wall_start + float(data.time)
                - time.monotonic()
            )
            if 0 < delay < 0.1:
                time.sleep(delay)

    finally:
        server.shutdown()
        viewer.close()
        depth_renderer.close()


if __name__ == "__main__":
    main()
