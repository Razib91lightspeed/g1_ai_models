#!/usr/bin/env python3
"""Browser preview of the G1 obstacle course. Simulation only."""

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

SCENE = (
    "/workspace/unitree_mujoco/unitree_robots/g1/"
    "scene_nav_lidar_29dof.xml"
)

PORT = 8771

model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, height=480, width=640)

camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_FREE
camera.lookat[:] = [3.2, 0.0, 0.6]
camera.distance = 9.5
camera.azimuth = 145
camera.elevation = -35

renderer.update_scene(data, camera=camera)
rgb = renderer.render().copy()

output = io.BytesIO()
Image.fromarray(rgb).save(output, format="JPEG", quality=85)
scene_jpg = output.getvalue()

renderer.enable_depth_rendering()
renderer.update_scene(data, camera="g1_depth_camera")
depth = renderer.render().copy()

depth_view = np.clip(depth, 0, 8) / 8
depth_view = ((1 - depth_view) * 255).astype(np.uint8)

output = io.BytesIO()
Image.fromarray(depth_view).save(output, format="PNG")
depth_png = output.getvalue()

map_path = Path("/tmp/g1_nav_map/occupancy_map.png")

HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>G1 Autonomous Navigation</title>
<style>
body {
    background: #101820;
    color: #edf2f7;
    font-family: system-ui, sans-serif;
    margin: 25px auto;
    max-width: 1200px;
    padding: 0 20px;
}
h1 { margin-bottom: 6px; }
p { color: #aab9c6; }
.grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
}
.panel {
    background: #1c2833;
    padding: 16px;
    border-radius: 12px;
}
img {
    width: 100%;
    border-radius: 8px;
}
.full { grid-column: 1 / -1; }
@media(max-width: 700px) {
    .grid { grid-template-columns: 1fr; }
    .full { grid-column: auto; }
}
</style>
</head>
<body>
<h1>Unitree G1 — Autonomous Navigation Lab</h1>
<p>MuJoCo simulation | 10 obstacles | LiDAR + depth camera</p>

<div class="grid">
    <div class="panel full">
        <h3>MuJoCo obstacle course</h3>
        <img src="/scene.jpg">
    </div>

    <div class="panel">
        <h3>LiDAR occupancy map</h3>
        <img src="/map.png">
        <p>Orange: occupied | White: observed free | Gray: unknown</p>
    </div>

    <div class="panel">
        <h3>G1 depth camera</h3>
        <img src="/depth.png">
        <p>Brighter pixels represent closer surfaces.</p>
    </div>
</div>

<p>Preview mode: robot walking and autonomous planning are not active yet.</p>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            body = HTML.encode()
            content_type = "text/html"
        elif self.path == "/scene.jpg":
            body = scene_jpg
            content_type = "image/jpeg"
        elif self.path == "/depth.png":
            body = depth_png
            content_type = "image/png"
        elif self.path == "/map.png" and map_path.exists():
            body = map_path.read_bytes()
            content_type = "image/png"
        else:
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


print(f"G1 browser preview ready on port {PORT}", flush=True)
ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
