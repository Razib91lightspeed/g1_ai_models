#!/usr/bin/env python3
"""Simulation-only browser visualization of the Dex3 V4 test."""

import io
import json
import math
import runpy
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

PORT = 8784
SOURCE = Path(__file__).with_name(
    "dex3_closed_hands_return_v10.py"
)

lock = threading.Lock()
start = threading.Event()
frame = b""
status = {
    "phase": "ready",
    "time": 0.0,
    "height": 0.0,
    "tilt": 0.0,
    "message": "Press Start to run the Dex3 test"
}

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Unitree G1 Dex3</title>
<style>
body{background:#101722;color:white;font-family:Arial;
margin:0;padding:24px}
main{max-width:1000px;margin:auto}
h1{margin-bottom:8px}
p{color:#b9c7d9}
button{padding:13px 22px;border:0;border-radius:8px;
background:#3676e7;color:white;font-weight:bold;
font-size:16px;cursor:pointer}
button:disabled{opacity:.5;cursor:default}
img{display:block;width:100%;max-width:800px;
min-height:320px;background:#202b39;border-radius:12px;
margin-top:18px}
.stats{background:#202b39;padding:16px;border-radius:10px;
margin-top:12px}
</style>
</head>
<body>
<main>
<h1>Unitree G1 — Dex3 Hands</h1>
<p>29 body motors + 14 dexterous-hand motors</p>
<button id="start" onclick="begin()">Start 11.4 m Dex3 Mission</button>
<img id="scene" alt="G1 dexterous-hand simulation">
<div class="stats" id="stats">Ready</div>
<p>Simulation only. Physical G1 is not connected.</p>
</main>
<script>
async function begin(){
 const r=await fetch("/start",{method:"POST"});
 if(!r.ok) alert(await r.text());
 refresh();
}
async function refresh(){
 try{
  const r=await fetch("/status");
  const s=await r.json();
  document.getElementById("start").disabled=s.phase!=="ready";
  document.getElementById("stats").textContent=
    "Status: "+s.phase+
    " | Simulation: "+s.time.toFixed(1)+" s"+
    " | Height: "+s.height.toFixed(3)+" m"+
    " | Tilt: "+s.tilt.toFixed(2)+"°"+
    " | "+s.message;
  document.getElementById("scene").src="/scene.jpg?t="+Date.now();
 }catch(e){
  document.getElementById("stats").textContent="Connection error";
 }
}
setInterval(refresh,250);
refresh();
</script>
</body>
</html>"""

class Handler(BaseHTTPRequestHandler):
    def reply(self, body, kind, code=200):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        with lock:
            current_status = dict(status)
            current_frame = frame

        if path == "/":
            self.reply(PAGE.encode(), "text/html")
        elif path == "/status":
            self.reply(json.dumps(current_status).encode(),
                       "application/json")
        elif path == "/scene.jpg":
            if current_frame:
                self.reply(current_frame, "image/jpeg")
            else:
                self.send_error(503, "First frame not ready")
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/start":
            self.send_error(404)
            return

        with lock:
            if status["phase"] != "ready":
                self.reply(b"Trial already started",
                           "text/plain", 409)
                return
            status["phase"] = "running"

        start.set()
        self.reply(b"Started", "text/plain")

    def log_message(self, *args):
        pass

def worker():
    global frame
    start.wait()

    # Renderer creation, rendering and closure all occur
    # in this same worker thread.
    renderer = None
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 2.4
    camera.azimuth = 135
    camera.elevation = -15

    original_step = mujoco.mj_step
    clock = None
    next_frame = 0.0

    def observed_step(model, data):
        nonlocal renderer, clock, next_frame
        global frame

        original_step(model, data)

        t = float(data.time)
        if clock is None:
            clock = time.monotonic() - t
        delay = clock + t - time.monotonic()
        if delay > 0:
            time.sleep(delay)

        if t < next_frame:
            return
        next_frame = t + 0.12

        if renderer is None:
            renderer = mujoco.Renderer(
                model, width=640, height=480
            )

        camera.lookat[:] = [
            float(data.qpos[0]),
            float(data.qpos[1]),
            0.75
        ]

        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()

        buffer = io.BytesIO()
        Image.fromarray(pixels).save(
            buffer, format="JPEG", quality=85
        )

        pelvis = model.body("pelvis").id
        rotation = data.xmat[pelvis].reshape(3, 3)
        tilt = math.degrees(math.acos(
            float(np.clip(rotation[2,2], -1, 1))
        ))

        with lock:
            frame = buffer.getvalue()
            status.update(
                time=t,
                height=float(data.qpos[2]),
                tilt=tilt,
                message=(
                    "Stabilizing" if t < 4
                    else "Closing fingers" if t < 8
                    else "Holding finger targets"
                )
            )

    try:
        mujoco.mj_step = observed_step
        print("DEX3 BROWSER TRIAL START", flush=True)
        runpy.run_path(str(SOURCE), run_name="__main__")
        with lock:
            status.update(
                phase="finished",
                message="Trial ended; inspect terminal diagnostics"
            )
    except Exception as exc:
        with lock:
            status.update(
                phase="failed",
                message=str(exc)[:200]
            )
        print("VIEWER ERROR:", repr(exc), flush=True)
    finally:
        mujoco.mj_step = original_step
        if renderer is not None:
            renderer.close()

def main():
    thread = threading.Thread(
        target=worker, daemon=True
    )
    thread.start()

    server = ThreadingHTTPServer(
        ("0.0.0.0", PORT), Handler
    )

    print(f"DEX3 VIEWER READY: port {PORT}", flush=True)
    print("Press Start in the browser.", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == "__main__":
    main()
