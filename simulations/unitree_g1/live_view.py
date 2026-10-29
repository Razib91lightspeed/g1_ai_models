#!/usr/bin/env python3
"""Headless Unitree G1 MuJoCo simulation and live MJPEG viewer."""

import io
import os
import time
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mujoco
from PIL import Image

PORT = 8766

MODEL_PATH = "/workspace/preview/g1_fixed_base.xml"

latest_frame = None
frame_lock = threading.Lock()
ready = threading.Event()


def simulation_loop():
    global latest_frame

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    print("G1 model loaded", flush=True)
    print("Bodies:", model.nbody, flush=True)
    print("Actuators:", model.nu, flush=True)

    renderer = mujoco.Renderer(
        model,
        height=480,
        width=640
    )

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.8]
    camera.distance = 3.2
    camera.azimuth = 135
    camera.elevation = -15

    def reset_robot():
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

    reset_robot()

    try:
        while True:
            # Physics simulation only.
            # No controller or physical robot connection.
            for _ in range(25):
                mujoco.mj_step(model, data)

            renderer.update_scene(data, camera=camera)

            frame = renderer.render()

            buffer = io.BytesIO()
            Image.fromarray(frame).save(
                buffer,
                format="JPEG",
                quality=80
            )

            with frame_lock:
                latest_frame = buffer.getvalue()

            ready.set()

            # Without a balancing policy, the G1 may fall.
            # Reset periodically so the scene remains visible.
            if data.time >= 3.0:
                reset_robot()

            time.sleep(0.05)

    finally:
        renderer.close()


def run_simulation():
    try:
        simulation_loop()
    except Exception:
        traceback.print_exc()
        os._exit(1)


class Viewer(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/":
            page = """
            <!DOCTYPE html>
            <html>
            <head>
              <title>Unitree G1 Simulation</title>
              <meta name="viewport"
                    content="width=device-width, initial-scale=1">
              <style>
                body {
                  margin: 0;
                  padding: 30px 15px;
                  background: #101827;
                  color: white;
                  font-family: system-ui, sans-serif;
                  text-align: center;
                }
                img {
                  width: min(95vw, 960px);
                  border-radius: 12px;
                  margin-top: 15px;
                }
                p { color: #a9b8ce; }
                .status { color: #86efac; }
              </style>
            </head>
            <body>
              <h2>Unitree G1 — MuJoCo Simulation</h2>
              <p class="status">Live simulation</p>
              <img src="/stream">
              <p>Robot: G1 29-actuator model</p>
              <p>Mode: Physics only, no balancing controller</p>
              <p>Running on NVIDIA DGX Spark</p>
            </body>
            </html>
            """

            data = page.encode()

            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        elif self.path == "/stream":
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame"
            )
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            try:
                while True:
                    with frame_lock:
                        frame = latest_frame

                    if frame:
                        self.wfile.write(
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            + f"Content-Length: {len(frame)}\r\n\r\n".encode()
                            + frame
                            + b"\r\n"
                        )
                        self.wfile.flush()

                    time.sleep(0.1)

            except (BrokenPipeError, ConnectionResetError):
                pass

        else:
            self.send_error(404)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    worker = threading.Thread(
        target=run_simulation,
        daemon=True
    )
    worker.start()

    if not ready.wait(timeout=120):
        raise RuntimeError("No G1 simulation frames produced")

    server = ThreadingHTTPServer(
        ("0.0.0.0", PORT),
        Viewer
    )

    print("G1 VIEWER READY on port 8766", flush=True)
    server.serve_forever()
