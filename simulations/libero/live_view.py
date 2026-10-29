#!/usr/bin/env python3
"""Read-only live LIBERO simulation viewer."""

import io
import os
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

PORT = 8765
latest_frame = None
frame_lock = threading.Lock()
ready = threading.Event()

def simulation_loop():
    global latest_frame

    suite = benchmark.get_benchmark("libero_spatial")()
    task = suite.get_task(0)

    bddl = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file
    )

    print("Task:", task.language, flush=True)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=256,
        camera_widths=256
    )

    try:
        env.seed(0)
        obs = env.reset()
        steps = 0

        while True:
            # Zero-action simulation only.
            # No VLA model or real robot is connected.
            obs, reward, done, info = env.step(
                np.zeros(7, dtype=np.float32)
            )

            frame = np.flipud(
                np.asarray(obs["agentview_image"])
            ).copy()

            buffer = io.BytesIO()
            Image.fromarray(frame).save(
                buffer, format="JPEG", quality=80
            )

            with frame_lock:
                latest_frame = buffer.getvalue()

            ready.set()
            steps += 1

            if done or steps >= 500:
                obs = env.reset()
                steps = 0

            time.sleep(0.15)

    finally:
        env.close()

class Viewer(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            page = """
            <!DOCTYPE html>
            <html>
            <head>
              <meta name="viewport"
                    content="width=device-width, initial-scale=1">
              <title>LIBERO Live Simulation</title>
              <style>
                body {
                  background: #111827;
                  color: #f3f4f6;
                  font-family: system-ui, sans-serif;
                  text-align: center;
                  margin: 0;
                  padding: 35px 15px;
                }
                img {
                  width: min(90vw, 720px);
                  image-rendering: auto;
                  border-radius: 12px;
                  margin-top: 18px;
                }
                p { color: #aeb7c6; }
                .status { color: #86efac; }
              </style>
            </head>
            <body>
              <h2>LIBERO — Live Simulation</h2>
              <p class="status">Live camera stream</p>
              <img src="/stream" alt="LIBERO robot scene">
              <p>Task: Pick up the black bowl and place it on the plate</p>
              <p>Mode: Zero-action physics test</p>
              <p>NVIDIA DGX Spark | Unitree VLA evaluation workspace</p>
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
                            + frame + b"\r\n"
                        )
                        self.wfile.flush()

                    time.sleep(0.2)

            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass

if __name__ == "__main__":
    thread = threading.Thread(
        target=simulation_loop, daemon=True
    )
    thread.start()

    if not ready.wait(timeout=120):
        raise RuntimeError(
            "LIBERO did not produce its first camera frame"
        )

    server = ThreadingHTTPServer(
        ("0.0.0.0", PORT), Viewer
    )

    print(f"VIEWER READY on port {PORT}", flush=True)
    server.serve_forever()
