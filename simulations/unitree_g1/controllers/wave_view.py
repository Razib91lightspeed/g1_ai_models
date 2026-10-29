#!/usr/bin/env python3
"""Unitree G1 fixed-base right-hand waving demonstration.

Simulation only. No Unitree SDK2, robot network, or hardware commands.
"""

import io
import math
import time
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mujoco
import numpy as np
from PIL import Image

MODEL = "/workspace/preview/g1_fixed_base.xml"
PORT = 8767

frame_lock = threading.Lock()
latest_frame = None
ready = threading.Event()


def smoothstep(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def wave_targets(t):
    """Generate smooth joint targets in radians."""

    cycle = t % 11.0

    if cycle < 1.0:
        amount = 0.0
    elif cycle < 2.5:
        amount = smoothstep((cycle - 1.0) / 1.5)
    elif cycle < 7.5:
        amount = 1.0
    elif cycle < 9.0:
        amount = 1.0 - smoothstep((cycle - 7.5) / 1.5)
    else:
        amount = 0.0

    waving = 2.5 <= cycle < 7.5

    targets = {
        "right_shoulder_pitch": -1.50 * amount,
        "right_shoulder_roll": -1.20 * amount,
        "right_shoulder_yaw": 0.00,
        "right_elbow": 0.00,
        "right_wrist_pitch": 0.00,
    }

    if waving:
        phase = 2.0 * math.pi * 1.2 * (cycle - 2.5)
        targets["right_wrist_yaw"] = 0.45 * math.sin(phase)
        targets["right_elbow"] = 0.35 * math.sin(phase)

    return targets


def simulation_loop():
    global latest_frame

    model = mujoco.MjModel.from_xml_path(MODEL)
    data = mujoco.MjData(model)

    assert model.nu == 29, "Expected 29 G1 actuators"
    assert model.nq == 29, "Expected fixed-base G1 model"

    actuator_info = []

    for i in range(model.nu):
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, i
        )

        joint_id = int(model.actuator_trnid[i, 0])
        qadr = int(model.jnt_qposadr[joint_id])
        dadr = int(model.jnt_dofadr[joint_id])

        # Conservative initial simulation PD gains.
        if "shoulder" in name:
            kp, kd = 24.0, 2.5
        elif "elbow" in name:
            kp, kd = 20.0, 2.0
        elif "wrist" in name:
            kp, kd = 6.0, 0.7
        elif name.startswith("waist_"):
            # Simulation-only waist stabilization.
            kp, kd = 180.0, 12.0
        else:
            kp, kd = 35.0, 3.0

        actuator_info.append(
            (i, name, joint_id, qadr, dadr, kp, kd)
        )

    renderer = mujoco.Renderer(
        model, height=480, width=640
    )

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.8]
    camera.distance = 2.8
    camera.azimuth = 135
    camera.elevation = -12

    mujoco.mj_forward(model, data)

    print("G1 fixed-base model loaded", flush=True)
    print("29 torque actuators available", flush=True)
    print("Simulation-only right-arm wave enabled", flush=True)

    try:
        while True:
            # Advance approximately 50 ms of simulated physics.
            for _ in range(25):
                targets = wave_targets(data.time)

                for (
                    i, name, jid, qadr, dadr, kp, kd
                ) in actuator_info:

                    target = targets.get(name, 0.0)

                    # Respect joint position limits.
                    if model.jnt_limited[jid]:
                        low, high = model.jnt_range[jid]
                        target = float(
                            np.clip(target, low, high)
                        )

                    q = data.qpos[qadr]
                    dq = data.qvel[dadr]

                    torque = kp * (target - q) - kd * dq

                    # Respect actuator command limits.
                    if model.actuator_ctrllimited[i]:
                        low, high = model.actuator_ctrlrange[i]
                        torque = np.clip(torque, low, high)

                    data.ctrl[i] = torque

                mujoco.mj_step(model, data)


            # TORSO_DIAGNOSTICS
            # Print actual waist angles and torso tilt approximately
            # once per simulated second.
            if int(data.time * 20) % 20 == 0:
                torso_id = model.body("torso_link").id
                rotation = data.xmat[torso_id].reshape(3, 3)

                tilt = math.degrees(
                    math.acos(
                        float(np.clip(rotation[2, 2], -1, 1))
                    )
                )

                readings = {}
                for joint_name in (
                    "waist_pitch_joint",
                    "waist_roll_joint",
                    "waist_yaw_joint",
                ):
                    joint = model.joint(joint_name)
                    readings[joint_name] = round(
                        float(data.qpos[joint.qposadr[0]]),
                        3
                    )

                print(
                    f"Simulation {data.time:.2f}s | "
                    f"Torso tilt: {tilt:.2f} degrees | "
                    f"Waist: {readings}",
                    flush=True
                )

            renderer.update_scene(data, camera=camera)
            image = renderer.render()

            buffer = io.BytesIO()
            Image.fromarray(image).save(
                buffer, format="JPEG", quality=85
            )

            with frame_lock:
                latest_frame = buffer.getvalue()

            ready.set()
            time.sleep(0.04)

    finally:
        renderer.close()


class Viewer(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/":
            html = """
            <!DOCTYPE html>
            <html>
            <head>
              <title>Unitree G1 Waving Demo</title>
              <style>
                body {
                  background: #101827;
                  color: white;
                  font-family: system-ui, sans-serif;
                  text-align: center;
                  padding: 24px;
                }
                img {
                  width: min(95vw, 960px);
                  border-radius: 12px;
                }
                p { color: #a9b8ce; }
              </style>
            </head>
            <body>
              <h2>Unitree G1 — Waving Demonstration</h2>
              <p>Live MuJoCo simulation on NVIDIA DGX Spark</p>
              <img src="/stream">
              <p>Right arm: raise, wave, lower, repeat</p>
              <p>Fixed-base simulation | No physical robot control</p>
            </body>
            </html>
            """

            payload = html.encode()

            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header(
                "Content-Length", str(len(payload))
            )
            self.end_headers()
            self.wfile.write(payload)

        elif self.path == "/stream":
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame"
            )
            self.send_header(
                "Cache-Control", "no-cache"
            )
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

                    time.sleep(0.1)

            except (BrokenPipeError, ConnectionResetError):
                pass

        else:
            self.send_error(404)

    def log_message(self, *args):
        pass


def start_simulation():
    try:
        simulation_loop()
    except Exception:
        traceback.print_exc()
        ready.set()


if __name__ == "__main__":
    worker = threading.Thread(
        target=start_simulation, daemon=True
    )
    worker.start()

    if not ready.wait(timeout=120) or latest_frame is None:
        raise RuntimeError(
            "Failed to initialize G1 waving simulation"
        )

    print("G1 WAVE VIEWER READY on port 8767", flush=True)

    server = ThreadingHTTPServer(
        ("0.0.0.0", PORT), Viewer
    )
    server.serve_forever()
