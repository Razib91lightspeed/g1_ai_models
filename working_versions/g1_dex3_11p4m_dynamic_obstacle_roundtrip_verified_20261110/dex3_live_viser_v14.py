"""Live Viser visualization of verified MuJoCo Dex3 mission.

The original V10 controller is executed without source edits.
No random obstacles are enabled during this baseline test.
"""

import os
import time
import runpy
import threading
import traceback

import numpy as np
import mujoco
import trimesh
import viser

MISSION = "/workspace/obstacle_navigation/experiments/dex3_random_navigation_v1/dex3_astar_mission_v24.py"

PORT = 8080
FPS = 15.0

server = viser.ViserServer(
    host="0.0.0.0",
    port=PORT
)

server.scene.set_up_direction("+z")
server.scene.add_grid(
    "/ground",
    width=60.0,
    height=40.0,
    position=(5.7, 0.0, -0.015),
)

server.initial_camera.position = (2.5, -7.0, 3.5)
server.initial_camera.look_at = (0.0, 0.0, 0.9)

import dex3_runtime_control_v1 as runtime_control

spawn_button = server.gui.add_button(
    "Spawn Obstacle Ahead"
)

@spawn_button.on_click
def request_spawn(_event):
    runtime_control.spawn_request.set()
    print("MANUAL SPAWN REQUESTED", flush=True)

start_button = server.gui.add_button(
    "Restart Dex3 11.4 m Mission"
)
status = server.gui.add_text(
    "Mission status",
    initial_value="Ready — click Start"
)
progress = server.gui.add_text(
    "Simulation",
    initial_value="0.0 s"
)

start_event = threading.Event()
running = threading.Event()

handles = []
last_model = None
last_data = None
last_render_time = -1.0
wall_start = None
original_step = mujoco.mj_step


@start_button.on_click
def _(_event):
    if running.is_set():
        return
    start_event.set()


def make_mesh(model, geom_id):
    kind = int(model.geom_type[geom_id])
    size = np.asarray(
        model.geom_size[geom_id], dtype=float
    )

    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            return None

        va = int(model.mesh_vertadr[mesh_id])
        vn = int(model.mesh_vertnum[mesh_id])
        fa = int(model.mesh_faceadr[mesh_id])
        fn = int(model.mesh_facenum[mesh_id])

        vertices = np.asarray(
            model.mesh_vert[va:va + vn],
            dtype=np.float32
        ).copy()

        faces = np.asarray(
            model.mesh_face[fa:fa + fn],
            dtype=np.int32
        ).copy()

        return trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False
        )

    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return trimesh.creation.box(extents=2 * size)

    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(
            subdivisions=2,
            radius=float(size[0])
        )

    if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        mesh = trimesh.creation.icosphere(subdivisions=2)
        mesh.apply_scale(size)
        return mesh

    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(
            radius=float(size[0]),
            height=float(2 * size[1]),
            sections=20
        )

    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return trimesh.creation.capsule(
            radius=float(size[0]),
            height=float(2 * size[1]),
            count=[8, 16]
        )

    return None


def clear_scene():
    global handles
    for geom_id, handle in handles:
        handle.remove()
    handles = []


def initialize_scene(model, data):
    clear_scene()

    rendered = 0

    for geom_id in range(model.ngeom):
        # The infinite ground plane is drawn as a Viser grid.
        if int(model.geom_type[geom_id]) == \
                mujoco.mjtGeom.mjGEOM_PLANE:
            continue

        mesh = make_mesh(model, geom_id)
        if mesh is None or len(mesh.faces) == 0:
            continue

        rgba = model.geom_rgba[geom_id]
        color = tuple(
            int(np.clip(v * 255, 0, 255))
            for v in rgba[:3]
        )

        handle = server.scene.add_mesh_simple(
            f"/robot/geom_{geom_id}",
            vertices=np.asarray(
                mesh.vertices, dtype=np.float32
            ),
            faces=np.asarray(
                mesh.faces, dtype=np.uint32
            ),
            color=color,
            side="double",
        )

        handles.append((geom_id, handle))
        rendered += 1

    if rendered < 20:
        raise RuntimeError(
            f"Too few robot geometries: {rendered}"
        )

    print(
        f"LIVE 3D: {rendered} MuJoCo geometries loaded",
        flush=True
    )
    update_scene(data)


def update_scene(data):
    quat = np.zeros(4, dtype=np.float64)

    with server.atomic():
        for geom_id, handle in handles:
            handle.position = np.asarray(
                data.geom_xpos[geom_id],
                dtype=float
            ).copy()

            mujoco.mju_mat2Quat(
                quat, data.geom_xmat[geom_id]
            )
            handle.wxyz = quat.copy()


def observed_step(model, data, *args, **kwargs):
    global last_model, last_data
    global last_render_time, wall_start

    result = original_step(
        model, data, *args, **kwargs
    )

    if last_model is not model:
        last_model = model
        last_data = data
        last_render_time = -1.0
        wall_start = time.monotonic() - float(data.time)
        initialize_scene(model, data)

    t = float(data.time)

    if t - last_render_time >= 1.0 / FPS:
        update_scene(data)
        last_render_time = t

        progress.value = f"{t:.1f} simulated seconds"

        elapsed = time.monotonic() - wall_start
        delay = t - elapsed

        if delay > 0:
            time.sleep(min(delay, 0.1))

    return result


def worker():
    global last_model, last_data

    while True:
        start_event.wait()
        start_event.clear()
        running.set()
        start_button.disabled = True
        status.value = "Running verified MuJoCo mission"

        last_model = None
        last_data = None

        try:
            os.environ["DEX3_MISSION_DISTANCE"] = "11.4"

            # Instrument the existing physics step without
            # changing the verified V10 source file.
            mujoco.mj_step = observed_step

            runpy.run_path(
                MISSION,
                run_name="__main__"
            )

            status.value = "Mission ended — check terminal result"

        except Exception as exc:
            status.value = f"Simulation error: {exc}"
            traceback.print_exc()

        finally:
            mujoco.mj_step = original_step
            running.clear()
            start_button.disabled = False


print("Loading initial 43-actuator Dex3 scene...", flush=True)

try:
    from dex3_dynamic_obstacles_v3 import DynamicObstacles
    preview_model = DynamicObstacles(
        count=int(os.environ.get("DEX3_OBSTACLE_COUNT", "10")),
        seed=int(os.environ.get("DEX3_OBSTACLE_SEED", "42"))
    ).model
    preview_data = mujoco.MjData(preview_model)
    mujoco.mj_forward(preview_model, preview_data)

    if preview_model.nu != 43:
        raise RuntimeError(
            f"Expected 43 actuators, found {preview_model.nu}"
        )

    initialize_scene(preview_model, preview_data)

    status.value = (
        "Robot loaded — automatic mission starting"
    )
    print("INITIAL ROBOT DISPLAY: READY", flush=True)

except Exception:
    status.value = "ERROR loading initial robot"
    traceback.print_exc()
    raise

threading.Thread(
    target=worker,
    daemon=True
).start()

# Start immediately; the user does not need to find
# the GUI panel before seeing the moving robot.
start_event.set()
print("AUTOMATIC MISSION START REQUESTED", flush=True)

print("===== LIVE MUJOCO VISER =====", flush=True)
print("Port:", PORT, flush=True)
print("Viewer controls: mouse orbit / zoom / pan", flush=True)
print("Mission: original verified V10, 11.4 m", flush=True)
print("Obstacle avoidance: not enabled in this baseline", flush=True)
print("Physical G1: not involved", flush=True)

while True:
    time.sleep(1.0)
