import mujoco
import numpy as np
import onnxruntime as ort
import yaml
from collections import deque

MODEL = "/workspace/unitree_mujoco/unitree_robots/g1/scene_obstacles_29dof.xml"
POLICY = "/workspace/policy/exported/policy.onnx"
CONFIG = "/workspace/policy/params/deploy.yaml"

with open(CONFIG) as f:
    cfg = yaml.safe_load(f)

model = mujoco.MjModel.from_xml_path(MODEL)
data = mujoco.MjData(model)

session = ort.InferenceSession(
    POLICY, providers=["CPUExecutionProvider"]
)

ids = np.array(cfg["joint_ids_map"], dtype=int)
defaults = np.array(cfg["default_joint_pos"])
# Stiffness/damping are stored in motor order.
# Joint positions and actions are in policy order.
kp = np.asarray(cfg["stiffness"], dtype=float)[ids]
kd = np.asarray(cfg["damping"], dtype=float)[ids]

act_cfg = cfg["actions"]["JointPositionAction"]
scale = np.array(act_cfg["scale"])
offset = np.array(act_cfg["offset"])

qids = []
vids = []

for motor in ids:
    joint = model.actuator_trnid[motor, 0]
    qids.append(model.jnt_qposadr[joint])
    vids.append(model.jnt_dofadr[joint])

qids = np.array(qids)
vids = np.array(vids)

assert model.nu == len(ids) == 29
assert session.get_inputs()[0].shape == [1, 480]

# Initialize the G1 in Unitree's configured standing posture.
data.qpos[qids] = defaults
data.qvel[:] = 0
mujoco.mj_forward(model, data)

pelvis = model.body("pelvis").id
terms_cfg = cfg["observations"]

history = {
    name: deque(maxlen=int(settings["history_length"]))
    for name, settings in terms_cfg.items()
}

last_action = np.zeros(29)
target = defaults.copy()

# Zero velocity means request standing in place.
command = np.zeros(3)

def compute_observations():
    # Body-frame angular velocity from MuJoCo.
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(
        model, data,
        mujoco.mjtObj.mjOBJ_BODY,
        pelvis, velocity, 1
    )
    ang_vel = velocity[:3]

    # Gravity direction in the pelvis frame.
    rotation = data.xmat[pelvis].reshape(3, 3)
    gravity = rotation.T @ np.array([0.0, 0.0, -1.0])

    return {
        "base_ang_vel": ang_vel,
        "projected_gravity": gravity,
        "velocity_commands": command,
        "joint_pos_rel": data.qpos[qids] - defaults,
        "joint_vel_rel": data.qvel[vids],
        "last_action": last_action
    }

def policy_step():
    global last_action, target

    measurements = compute_observations()

    for name, settings in terms_cfg.items():
        values = np.asarray(measurements[name], dtype=np.float32)
        clipping = settings.get("clip")

        if clipping is not None:
            values = np.clip(values, clipping[0], clipping[1])

        values = values * np.asarray(settings["scale"])
        history[name].append(values)

    # Initialize missing history with the first observation.
    for name, settings in terms_cfg.items():
        length = int(settings["history_length"])
        while len(history[name]) < length:
            history[name].appendleft(history[name][0].copy())

    # Term-major, oldest-to-newest history.
    obs = np.concatenate([
        np.concatenate(list(history[name]))
        for name in terms_cfg
    ]).astype(np.float32).reshape(1, -1)

    assert obs.shape == (1, 480)

    raw = session.run(["actions"], {"obs": obs})[0][0]

    if not np.isfinite(raw).all():
        raise RuntimeError("Policy generated non-finite actions")

    last_action = raw.copy()
    target = raw * scale + offset

    # Enforce model joint-position limits.
    for i, motor in enumerate(ids):
        joint = model.actuator_trnid[motor, 0]
        if model.jnt_limited[joint]:
            lo, hi = model.jnt_range[joint]
            target[i] = np.clip(target[i], lo, hi)


leg_ids = np.array([
    i for i, motor in enumerate(ids)
    if model.actuator(motor).name.startswith(("left_", "right_"))
    and any(part in model.actuator(motor).name
            for part in ("hip", "knee", "ankle"))
])

leg_start = data.qpos[qids[leg_ids]].copy()
max_leg_motion = 0.0

dt = float(model.opt.timestep)
interval = round(float(cfg["step_dt"]) / dt)

assert interval > 0
assert abs(interval * dt - cfg["step_dt"]) < 1e-8

initial_z = float(data.qpos[2])
total_mass = float(np.sum(model.body_mass))

print("=== SUPPORTED STANDING EXPERIMENT ===")
print("Total model mass:", round(total_mass, 2), "kg")
print("Support: full 0-2s, gradual release 2-4s, off 4-8s")

print("MuJoCo:", round(1 / dt), "Hz")
print("Policy:", round(1 / cfg["step_dt"]), "Hz")
print("Commanded velocity: zero")
print("Floating pelvis: enabled")
print("Artificial support: active 0-4s, then disabled")
print("Physical robot connection: none")
print()

duration = 20.0
steps = round(duration / dt)

for step in range(steps):
    if step % interval == 0:
        # Stand until 6 seconds, then ramp forward to 0.1 m/s.
        # Command stays within the policy's configured range.
        # Navigation commands are expressed in the robot frame.
        # The first waypoint is to the side of the obstacle.
        waypoint = np.array([0.8, -1.2])
        obstacle_xy = np.array([1.5, 0.0])
        obstacle_half = np.array([0.25, 0.35])

        position = np.asarray(data.qpos[:2])
        rot = data.xmat[pelvis].reshape(3, 3)
        yaw = np.arctan2(rot[1, 0], rot[0, 0])

        desired_yaw = np.arctan2(
            waypoint[1] - position[1],
            waypoint[0] - position[0]
        )
        error = np.arctan2(
            np.sin(desired_yaw - yaw),
            np.cos(desired_yaw - yaw)
        )

        # Distance from pelvis XY to the obstacle footprint.
        delta = np.maximum(
            np.abs(position - obstacle_xy) - obstacle_half,
            0.0
        )
        clearance = float(np.linalg.norm(delta))

        # Conservative test: no obstacle crossing permitted.
        # This is a test guard, not a certified safety system.
        ramp = np.clip((data.time - 6.0) / 2.0, 0.0, 1.0)
        command[0] = 0.30 * ramp
        command[1] = 0.0
        command[2] = np.clip(0.20 * error, -0.05, 0.05) * ramp

        if clearance < 0.75 or data.time >= 18.0:
            command[:] = 0.0

        if step % round(1.0 / dt) == 0:
            print(
                f"NAV t={data.time:.1f} "
                f"clearance={clearance:.2f}m "
                f"heading={np.degrees(yaw):.1f}deg "
                f"error={np.degrees(error):.1f}deg "
                f"vx={command[0]:.2f} "
                f"wz={command[2]:.3f}",
                flush=True
            )

        mujoco.mj_forward(model, data)
        policy_step()

    torque = (
        kp * (target - data.qpos[qids])
        - kd * data.qvel[vids]
    )

    data.ctrl[ids] = np.clip(
        torque,
        model.actuator_ctrlrange[ids, 0],
        model.actuator_ctrlrange[ids, 1]
    )


    # Simulation-only virtual stabilization harness.
    # Active for 2 seconds, released gradually until 4 seconds.
    # From 4 to 8 seconds the policy must balance unsupported.
    time_now = float(data.time)
    support_factor = float(np.clip((4.0 - time_now) / 2.0, 0, 1))

    data.xfrc_applied[pelvis, :] = 0.0

    if support_factor > 0.0:
        mujoco.mj_forward(model, data)

        base_pos = data.qpos[:3]
        base_vel = data.qvel[:3]
        desired_pos = np.array([0.0, 0.0, initial_z])

        # World-frame translational spring with gravity support.
        force = (
            np.array([0.0, 0.0, total_mass * 9.81])
            + 250.0 * (desired_pos - base_pos)
            - 60.0 * base_vel
        )

        # Keep pelvis Z aligned with world vertical.
        rot = data.xmat[pelvis].reshape(3, 3)
        body_up = rot[:, 2]
        world_up = np.array([0.0, 0.0, 1.0])

        spatial_vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY,
            pelvis, spatial_vel, 0
        )

        torque_support = (
            180.0 * np.cross(body_up, world_up)
            - 35.0 * spatial_vel[:3]
        )

        data.xfrc_applied[pelvis, :3] = support_factor * force
        data.xfrc_applied[pelvis, 3:] = support_factor * torque_support

    mujoco.mj_step(model, data)

    obstacle_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM,
        "nav_obstacle_1"
    )
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        if obstacle_id in (contact.geom1, contact.geom2):
            print("NAV FAILURE: obstacle contact detected")
            raise SystemExit(2)


    max_leg_motion = max(
        max_leg_motion,
        float(np.max(np.abs(
            data.qpos[qids[leg_ids]] - leg_start
        )))
    )

    if (step + 1) % round(0.5 / dt) == 0:
        rot = data.xmat[pelvis].reshape(3, 3)
        tilt = np.degrees(
            np.arccos(np.clip(rot[2, 2], -1, 1))
        )

        print(
            f"t={data.time:4.1f}s | "
            f"pelvis_z={data.qpos[2]:.3f}m | "
            f"tilt={tilt:5.1f}deg | "
            f"x={data.qpos[0]:+.3f}m | "
            f"y={data.qpos[1]:+.3f}m | "
            f"vx_cmd={command[0]:.2f}m/s | "
            f"contacts={data.ncon} | "
            f"support={support_factor:.2f}",
            flush=True
        )

    if not np.isfinite(data.qpos).all():
        print("Stopped: non-finite simulation state")
        break

print("\nCLOSED-LOOP TEST COMPLETE")

print(
    "Maximum leg joint displacement:",
    round(max_leg_motion, 4),
    "rad"
)
print(
    "Final pelvis displacement:",
    round(float(data.qpos[0]), 4),
    "m"
)

rotation = np.zeros(9)
mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
rotation = rotation.reshape(3, 3)

heading = np.degrees(
    np.arctan2(rotation[1, 0], rotation[0, 0])
)

print("\n=== TURNING TEST RESULTS ===")
print("Forward command: 0.30 m/s")
print("Turning command: 0.05 rad/s")
print("Final x:", round(float(data.qpos[0]), 3), "m")
print("Final y:", round(float(data.qpos[1]), 3), "m")
print("Final heading:", round(float(heading), 2), "degrees")
print("Final pelvis height:", round(float(data.qpos[2]), 3), "m")
