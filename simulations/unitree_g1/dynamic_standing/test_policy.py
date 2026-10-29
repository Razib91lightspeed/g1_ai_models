import mujoco
import numpy as np
import onnxruntime as ort
import yaml
from collections import deque

MODEL = "/workspace/unitree_mujoco/unitree_robots/g1/scene_29dof.xml"
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
kp = np.array(cfg["stiffness"])
kd = np.array(cfg["damping"])

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

dt = float(model.opt.timestep)
interval = round(float(cfg["step_dt"]) / dt)

assert interval > 0
assert abs(interval * dt - cfg["step_dt"]) < 1e-8

print("=== UNITREE G1 CLOSED-LOOP POLICY TEST ===")
print("MuJoCo:", round(1 / dt), "Hz")
print("Policy:", round(1 / cfg["step_dt"]), "Hz")
print("Commanded velocity: zero")
print("Floating pelvis: enabled")
print("Artificial support: disabled")
print("Physical robot connection: none")
print()

duration = 5.0
steps = round(duration / dt)

for step in range(steps):
    if step % interval == 0:
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

    mujoco.mj_step(model, data)

    if (step + 1) % round(0.5 / dt) == 0:
        rot = data.xmat[pelvis].reshape(3, 3)
        tilt = np.degrees(
            np.arccos(np.clip(rot[2, 2], -1, 1))
        )

        print(
            f"t={data.time:4.1f}s | "
            f"pelvis_z={data.qpos[2]:.3f}m | "
            f"tilt={tilt:5.1f}deg | "
            f"contacts={data.ncon}",
            flush=True
        )

    if not np.isfinite(data.qpos).all():
        print("Stopped: non-finite simulation state")
        break

print("\nCLOSED-LOOP TEST COMPLETE")
