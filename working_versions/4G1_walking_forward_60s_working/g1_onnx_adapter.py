"""G1 29-DoF ONNX locomotion adapter.

Reproduces the observation history and action transformation
from the working MuJoCo controller.

Simulation only. Does not send commands to a physical robot.
"""

from collections import deque

import numpy as np
import onnxruntime as ort
import yaml


class G1PolicyAdapter:
    def __init__(self, config_path, model_path):
        with open(config_path, encoding="utf-8") as file:
            self.cfg = yaml.safe_load(file)

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        self.ids = np.asarray(
            self.cfg["joint_ids_map"], dtype=int
        )

        self.defaults = np.asarray(
            self.cfg["default_joint_pos"], dtype=np.float32
        )

        action_cfg = self.cfg["actions"]["JointPositionAction"]

        self.scale = np.asarray(
            action_cfg["scale"], dtype=np.float32
        )
        self.offset = np.asarray(
            action_cfg["offset"], dtype=np.float32
        )

        self.terms = self.cfg["observations"]

        self.history = {
            name: deque(maxlen=int(settings["history_length"]))
            for name, settings in self.terms.items()
        }

        self.last_action = np.zeros(29, dtype=np.float32)

        assert sorted(self.ids.tolist()) == list(range(29))
        assert self.defaults.shape == (29,)
        assert self.scale.shape == (29,)
        assert self.offset.shape == (29,)

        assert self.session.get_inputs()[0].shape == [1, 480]
        assert self.session.get_outputs()[0].shape == [1, 29]

    def reset(self):
        for history in self.history.values():
            history.clear()

        self.last_action.fill(0.0)

    def step(
        self,
        angular_velocity,
        projected_gravity,
        velocity_command,
        joint_position,
        joint_velocity,
    ):
        """Compute position targets in policy joint order.

        joint_position and joint_velocity must already be
        mapped to the policy's 29-joint ordering.
        """

        observations = {
            "base_ang_vel": angular_velocity,
            "projected_gravity": projected_gravity,
            "velocity_commands": velocity_command,
            "joint_pos_rel": (
                np.asarray(joint_position) - self.defaults
            ),
            "joint_vel_rel": joint_velocity,
            "last_action": self.last_action,
        }

        for name, settings in self.terms.items():
            values = np.asarray(
                observations[name], dtype=np.float32
            )

            clipping = settings.get("clip")

            if clipping is not None:
                values = np.clip(
                    values, clipping[0], clipping[1]
                )

            values = values * np.asarray(
                settings["scale"], dtype=np.float32
            )

            self.history[name].append(values.copy())

            while len(self.history[name]) < self.history[name].maxlen:
                self.history[name].appendleft(values.copy())

        obs = np.concatenate([
            np.concatenate(list(self.history[name]))
            for name in self.terms
        ]).astype(np.float32).reshape(1, -1)

        if obs.shape != (1, 480):
            raise ValueError(
                f"Invalid observation shape: {obs.shape}"
            )

        if not np.isfinite(obs).all():
            raise ValueError("Nonfinite policy observations")

        raw = self.session.run(
            ["actions"], {"obs": obs}
        )[0][0].astype(np.float32)

        if not np.isfinite(raw).all():
            raise ValueError("Nonfinite ONNX actions")

        # Safety limit for the first Newton compatibility tests.
        # Crucially, reject abnormal output BEFORE it enters
        # the recurrent last_action observation.
        maximum = float(np.max(np.abs(raw)))
        if maximum > 5.0:
            raise RuntimeError(
                f"ONNX_SAFETY_STOP: raw action {maximum:.3f} "
                f"exceeds diagnostic limit 5.0"
            )

        self.last_action = raw.copy()

        targets = raw * self.scale + self.offset

        return obs, raw, targets


if __name__ == "__main__":
    import sys

    adapter = G1PolicyAdapter(sys.argv[1], sys.argv[2])

    print("=== G1 ONNX COMPATIBILITY TEST ===")

    for step in range(5):
        obs, actions, targets = adapter.step(
            angular_velocity=np.zeros(3),
            projected_gravity=np.array([0, 0, -1]),
            velocity_command=np.zeros(3),
            joint_position=adapter.defaults.copy(),
            joint_velocity=np.zeros(29),
        )

        print(
            f"Step {step + 1}: "
            f"observations={obs.shape}, "
            f"actions={actions.shape}, "
            f"finite={np.isfinite(targets).all()}"
        )

    print("Joint mapping:", adapter.ids.tolist())
    print("Control timestep:", adapter.cfg["step_dt"])

    print("PASS: ONNX observations and inference verified")
    print(
        "NOTE: No physics or walking stability "
        "was tested."
    )
