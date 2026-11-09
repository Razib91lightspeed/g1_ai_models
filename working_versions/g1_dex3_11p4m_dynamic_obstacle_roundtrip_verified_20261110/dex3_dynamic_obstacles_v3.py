"""Dynamic obstacles for the 43-actuator Dex3 MuJoCo model.

Environment-side spawning only. Navigation must use sensors.
No physical G1 control.
"""

import os
import math
import numpy as np
import mujoco

SCENE = "/workspace/dex3/assets/scene_43dof.xml"


class DynamicObstacles:
    def __init__(self, count=10, seed=42):
        if not 1 <= count <= 100:
            raise ValueError("Obstacle slots must be 1..100")

        self.count = count
        self.rng = np.random.default_rng(seed)
        self.spec = mujoco.MjSpec.from_file(SCENE)
        self.sizes = []
        self.names = []
        self.active = []

        for i in range(count):
            radius = float(self.rng.uniform(0.18, 0.38))
            half_height = float(self.rng.uniform(0.40, 0.75))
            is_box = i % 2 == 0

            size = (
                [radius, radius, half_height]
                if is_box else
                [radius, half_height, 0.0]
            )

            name = f"dynamic_obstacle_{i}"

            body = self.spec.worldbody.add_body(
                name=f"{name}_body",
                mocap=True,
                pos=[100.0 + i, 100.0, half_height],
            )

            body.add_geom(
                name=name,
                type=(
                    mujoco.mjtGeom.mjGEOM_BOX
                    if is_box else
                    mujoco.mjtGeom.mjGEOM_CYLINDER
                ),
                size=size,
                group=3,
                contype=1,
                conaffinity=1,
                rgba=[0.98, 0.35, 0.09, 1.0],
            )

            self.sizes.append((radius, half_height))
            self.names.append(name)


        goal_distance = float(
            os.environ.get("DEX3_MISSION_DISTANCE", "11.4")
        )

        for name, x in (
            ("start", 0.0),
            ("goal", goal_distance),
        ):
            self.spec.worldbody.add_geom(
                name=f"nav_marker_{name}",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=[x, 0.0, 0.025],
                size=[0.32, 0.018, 0.0],
                group=0,
                contype=0,
                conaffinity=0,
                rgba=[0.05, 0.95, 0.10, 1.0],
            )

        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        if self.model.nu != 43:
            raise RuntimeError("Dex3 actuator count changed")

        self.mocap_ids = []

        for i in range(count):
            bid = self.model.body(
                f"dynamic_obstacle_{i}_body"
            ).id
            mid = int(self.model.body_mocapid[bid])

            if mid < 0:
                raise RuntimeError("Obstacle is not mocap-enabled")

            self.mocap_ids.append(mid)


    def clear_all(self):
        # Move every obstacle outside the arena.
        # The bodies and their collision geoms remain
        # available for reuse on the return journey.
        for i, mocap_id in enumerate(self.mocap_ids):
            half_height = self.sizes[i][1]
            self.data.mocap_pos[mocap_id] = [
                100.0 + i,
                100.0,
                half_height,
            ]

        self.active.clear()
        mujoco.mj_forward(self.model, self.data)

        print(
            "RETURN RESET: all obstacles removed",
            flush=True
        )

    def spawn_ahead(self, robot_xy, heading,
                    distance=None):
        """Spawn one obstacle in front of the current robot pose.

        The environment owns placement. The navigation planner
        must not access these positions.
        """
        index = len(self.active)

        if index >= self.count:
            return None

        if distance is None:
            distance = float(self.rng.uniform(2.0, 3.0))

        if distance < 1.8:
            raise ValueError("Obstacle spawning too close")

        radius, half_height = self.sizes[index]

        forward = np.array([
            math.cos(heading),
            math.sin(heading),
        ])
        lateral = np.array([
            -forward[1], forward[0],
        ])

        robot_xy = np.asarray(robot_xy, dtype=float)

        chosen = None
        for _ in range(100):
            offset = float(self.rng.uniform(-0.35, 0.35))
            xy = (
                robot_xy + distance * forward
                + offset * lateral
            )

            if not (-2.5 < xy[1] < 2.5):
                continue
            if not (-1.0 < xy[0] < 13.0):
                continue

            overlap = False
            for old_xy, old_radius in self.active:
                if np.linalg.norm(xy-old_xy) < (
                    radius + old_radius + 0.45
                ):
                    overlap = True
                    break

            if not overlap:
                chosen = xy
                break

        if chosen is None:
            return None

        self.data.mocap_pos[self.mocap_ids[index]] = [
            float(chosen[0]),
            float(chosen[1]),
            half_height,
        ]

        self.active.append((chosen.copy(), radius))
        mujoco.mj_forward(self.model, self.data)

        return {
            "name": self.names[index],
            "position": chosen.tolist(),
            "diameter": round(2 * radius, 3),
            "height": round(2 * half_height, 3),
        }


if __name__ == "__main__":
    count = int(os.getenv("DEX3_OBSTACLE_COUNT", "10"))
    seed = int(os.getenv("DEX3_OBSTACLE_SEED", "42"))

    world = DynamicObstacles(count=count, seed=seed)

    print("===== DYNAMIC OBSTACLE SMOKE TEST =====")
    print("Dex3 actuators:", world.model.nu)
    print("Prepared hidden slots:", world.count)

    result = world.spawn_ahead(
        robot_xy=[0.0, 0.0],
        heading=0.0,
        distance=2.5,
    )

    if result is None:
        raise SystemExit("FAIL: obstacle placement")

    print("Spawned:", result)

    groups = np.zeros(6, dtype=np.uint8)
    groups[3] = 1
    hit_id = np.array([-1], dtype=np.int32)

    origin = np.array(
        [0.0, result["position"][1], 0.60],
        dtype=np.float64
    )
    direction = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float64
    )

    distance = mujoco.mj_ray(
        world.model,
        world.data,
        origin,
        direction,
        groups,
        1,
        -1,
        hit_id,
    )

    if distance < 0 or hit_id[0] < 0:
        raise SystemExit("FAIL: spawned obstacle not detected")

    hit_name = world.model.geom(int(hit_id[0])).name
    print("LiDAR ray hit:", hit_name)
    print("Distance:", round(float(distance), 3), "m")

    assert hit_name == result["name"]

    print("PASS: dynamic obstacle appeared and was detected")
    print("Navigation controller: not activated in this test")
    print("Physical G1: not involved")
