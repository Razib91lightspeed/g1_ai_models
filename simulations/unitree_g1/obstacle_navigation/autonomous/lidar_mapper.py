"""2D LiDAR occupancy mapping for the G1 MuJoCo experiment."""

import math
import numpy as np
import mujoco


class LidarMapper:
    UNKNOWN = -1
    FREE = 0
    OCCUPIED = 100

    def __init__(
        self,
        model,
        xlim=(-1.0, 4.5),
        ylim=(-2.0, 2.0),
        resolution=0.10,
        max_range=5.0,
    ):
        self.model = model
        self.xmin, self.xmax = xlim
        self.ymin, self.ymax = ylim
        self.res = resolution
        self.max_range = max_range

        self.width = math.ceil((self.xmax-self.xmin)/resolution)
        self.height = math.ceil((self.ymax-self.ymin)/resolution)

        self.grid = np.full(
            (self.height, self.width),
            self.UNKNOWN,
            dtype=np.int16
        )

        # Navigation obstacles are in MuJoCo geometry group 3.
        self.groups = np.zeros(6, dtype=np.uint8)
        self.groups[3] = 1

        self.sensor_body = model.body("torso_link").id

    def world_to_cell(self, x, y):
        col = math.floor((x-self.xmin)/self.res)
        row = math.floor((y-self.ymin)/self.res)

        if 0 <= row < self.height and 0 <= col < self.width:
            return (row, col)
        return None

    def cell_to_world(self, row, col):
        return np.array([
            self.xmin + (col+0.5)*self.res,
            self.ymin + (row+0.5)*self.res
        ])

    def scan(self, data):
        """Return measured ray endpoints, ranges and hit flags.

        Only MuJoCo's ray results are used to build the map.
        Geometry coordinates are not read by this mapper.
        """
        mujoco.mj_forward(self.model, data)

        rot = data.xmat[self.sensor_body].reshape(3, 3)
        origin = (
            data.xpos[self.sensor_body]
            + rot @ np.array([0.0, 0.0, 0.15])
        ).copy()

        measurements = []

        # Horizontal 360-degree LiDAR, 2-degree spacing.
        for angle_deg in range(0, 360, 2):
            a = math.radians(angle_deg)
            local_direction = np.array([
                math.cos(a), math.sin(a), 0.0
            ])
            direction = rot @ local_direction

            hit_id = np.array([-1], dtype=np.int32)
            distance = float(mujoco.mj_ray(
                self.model,
                data,
                origin,
                direction,
                self.groups,
                1,
                -1,
                hit_id
            ))

            hit = hit_id[0] >= 0 and distance >= 0
            usable_hit = hit and distance <= self.max_range
            measured_range = (
                distance if usable_hit else self.max_range
            )
            endpoint = (
                origin + measured_range * direction
            )

            measurements.append({
                "angle_deg": angle_deg,
                "range": measured_range,
                "hit": bool(usable_hit),
                "endpoint": endpoint[:2].copy(),
            })

        return origin[:2].copy(), measurements

    def update(self, origin, measurements):
        """Integrate observed 2D rays into the map.

        A hit marks the endpoint occupied. A no-hit ray marks
        only its observed line free, not the whole map.
        """
        for measurement in measurements:
            end = measurement["endpoint"]
            distance = float(np.linalg.norm(end-origin))
            steps = max(1, math.ceil(distance/(self.res*0.4)))

            cells = []
            for i in range(steps+1):
                p = origin + (end-origin)*(i/steps)
                cell = self.world_to_cell(p[0], p[1])

                if cell is not None and (
                    not cells or cell != cells[-1]
                ):
                    cells.append(cell)

            hit = measurement["hit"]

            free_cells = cells[:-1] if hit else cells

            for row, col in free_cells:
                if self.grid[row, col] != self.OCCUPIED:
                    self.grid[row, col] = self.FREE

            if hit and cells:
                row, col = cells[-1]
                self.grid[row, col] = self.OCCUPIED

    def observe(self, data):
        origin, measurements = self.scan(data)
        self.update(origin, measurements)
        return origin, measurements

    def reset(self):
        self.grid.fill(self.UNKNOWN)
