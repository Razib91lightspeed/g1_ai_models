
import sys
import math
import numpy as np

sys.path.insert(
    0, "/workspace/obstacle_navigation/autonomous"
)

from lidar_mapper import LidarMapper
from path_planner import AStarPlanner


class RouteFollower:
    def __init__(self, model):
        self.mapper = LidarMapper(
            model,
            xlim=(-3.0, 16.0),
            ylim=(-5.0, 5.0),
            resolution=0.10,
            max_range=5.0,
        )
        self.planner = AStarPlanner(
            self.mapper,
            clearance=0.50,
            allow_unknown=False,
        )
        self.active = False
        self.path = None
        self.waypoint = 0
        self.next_scan = 0.0
        self.next_plan = 0.0
        self.nearest = float("inf")
        self.replans = 0
        self.last_debug = -1.0

    @staticmethod
    def angle_error(a):
        return math.atan2(math.sin(a), math.cos(a))

    def choose_route(self, position, home, heading, distance):
        forward = np.array([
            math.cos(heading), math.sin(heading)
        ])
        side = np.array([-forward[1], forward[0]])

        goal = home + forward * distance
        remaining = float(np.dot(goal-position, forward))

        candidates = []

        # The actual goal is preferred whenever observed.
        if np.linalg.norm(goal-position) <= 4.0:
            direct_path, reason = self.planner.plan(
                position, goal
            )
            if direct_path is not None and len(direct_path) >= 2:
                self.path = np.asarray(
                    direct_path, dtype=float
                )
                self.waypoint = 1
                self.replans += 1
                length = float(np.sum(
                    np.linalg.norm(
                        np.diff(self.path, axis=0),
                        axis=1
                    )
                ))
                print(
                    f"NAV DIRECT GOAL #{self.replans}: "
                    f"path={length:.2f}m "
                    f"goal={goal.round(2)}",
                    flush=True
                )
                return

        if remaining <= 4.0:
            candidates.append(goal)

        # Progressively nearer observed local targets.
        for advance in (3.2, 2.5, 1.8, 1.1):
            for offset in (0.0, -0.7, 0.7, -1.2, 1.2):
                target = (
                    position
                    + forward * min(advance, max(0.4, remaining))
                    + side * offset
                )
                candidates.append(target)

        choices = []

        for target in candidates:
            path, reason = self.planner.plan(
                position, target
            )
            if path is None or len(path) < 2:
                continue

            path = np.asarray(path, dtype=float)
            length = float(np.sum(
                np.linalg.norm(
                    np.diff(path, axis=0), axis=1
                )
            ))

            progress = float(
                np.dot(path[-1]-position, forward)
            )
            if progress < 0.35:
                continue

            lateral = abs(float(
                np.dot(path[-1]-home, side)
            ))

            # Reward progress, penalize distance and
            # excessive deviation from the goal centreline.
            score = (
                length - 1.7*progress + 0.25*lateral
            )
            choices.append((score, path, length))

        if not choices:
            self.path = None
            print(
                "NAV: no safe observed route; stopping",
                flush=True
            )
            return

        _, self.path, length = min(
            choices, key=lambda x: x[0]
        )
        self.waypoint = 1
        self.replans += 1

        print(
            f"NAV PLAN #{self.replans}: "
            f"points={len(self.path)} "
            f"length={length:.2f}m "
            f"target={self.path[-1].round(2)}",
            flush=True
        )

    def command(self, data, nav, home, heading,
                distance, t):
        position = data.qpos[:2].copy()

        if t >= self.next_scan:
            _, rays = self.mapper.observe(data)
            self.next_scan = t + 0.20

            front_hits = [
                float(r["range"])
                for r in rays
                if r["hit"]
                and (
                    r["angle_deg"] <= 35
                    or r["angle_deg"] >= 325
                )
            ]
            self.nearest = (
                min(front_hits)
                if front_hits else float("inf")
            )

            if not self.active and self.nearest < 3.5:
                self.active = True
                print(
                    f"NAV ACTIVE: obstacle at "
                    f"{self.nearest:.2f}m",
                    flush=True
                )

        if not self.active:
            return None

        if t >= self.next_plan:
            self.choose_route(
                position, home, heading, distance
            )
            self.next_plan = t + 2.00

        if self.path is None:
            return np.zeros(3)

        # Follow a point ahead on the planned path,
        # avoiding rapid commands for every 10 cm cell.
        distances = np.linalg.norm(
            self.path-position, axis=1
        )
        near = int(np.argmin(distances))
        self.waypoint = max(self.waypoint, near)

        index = self.waypoint
        while index < len(self.path)-1:
            if np.linalg.norm(
                self.path[index]-position
            ) >= 0.55:
                break
            index += 1

        self.waypoint = index
        target = self.path[index]

        delta = target-position
        bearing = math.atan2(delta[1], delta[0])
        error = self.angle_error(
            bearing-float(nav.yaw_angle())
        )

        yaw = float(np.clip(
            0.85*error, -0.20, 0.20
        ))

        # Keep a small translation command during turns.
        # This is an experimental command-response test.
        if abs(error) > 0.45:
            vx = 0.12
        elif abs(error) > 0.25:
            vx = 0.18
        else:
            vx = 0.32

        # Try bounded lateral motion toward the waypoint.
        vy = float(np.clip(
            0.22 * math.sin(error), -0.12, 0.12
        ))

        # Reduce forward speed near an obstacle.
        # The existing 0.60 m emergency gate remains active.
        if self.nearest < 1.0:
            vx = 0.0

        cmd = np.array(
            [vx, vy, yaw], dtype=float
        )

        if t - self.last_debug >= 1.0:
            self.last_debug = t
            print(
                f"NAV STEER t={t:.1f}s "
                f"pos={position.round(2)} "
                f"waypoint={target.round(2)} "
                f"heading_error={math.degrees(error):+.1f}deg "
                f"cmd=[{vx:.2f},{vy:+.2f},{yaw:+.2f}] "
                f"lidar={self.nearest:.2f}m",
                flush=True
            )

        return cmd
