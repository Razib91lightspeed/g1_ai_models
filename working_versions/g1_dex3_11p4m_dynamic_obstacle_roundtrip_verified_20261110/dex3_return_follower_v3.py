
import math
import numpy as np

from path_planner import AStarPlanner


class ReturnFollower:
    def __init__(self, mapper):
        self.mapper = mapper
        self.planner = AStarPlanner(
            mapper,
            clearance=0.50,
            allow_unknown=False,
        )
        self.path = None
        self.next_scan = 0.0
        self.next_plan = 0.0
        self.last_log = -1.0
        self.replans = 0
        self.aligning = False
        self.align_since = None
        self.turn_command = 0.0
        self.heading_error_deg = 0.0

    def command(self, data, nav, home, t):
        pos = data.qpos[:2].copy()
        home = np.asarray(home, dtype=float)
        remaining = float(np.linalg.norm(home-pos))

        if remaining <= 0.23:
            return np.zeros(3)

        if t >= self.next_scan:
            self.mapper.observe(data)
            self.next_scan = t + 0.20

        if self.path is None or t >= self.next_plan:
            direction = (home-pos) / max(remaining, 1e-9)
            side = np.array([-direction[1], direction[0]])

            candidates = [home]

            for advance in (3.0, 2.0, 1.0):
                if advance >= remaining:
                    continue
                for offset in (0.0, -0.6, 0.6, -1.0, 1.0):
                    candidates.append(
                        pos + advance*direction + offset*side
                    )

            best = None

            for goal in candidates:
                path, reason = self.planner.plan(pos, goal)

                if path is None or len(path) < 2:
                    continue

                path = np.asarray(path, dtype=float)
                length = float(np.sum(np.linalg.norm(
                    np.diff(path, axis=0), axis=1
                )))
                residual = float(np.linalg.norm(goal-home))

                if residual >= remaining - 0.15:
                    continue

                score = length + residual

                if best is None or score < best[0]:
                    best = (score, path, length)

            if best is None:
                self.path = None
                print(
                    "RETURN: no safe observed path; holding",
                    flush=True
                )
            else:
                _, self.path, length = best
                self.replans += 1
                print(
                    f"RETURN A* #{self.replans}: "
                    f"path={length:.2f}m "
                    f"home_remaining={remaining:.2f}m",
                    flush=True
                )

            self.next_plan = t + 2.0

        if self.path is None:
            return np.zeros(3)

        distances = np.linalg.norm(
            self.path-pos, axis=1
        )
        index = int(np.argmin(distances))

        while index < len(self.path)-1:
            if np.linalg.norm(self.path[index]-pos) >= 0.55:
                break
            index += 1

        target = self.path[index]
        delta = target-pos
        bearing = math.atan2(delta[1], delta[0])
        yaw = float(nav.yaw_angle())

        error = math.atan2(
            math.sin(bearing-yaw),
            math.cos(bearing-yaw)
        )


        # Hysteresis prevents repeated switching between
        # walking and omnidirectional turning.
        self.heading_error_deg = math.degrees(error)

        if self.aligning:
            if abs(error) < 0.15:
                self.aligning = False
                self.align_since = None
                print(
                    "RETURN OMNI ALIGNMENT COMPLETE",
                    flush=True
                )
        elif abs(error) > 0.35:
            self.aligning = True
            self.align_since = t
            print(
                "RETURN OMNI ALIGNMENT START: "
                f"error={self.heading_error_deg:+.1f}deg",
                flush=True
            )

        if self.aligning:
            self.turn_command = float(
                np.sign(error) * 0.60
            )
            return np.zeros(3)

        self.turn_command = 0.0

        cmd_yaw = float(np.clip(
            0.85*error, -0.20, 0.20
        ))

        if abs(error) > 1.0:
            vx = 0.0
            vy = 0.0
        elif abs(error) > 0.45:
            vx = 0.12
            vy = float(np.clip(
                0.22*math.sin(error), -0.12, 0.12
            ))
        else:
            vx = 0.32
            vy = float(np.clip(
                0.22*math.sin(error), -0.12, 0.12
            ))

        if t-self.last_log >= 1.0:
            self.last_log = t
            print(
                f"RETURN STEER t={t:.1f}s "
                f"remaining={remaining:.2f}m "
                f"heading_error={math.degrees(error):+.1f}deg "
                f"cmd=[{vx:.2f},{vy:+.2f},{cmd_yaw:+.2f}]",
                flush=True
            )

        return np.array([vx, vy, cmd_yaw])
