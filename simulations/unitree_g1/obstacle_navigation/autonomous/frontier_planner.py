"""Find safe, reachable viewpoints near unknown map regions."""

import math
import numpy as np


def select_frontier(mapper, planner, robot_xy, goal_xy):
    grid = mapper.grid
    blocked = planner.build_blocked_map()

    robot_xy = np.asarray(robot_xy, dtype=float)
    goal_xy = np.asarray(goal_xy, dtype=float)

    candidates = []

    for row in range(2, mapper.height - 2):
        for col in range(2, mapper.width - 2):

            if grid[row, col] != mapper.FREE:
                continue

            if blocked[row, col]:
                continue

            neighborhood = grid[
                row - 2:row + 3,
                col - 2:col + 3
            ]

            unknown_count = int(
                np.count_nonzero(
                    neighborhood == mapper.UNKNOWN
                )
            )

            if unknown_count == 0:
                continue

            xy = mapper.cell_to_world(row, col)

            distance_from_robot = float(
                np.linalg.norm(xy - robot_xy)
            )

            if distance_from_robot < 0.45:
                continue

            distance_to_goal = float(
                np.linalg.norm(xy - goal_xy)
            )

            # Prefer viewpoints toward the goal and
            # with useful unknown space nearby.
            score = (
                distance_to_goal
                - 0.04 * unknown_count
                + 0.10 * distance_from_robot
            )

            candidates.append((score, xy))

    candidates.sort(key=lambda item: item[0])

    # Validate reachability using the real A* planner.
    # Limit candidates to keep the test responsive.
    for score, xy in candidates[:100]:
        path, status = planner.plan(robot_xy, xy)

        if status == "path_found":
            return {
                "status": "frontier_found",
                "target": xy,
                "path": path,
                "score": score,
            }

    return {
        "status": "no_reachable_frontier",
        "target": None,
        "path": None,
        "score": None,
    }
