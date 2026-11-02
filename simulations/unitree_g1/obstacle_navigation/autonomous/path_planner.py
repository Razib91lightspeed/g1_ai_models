"""Clearance-aware A* planner for a 2D occupancy grid."""

import heapq
import math
import numpy as np


class AStarPlanner:
    def __init__(
        self,
        mapper,
        clearance=0.40,
        allow_unknown=False,
    ):
        self.mapper = mapper
        self.clearance = clearance
        self.allow_unknown = allow_unknown

    def build_blocked_map(self):
        grid = self.mapper.grid
        blocked = grid == self.mapper.OCCUPIED

        # Expand each detected obstacle by robot clearance.
        radius = math.ceil(self.clearance/self.mapper.res)
        inflated = blocked.copy()

        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                if math.hypot(dx, dy)*self.mapper.res > self.clearance:
                    continue

                src_y0 = max(0, -dy)
                src_y1 = min(grid.shape[0], grid.shape[0]-dy)
                src_x0 = max(0, -dx)
                src_x1 = min(grid.shape[1], grid.shape[1]-dx)

                if src_y1 <= src_y0 or src_x1 <= src_x0:
                    continue

                inflated[
                    src_y0+dy:src_y1+dy,
                    src_x0+dx:src_x1+dx
                ] |= blocked[
                    src_y0:src_y1,
                    src_x0:src_x1
                ]

        if not self.allow_unknown:
            inflated |= grid == self.mapper.UNKNOWN

        return inflated

    @staticmethod
    def heuristic(a, b):
        return math.hypot(
            a[0]-b[0], a[1]-b[1]
        )

    def plan(self, start_xy, goal_xy):
        start = self.mapper.world_to_cell(*start_xy)
        goal = self.mapper.world_to_cell(*goal_xy)

        if start is None or goal is None:
            return None, "start_or_goal_outside_map"

        blocked = self.build_blocked_map()

        if blocked[start]:
            return None, "start_blocked_or_unknown"

        if blocked[goal]:
            return None, "goal_blocked_or_unknown"

        moves = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]

        frontier = [(self.heuristic(start, goal), 0, start)]
        cost = {start: 0.0}
        parents = {}
        counter = 0
        found = False

        while frontier:
            _, _, current = heapq.heappop(frontier)

            if current == goal:
                found = True
                break

            for dy, dx in moves:
                nr = current[0]+dy
                nc = current[1]+dx

                if not (
                    0 <= nr < blocked.shape[0]
                    and 0 <= nc < blocked.shape[1]
                ):
                    continue

                nxt = (nr, nc)

                if blocked[nxt]:
                    continue

                # Never cut diagonally between blocked cells.
                if dy != 0 and dx != 0:
                    if (
                        blocked[current[0]+dy, current[1]]
                        or blocked[current[0], current[1]+dx]
                    ):
                        continue

                step = math.hypot(dx, dy)
                new_cost = cost[current]+step

                if new_cost < cost.get(nxt, float("inf")):
                    cost[nxt] = new_cost
                    parents[nxt] = current
                    counter += 1

                    priority = (
                        new_cost+self.heuristic(nxt, goal)
                    )

                    heapq.heappush(
                        frontier,
                        (priority, counter, nxt)
                    )

        if not found:
            return None, "no_observed_safe_path"

        path = [goal]
        while path[-1] != start:
            path.append(parents[path[-1]])

        path.reverse()

        world_path = np.array([
            self.mapper.cell_to_world(*cell)
            for cell in path
        ])

        return world_path, "path_found"
