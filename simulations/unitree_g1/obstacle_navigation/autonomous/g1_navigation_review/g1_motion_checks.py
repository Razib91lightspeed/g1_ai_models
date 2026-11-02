"""Small, simulator-independent helpers for the G1 navigation experiment."""
import math


class PolylineTracker:
    """Monotonic distance along segments, independent of waypoint spacing."""
    def __init__(self, lookahead=0.42, search_ahead=1.0):
        self.lookahead = lookahead
        self.search_ahead = search_ahead
        self.reset()

    def reset(self):
        self.source = None
        self.points = []
        self.distances = []
        self.progress = 0.0

    def advance(self, points, position):
        if points is not self.source:
            self.source = points
            self.points = [(float(p[0]), float(p[1])) for p in points]
            if len(self.points) < 2 or not all(math.isfinite(v) for p in self.points for v in p):
                raise ValueError('A path needs at least two finite points')
            self.distances = [0.0]
            for a, b in zip(self.points, self.points[1:]):
                self.distances.append(self.distances[-1]+math.hypot(b[0]-a[0], b[1]-a[1]))
            self.progress = 0.0
        low = self.progress
        high = min(self.distances[-1], low+self.search_ahead)
        best = None
        for i, (a, b) in enumerate(zip(self.points, self.points[1:])):
            start, end = self.distances[i:i+2]
            length = end-start
            if length <= 1e-12 or end < low or start > high:
                continue
            projection = ((position[0]-a[0])*(b[0]-a[0])+
                          (position[1]-a[1])*(b[1]-a[1]))/length
            s = max(low, start, min(high, end, start+projection))
            f = (s-start)/length
            x, y = a[0]+f*(b[0]-a[0]), a[1]+f*(b[1]-a[1])
            candidate = ((x-position[0])**2+(y-position[1])**2, s)
            if best is None or candidate < best:
                best = candidate
        if best is not None:
            self.progress = max(self.progress, best[1])
        return self.at(self.progress+self.lookahead)

    def at(self, distance):
        distance = max(0.0, min(distance, self.distances[-1]))
        for i, (start, end) in enumerate(zip(self.distances, self.distances[1:])):
            if end >= distance and end-start > 1e-12:
                a, b = self.points[i:i+2]
                f = (distance-start)/(end-start)
                return (a[0]+f*(b[0]-a[0]), a[1]+f*(b[1]-a[1]))
        return self.points[-1]

    def remaining(self, horizon=1.5):
        end = min(self.distances[-1], self.progress+horizon)
        points = [self.at(self.progress)]
        points.extend(p for p, s in zip(self.points, self.distances)
                      if self.progress < s < end)
        points.append(self.at(end))
        return points


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def adaptive_speed(error, clearance, remaining, slow_trials=False):
    """Experimental command envelope; does not establish stopping distance."""
    if clearance is not None and clearance <= 0.80:
        return 0.0
    if not slow_trials:
        return 0.40
    # 0.25 m/s right turn stalled in the first screening run. Keep moving
    # commands at 0.32–0.40 until a lower-speed regime is demonstrated.
    def interpolate(value, knots):
        for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
            if value <= x1:
                f = max(0.0, min(1.0, (value-x0)/(x1-x0)))
                return y0 + f*(y1-y0)
        return knots[-1][1]
    near = float('inf') if clearance is None else clearance
    by_clearance = interpolate(near, [(0.8, 0.32), (1.6, 0.32), (2.4, 0.40)])
    by_heading = interpolate(abs(error), [(0.0, 0.40),
                            (math.radians(25), 0.32), (math.radians(55), 0.32)])
    by_goal = interpolate(remaining, [(0.2, 0.32), (0.8, 0.32), (1.5, 0.40)])
    return min(by_clearance, by_heading, by_goal)


def slew(current, desired, dt, acceleration=0.15, deceleration=0.35):
    if dt < 0 or not all(math.isfinite(v) for v in (current, desired, dt)):
        raise ValueError('Invalid slew input')
    rate = acceleration if desired > current else deceleration
    return current + max(-rate*dt, min(rate*dt, desired-current))


def arc_points(xy, yaw, direction, radius, angle=math.pi, samples=91):
    if direction not in (-1, 1) or radius <= 0 or samples < 2:
        raise ValueError('Invalid arc parameters')
    points = []
    for i in range(samples):
        theta = yaw + direction*angle*i/(samples-1)
        points.append((xy[0] + radius/direction*(math.sin(theta)-math.sin(yaw)),
                       xy[1] - radius/direction*(math.cos(theta)-math.cos(yaw))))
    return points


def arc_margin(points, bounds, buffer):
    xmin, xmax, ymin, ymax = bounds
    return min(min(x-xmin, xmax-x, y-ymin, ymax-y)-buffer for x, y in points)


def reverse_segment(trace, cursor, target, position, joining):
    """A pinned join or a sequential reverse segment, without the reached point."""
    if not 0 <= target <= cursor < len(trace):
        raise ValueError('Return indices are out of order')
    if joining:
        return [tuple(position), tuple(trace[target][:2])]
    # cursor has already been reached; putting it back ahead can command a reversal.
    return [tuple(position)] + [tuple(trace[i][:2])
                               for i in range(cursor-1, target-1, -1)]


def circular_fit(samples):
    """Least-squares diagnostic only. Requires a substantial measured turn."""
    import numpy as np
    p = np.asarray(samples, dtype=float)
    if len(p) < 20 or p.shape[1] != 2 or not np.isfinite(p).all():
        return None
    centre0 = p.mean(axis=0)
    q = p-centre0
    matrix = np.column_stack((2*q[:, 0], 2*q[:, 1], np.ones(len(q))))
    solution, _, rank, _ = np.linalg.lstsq(matrix, np.sum(q*q, axis=1), rcond=None)
    if rank < 3:
        return None
    centre = solution[:2] + centre0
    distances = np.linalg.norm(p-centre, axis=1)
    radius = float(distances.mean())
    return {'radius_m': radius,
            'radial_rmse_m': float(np.sqrt(np.mean((distances-radius)**2))),
            'radial_max_error_m': float(np.max(np.abs(distances-radius)))}
