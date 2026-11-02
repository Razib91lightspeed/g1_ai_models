#!/usr/bin/env python3
"""Unattended MuJoCo round-trip evaluation. No policy training."""

import argparse
import importlib
import json
import random
import sys
import threading
import time
from datetime import datetime, timezone


def report(event, **fields):
    record = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    print("EPISODE_JSON " + json.dumps(record), flush=True)


parser = argparse.ArgumentParser()
parser.add_argument("--max-episodes", type=int, default=50)
parser.add_argument("--max-hours", type=float, default=4.0)
parser.add_argument("--episode-seconds", type=float, default=180.0)
parser.add_argument("--obstacles", type=int, default=8)
parser.add_argument("--seed", type=int, default=20261030)
args = parser.parse_args()

if (
    args.max_episodes < 1
    or args.max_hours <= 0
    or args.episode_seconds <= 0
    or args.obstacles < 0
):
    parser.error("Invalid experiment limits")

nav = importlib.import_module("g1_dynamic_browser_v19_local_detour")

# These are terminal outcomes from the existing navigation controller.
failures = {
    "blocked",
    "turn_blocked",
    "turn_required",
    "turn_timeout",
    "stalled",
    "fallen",
    "collision",
    "boundary_blocked",
    "geofence_violation",
    "assistance_required",
    "home_heading_unverified",
}

def queue_action(action):
    with nav.lock:
        nav.requests.append(action)


simulation = threading.Thread(
    target=nav.main,
    daemon=True,
    name="g1-mujoco"
)
simulation.start()

started = time.monotonic()
deadline = started + args.max_hours * 3600.0

success = False
attempted = 0

report(
    "experiment_started",
    max_episodes=args.max_episodes,
    max_hours=args.max_hours,
    obstacles_per_episode=args.obstacles,
    policy_training=False,
)

# Allow the simulator to initialize.
time.sleep(2.0)

for episode in range(1, args.max_episodes + 1):
    if time.monotonic() >= deadline or not simulation.is_alive():
        break

    attempted = episode
    rng = random.Random(args.seed + episode)

    # Reset all simulation state and begin a new mission.
    queue_action("restart")
    queue_action("start")

    episode_start = time.monotonic()
    spawned = 0
    next_spawn = episode_start + rng.uniform(7.0, 12.0)
    reason = None
    saw_motion = False
    saw_goal = False

    report(
        "episode_started",
        episode=episode,
        seed=args.seed + episode,
    )

    while True:
        now = time.monotonic()

        if not simulation.is_alive():
            reason = "simulation_thread_exited"
            break

        if now >= deadline:
            reason = "experiment_time_limit"
            break

        state = nav.status
        phase = nav.phase

        if state.startswith("walking") or state == "turnaround":
            saw_motion = True

        if phase == "returning":
            saw_goal = True

        if state == "finished":
            reason = "success"
            success = True
            break

        if state in failures:
            reason = state
            break

        if now - episode_start > args.episode_seconds:
            reason = "episode_timeout"
            break

        # Introduce obstacles only during active forward navigation.
        # The existing simulator handles spawn-position rejection.
        if (
            spawned < args.obstacles
            and state.startswith("walking")
            and phase == "outbound"
            and now >= next_spawn
        ):
            queue_action("spawn")
            spawned += 1
            next_spawn = now + rng.uniform(6.0, 12.0)

            report(
                "obstacle_requested",
                episode=episode,
                request_number=spawned,
            )

        time.sleep(0.20)

    # Prevent the old episode from continuing while results are recorded.
    queue_action("stop")

    report(
        "episode_finished",
        episode=episode,
        outcome=reason,
        reached_goal=saw_goal,
        movement_observed=saw_motion,
        obstacles_requested=spawned,
        position=[
            round(float(nav.data.qpos[0]), 3),
            round(float(nav.data.qpos[1]), 3),
        ],
        status=nav.status,
        elapsed_seconds=round(time.monotonic() - episode_start, 2),
    )

    if success or reason in (
        "experiment_time_limit",
        "simulation_thread_exited",
    ):
        break

    time.sleep(1.0)

report(
    "experiment_finished",
    episodes=attempted,
    success=success,
    elapsed_hours=round((time.monotonic() - started) / 3600, 3),
)

# Main simulation thread is daemonized and exits with this process.
sys.stdout.flush()
sys.stderr.flush()
