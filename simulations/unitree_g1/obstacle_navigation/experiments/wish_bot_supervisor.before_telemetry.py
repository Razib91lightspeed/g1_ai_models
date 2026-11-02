#!/usr/bin/env python3
"""Simulator-only WISH_BOT supervisor with persistent browser viewer."""

import importlib
import json
import math
import random
import threading
import time
from datetime import datetime, timezone

import numpy as np

# The simulation itself runs in the MAIN thread.
# Only the episode supervisor runs in a background thread.
nav = importlib.import_module("g1_dynamic_browser_v20_multistep_turn")

MAX_EPISODES = 30
EPISODE_TIMEOUT = 180.0
SPAWN_INTERVAL = 5.0
MAX_SPAWN_REQUESTS = 20
MIN_OBSTACLES = 3

FAILURE_STATES = {
    "turn_blocked",
    "turn_timeout",
    "turn_required",
    "blocked",
    "collision",
    "fallen",
    "stalled",
    "boundary_blocked",
    "geofence_violation",
    "assistance_required",
    "home_heading_unverified",
}

run_state = {
    "experiment": "WISH_BOT",
    "episode": 0,
    "reward": None,
    "downgrading": 0,
}


def event(name, **fields):
    record = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "event": name,
        **fields,
    }
    print("WISH_BOT_JSON " + json.dumps(record), flush=True)


def command(action):
    """Send the same action used by the browser buttons."""
    queue = nav.requests

    if hasattr(queue, "append"):
        queue.append(action)
    elif hasattr(queue, "put"):
        queue.put(action)
    else:
        raise RuntimeError("Unsupported V20 action queue")


def robot_position():
    try:
        return [
            round(float(nav.data.qpos[0]), 3),
            round(float(nav.data.qpos[1]), 3),
        ]
    except Exception:
        return None


def supervise():
    # Allow the simulation's HTTP server to initialize.
    time.sleep(2.0)

    for episode in range(1, MAX_EPISODES + 1):
        seed = 20261030 + episode
        random.seed(seed)
        np.random.seed(seed % (2**32))

        run_state["episode"] = episode

        event(
            "episode_starting",
            episode=episode,
            seed=seed,
            minimum_obstacles=MIN_OBSTACLES,
        )

        command("restart")
        time.sleep(0.7)
        command("start")

        start = time.monotonic()
        last_spawn = start
        requests_sent = 0
        reached_goal = False
        saw_motion = False
        outcome = None

        while time.monotonic() - start < EPISODE_TIMEOUT:
            time.sleep(0.25)

            status = str(nav.status)
            phase = str(nav.phase)

            if status.startswith("walking"):
                saw_motion = True

            if phase == "returning":
                reached_goal = True

            actual_obstacles = int(nav.spawn_count)

            # Success requires the existing navigation system
            # to verify the complete return journey.
            if status == "finished":
                if reached_goal and actual_obstacles >= MIN_OBSTACLES:
                    outcome = "wish_bot"
                else:
                    outcome = "invalid_success"
                break

            if status in FAILURE_STATES:
                outcome = status
                break

            # Request obstacles during the outbound mission,
            # counting only those actually spawned by V20.
            now = time.monotonic()

            if (
                phase == "outbound"
                and saw_motion
                and actual_obstacles < MIN_OBSTACLES
                and requests_sent < MAX_SPAWN_REQUESTS
                and now - last_spawn >= SPAWN_INTERVAL
            ):
                command("spawn")
                requests_sent += 1
                last_spawn = now

                event(
                    "obstacle_requested",
                    episode=episode,
                    request_number=requests_sent,
                    actual_obstacles=actual_obstacles,
                    position=robot_position(),
                )

        if outcome is None:
            outcome = "timeout"

        actual_obstacles = int(nav.spawn_count)

        event(
            "episode_finished",
            episode=episode,
            outcome=outcome,
            reached_goal=reached_goal,
            obstacles_placed=actual_obstacles,
            obstacle_requests=requests_sent,
            position=robot_position(),
            status=str(nav.status),
            elapsed_seconds=round(time.monotonic() - start, 2),
        )

        if outcome == "wish_bot":
            run_state["reward"] = "wish_bot"

            print(
                "\n"
                "============================================\n"
                "       WISH_BOT — MISSION COMPLETED\n"
                "       HOME -> GOAL -> HOME\n"
                "       REWARD: wish_bot\n"
                "============================================\n",
                flush=True,
            )

            event(
                "experiment_finished",
                result="wish_bot",
                episode=episode,
            )

            # Keep the browser alive on the final scene.
            return

        run_state["downgrading"] += 1

        event(
            "downgrading",
            episode=episode,
            reason=outcome,
            total_failures=run_state["downgrading"],
            punishment="restart_episode",
        )

        # A reset on the next iteration reinitializes
        # the simulation, occupancy map, and obstacle state.
        time.sleep(1.0)

    event(
        "experiment_finished",
        result="no_success",
        attempts=MAX_EPISODES,
        downgrading=run_state["downgrading"],
    )

    print(
        "WISH_BOT: trial limit reached. Browser remains available.",
        flush=True,
    )


def main():
    required = ("main", "requests", "status", "phase", "data", "spawn_count")

    missing = [name for name in required if not hasattr(nav, name)]

    if missing:
        raise RuntimeError(
            "Missing V20 interfaces: " + ", ".join(missing)
        )

    event(
        "experiment_started",
        episodes=MAX_EPISODES,
        timeout_seconds=EPISODE_TIMEOUT,
        minimum_obstacles=MIN_OBSTACLES,
        browser_port=8771,
        note="evaluation_only_no_policy_training",
    )

    worker = threading.Thread(
        target=supervise,
        name="wish-bot-supervisor",
        daemon=True,
    )
    worker.start()

    # MuJoCo and EGL stay on the main Python thread.
    nav.main()


if __name__ == "__main__":
    main()
