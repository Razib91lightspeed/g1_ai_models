#!/usr/bin/env python3
"""Simulation-only G1 skill command validation and scheduling."""

import argparse
import json
import math

MAX_SPEED = 0.3
MAX_DURATION = 10.0

def validate_command(command):
    if not isinstance(command, dict):
        raise ValueError("Command must be a JSON object")

    allowed_fields = {"skill", "speed_mps", "duration_s", "stop_after"}
    unknown = set(command) - allowed_fields
    if unknown:
        raise ValueError(f"Unknown fields: {sorted(unknown)}")

    skill = command.get("skill")

    if skill not in ("stand", "walk_forward", "stop"):
        raise ValueError(f"Unsupported skill: {skill}")

    if skill == "walk_forward":
        if "speed_mps" not in command or "duration_s" not in command:
            raise ValueError("Walking requires speed_mps and duration_s")

        speed = command["speed_mps"]
        duration = command["duration_s"]

        if isinstance(speed, bool) or not isinstance(speed, (int, float)):
            raise ValueError("speed_mps must be numeric")

        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise ValueError("duration_s must be numeric")

        if not math.isfinite(speed) or not (0 < speed <= MAX_SPEED):
            raise ValueError("speed_mps must be between 0 and 0.3")

        if not math.isfinite(duration) or not (0 < duration <= MAX_DURATION):
            raise ValueError("duration_s must be between 0 and 10")

        if command.get("stop_after", True) is not True:
            raise ValueError("Walking must finish with a stop")

        return {
            "skill": "walk_forward",
            "speed_mps": float(speed),
            "duration_s": float(duration),
            "stop_after": True,
        }

    if set(command) != {"skill"}:
        raise ValueError(f"{skill} accepts no additional fields")

    return {"skill": skill}


def create_schedule(command):
    command = validate_command(command)
    skill = command["skill"]

    if skill in ("stand", "stop"):
        return {
            "skill": skill,
            "velocity_command": [0.0, 0.0, 0.0],
            "status": "validated",
        }

    return {
        "skill": skill,
        "velocity_command": [
            command["speed_mps"], 0.0, 0.0
        ],
        "duration_s": command["duration_s"],
        "stop_after": True,
        "status": "validated",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="JSON skill request")
    args = parser.parse_args()

    try:
        command = json.loads(args.command)
        result = create_schedule(command)
        print(json.dumps(result, indent=2))
    except (ValueError, json.JSONDecodeError) as exc:
        parser.exit(1, f"REJECTED: {exc}\n")


if __name__ == "__main__":
    main()
