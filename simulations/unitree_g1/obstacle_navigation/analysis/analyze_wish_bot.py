#!/usr/bin/env python3

import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

BASE = Path(__file__).resolve().parents[1]
LOG = BASE / "reports/wish_bot_live.log"
OUT = BASE / "analysis"
OUT.mkdir(exist_ok=True)

events = []

if not LOG.exists():
    print("ERROR: log not found")
    raise SystemExit(0)

for line in LOG.read_text(errors="replace").splitlines():
    if not line.startswith("WISH_BOT_JSON "):
        continue
    try:
        events.append(json.loads(
            line.split("WISH_BOT_JSON ", 1)[1]
        ))
    except json.JSONDecodeError:
        pass

samples = [
    e for e in events
    if e.get("event") == "telemetry"
    and isinstance(e.get("position"), list)
]

results = [
    e for e in events
    if e.get("event") == "episode_finished"
]

with (OUT / "trajectory.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "episode", "x_m", "y_m", "heading_deg",
        "phase", "status", "obstacles", "sim_time"
    ])
    for e in samples:
        writer.writerow([
            e.get("episode"),
            e["position"][0],
            e["position"][1],
            e.get("heading_deg"),
            e.get("phase"),
            e.get("status"),
            e.get("obstacles"),
            e.get("sim_time"),
        ])

(OUT / "episode_summary.json").write_text(
    json.dumps(results, indent=2)
)

# Plot coordinates in metres, preserving physical aspect ratio.
# Background bounds are the requested 11.4 m x 4 m
# reference, NOT proof the actual MuJoCo geofence was resized.

W, H = 1140, 400

def px(x):
    return 50 + (float(x) + 0.5) * 90

def py(y):
    return 200 - float(y) * 90

parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" '
    f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    '<rect width="100%" height="100%" fill="#f4f7fb"/>',
    '<rect x="50" y="20" width="1026" height="360" '
    'fill="#ffffff" stroke="#64748b" stroke-width="2"/>',
    '<text x="60" y="16" font-size="13" fill="#334155">'
    'WISH_BOT trajectory — reference field 11.4 x 4 m</text>',
]

episodes = sorted(set(e.get("episode") for e in samples))

for episode in episodes:
    group = [
        e for e in samples if e.get("episode") == episode
    ]

    for phase, color in [
        ("outbound", "#0891b2"),
        ("returning", "#16a34a"),
    ]:
        points = [
            f'{px(e["position"][0]):.1f},'
            f'{py(e["position"][1]):.1f}'
            for e in group if e.get("phase") == phase
        ]

        if len(points) >= 2:
            parts.append(
                f'<polyline points="{" ".join(points)}" '
                f'fill="none" stroke="{color}" '
                f'stroke-width="2.5" opacity=".8"/>'
            )

    if group:
        last = group[-1]
        x, y = last["position"]
        parts.append(
            f'<circle cx="{px(x)}" cy="{py(y)}" '
            f'r="4" fill="#dc2626"/>'
        )
        parts.append(
            f'<text x="{px(x)+7}" y="{py(y)-7}" '
            f'font-size="11" fill="#111827">'
            f'Episode {escape(str(episode))}</text>'
        )

# Mark request positions, not confirmed physical obstacle centres.
for e in events:
    if e.get("event") != "obstacle_requested":
        continue
    pos = e.get("position")
    if isinstance(pos, list) and len(pos) >= 2:
        parts.append(
            f'<circle cx="{px(pos[0])}" cy="{py(pos[1])}" '
            f'r="5" fill="none" stroke="#f59e0b" '
            f'stroke-width="2"/>'
        )

parts += [
    f'<circle cx="{px(0)}" cy="{py(0)}" r="7" fill="#2563eb"/>',
    f'<circle cx="{px(8.5)}" cy="{py(0)}" r="7" fill="#7c3aed"/>',
    f'<text x="{px(0)+9}" y="{py(0)-9}" font-size="13">HOME</text>',
    f'<text x="{px(8.5)+9}" y="{py(0)-9}" font-size="13">GOAL</text>',
    '</svg>',
]

(OUT / "trajectory.svg").write_text("\n".join(parts))

print("=== WISH_BOT ANALYSIS ===")
print("Telemetry samples:", len(samples))
print("Completed episodes:", len(results))
print("Successful missions:", sum(
    e.get("outcome") == "wish_bot" for e in results
))
print("Files:")
print(OUT / "trajectory.svg")
print(OUT / "trajectory.csv")
print(OUT / "episode_summary.json")
