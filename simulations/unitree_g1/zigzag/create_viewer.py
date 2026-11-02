#!/usr/bin/env python3
"""Create an independent browser replay of the verified G1 zigzag test."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / "skills/g1_interactive.py"
output = Path(__file__).with_name("zigzag_viewer.py")

code = source.read_text()

# Independent HTTP port.
assert "8769" in code
code = code.replace("8769", "8770")

# Replace interactive page with automatic zigzag replay.
start = code.index("HTML = '''")
end = code.index("\nclass Handler", start)

page = """HTML = '''<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>G1 Zigzag Simulation</title>
<style>
body {
  background:#101419; color:#eef2f5;
  font:16px system-ui; margin:20px auto;
  max-width:1050px; padding:0 16px;
}
img {width:100%;border-radius:12px;background:#202730}
pre {background:#202730;padding:16px;border-radius:9px}
small {color:#aeb8c5}
</style>
</head>
<body>
<h2>Unitree G1 — Zigzag Walking</h2>
<p><small>DGX Spark · MuJoCo · ONNX · Simulation only</small></p>
<img src="/stream" alt="G1 zigzag live viewer">
<h3>Live telemetry</h3>
<pre id="info">Initializing simulation...</pre>
<p>0–6s: Stand | 6–8s: Accelerate | 8–11s: Left turn |
11–15s: Right turn | 15–20s: Left turn | 20–22s: Stop</p>
<script>
async function refresh() {
  try {
    const response = await fetch('/status');
    const data = await response.json();
    document.getElementById('info').textContent =
      JSON.stringify(data, null, 2);
  } catch(e) {
    document.getElementById('info').textContent = String(e);
  }
}
setInterval(refresh, 500);
refresh();
</script>
</body>
</html>'''
"""
code = code[:start] + page + code[end:]

# This viewer is a read-only replay.
start = code.index("    def do_POST(self):")
end = code.index("    def send_bytes(", start)
code = (
    code[:start]
    + "    def do_POST(self):\n"
      "        self.send_error(405, 'Read-only replay')\n\n"
    + code[end:]
)

# Replace the interactive task state machine.
start = code.index("    # A velocity-command stop is not an emergency stop.")
end = code.index("    if steps % policy_interval == 0:", start)

schedule = """    # Replay the validated zigzag experiment.
    # Keep the same 500 Hz physics and 50 Hz policy.
    ramp = float(np.clip((t - 6.0) / 2.0, 0.0, 1.0))

    if status == 'fallen':
        command[:] = 0.0
    elif t < 4.0:
        status = 'initializing'
        command[:] = 0.0
    elif t < 6.0:
        status = 'standing'
        command[:] = 0.0
    elif t < 20.0:
        status = 'zigzag'
        turn = 0.05 if t < 11.0 or t >= 15.0 else -0.05
        command[:] = [0.30 * ramp, 0.0, turn * ramp]
    elif t < 22.0:
        status = 'stopping'
        factor = float(np.clip((22.0 - t) / 2.0, 0.0, 1.0))
        command[:] = [0.30 * factor, 0.0, 0.0]
    else:
        status = 'standing'
        command[:] = 0.0

    v_cmd = float(command[0])

"""
code = code[:start] + schedule + code[end:]

# Add heading and angular-command telemetry.
old = "'speed_command_mps': round(v_cmd, 3)"
new = (
    "'speed_command_mps': round(v_cmd, 3), "
    "'turn_command_rad_s': round(float(command[2]), 3), "
    "'heading_deg': round(math.degrees("
    "math.atan2(rot[1, 0], rot[0, 0])), 2)"
)
assert old in code
code = code.replace(old, new)

# Wider camera view for walking.
code = code.replace("camera.distance = 2.8", "camera.distance = 4.5")

code = code.replace(
    "G1 INTERACTIVE VIEWER READY",
    "G1 ZIGZAG VIEWER READY"
)

output.write_text(code)
print("Created:", output)
print("Existing interactive controller: unchanged")
