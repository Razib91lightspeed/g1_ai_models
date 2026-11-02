from pathlib import Path
p = Path('simulations/unitree_g1/zigzag/zigzag_viewer.py')
s = p.read_text()
if 'ZIGZAG_REPLAY_PATCH_V1' in s:
    raise SystemExit('Already patched; no changes made')
if 'G1 ZIGZAG VIEWER READY' not in s:
    raise SystemExit('Expected zigzag viewer not found')
# Create an atomic backup and only commit after all assertions pass.
# Replace the page entirely, avoid depending on CSS/JS internals.
a = s.index("HTML = '''")
b = s.index('\nclass Handler', a)
page = '''HTML = \'\'\'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>G1 Zigzag Replay</title><style>
body{background:#101419;color:#eef2f5;font:16px system-ui;margin:20px auto;max-width:950px;padding:0 16px}
img{width:100%;border-radius:12px;background:#202730}button{padding:12px 20px;margin:8px 5px 8px 0;cursor:pointer;font:inherit;border:0;border-radius:8px}
pre{background:#202730;padding:15px;border-radius:9px;white-space:pre-wrap}small{color:#acb9c6}
</style></head><body><h2>Unitree G1 — Zigzag Replay</h2>
<p><small>MuJoCo + ONNX · Simulation only · port 8770</small></p>
<img src="/stream" alt="G1 simulation feed">
<p><button onclick="control('start')">Start</button><button onclick="control('replay')">Replay from start</button><button onclick="control('stop')">Stop</button></p>
<p id="message">Press Start to begin. Replay resets the simulation.</p>
<pre id="info">Loading...</pre>
<script>
async function control(action){
 try{const r=await fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action})});
 const d=await r.json();document.getElementById('message').textContent=(r.ok?'OK: ':'ERROR: ')+(d.message||d.error);
 }catch(e){document.getElementById('message').textContent=String(e)}
}
async function refresh(){try{let r=await fetch('/status');let d=await r.json();document.getElementById('info').textContent=JSON.stringify(d,null,2)}catch(e){}}
setInterval(refresh,500);refresh();
</script></body></html>\'\'\'\n'''
s = s[:a] + page + s[b:]
# Convert disabled POST handler to bounded control handler.
a = s.index('    def do_POST(self):')
b = s.index('    def send_bytes(', a)
handler = '''    def do_POST(self):
        global control_request
        if self.path != '/control':
            self.send_error(404)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 256:
                raise ValueError('Invalid request length')
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict) or set(payload) != {'action'}:
                raise ValueError('Expected {"action": ...}')
            action = payload['action']
            if action not in ('start', 'replay', 'stop'):
                raise ValueError('Unknown control action')
            with state_lock:
                control_request = action
            self.send_bytes(202, 'application/json', json.dumps({'message': action + ' requested'}).encode())
        except (ValueError, TypeError) as exc:
            self.send_bytes(400, 'application/json', json.dumps({'error': str(exc)}).encode())

'''
s = s[:a] + handler + s[b:]
# Augment global variables before the web server is exposed.
needle = "telemetry = {'status': 'initializing'"
assert s.count(needle) == 1
idx = s.index(needle)
s = s[:idx] + "# ZIGZAG_REPLAY_PATCH_V1\ncontrol_request = None\nmode = 'idle'\nplay_start = None\nstop_start = None\nstop_initial = 0.0\nauto_start = False\n" + s[idx:]
# Replace entire main loop preamble (including inherited interactive task logic), up to policy inference.
a = s.index('while True:\n    t = float(data.time)', s.index("print('G1 ZIGZAG VIEWER READY"))
b = s.index('    if steps % policy_interval == 0:', a)
loop = '''while True:
    t = float(data.time)
    with state_lock:
        requested = control_request
        control_request = None

    if requested == 'replay':
        # Reset simulation and *all* recurrent observation state, only in the physics thread.
        mujoco.mj_resetData(model, data)
        data.qpos[qids] = defaults
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(model, data)
        for h in history.values():
            h.clear()
        last_action = np.zeros_like(last_action)
        target = defaults.copy()
        command[:] = 0.0
        v_cmd = 0.0
        steps = 0
        mode = 'idle'
        status = 'initializing'
        play_start = None
        stop_start = None
        auto_start = True
        t = float(data.time)
        wall_start = time.monotonic() - t
        print('ZIGZAG: replay reset complete', flush=True)

    if requested == 'start' and mode in ('idle', 'finished'):
        if mode == 'finished':
            # Starting a finished run must reset policy history too.
            with state_lock:
                control_request = 'replay'
        else:
            auto_start = True

    if requested == 'stop':
        auto_start = False
        if mode == 'playing':
            mode = 'stopping'
            stop_start = t
            stop_initial = float(command[0])
            print('ZIGZAG: stopping requested', flush=True)

    if mode == 'fallen':
        status = 'fallen'
        command[:] = 0.0
    elif t < 4.0:
        status = 'initializing'
        command[:] = 0.0
    else:
        if auto_start and mode == 'idle':
            mode = 'playing'
            play_start = t
            auto_start = False
            print('ZIGZAG: started', flush=True)
        if mode == 'playing':
            elapsed = t - play_start
            if elapsed < 2.0:
                status = 'standing'
                command[:] = 0.0
            elif elapsed < 16.0:
                status = 'zigzag'
                ramp = float(np.clip((elapsed - 2.0) / 2.0, 0.0, 1.0))
                turn = 0.05 if elapsed < 7.0 or elapsed >= 11.0 else -0.05
                command[:] = [0.30 * ramp, 0.0, turn * ramp]
            else:
                mode = 'stopping'
                stop_start = t
                stop_initial = float(command[0])
        if mode == 'stopping':
            status = 'stopping'
            remaining = max(0.0, stop_initial - 0.15 * (t - stop_start))
            command[:] = [remaining, 0.0, 0.0]
            if remaining == 0.0:
                mode = 'finished'
        if mode == 'finished':
            status = 'standing'
            command[:] = 0.0
        elif mode == 'idle':
            status = 'standing'
            command[:] = 0.0
    v_cmd = float(command[0])

'''
s = s[:a] + loop + s[b:]
# Latch a detected fall until an explicit replay reset.
fall = "        status = 'fallen'\n        current_task = None"
assert fall in s, 'Fall detection block not found'
s = s.replace(fall, "        status = 'fallen'\n        mode = 'fallen'\n        current_task = None", 1)
# Let the UI see replay phase.
needle = "'turn_command_rad_s': round(float(command[2]), 3)"
assert needle in s, 'Telemetry angular field absent'
s = s.replace(needle, "'replay_mode': mode, 'replay_elapsed_s': round(t - play_start, 2) if play_start is not None else None, " + needle, 1)
p.with_suffix('.py.pre_replay.bak').write_text(p.read_text())
p.write_text(s)
print('Patched:', p)
print('Backup:', p.with_suffix('.py.pre_replay.bak'))
