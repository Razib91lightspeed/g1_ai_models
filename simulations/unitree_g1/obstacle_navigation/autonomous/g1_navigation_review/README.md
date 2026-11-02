# G1 navigation review trial

This is a separate simulation controller based on the uploaded V12 source. It
imports the existing `live_demo_v8.py`, ONNX policy, deployment YAML, MuJoCo scene
and sensor/planner modules. V12 and the locomotion controller are not overwritten.

## Latest round-trip report and repair

The uploaded round-trip report stopped with `turn_required` at 52.594 s, about
4.393 m from HOME, without a robot-obstacle collision. It completed approximately
205.6 degrees of active turning and retained return checkpoint #14. Turning used
a 0.32 m/s command. The turn samples covered X=4.802–6.541 m and Y=-1.045–1.770 m.

The failed return connector had two endpoints roughly 3.78 m apart. After moving
about 0.45 m from its first endpoint, the old waypoint follower selected that first
endpoint behind the robot. This was a controller bug, not evidence that checkpoint
#14 was lost. The updated follower projects onto path segments, tracks forward
distance along the path, and interpolates a lookahead target. It handles sparse
and dense paths consistently. Tests include the exact reported failure pose.

The dashboard now disables the Spawn button visibly when `G1_ENABLE_OBSTACLES=0`.
Setting that flag to `1` enables the button while walking. The initial navigation
test keeps it at zero to isolate route-following problems. Mocap physics is unchanged.

## Next measurement: tighter turns

The 0.25 m/s right-turn stall means lowering forward speed is not a demonstrated
general solution. Measure higher yaw commands at the working 0.32 m/s speed first.
The following four cases test left/right at 0.40 and 0.50 rad/s commanded yaw.
These are new simulation experiments, not validated policy settings. The navigation
controller retains its measured 0.30 rad/s yaw limit until the results are reviewed.

After installing this updated package, stop the browser Python process, leaving
Docker running, and run on Spark:

```bash
docker exec -it \
  -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
  -e PYTHONDONTWRITEBYTECODE=1 \
  g1-nav-sim bash -lc '
    cd /workspace/obstacle_navigation/autonomous &&
    sha256sum -c g1_navigation_review/BASELINE_SHA256SUMS.txt &&
    cd g1_navigation_review &&
    sha256sum -c SHA256SUMS.txt &&
    python -B -u measure_g1_motion.py \
      --speeds 0.32 --yaw-rates 0.40 0.50 --turns-only
  '
```

When `REPORT READY` appears, on Spark:

```bash
docker cp g1-nav-sim:/tmp/g1_motion_report_latest.tar.gz /tmp/g1_motion_report_latest.tar.gz
```

Then on the Mac:

```bash
scp rhasan@10.11.0.50:/tmp/g1_motion_report_latest.tar.gz ~/Downloads/
```

Attach this new motion report. It is a different measurement run from the first
nine-case report and does not replace the embedded evidence for the current controller.

## Measured evidence

Source: the supplied `g1_motion_report_latest.tar.gz`, run ID
`20261029T231656_860407Z`, MuJoCo 3.3.5. Each command was screened once.

| Forward command | Left turn | Right turn | Straight measured mean speed |
|---|---|---|---|
| 0.40 m/s | Boundary stop; fitted radius 1.81 m | Boundary stop; fitted radius 1.86 m | 0.376 m/s |
| 0.32 m/s | Completed and stood; fitted radius 1.40 m | Completed and stood; fitted radius 1.51 m | 0.296 m/s |
| 0.25 m/s | Completed and stood; fitted radius 1.09 m | Timed out with little translation | 0.131 m/s |

All nine cases met the runner's standing check after stopping. Maximum observed
post-support tilt was 4.07 degrees. Maximum measured braking displacement was
0.316 m across all cases and 0.216 m among the 0.32 m/s cases. These are observed
values, not guaranteed bounds. The 0.40 left case reached about 181 degrees only
after its boundary-triggered braking phase; it did not pass the turn trial.
The small circle fit in the failed 0.25 right case is not a usable turning radius.

## Controller changes

- Retain the exact checkpoint chosen by turnaround through replanning and retries.
  Advance the return cursor only after physically reaching that checkpoint.
- Retrace saved points in reverse without reintroducing the already-reached cursor
  point. A* reconnection is attempted only when the remembered segment is blocked,
  and keeps the same checkpoint target.
- Use a 0.32 m/s turn command with +/-0.30 rad/s commanded yaw. Direction-specific
  screening radii use the measured fits plus a provisional 0.15 m allowance:
  approximately 1.55 m left and 1.66 m right. That allowance is an engineering
  trial setting, not a measured confidence bound.
- Screen up to 230 degrees to reserve space for aligning with a forward-approachable
  return checkpoint. The initial tests covered 180-degree turns followed by stopping;
  continuation to 230 degrees is an experimental controller maneuver. It stops if
  no verified checkpoint is found, the remaining arc becomes blocked, the boundary
  check fails, or measured yaw progress stalls.
- Accumulate signed yaw through the +/-180-degree wrap. Compare turn progress with
  turn entry, not HOME heading. Pin the selected join before handing it to the planner.
- Use a continuous 0.32–0.40 m/s moving-speed schedule, with slower commands near
  obstacles, heading changes and the destination. Exclude sustained 0.25 m/s from
  this navigation trial because the right-turn test stalled.
- Ramp forward commands at 0.15 m/s² acceleration and 0.35 m/s² deceleration, and yaw
  commands at 0.30 rad/s². These are the settings used in the supplied measurements.
- Keep physics running during normal deceleration and standing checks. A zero
  command alone is not a successful stop. A failed standing check is reported.
- Preserve all obstacle collision masks and mocap placement. Detect robot-obstacle
  contacts without mistaking an obstacle resting on the floor for a robot collision.
- Check the primary scene XML, deployment YAML, policy and locomotion-source hashes
  against the supplied report before starting. Included assets referenced by the XML
  are not individually fingerprinted by that original report.
- Write a compact trajectory/status report outside the repository when a trial
  finishes or stops. A full success requires the requested HOME position, heading
  and standing criteria.

## What remains unresolved

The complete round trip has not yet been run with this revised controller. The
review environment has Python source but does not have the model assets or weights.
Twenty-three local regression tests passed; they do not establish physical behavior.

Final heading restoration at HOME is not implemented as a validated maneuver.
Retracing the route can return to HOME facing the opposite direction. In that case
the controller reports `home_heading_unverified`, with `completed_all_criteria=false`.
The next trial is intended to validate the destination turn and recorded-route
return, while retaining this final-heading requirement as an explicit open issue.

The nominal arcs and 2D obstacle inflation are screening models, not full-body
swept-volume proofs. Room length, wall margin and extra turn buffer remain
provisional. Realized movement differs from commanded speed and yaw. The existing
single-height LiDAR can miss objects outside its scan plane. No physical robot
connection or deployment is implemented by these scripts.

## Install the complete updated directory

Download the new archive and upload it from the Mac:

```bash
scp ~/Downloads/g1_navigation_review.tar.gz rhasan@10.11.0.50:/tmp/
```

On Spark, back up the preceding review directory, then replace only that directory's
files. This is a full-file update, with no text-replacement patch installer:

```bash
cd ~/Desktop/razib/ai_models
review_dir=simulations/unitree_g1/obstacle_navigation/autonomous/g1_navigation_review
review_backup="$HOME/g1_nav_backups/review_$(date +%Y%m%dT%H%M%S_%N)"
mkdir -p "$review_backup" &&
cp -a "$review_dir" "$review_backup/" &&
tar -xzf /tmp/g1_navigation_review.tar.gz \
  -C simulations/unitree_g1/obstacle_navigation/autonomous/
```

## Run the obstacle-free browser trial

Stop any old browser Python process with Ctrl+C in its terminal. Keep Docker running.
The measurement program exits on completion and does not need to be stopped.

```bash
docker exec -it \
  -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e G1_ENABLE_SLOW_TRIALS=1 -e G1_ENABLE_OBSTACLES=0 \
  g1-nav-sim bash -lc '
    cd /workspace/obstacle_navigation/autonomous &&
    sha256sum -c g1_navigation_review/BASELINE_SHA256SUMS.txt &&
    cd g1_navigation_review &&
    sha256sum -c SHA256SUMS.txt &&
    python -B -u g1_navigation_trial.py
  '
```

Keep the existing Mac tunnel running:

```bash
ssh -N -L 8766:127.0.0.1:8771 rhasan@10.11.0.50
```

Open `http://localhost:8766`, refresh, and press Start once. Obstacle spawning is
blocked by default for this test. Do not enable it before validating the basic route.
The terminal prints commanded/measured speeds and checkpoint handoffs.

When the robot stops, a report is written in the container, even if a controller
check blocks progress. Use a separate Spark terminal while the browser process
remains running:

```bash
docker cp g1-nav-sim:/tmp/g1_roundtrip_report_latest.json /tmp/g1_roundtrip_report_latest.json
```

Then, on the Mac:

```bash
scp rhasan@10.11.0.50:/tmp/g1_roundtrip_report_latest.json ~/Downloads/
```

Attach that JSON report. Do not treat `home_heading_unverified` as a successful
completion of the original heading requirement.

## Tests, repeated measurements and Git

Run code regressions using Python with NumPy:

```bash
cd g1_navigation_review
python3 -B -m unittest -v test_navigation_review.py
```

The original measurement runner remains included. It runs nine cases by default;
`--repeats 3` repeats identical deterministic initial conditions and is not a
substitute for testing varied poses, gait phase, perturbations or obstacle layouts.
It uses a fresh headless simulation, retains collisions, and parks obstacles
through mocap bodies. Test fixture placement occurs only before stabilization.

No commit or push has been made. Update the existing repository after navigation
validation. Commit only reviewed source, tests, documentation and the small
motion-screening evidence file required by the runtime checks. Do not commit raw
CSV traces, trial reports, backups, weights, secrets or generated caches.
