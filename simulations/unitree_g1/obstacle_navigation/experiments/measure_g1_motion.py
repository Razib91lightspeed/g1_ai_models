#!/usr/bin/env python3
"""Headless simulation measurements. Never connects to a physical robot.

Run inside the existing g1-nav-sim container. No collision masks are modified.
The output is evidence for selecting a maneuver, not automatic certification.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import mujoco
import live_demo_v8 as nav
from g1_motion_checks import circular_fit, slew, wrap

BOUNDS = (-1.30, 11.98, -2.80, 2.80)
BODY_BUFFER = 0.55
FIELDS = ('time_s', 'stage', 'x_m', 'y_m', 'yaw_rad', 'yaw_unwrapped_rad',
          'height_m', 'tilt_deg', 'command_v_mps', 'command_yaw_rps',
          'measured_speed_mps', 'measured_forward_mps', 'measured_yaw_rps',
          'boundary_clearance_m')


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def clearance(xy):
    x, y = xy
    xmin, xmax, ymin, ymax = BOUNDS
    return min(x-xmin, xmax-x, y-ymin, ymax-y)-BODY_BUFFER


def robot_geoms(model):
    bodies = {nav.pelvis}
    for body in range(1, model.nbody):
        if int(model.body_parentid[body]) in bodies:
            bodies.add(body)
    return {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies}


def step_physics(step, support_pose):
    model, data = nav.model, nav.data
    t = float(data.time)
    if step % nav.interval == 0:
        nav.update_policy()
    torque = nav.kp*(nav.target-data.qpos[nav.qids])-nav.kd*data.qvel[nav.vids]
    data.ctrl[nav.ids] = np.clip(torque, model.actuator_ctrlrange[nav.ids, 0],
                               model.actuator_ctrlrange[nav.ids, 1])
    data.xfrc_applied[nav.pelvis, :] = 0
    support = nav.clamp((4.0-t)/2.0, 0.0, 1.0)
    if support > 0:
        mujoco.mj_forward(model, data)
        force = (np.array([0., 0., nav.total_mass*9.81])
                 + 250*(support_pose-data.qpos[:3])-60*data.qvel[:3])
        rotation = data.xmat[nav.pelvis].reshape(3, 3)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY,
                                 nav.pelvis, velocity, 0)
        angular = 180*np.cross(rotation[:, 2], [0., 0., 1.])-35*velocity[:3]
        data.xfrc_applied[nav.pelvis, :3] = support*force
        data.xfrc_applied[nav.pelvis, 3:] = support*angular
    mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)


def run_case(speed, direction, output, repeat, yaw_rate=0.30):
    model, data = nav.model, nav.data
    name = f'v{speed:.2f}_{"straight" if direction == 0 else "left" if direction > 0 else "right"}_r{repeat}'
    if yaw_rate != 0.30:
        name += f'_w{yaw_rate:.2f}'
    nav.reset_sim()
    # Test-fixture placement occurs only at reset, before stabilization or walking.
    # There is no teleportation during any maneuver or stop.
    data.qpos[0:2] = [2.0, 0.0] if direction == 0 else [4.0, -1.70*direction]
    support_pose = data.qpos[:3].copy()
    for i, gid in enumerate(sorted(nav.obstacle_ids)):
        body = int(model.geom_bodyid[gid])
        mocap = int(model.body_mocapid[body])
        if mocap < 0:
            raise RuntimeError(f'Obstacle {gid} is not a mocap body')
        data.mocap_pos[mocap] = [100.+i, 100., float(model.geom_size[gid, 2])]
    mujoco.mj_forward(model, data)
    collision_masks = (model.geom_contype.copy(), model.geom_conaffinity.copy())
    robot = robot_geoms(model)
    yaw_previous = nav.yaw_angle()
    yaw_unwrapped = 0.0
    yaw_at_motion = None
    rows = []
    next_sample = next_print = 0.0
    braking_at = brake_pose = None
    result_reason = None
    stage = 'stabilizing'
    print(f'\nCASE {name}: speed={speed:.2f}, yaw command={direction*yaw_rate:+.2f}', flush=True)
    for step in range(int(55.0/nav.dt)):
        t = float(data.time)
        progress = 0.0 if yaw_at_motion is None else direction*(yaw_unwrapped-yaw_at_motion)
        if t >= 4 and yaw_at_motion is None:
            yaw_at_motion = yaw_unwrapped
        if t >= 4 and braking_at is None:
            stage = 'moving'
            if direction == 0 and t >= 11:
                result_reason = 'straight_completed'
            elif direction and progress >= math.pi:
                result_reason = 'turn_completed'
            elif t >= 45:
                result_reason = 'motion_timeout'
            # Conservative experimental stopping allowance, not a measured bound.
            allowance = float(np.linalg.norm(data.qvel[:2]))**2/(2*0.35) + 0.15
            if clearance(data.qpos[:2]) <= allowance:
                result_reason = 'boundary_stop'
            if result_reason is not None:
                braking_at = t
                brake_pose = data.qpos[:2].copy()
                stage = 'braking'
                print(f'{name}: {result_reason}; controlled deceleration', flush=True)
        desired_v = speed if stage == 'moving' else 0.0
        desired_yaw = direction*yaw_rate if stage == 'moving' else 0.0
        nav.command[0] = slew(float(nav.command[0]), desired_v, nav.dt)
        nav.command[1] = 0.0
        nav.command[2] += np.clip(desired_yaw-nav.command[2], -0.30*nav.dt, 0.30*nav.dt)
        step_physics(step, support_pose)
        yaw = nav.yaw_angle()
        delta_yaw = wrap(yaw-yaw_previous)
        yaw_unwrapped += delta_yaw
        yaw_previous = yaw
        rotation = data.xmat[nav.pelvis].reshape(3, 3)
        tilt = math.degrees(math.acos(np.clip(rotation[2, 2], -1, 1)))
        measured_speed = float(np.linalg.norm(data.qvel[:2]))
        measured_forward = float(np.dot(data.qvel[:2], [math.cos(yaw), math.sin(yaw)]))
        measured_yaw_rate = delta_yaw/nav.dt
        contact = any((int(c.geom1) in robot and int(c.geom2) in nav.obstacle_ids)
                      or (int(c.geom2) in robot and int(c.geom1) in nav.obstacle_ids)
                      for c in (data.contact[i] for i in range(data.ncon)))
        fatal = None
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            fatal = 'nonfinite'
        elif tilt > 40 or data.qpos[2] < 0.50:
            fatal = 'fallen'
        elif contact:
            fatal = 'obstacle_contact'
        elif clearance(data.qpos[:2]) < 0:
            fatal = 'buffer_boundary_violation'
        if float(data.time) >= next_sample or fatal:
            row = dict(zip(FIELDS, (float(data.time), stage, float(data.qpos[0]),
                       float(data.qpos[1]), yaw, yaw_unwrapped, float(data.qpos[2]),
                       tilt, float(nav.command[0]), float(nav.command[2]), measured_speed,
                       measured_forward, measured_yaw_rate, clearance(data.qpos[:2]))))
            rows.append(row)
            next_sample = float(data.time)+0.05
        if float(data.time) >= next_print:
            print(f'{name}: t={data.time:.1f}s {stage}; cmd={nav.command[0]:.2f} '
                  f'measured={measured_speed:.2f}m/s; turn={math.degrees(progress):.1f}deg; '
                  f'tilt={tilt:.1f}deg; boundary buffer={clearance(data.qpos[:2]):.2f}m', flush=True)
            next_print = float(data.time)+2.0
        if fatal:
            result_reason = fatal
            nav.command[:] = 0
            # Freeze a failed trial for diagnosis; never classify it as a standing stop.
            break
        if braking_at is not None and float(data.time)-braking_at >= 6.0:
            break
    nav.command[:] = 0
    if not (np.array_equal(collision_masks[0], model.geom_contype)
            and np.array_equal(collision_masks[1], model.geom_conaffinity)):
        raise RuntimeError('Collision masks changed')
    with (output/f'{name}.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    tail = [r for r in rows if r['time_s'] >= rows[-1]['time_s']-2.0]
    settled = (braking_at is not None and rows[-1]['time_s']-braking_at >= 5.9
               and all(r['measured_speed_mps'] < 0.08 and abs(r['measured_yaw_rps']) < 0.10
                       and r['tilt_deg'] < 12 and r['height_m'] > max(0.5, nav.initial_z-0.12)
                       for r in tail))
    moving = [r for r in rows if r['stage'] == 'moving' and r['time_s'] >= 7.0]
    fit_rows = [r for r in moving if direction and yaw_at_motion is not None
                and math.radians(30) <= direction*(r['yaw_unwrapped_rad']-yaw_at_motion) <= math.radians(170)]
    fit = circular_fit([(r['x_m'], r['y_m']) for r in fit_rows]) if len(fit_rows) >= 20 else None
    completed = result_reason in ('straight_completed', 'turn_completed')
    summary = {'name': name, 'speed_command_mps': speed, 'yaw_command_rps': direction*yaw_rate,
               'reason': result_reason or 'test_timeout', 'completed': completed,
               'standing_after_stop': settled, 'trial_passed': completed and settled,
               'yaw_change_deg': math.degrees(yaw_unwrapped-(yaw_at_motion or 0.0)),
               'measured_mean_speed_mps': float(np.mean([r['measured_speed_mps'] for r in moving])) if moving else None,
               'max_tilt_after_support_deg': max((r['tilt_deg'] for r in rows if r['time_s'] >= 4), default=None),
               'minimum_boundary_buffer_m': min(r['boundary_clearance_m'] for r in rows),
               'circle_fit_diagnostic': fit,
               'braking_displacement_max_m': max((math.hypot(r['x_m']-brake_pose[0], r['y_m']-brake_pose[1])
                       for r in rows if r['time_s'] >= braking_at), default=None) if brake_pose is not None else None,
               'csv': f'{name}.csv'}
    summary = finite_json(summary)
    print('RESULT ' + json.dumps(summary, allow_nan=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--speeds', nargs='+', type=float, choices=(0.25, 0.32, 0.40),
                        default=[0.40, 0.32, 0.25])
    parser.add_argument('--yaw-rates', nargs='+', type=float, choices=(0.30, 0.40, 0.50),
                        default=[0.30], help='Simulation-only command trials; larger yaw commands are unvalidated')
    parser.add_argument('--turns-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.repeats <= 5:
        parser.error('--repeats must be 1 to 5')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = args.output or Path('/workspace/g1_experiment_reports')/f'g1_motion_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 1, 'simulation_only': True, 'created_utc': stamp,
              'mujoco_version': mujoco.__version__, 'bounds_xy': BOUNDS,
              'extra_body_buffer_m': BODY_BUFFER, 'collision_masks_preserved': True,
              'policy_sha256': sha256(nav.POLICY), 'config_sha256': sha256(nav.CONFIG),
              'scene_xml_sha256': sha256(nav.SCENE),
              'locomotion_source_sha256': sha256(nav.__file__),
              'note': 'One repeat is a screening test. Circle fits are diagnostics, not safe maneuver bounds.',
              'cases': []}
    for repeat in range(1, args.repeats+1):
        for speed in args.speeds:
            for yaw_rate in args.yaw_rates:
                for direction in ((1, -1) if args.turns_only else (0, 1, -1)):
                    report['cases'].append(run_case(speed, direction, output, repeat, yaw_rate))
                    (output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    archive = Path('/workspace/g1_experiment_reports/g1_motion_report_latest.tar.gz')
    temporary = archive.with_suffix('.pending')
    with tarfile.open(temporary, 'w:gz') as bundle:
        bundle.add(output, arcname=output.name)
    temporary.replace(archive)
    print(f'\nREPORT READY: {archive}', flush=True)
    print('Send this report for review before selecting a new turning radius.', flush=True)


if __name__ == '__main__':
    main()
