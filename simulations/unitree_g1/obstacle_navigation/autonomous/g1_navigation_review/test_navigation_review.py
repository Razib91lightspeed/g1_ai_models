"""Regression tests without MuJoCo, model assets, ONNX or a GPU."""
import ast
from contextlib import redirect_stdout
import io
import math
import hashlib
import json
from pathlib import Path
import tempfile
import argparse
import tarfile
from datetime import datetime, timezone
from unittest.mock import patch
from types import SimpleNamespace
import unittest
import numpy as np
from g1_motion_checks import adaptive_speed, arc_margin, arc_points, circular_fit, reverse_segment, slew, wrap, PolylineTracker


def controller_functions(names, namespace):
    source = Path(__file__).with_name('g1_navigation_trial.py').read_text()
    tree = ast.parse(source)
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'actual_trial_functions', 'exec'), namespace)
    return namespace


class ReturnRegression(unittest.TestCase):
    def state(self):
        calls = []
        def plan(pos, target):
            calls.append(tuple(target))
            return np.array([pos, target]), 'test_plan'
        nav = SimpleNamespace(command=np.zeros(3), yaw_angle=lambda: math.pi, wrap=wrap)
        ns = dict(np=np, math=math, nav=nav, phase='returning', trace_cursor=53,
                  trace_target_index=18, return_join_index=18, return_joined=False,
                  outbound_trace=[(i/10, 0., 0.) for i in range(54)],
                  data=SimpleNamespace(qpos=np.array([3., 0., .8]), time=20.),
                  path=None, path_kind=None, plan_target=None, status='replanning',
                  return_mode='none', replans=0, trace_replans=0, trace_detours=0,
                  planner=SimpleNamespace(plan=plan), path_cells_clear=lambda p: True,
                  in_lab=lambda p: True, reverse_segment=reverse_segment,
                  RETRACE_CHUNK_POINTS=7, REJOIN_MAX_HEADING_DEG=75,
                  make_plan_original=lambda: self.fail('Unexpected original planner'))
        controller_functions({'make_plan', 'initial_heading_error'}, ns)
        return ns, calls

    def plan(self, ns):
        with redirect_stdout(io.StringIO()):
            ns['make_plan']()

    def test_handoff_18_is_not_replaced_by_52(self):
        ns, calls = self.state()
        self.plan(ns)
        self.assertEqual(ns['trace_target_index'], 18)
        np.testing.assert_allclose(ns['path'], [[3, 0], [1.8, 0]])
        self.assertEqual(ns['trace_cursor'], 53)
        self.assertEqual(calls, [])

    def test_blockage_reconnects_to_same_checkpoint(self):
        ns, calls = self.state()
        count = [0]
        def clear(points):
            count[0] += 1
            return count[0] > 1
        ns['path_cells_clear'] = clear
        self.plan(ns)
        self.assertEqual(calls, [(1.8, 0)])
        self.assertEqual(ns['return_join_index'], 18)
        self.assertEqual(ns['return_mode'], 'astar_reconnection')

    def test_failed_replan_preserves_cursor_and_pending_target(self):
        ns, calls = self.state()
        ns['path_cells_clear'] = lambda p: False
        ns['planner'].plan = lambda p, t: (None, 'blocked')
        self.plan(ns)
        self.assertEqual((ns['trace_cursor'], ns['trace_target_index'], ns['return_join_index']), (53, 18, 18))
        self.assertIsNone(ns['path'])

    def test_retrace_excludes_already_reached_point(self):
        ns, calls = self.state()
        ns.update(return_joined=True, return_join_index=None, trace_cursor=18,
                  trace_target_index=None)
        ns['data'].qpos[:2] = [1.75, 0]
        self.plan(ns)
        np.testing.assert_allclose(ns['path'][:, 0], [1.75, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1])
        self.assertEqual(ns['trace_cursor'], 18)
        self.assertEqual(ns['trace_target_index'], 11)

    def test_existing_target_survives_a_second_replan(self):
        ns, calls = self.state()
        ns.update(return_joined=True, trace_cursor=30, trace_target_index=25)
        self.plan(ns)
        self.assertEqual(ns['trace_target_index'], 25)

    def test_clear_route_heading_failure_does_not_invent_detour(self):
        ns, calls = self.state()
        ns['nav'].yaw_angle = lambda: 0.
        self.plan(ns)
        self.assertIsNone(ns['path'])
        self.assertEqual(calls, [])

    def test_turn_handoff_uses_turn_entry_heading_not_home_heading(self):
        ns, calls = self.state()
        ns['data'].qpos[:2] = [2.5, 0]
        ns.update(turn_start=10., TURN_TIMEOUT=45., turn_entry_yaw=0.,
                  REJOIN_LOOKAHEAD=35, return_join_index=None, trace_target_index=None,
                  turn_direction=1., turn_previous_yaw=0., turn_angle=0.,
                  TURN_MAX_ANGLE=math.radians(230), turn_progress_time=19., turn_progress_angle=0.)
        ns['nav'].start_yaw = math.pi  # Comparing against HOME would incorrectly see 0 reversal.
        controller_functions({'command_turnaround'}, ns)
        with redirect_stdout(io.StringIO()):
            ns['command_turnaround']()
        chosen = ns['return_join_index']
        self.assertIsNotNone(chosen)
        self.assertEqual(ns['status'], 'replanning')
        self.plan(ns)
        self.assertEqual(ns['trace_target_index'], chosen)
        self.assertEqual(ns['trace_cursor'], 53)

    def test_turn_angle_keeps_increasing_after_180_degrees(self):
        ns, _ = self.state()
        ns['nav'].yaw_angle = lambda: math.radians(-140)
        ns.update(turn_start=10., TURN_TIMEOUT=45., turn_entry_yaw=0., REJOIN_LOOKAHEAD=35,
                  turn_direction=1., turn_previous_yaw=math.radians(-141), turn_angle=math.radians(219),
                  TURN_MAX_ANGLE=math.radians(230), turn_progress_time=19., turn_progress_angle=0.,
                  cached_rays=[], TURN_BODY_BUFFER_M=.55,
                  in_lab=lambda *a, **kw: False)
        # Suppress a join so this test examines turn progress across the wrap boundary.
        ns['path_cells_clear'] = lambda p: False
        controller_functions({'command_turnaround'}, ns)
        ns['command_turnaround']()
        self.assertAlmostEqual(math.degrees(ns['turn_angle']), 220)
        self.assertEqual(ns['status'], 'turn_blocked')

    def test_230_degree_limit_stops_without_looping_again(self):
        ns, _ = self.state()
        ns['nav'].yaw_angle = lambda: math.radians(-129)
        ns.update(turn_start=10., TURN_TIMEOUT=45., turn_direction=1.,
                  turn_previous_yaw=math.radians(-130), turn_angle=math.radians(230),
                  TURN_MAX_ANGLE=math.radians(230))
        controller_functions({'command_turnaround'}, ns)
        with redirect_stdout(io.StringIO()):
            ns['command_turnaround']()
        self.assertEqual(ns['status'], 'turn_blocked')
        np.testing.assert_array_equal(ns['nav'].command, np.zeros(3))


class MotionRegression(unittest.TestCase):
    def test_tighter_turn_cli_schedules_only_four_requested_cases(self):
        calls = []
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def local_path(value):
                value = str(value)
                return root if value == '/tmp' else root/Path(value).name
            def run_case(speed, direction, output, repeat, yaw_rate):
                calls.append((speed, direction, yaw_rate))
                return dict(name=str(len(calls)))
            source = Path(__file__).with_name('measure_g1_motion.py').read_text()
            fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
            ns = dict(__doc__='Test', argparse=argparse, Path=local_path, json=json,
                      datetime=datetime, timezone=timezone, tarfile=tarfile,
                      mujoco=SimpleNamespace(__version__='test'), BOUNDS=(-1.3,11.98,-2.8,2.8),
                      BODY_BUFFER=.55, nav=SimpleNamespace(POLICY='p',CONFIG='c',SCENE='s',__file__='l'),
                      sha256=lambda p:'test', run_case=run_case)
            exec(compile(ast.Module(body=[fn], type_ignores=[]), 'actual_measurement_main', 'exec'), ns)
            with patch('sys.argv', ['measure_g1_motion.py','--speeds','0.32','--yaw-rates','0.40','0.50','--turns-only']):
                with redirect_stdout(io.StringIO()):
                    ns['main']()
            self.assertEqual(calls, [(.32,1,.4),(.32,-1,.4),(.32,1,.5),(.32,-1,.5)])
            self.assertTrue((root/'g1_motion_report_latest.tar.gz').is_file())

    def test_reported_sparse_return_segment_stays_forward(self):
        # Exact path and stop-trigger pose from the user's round-trip report.
        path = np.array([[4.730671665195341, 1.7086381239680926],
                         [1.3906247193626813, -0.06506136040146338]])
        position = np.array([4.298517830871044, 1.5689609741559318])
        yaw = -2.7081228886088153
        tracker = PolylineTracker()
        tracker.advance(path, path[0])
        target = np.asarray(tracker.advance(path, position))
        delta = target-position
        self.assertLess(abs(wrap(math.atan2(delta[1], delta[0])-yaw)), math.radians(25))
        self.assertGreater(tracker.progress, .4)
        self.assertNotEqual(tuple(target), tuple(path[0]))

    def test_actual_controller_does_not_stop_at_reported_failure_pose(self):
        path = np.array([[4.730671665195341, 1.7086381239680926],
                         [1.3906247193626813, -0.06506136040146338]])
        ns = dict(np=np, math=math, nav=SimpleNamespace(command=np.array([.38, 0., .06]),
                  yaw_angle=lambda:-2.7081228886088153, wrap=wrap),
                  data=SimpleNamespace(qpos=np.array([4.298517830871044,1.5689609741559318,.79]),time=48.834),
                  phase='returning', HOME=np.array([0.,0.]), GOAL=np.array([5.,-1.]),
                  in_lab=lambda p:True, path=path, path_kind='home_trace',
                  plan_target=path[-1], status='walking_to_home_trace', TRACE_REACHED_M=.12,
                  path_tracker=PolylineTracker(), path_cells_clear=lambda p:True,
                  motion_clearance=lambda:None, STOP_RANGE=.8, YAW_LIMIT=.3,
                  variable_cruise=lambda e,n:.32, swept_lab_check=lambda *a,**kw:True)
        controller_functions({'command_for_path'}, ns)
        ns['command_for_path']()
        self.assertEqual(ns['status'], 'walking_to_home_trace')
        self.assertGreater(ns['nav'].command[0], 0)

    def test_sparse_and_dense_straight_paths_have_identical_targets(self):
        sparse=np.array([[5.,1.],[1.,-1.]])
        dense=np.linspace(sparse[0],sparse[-1],81)
        a,b=PolylineTracker(),PolylineTracker()
        for position in np.linspace(sparse[0],sparse[-1],60):
            np.testing.assert_allclose(a.advance(sparse,position),b.advance(dense,position),atol=1e-10)

    def test_projection_does_not_jump_to_a_later_crossing(self):
        points=np.array([[0,0],[1,0],[1,1],[0,1],[0,0],[-1,0]],dtype=float)
        tracker=PolylineTracker()
        tracker.advance(points,[0,0])
        self.assertEqual(tracker.progress,0)
        tracker.advance(points,[.4,0])
        before=tracker.progress
        tracker.advance(points,[.35,0])
        self.assertEqual(tracker.progress,before)
        self.assertLess(tracker.progress,1)

    def test_motion_provenance_rejects_a_changed_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            files = {name:root/name for name in ('policy', 'config', 'scene', 'locomotion')}
            for name, p in files.items():
                p.write_text(name)
            evidence = dict(mujoco_version='3.3.5', cases=[
                dict(name='v0.32_left_r1', trial_passed=True, speed_command_mps=.32,
                     yaw_command_rps=.3, circle_fit_diagnostic=dict(radius_m=1.4)),
                dict(name='v0.32_right_r1', trial_passed=True, speed_command_mps=.32,
                     yaw_command_rps=-.3, circle_fit_diagnostic=dict(radius_m=1.5))])
            for name, key in (('policy','policy_sha256'),('config','config_sha256'),
                              ('scene','scene_xml_sha256'),('locomotion','locomotion_source_sha256')):
                evidence[key] = hashlib.sha256(files[name].read_bytes()).hexdigest()
            (root/'motion_screening_summary.json').write_text(json.dumps(evidence))
            ns = dict(Path=Path, __file__=str(root/'controller.py'), json=json, hashlib=hashlib,
                      nav=SimpleNamespace(POLICY=files['policy'], CONFIG=files['config'],
                                          SCENE=files['scene'], __file__=files['locomotion']),
                      mujoco=SimpleNamespace(__version__='3.3.5'), TURN_SPEED_MPS=.32,
                      YAW_LIMIT=.3, TURN_RADII={})
            controller_functions({'verify_motion_evidence'}, ns)
            ns['verify_motion_evidence']()
            self.assertAlmostEqual(ns['TURN_RADII'][1], 1.55)
            files['policy'].write_text('changed policy')
            with self.assertRaisesRegex(RuntimeError, 'policy_sha256'):
                ns['verify_motion_evidence']()

    def test_speed_is_continuous_at_old_thresholds(self):
        for threshold in (1.6, 2.4):
            low = adaptive_speed(0, threshold-1e-5, 5, True)
            high = adaptive_speed(0, threshold+1e-5, 5, True)
            self.assertLess(abs(high-low), 1e-4)
        self.assertEqual(adaptive_speed(0, 0.79, 5, True), 0.)

    def test_slow_speeds_require_explicit_trial_flag(self):
        self.assertEqual(adaptive_speed(math.radians(50), 1.1, .5), .4)
        self.assertLessEqual(adaptive_speed(math.radians(50), 1.1, .5, True), .32)

    def test_failed_quarter_speed_is_excluded_from_navigation(self):
        for heading in np.linspace(0, math.pi/2, 20):
            for clearance in (0.81, 1.0, 1.6, 2.4, None):
                self.assertGreaterEqual(adaptive_speed(heading, clearance, .21, True), .32)

    def test_measured_turn_screens_left_at_current_goal(self):
        ns = dict(np=np, math=math, arc_points=arc_points, arc_margin=arc_margin,
                  TURN_RADII={1:1.398361142713348+.15, -1:1.5135019872916735+.15},
                  TURN_MAX_ANGLE=math.radians(230), TURN_BODY_BUFFER_M=.55,
                  GEOFENCE_X_MIN=-1.3, GEOFENCE_X_MAX=11.98,
                  GEOFENCE_Y_MIN=-2.8, GEOFENCE_Y_MAX=2.8,
                  cached_rays=[dict(range=5, hit=False, angle_deg=0)],
                  nav=SimpleNamespace(wrap=wrap), path_cells_clear=lambda p:True,
                  mapper=SimpleNamespace(max_range=5), TURN_STOP_RANGE=1.35)
        controller_functions({'predicted_turn_arc','select_lab_turn_direction'}, ns)
        with redirect_stdout(io.StringIO()):
            choice = ns['select_lab_turn_direction'](np.array([5., -1.]), -.2)
        self.assertEqual(choice, 1)

    def test_acceleration_and_braking_at_different_step_sizes(self):
        for dt in (.002, .01, .02):
            v = 0.
            for _ in range(round(1/dt)):
                v = slew(v, .4, dt)
            self.assertAlmostEqual(v, .15)
            for _ in range(round(2/dt)):
                v = slew(v, 0., dt)
            self.assertEqual(v, 0.)

    def test_old_estimate_does_not_fit_goal_but_fits_staged_fixture(self):
        bounds = (-1.3, 11.98, -2.8, 2.8)
        goal_arc = arc_points((5, -1), -.2, 1, 1.8)
        staged_arc = arc_points((4, -1.7), 0, 1, 1.8)
        self.assertLess(arc_margin(goal_arc, bounds, .55), 0)
        self.assertGreater(arc_margin(staged_arc, bounds, .55), 0)

    def test_circle_fit_recovers_measured_geometry(self):
        points = arc_points((4, -1.7), .2, -1, 1.37)
        fit = circular_fit(points)
        self.assertAlmostEqual(fit['radius_m'], 1.37, places=8)
        self.assertLess(fit['radial_rmse_m'], 1e-8)

    def test_no_collision_mask_mutations_in_delivered_sources(self):
        for filename in ('g1_navigation_trial.py', 'measure_g1_motion.py'):
            tree = ast.parse(Path(__file__).with_name(filename).read_text())
            for n in ast.walk(tree):
                if isinstance(n, (ast.Assign, ast.AugAssign)):
                    targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                    for target in targets:
                        rendered = ast.unparse(target)
                        self.assertNotIn('geom_contype', rendered)
                        self.assertNotIn('geom_conaffinity', rendered)


if __name__ == '__main__':
    unittest.main(verbosity=2)
