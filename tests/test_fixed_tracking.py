"""固定基座单向协同核心合同；由用户手动执行。"""
import unittest
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import replace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from confined_dual_continuum.domain.camera import CameraModel
from confined_dual_continuum.task.timed_trajectory import TimedTrajectory
from confined_dual_continuum.kinematics.constrained_velocity import constrained_velocity
from confined_dual_continuum.application.fixed_tracking import FixedTrackingController, TrackingConfig
from confined_dual_continuum.domain.geometry import Pose
from confined_dual_continuum.domain.state import ArmState, RobotState
from confined_dual_continuum.domain.tracking import TrackingGeometry, TrackingClearance
from confined_dual_continuum.robot.configuration import load_robot_model
from confined_dual_continuum.adapters.tracking_backend import build_tracking_scene
from confined_dual_continuum.coordination.camera_follower import camera_differential
from confined_dual_continuum.coordination.observation_candidates import select_observation_candidate
from confined_dual_continuum.domain.tracking import ObservationCandidate
from confined_dual_continuum.kinematics.pcc import forward_kinematics


class GeometryContracts(unittest.TestCase):
    def test_sample_selection_uses_pose_not_position_box(self):
        camera = CameraModel(70, 640, 480, .01, .2, .1, .001)
        forward = np.eye(4)
        backward = np.diag([-1., 1., -1., 1.])
        candidates = (ObservationCandidate(1, np.zeros(6), backward),
                      ObservationCandidate(2, np.ones(6), forward))
        chosen = select_observation_candidate(candidates, np.eye(4), np.eye(4),
                                              np.array([0., 0., .05]), np.zeros(6), camera, .05)
        self.assertEqual(chosen.sample, 2)

    def test_camera_differential_honors_actual_mount_offset(self):
        lengths, offsets = (.0365,)*3, (.0035,)*3
        q = np.zeros(6)
        tip = forward_kinematics(q, lengths, offsets).tip_transform
        camera = tip.copy()
        camera[:3, 3] += [.003, 0., .002]
        target = camera[:3, 3]+[0., 0., .05]
        point, jacobian, position = camera_differential(q, lengths, offsets, tip, camera, target)
        np.testing.assert_allclose(point, [0., 0., .05])
        self.assertEqual(jacobian.shape, (3, 6))
        _, _, no_offset = camera_differential(q, lengths, offsets, tip, tip, target)
        self.assertGreater(np.linalg.norm(position-no_offset), 1e-6)

    def test_scene_keeps_physical_interarm_collision_and_source(self):
        source = ROOT/'assets/mujoco/multi_task_patent_demo.xml'
        original = source.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory)/'scene.xml'
            build_tracking_scene(source, scene, [dict(type='sphere', position_base_m=[.02, -.03, .105],
                                                     size_m=[.006], name='test')])
            root = ET.parse(scene).getroot()
            self.assertIsNotNone(root.find(".//camera[@name='observer_camera']"))
            self.assertIsNotNone(root.find(".//geom[@name='tracking_test']"))
            for arm in ('executor', 'observer'):
                geom = root.find(f".//geom[@name='{arm}_segment_1_link_1_collision']")
                self.assertEqual(geom.get('contype'), '1')
                self.assertEqual(geom.get('conaffinity'), '1')
            for pair in root.findall('./contact/exclude'):
                self.assertEqual(pair.get('body1').split('_')[0], pair.get('body2').split('_')[0])
        self.assertEqual(source.read_bytes(), original)

    def test_observer_fault_does_not_change_executor_command_or_target(self):
        robot = load_robot_model(ROOT / 'configs/robot/three_segment_nine_tendon.yaml')
        def arm(y):
            return ArmState(Pose(np.array([0., y, .14])), np.zeros(6), np.zeros(9),
                            np.array([[0., y, .02], [0., y, .14]]), np.zeros(3))
        state = RobotState(Pose(np.zeros(3)), arm(.01), arm(-.01))
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = [0., -.01, .142]
        geometry = TrackingGeometry(camera_pose, np.eye(4), np.eye(4), (), False,
                                    {'executor': 1., 'observer': 1.})
        bad = replace(geometry, occluded=True, joint_margin_rad={'executor': 1., 'observer': 0.},
                      clearances=(TrackingClearance('observer', .12, -.001,
                                                   np.array([1., 0., 0.])),))
        controllers = [FixedTrackingController(robot, CameraModel(70, 640, 480, .01, .2, .1, .001),
                                               TrackingConfig()) for _ in range(2)]
        for c in controllers:
            c.reset(state)
        for t in (0., .01, .02):
            outputs = [c.step(state, g, [.001+t*.001, .01, .14], [.001, 0., 0.], .01, t)
                       for c, g in zip(controllers, (geometry, bad))]
            np.testing.assert_array_equal(outputs[0].command.executor.tendon_target_m,
                                          outputs[1].command.executor.tendon_target_m)
            np.testing.assert_array_equal(outputs[0].command.executor.bending_rate,
                                          outputs[1].command.executor.bending_rate)
            self.assertEqual(outputs[1].diagnostics['observer_status'], 'observer_joint_guard')
            for output in outputs:
                np.testing.assert_array_equal(output.command.base_twist_world, np.zeros(6))
                for name in ('executor', 'observer'):
                    self.assertTrue(robot.bending_model(name).is_compatible(
                        getattr(output.command, name).tendon_rate_mps))
        before = controllers[0].targets['executor'].copy()
        controllers[0].observe(state, geometry, np.array([.1, .01, .14]), .03)
        np.testing.assert_array_equal(controllers[0].targets['executor'], before)

    def test_constraint_projection_and_infeasible_result(self):
        rate, feasible = constrained_velocity(np.eye(2), np.zeros(2),
                                               [[1., 0.]], [1.])
        self.assertTrue(feasible)
        np.testing.assert_allclose(rate, [1., 0.], atol=1e-7)
        _, feasible = constrained_velocity(np.eye(2), np.zeros(2),
                                          [[1., 0.], [-1., 0.]], [1., 1.])
        self.assertFalse(feasible)

    def test_trajectory_clock_and_endpoints(self):
        path = TimedTrajectory([0., 2., 4.], [[0., 0., 0.], [1., 0., 0.], [1., 1., 0.]])
        np.testing.assert_allclose(path.sample(0.)[0], [0., 0., 0.])
        np.testing.assert_allclose(path.sample(2.)[0], [1., 0., 0.])
        np.testing.assert_allclose(path.sample(4.)[1], 0.)
        np.testing.assert_allclose(path.sample(1.)[0], [.5, 0., 0.])
        with self.assertRaises(ValueError):
            TimedTrajectory([0., 0.], [[0., 0., 0.], [1., 0., 0.]])

    def test_region_visibility_rejects_behind_edge_and_occlusion(self):
        camera = CameraModel(70., 640, 480, .01, .2, .1, .001)
        self.assertTrue(camera.visible([0., 0., .05], False))
        self.assertFalse(camera.visible([0., 0., -.05], False))
        self.assertFalse(camera.visible([.1, 0., .05], False))
        self.assertFalse(camera.visible([0., 0., .05], True))
        # 中心尚在边界内，但半径覆盖区域已越界。
        edge = .05 * camera.slopes[0]
        self.assertFalse(camera.visible([edge-.0001, 0., .05], False))


if __name__ == '__main__':
    unittest.main()
