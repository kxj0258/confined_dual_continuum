"""独立主臂核心合同；手动运行，不启动动力学或窗口。"""
from pathlib import Path
import sys
import unittest
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import replace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from confined_dual_continuum.kinematics.fast_pcc import tip_position_jacobian
from confined_dual_continuum.kinematics.pcc import material_point, position_jacobian
from confined_dual_continuum.application.executor_tracking import ExecutorTrackingController
from confined_dual_continuum.domain.executor_tracking import ExecutorFeedback
from confined_dual_continuum.robot.configuration import load_robot_model
from confined_dual_continuum.adapters.executor_backend import build_executor_scene
from confined_dual_continuum.diagnostics.executor_records import RunRecords, COLUMNS


class ExecutorContracts(unittest.TestCase):
    def test_error_record_uses_same_time_reference_and_feedback(self):
        records = RunRecords()
        state = ExecutorFeedback(np.array([.001, .002, .003]), np.zeros(9), .2)
        records.append(.02, np.array([.004, .006, .003]), state, control_time_s=0.)
        rows, metrics = records.snapshot()
        self.assertEqual(len(rows[0]), len(COLUMNS))
        self.assertAlmostEqual(rows[0][COLUMNS.index('error_norm_m')], .005)
        self.assertEqual(rows[0][COLUMNS.index('time_s')], .02)
        self.assertEqual(rows[0][COLUMNS.index('control_time_s')], 0.)
        self.assertAlmostEqual(metrics['rmse_m'], .005)

    def test_scene_removes_observer_without_changing_executor_physics(self):
        source = ROOT/'assets/mujoco/multi_task_patent_demo.xml'
        original = source.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'executor.xml'
            build_executor_scene(source, path)
            tree = ET.parse(path).getroot()
            self.assertFalse(any('observer' in value for element in tree.iter()
                                 for value in element.attrib.values()))
            self.assertEqual(len(tree.findall('./actuator/*')), 9)
            self.assertEqual(tree.find('./option').get('timestep'), '0.001')
            self.assertIsNotNone(tree.find(".//site[@name='executor_tip']"))
            self.assertEqual(tree.find(".//geom[@name='executor_segment_1_link_1_collision']").get('contype'), '1')
            self.assertEqual(len(tree.findall('./contact/exclude')), 10)
        self.assertEqual(source.read_bytes(), original)

    def test_analytic_jacobian_matches_independent_difference(self):
        lengths, offsets = (.0365,)*3, (.0035,)*3
        cases = (np.zeros(6), np.full(6, 1e-7), np.array([2., -3., 0., 0., -6., 4.]),
                 np.array([12., -3., -8., 9., 5., -10.]))
        for q in cases:
            with self.subTest(q=q):
                point, jac = tip_position_jacobian(q, lengths, offsets)
                np.testing.assert_allclose(point, material_point(q, lengths, offsets), atol=1e-12)
                np.testing.assert_allclose(jac, position_jacobian(q, lengths, offsets),
                                           atol=2e-9, rtol=1e-5)

    def test_public_limits_are_used_without_task_override(self):
        robot = load_robot_model(ROOT/'configs/robot/three_segment_nine_tendon.yaml')
        robot = replace(robot, limits=replace(robot.limits,
                        maximum_rate_mps=np.full(9, .0037), rate_enabled=True))
        controller = ExecutorTrackingController(robot, np.eye(3))
        self.assertIs(controller.actuation.limits, robot.limits)
        state = ExecutorFeedback(np.array([0., 0., .12]), np.zeros(9), .2)
        controller.reset(state)
        result = controller.step(state, [.1, 0., .12], [0., 0., 0.], .02)
        self.assertTrue(result.step.feasible)
        self.assertAlmostEqual(float(np.max(np.abs(result.command.tendon_rate_mps))), .0037, places=10)
        self.assertTrue(controller.actuation.model.is_compatible(result.command.tendon_rate_mps))

    def test_joint_guard_does_not_integrate(self):
        robot = load_robot_model(ROOT/'configs/robot/three_segment_nine_tendon.yaml')
        controller = ExecutorTrackingController(robot, np.eye(3))
        state = ExecutorFeedback(np.array([0., 0., .12]), np.zeros(9), 0.)
        controller.reset(state)
        result = controller.step(state, [.1, 0., .12], [0., 0., 0.], .02)
        self.assertEqual(result.status, 'joint_guard')
        np.testing.assert_array_equal(result.command.tendon_rate_mps, np.zeros(9))
        np.testing.assert_array_equal(result.command.tendon_target_m, np.zeros(9))


if __name__ == '__main__':
    unittest.main()
