"""响应轨迹与失败分类合同；不启动 MuJoCo。"""
from pathlib import Path
import sys
import unittest
import json
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from confined_dual_continuum.application.response_protocol import ResponsePlan
from confined_dual_continuum.actuation.bending_space import BendingSpaceModel
from confined_dual_continuum.actuation.compatible_tendon_control import CompatibleTendonController, TendonLimits
from confined_dual_continuum.robot.configuration import load_robot_model


class ResponseProtocolTests(unittest.TestCase):
    def test_small_probe_returns_to_zero_and_covers_both_signs(self):
        plan = ResponsePlan(.01, 1., (0,), 1., 1., 1., 1., 1.)
        samples = list(plan.samples())
        self.assertEqual(len(samples), 900)
        self.assertEqual(plan.duration_s, 9.)
        references = np.array([item[3] for item in samples])
        self.assertAlmostEqual(references[:,0].max(), 1.)
        self.assertAlmostEqual(references[:,0].min(), -1.)
        np.testing.assert_array_equal(references[-1], np.zeros(6))
        np.testing.assert_array_equal(references[:,1:], np.zeros((900,5)))
        self.assertLessEqual(np.max(np.abs(np.diff(references, axis=0))), .010000000001)

    def test_invalid_duration_or_axis_rejected(self):
        with self.assertRaises(ValueError):
            ResponsePlan(.01, 1., (0,), 1., .015, 1., 1., 1.)
        with self.assertRaises(ValueError):
            ResponsePlan(.01, 1., (0,0), 1., 1., 1., 1., 1.)

    def test_old_target_lead_excursion_has_specific_cause(self):
        model = BendingSpaceModel(np.array([[1.,0.],[0.,1.],[-1.,-1.]]))
        limits = TendonLimits(np.full(3,-.02), np.full(3,.02), np.full(3,.05), np.full(3,.0005))
        controller = CompatibleTendonController(model, limits)
        first = controller.step(np.array([.05,0.]), np.zeros(3), .01)
        self.assertTrue(first.feasible)
        result = controller.step(np.zeros(2), np.array([-3.57e-7,0.,3.57e-7]), .01, allow_recovery=False)
        self.assertFalse(result.feasible)
        self.assertEqual(result.constraint_status, "previous_target_outside_lead")
        self.assertEqual(result.limiting_tendon_index, 0)
        self.assertAlmostEqual(result.lead_excess_m, 3.57e-7, places=12)
        self.assertEqual(result.displacement_excess_m, 0.)

    def test_lead_recovery_preserves_all_hard_constraints(self):
        model = BendingSpaceModel(np.array([[1.,0.],[0.,1.],[-1.,-1.]]))
        limits = TendonLimits(np.full(3,-.02), np.full(3,.02), np.full(3,.05), np.full(3,.0005))
        controller = CompatibleTendonController(model, limits)
        previous = controller.step(np.array([.05,0.]), np.zeros(3), .01).target_m
        actual = np.array([-3.57e-7,0.,3.57e-7])
        result = controller.step(np.zeros(2), actual, .01)
        self.assertTrue(result.feasible)
        self.assertEqual(result.constraint_status, "recovered_lead")
        self.assertTrue(model.is_compatible(result.target_m))
        self.assertTrue(model.is_compatible(result.tendon_rate_mps))
        np.testing.assert_allclose(previous+.01*result.tendon_rate_mps, result.target_m, atol=1e-12)
        self.assertLessEqual(np.max(np.abs(result.target_m-actual)), .0005+1e-12)
        self.assertLessEqual(np.max(np.abs(result.tendon_rate_mps)), .05+1e-10)

    def test_unrecoverable_rate_interval_stops(self):
        model = BendingSpaceModel(np.eye(2))
        limits = TendonLimits(np.full(2,-.02), np.full(2,.02), np.full(2,.05), np.full(2,.0005))
        controller = CompatibleTendonController(model, limits)
        controller.reset(np.zeros(2))
        result = controller.step(np.zeros(2), np.array([.01,0.]), .01)
        self.assertFalse(result.feasible)
        self.assertEqual(result.constraint_status, "recovery_rate_interval_empty")
        np.testing.assert_array_equal(result.target_m, np.zeros(2))

    def test_recorded_executor_failure_has_a_compatible_recovery(self):
        from dataclasses import replace
        root = Path(__file__).resolve().parents[1]
        fixture = json.loads((root/"tests/fixtures/executor_lead_boundary.json").read_text(encoding="utf-8-sig"))
        robot = load_robot_model(root/"configs/robot/three_segment_nine_tendon.yaml")
        model = robot.bending_model("executor")
        controller = CompatibleTendonController(model, replace(robot.limits, enabled=True))
        previous = np.asarray(fixture["previous_target_m"])
        actual = np.asarray(fixture["actual_m"])
        controller.reset(previous)
        result = controller.step(np.zeros(6), actual, fixture["dt"])
        self.assertTrue(result.feasible)
        self.assertEqual(result.constraint_status, "recovered_lead")
        self.assertTrue(model.is_compatible(result.target_m))
        self.assertTrue(np.all(np.abs(result.target_m-actual) <= robot.limits.maximum_lead_m+1e-12))
        self.assertTrue(np.all(np.abs(result.tendon_rate_mps) <= robot.limits.maximum_rate_mps+1e-10))

    def test_incompatible_intervals_do_not_return_false_success(self):
        model = BendingSpaceModel(np.array([[1.],[-1.]]))
        limits = TendonLimits(np.full(2,-.02), np.full(2,.02), np.full(2,.05), np.full(2,.0005))
        controller = CompatibleTendonController(model, limits)
        controller.reset(np.zeros(2))
        result = controller.step(np.zeros(1), np.full(2,.001), .01)
        self.assertFalse(result.feasible)
        self.assertEqual(result.constraint_status, "recovery_search_exhausted")
        np.testing.assert_array_equal(result.tendon_rate_mps, np.zeros(2))


if __name__ == "__main__":
    unittest.main()
