"""只覆盖事件独立性、相容绳速及验收不冒充原请求达标的合同。"""
import unittest
import numpy as np
from confined_dual_continuum.actuation.bending_space import BendingSpaceModel
from confined_dual_continuum.diagnostics.drive_metrics import DriveEvents, TrapezoidProfile, tracking_statistics
from confined_dual_continuum.drive_workspace_probe import dynamic_targets, workspace_targets


class DriveMetricsTests(unittest.TestCase):
    def test_disabled_limits_pass_large_command_despite_feedback_drift(self):
        from confined_dual_continuum.actuation.compatible_tendon_control import CompatibleTendonController, TendonLimits
        from confined_dual_continuum.actuation.independent_tendon_control import IndependentTendonController
        model = BendingSpaceModel(np.array([[1.], [-1.]]))
        limits = TendonLimits(np.full(2, -.02), np.full(2, .02), np.full(2, .05), np.full(2, .0005), enabled=False)
        for controller_type in (CompatibleTendonController, IndependentTendonController):
            controller = controller_type(model, limits)
            controller.reset(np.zeros(2))
            result = controller.step(np.array([10.]), np.array([.3, .3]), .01)
            np.testing.assert_allclose(result.tendon_rate_mps, [10., -10.])
            np.testing.assert_allclose(result.target_m, [.1, -.1])
            self.assertEqual(result.constraint_status, 'limits_disabled')
            self.assertEqual(result.common_scale, 1.)
            self.assertTrue(result.feasible)

    def test_stability_must_remain_continuous_through_dwell(self):
        e = DriveEvents(.2, .3)
        for i in range(4):
            e.update((i+1)*.1, .1, True, True)
        self.assertFalse(e.ready)
        e.update(.5, .1, True, False)
        for i in range(4):
            e.update(.6+i*.1, .1, True, True)
        self.assertFalse(e.ready)
        e.update(1., .1, True, True)
        self.assertTrue(e.ready)

    def test_trapezoid_and_triangle_integrate_without_overshoot(self):
        model = BendingSpaceModel(np.array([[.01, 0], [0, .02], [-.01, -.02]]))
        for speed in (.01, .5):
            start, goal, dt = np.zeros(2), np.array([1., 2.]), .001
            delta = goal-start
            distance = max(abs(model.to_tendon(delta)))
            profile = TrapezoidProfile(distance, speed, .1)
            times = np.arange(0., profile.duration_s+2*dt, dt)
            positions = np.array([profile.sample(t)[0] for t in times])
            rates = np.diff(positions)/dt
            self.assertTrue(np.all(np.diff(positions) >= -1e-12))
            self.assertLessEqual(max(rates), speed+1e-10)
            self.assertLessEqual(max(abs(np.diff(np.r_[0., rates, 0.])/dt)), .1+1e-8)
            q = start.copy()
            for ds in np.diff(positions):
                q += delta*ds/distance
                self.assertTrue(model.is_compatible(model.to_tendon(delta*ds/distance)))
            np.testing.assert_allclose(q, goal, atol=1e-10)
            self.assertEqual(profile.sample(profile.duration_s), (distance, 0., 0., 'hold'))
        self.assertEqual(TrapezoidProfile(.001, 1., .01).cruise_time_s, 0.)

    def test_statistics_separate_motion_from_hold_without_verdict(self):
        def record(phase, actual):
            return dict(phase=phase, actual_velocity_mps=[actual], sent_velocity_mps=[0.],
                        actual_acceleration_mps2=[0.], sent_acceleration_mps2=[0.])
        stats = tracking_statistics([record('cruise', 1.)]+[record('hold', 0.)]*100)
        self.assertEqual(stats['cruise']['velocity']['rmse'], [1.])
        self.assertNotIn('passed', stats)

    def test_sampling_is_reproducible_and_bounded(self):
        config = dict(seed=9, axes=[0,1,2,3,4,5], amplitude_per_m=6., combination_samples=4)
        a, b = list(dynamic_targets(config)), list(dynamic_targets(config))
        self.assertEqual(len(a),10)
        np.testing.assert_array_equal(a,b)
        self.assertTrue(np.all(np.linalg.norm(np.array(a).reshape(-1,3,2),axis=2) <= 6.+1e-12))
        w = dict(seed=9, amplitudes_per_m=[3.,6.], combination_samples=4, boundary_samples=8)
        points = list(workspace_targets(w))
        self.assertEqual(len(points), 1+24+4+8)
        np.testing.assert_allclose(np.linalg.norm(np.array(points[-8:]).reshape(-1,3,2),axis=2),6.)


if __name__ == "__main__":
    unittest.main()
