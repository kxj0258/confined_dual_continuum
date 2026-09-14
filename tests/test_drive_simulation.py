"""仿真驱动的少量跨层合同；不启动 viewer。"""
import io
from pathlib import Path
import unittest
import numpy as np
import yaml
from confined_dual_continuum.adapters.mujoco_backend import MujocoRobotBackend
from confined_dual_continuum.robot.configuration import load_robot_model
from confined_dual_continuum.drive_workspace_probe import collision_scope, run_path, _self_contact
from confined_dual_continuum.coordination.interarm_safety import InterArmSafety


class DriveSimulationTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.robot = load_robot_model(root/'configs/robot/three_segment_nine_tendon.yaml')
        self.backend = MujocoRobotBackend(root/'assets/mujoco/multi_task_patent_demo.xml', self.robot)
        self.config = yaml.safe_load((root/'configs/experiments/drive_workspace_probe.yaml').read_text(encoding='utf-8'))['criteria']

    def tearDown(self):
        self.backend.stop()
        self.backend.close()

    def test_neutral_has_no_artificial_self_contact(self):
        self.backend.reset()
        for arm in ('executor', 'observer'):
            self.assertFalse(_self_contact(self.backend, arm))

    def test_native_capsule_gap_matches_centerline_safety(self):
        b = self.backend
        rng = np.random.default_rng(7)
        for _ in range(8):
            b.data.qpos[:] = rng.uniform(-.12, .12, b.model.nq)
            b._mj.mj_forward(b.model, b.data)
            expected = InterArmSafety().evaluate(b._physical_centerline('executor'), b._physical_centerline('observer')).distance_m
            actual = min(b._mj.mj_geomDistance(b.model, b.data, a, c, .05, None)
                         for a in b._collision_geoms['executor'] for c in b._collision_geoms['observer'])
            self.assertAlmostEqual(actual, expected, places=8)

    def test_collision_scope_restores_even_after_exception(self):
        types = self.backend.model.geom_contype.copy()
        affinity = self.backend.model.geom_conaffinity.copy()
        with self.assertRaises(RuntimeError):
            with collision_scope(self.backend, 'executor', 'mechanical'):
                self.assertFalse(any(self.backend.model.geom_contype[self.backend._collision_geoms['observer']]))
                raise RuntimeError('test')
        np.testing.assert_array_equal(self.backend.model.geom_contype, types)
        np.testing.assert_array_equal(self.backend.model.geom_conaffinity, affinity)

    def test_zero_reference_collects_stable_pose_without_tracking_verdict(self):
        with collision_scope(self.backend, 'executor', 'mechanical'):
            row = run_path(self.backend, self.robot, 'executor', 'mechanical', [np.zeros(6)], .001, self.config, io.StringIO())
        self.assertTrue(row['path_completed'])
        self.assertEqual(row['status'], 'stable')
        self.assertIsNotNone(row['legs'][0]['stable_s'])
        self.assertNotIn('speed_verdict', row)
        self.assertAlmostEqual(row['comparison']['command']['tip_error_m'], 0., places=9)

    def test_worker_reuse_does_not_leak_previous_path_or_collision_scope(self):
        robot, backend = self.robot, self.backend
        q = np.zeros(6)
        q[4] = .1
        with collision_scope(backend, 'executor', 'mechanical'):
            first = run_path(backend, robot, 'executor', 'mechanical', [q], .001, self.config, io.StringIO(), command_profile='trapezoid', acceleration_time=.5)
        with collision_scope(backend, 'observer', 'scene'):
            run_path(backend, robot, 'observer', 'scene', [np.zeros(6)], .001, self.config, io.StringIO())
        with collision_scope(backend, 'executor', 'mechanical'):
            second = run_path(backend, robot, 'executor', 'mechanical', [q], .001, self.config, io.StringIO(), command_profile='trapezoid', acceleration_time=.5)
        np.testing.assert_allclose(first['T_root_tip'], second['T_root_tip'], atol=1e-12, rtol=0)


if __name__ == '__main__':
    unittest.main()
