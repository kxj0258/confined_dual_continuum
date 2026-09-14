"""只检查轨迹与采集点云合同，不启动动力学或窗口。"""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np
from confined_dual_continuum.task.catalog_trajectory import CatalogTrajectory
from confined_dual_continuum.trajectory_catalog import DEFAULT_DIRECTORY, load_approved
from confined_dual_continuum.workspace.pointcloud_coverage import check_coverage


class PointcloudCatalogTests(unittest.TestCase):
    def test_builtin_curves_and_entry_are_covered(self):
        policy = json.loads((DEFAULT_DIRECTORY/'workspace.json').read_text(encoding='utf-8'))
        with np.load(DEFAULT_DIRECTORY/'workspace.npz') as archive:
            cloud = archive['points']
        for name in ('circle', 'figure8', 'rectangle'):
            with self.subTest(name=name):
                definition, _ = load_approved(DEFAULT_DIRECTORY/f'{name}.yaml', DEFAULT_DIRECTORY)
                trajectory = CatalogTrajectory(definition)
                self.assertTrue(check_coverage(trajectory, cloud, policy)['passed'])
                np.testing.assert_allclose(trajectory.sample(0)[0], policy['initial_tip_base_m'])
                np.testing.assert_allclose(trajectory.sample(trajectory.entry_s)[0], trajectory.point(0))
                np.testing.assert_allclose(trajectory.sample(trajectory.duration_s)[0], trajectory.point(0), atol=1e-12)
                outside = dict(definition, center_m=(trajectory.center+1).tolist(),
                               entry_points_m=(trajectory.entry.points+1).tolist())
                self.assertFalse(check_coverage(CatalogTrajectory(outside), cloud, policy)['passed'])

    def test_edited_definition_invalidates_approval(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ('workspace.json', 'workspace.npz', 'circle.yaml', 'circle.coverage.json'):
                shutil.copy2(DEFAULT_DIRECTORY/name, directory/name)
            path = directory/'circle.yaml'
            path.write_text(path.read_text(encoding='utf-8')+'\n# edited\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                load_approved(path, directory)


if __name__ == '__main__':
    unittest.main()
