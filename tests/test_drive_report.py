"""报告不按跟踪误差筛除稳定点，旧协议必须显式分开。"""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from confined_dual_continuum.diagnostics.drive_report import create_report


class DriveReportTests(unittest.TestCase):
    def test_large_error_stable_point_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root/'run'
            run.mkdir()
            summary = dict(protocol_version=5, completed=True, workspace=[dict(scope='mechanical',
                arm='executor', sampled=1, stable=1)], criteria={}, sampling={}, dynamic={})
            (run/'summary.json').write_text(json.dumps(summary), encoding='utf-8')
            row = dict(kind='workspace', scope='mechanical', arm='executor', sample=0,
                status='stable', path_completed=True, T_root_tip=np.eye(4).tolist(),
                comparison={'command': {'tip_error_m': .02}},
                legs=[dict(leg=0, status='stable', sent_s=.01, stable_s=1., duration_s=1.,
                           planned_motion_s=.01, planned_peak_speed_mps=None, planned_cruise_s=None)])
            (run/'samples.jsonl').write_text(json.dumps(row)+'\n', encoding='utf-8')
            report = create_report([run], root/'report')
            self.assertNotIn('stable_tracked', report.read_text(encoding='utf-8'))
            stored = json.loads((report.parent/'samples.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(stored['comparison']['command']['tip_error_m'], .02)
            self.assertEqual(stored['status'], 'stable')

    def test_old_protocol_is_not_silently_reinterpreted(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run/'summary.json').write_text(json.dumps({'protocol_version': 4}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '协议 5'):
                create_report([run], run/'report')


if __name__ == '__main__':
    unittest.main()
