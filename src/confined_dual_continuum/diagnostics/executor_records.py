"""主臂全频率轻量记录；窗口只复制有限历史，运行结束批量写盘。"""
import csv
from threading import Lock
import numpy as np

COLUMNS = ('time_s', 'control_time_s', 'command_applied',
    'reference_x_m', 'reference_y_m', 'reference_z_m',
    'actual_x_m', 'actual_y_m', 'actual_z_m', 'error_x_m', 'error_y_m', 'error_z_m', 'error_norm_m',
    'control_wall_s', 'physics_wall_s', 'feedback_wall_s', 'display_wall_s',
    'saturated', 'common_scale', 'compatibility_residual', 'joint_margin_rad',
    'jacobian_sigma_min', 'jacobian_condition_number') + tuple(
    f'{kind}_{i}'
    for kind, count in (
        ('tendon_command_mps', 9), ('tendon_target_m', 9), ('tendon_actual_m', 9),
        ('raw_bending_rate', 6), ('applied_bending_rate', 6),
        ('raw_tendon_rate_mps', 9), ('applied_tendon_rate_mps', 9),
        ('curvature_norm', 3), ('requested_curvature_norm', 3),
        ('applied_curvature_norm', 3), ('curvature_margin', 3),
    ) for i in range(1, count + 1)) + tuple(f'jacobian_sigma_{i}' for i in range(1, 4))


class RunRecords:
    def __init__(self):
        self.lock = Lock()
        self.rows = []
        self.status = '初始化'
        self.error_squared = self.maximum_error = 0.
        self.saturated = self.commands = 0
        self.rtf = 0.
        self.interarm_rows = []

    def set_interarm_clearance(self, time_s, clearance_m):
        if clearance_m is not None and np.isfinite(clearance_m):
            with self.lock:
                self.interarm_rows.append((float(time_s), float(clearance_m)))

    def interarm_snapshot(self):
        with self.lock:
            return list(self.interarm_rows[-5000:])
        self.interarm_rows = []

    def set_interarm_clearance(self, time_s, clearance_m):
        if clearance_m is not None and np.isfinite(clearance_m):
            with self.lock:
                self.interarm_rows.append((float(time_s), float(clearance_m)))

    def interarm_snapshot(self):
        with self.lock:
            return list(self.interarm_rows[-5000:])

    def set_status(self, status, rtf=None):
        with self.lock:
            self.status = str(status)
            if rtf is not None:
                self.rtf = float(rtf)

    def append(self, time_s, reference, feedback, result=None, timings=(0., 0., 0., 0.),
               control_time_s=None, command_applied=False):
        error = np.asarray(reference)-feedback.tip_world_m
        norm = float(np.linalg.norm(error))
        step = result.step if result else None
        diagnostics = result.diagnostics if result else None
        rate = result.command.tendon_rate_mps if result else np.zeros(9)
        target = result.command.tendon_target_m if result else np.zeros(9)
        raw_bending = diagnostics['raw_bending_rate'] if diagnostics else np.zeros(6)
        applied_bending = result.command.bending_rate if result else np.zeros(6)
        raw_tendon = diagnostics['raw_tendon_rate_mps'] if diagnostics else np.zeros(9)
        current_curvature = diagnostics['curvature_norm'] if diagnostics else np.zeros(3)
        requested_curvature = diagnostics['requested_curvature_norm'] if diagnostics else np.zeros(3)
        applied_curvature = diagnostics['applied_curvature_norm'] if diagnostics else np.zeros(3)
        curvature_margin = diagnostics['curvature_margin'] if diagnostics else np.zeros(3)
        sigma = diagnostics['jacobian_singular_values'] if diagnostics else np.zeros(3)
        sigma_min = diagnostics['jacobian_sigma_min'] if diagnostics else 0.
        condition = diagnostics['jacobian_condition_number'] if diagnostics else 0.
        row = (float(time_s), float(time_s if control_time_s is None else control_time_s), int(command_applied),
               *reference, *feedback.tip_world_m, *error, norm, *timings,
               int(step.saturated) if step else 0, float(step.common_scale) if step else 1.,
               float(step.compatibility_residual_norm) if step else 0., feedback.minimum_joint_margin_rad,
               sigma_min, condition, *rate, *target, *feedback.tendon_displacement_m,
               *raw_bending, *applied_bending, *raw_tendon, *rate,
               *current_curvature, *requested_curvature, *applied_curvature,
               *curvature_margin, *sigma)
        row = tuple(float(v) for v in row)
        with self.lock:
            self.rows.append(row)
            self.error_squared += norm*norm
            self.maximum_error = max(self.maximum_error, norm)
            self.commands += int(command_applied)
            self.saturated += int(command_applied and step is not None and step.saturated)

    def snapshot(self, count=5000):
        with self.lock:
            return (self.rows[-count:], dict(status=self.status, realtime_factor=self.rtf,
                rmse_m=np.sqrt(self.error_squared/len(self.rows)) if self.rows else 0.,
                maximum_error_m=self.maximum_error,
                saturation_fraction=self.saturated/self.commands if self.commands else 0.))

    def save(self, output):
        # 由仿真工作线程在结束后调用，此时不会再追加记录。
        with (output/'tracking.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.writer(stream)
            writer.writerow(COLUMNS)
            writer.writerows(self.rows)

    def metrics(self):
        _, summary = self.snapshot(1)
        summary.pop('status')
        array = np.asarray(self.rows)
        summary.update(samples=len(self.rows), commands=self.commands,
                       simulation_time_s=float(array[-1, 0]) if len(array) else 0.)
        if len(array):
            summary['final_error_m'] = float(array[-1, COLUMNS.index('error_norm_m')])
            summary['maximum_compatibility_residual'] = float(np.max(array[:, COLUMNS.index('compatibility_residual')]))
            for name in ('control_wall_s', 'physics_wall_s', 'feedback_wall_s', 'display_wall_s'):
                values = array[array[:, 2] > 0, COLUMNS.index(name)]
                summary[name] = {'mean': float(np.mean(values)) if len(values) else None,
                                 'p95': float(np.percentile(values, 95)) if len(values) else None,
                                 'maximum': float(np.max(values)) if len(values) else None}
        return summary
