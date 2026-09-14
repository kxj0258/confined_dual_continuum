"""双臂固定基座；主从无接触响应，碰撞包络仍用于独立距离查询。"""
import xml.etree.ElementTree as ET
import numpy as np
from confined_dual_continuum.adapters.robot_only_scene import robot_only_scene
from confined_dual_continuum.adapters.tracking_backend import TrackingBackend
from confined_dual_continuum.domain.executor_tracking import ExecutorFeedback


def build_scene(source, destination):
    robot_only_scene(source, destination)
    tree = ET.parse(destination)
    # 两臂使用不相交接触位，保留各自原有自碰撞和 exclude。
    for geom in tree.getroot().iter('geom'):
        name = geom.get('name', '')
        if name.endswith('_collision'):
            for arm, bit in (('executor', '1'), ('observer', '2')):
                if name.startswith(arm+'_'):
                    geom.set('contype', bit)
                    geom.set('conaffinity', bit)
    tree.write(destination, encoding='utf-8', xml_declaration=True)


class DualTrackingBackend(TrackingBackend):
    def margin(self, arm):
        joints = np.asarray(self._joint_ids[arm])
        joints = joints[self.model.jnt_limited[joints].astype(bool)]
        p = self.data.qpos[self.model.jnt_qposadr[joints]]
        bounds = self.model.jnt_range[joints]
        return float(np.min(np.minimum(p-bounds[:, 0], bounds[:, 1]-p)))

    def executor_feedback(self):
        return ExecutorFeedback(self._site_transform('executor_tip')[:3, 3],
            self.data.ten_length[self._tendon_ids['executor']]-self._neutral['executor'],
            self.margin('executor'))

    def snapshot(self):
        return {name: getattr(self.data, name).copy() for name in
                ('qpos', 'qvel', 'act', 'ctrl', 'mocap_pos', 'mocap_quat')} | {'time': float(self.data.time)}

    def restore(self, snapshot):
        for name, value in snapshot.items():
            if name == 'time':
                self.data.time = value
            else:
                getattr(self.data, name)[:] = value
        self._mj.mj_forward(self.model, self.data)

    def advance(self, command, dt, on_frame=None, video_stride=None):
        steps = int(round(dt/self.model.opt.timestep))
        if steps < 1 or not np.isclose(steps*self.model.opt.timestep, dt, atol=1e-12, rtol=0):
            raise ValueError('控制周期必须为物理步长整数倍。')
        if np.any(command.base_twist_world):
            raise ValueError('本程序禁止基座运动。')
        for arm in ('executor', 'observer'):
            cmd = getattr(command, arm)
            expected = self._targets[arm]-self._neutral[arm]+dt*cmd.tendon_rate_mps
            if not np.allclose(expected, cmd.tendon_target_m, atol=1e-10, rtol=0):
                raise ValueError(f'{arm} 积分目标不一致。')
        for _ in range(steps):
            for arm in ('executor', 'observer'):
                self._targets[arm] += self.model.opt.timestep*getattr(command, arm).tendon_rate_mps
            self._write_targets()
            self._mj.mj_step(self.model, self.data)
            tick = int(round(self.data.time/self.model.opt.timestep))
            if on_frame is not None and tick % video_stride == 0:
                on_frame(self.snapshot())
        self._mj.mj_forward(self.model, self.data)
