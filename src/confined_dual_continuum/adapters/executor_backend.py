"""固定基座单主臂场景和最小反馈/执行接口，不含轨迹控制算法。"""
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from confined_dual_continuum.adapters.robot_only_scene import robot_only_scene
from confined_dual_continuum.domain.executor_tracking import ExecutorFeedback


def build_executor_scene(source: Path, destination: Path):
    robot_only_scene(source, destination)
    tree = ET.parse(destination)
    # 同时移除从臂实体及引用它的肌腱、执行器、传感器、排除对和网格资源。
    def remove_observer(parent):
        for child in list(parent):
            if any('observer' in value for value in child.attrib.values()):
                parent.remove(child)
            else:
                remove_observer(child)
    remove_observer(tree.getroot())
    tree.write(destination, encoding='utf-8', xml_declaration=True)


@dataclass(frozen=True)
class VideoState:
    time_s: float
    qpos: np.ndarray
    qvel: np.ndarray


class ExecutorBackend:
    def __init__(self, xml_path, *, video_fps=None):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.timestep = float(self.model.opt.timestep)
        self.ids = {}
        for key, kind, names in (
            ('tendons', mujoco.mjtObj.mjOBJ_TENDON, [f'executor_tendon_{i}' for i in range(1, 10)]),
            ('actuators', mujoco.mjtObj.mjOBJ_ACTUATOR, [f'act_executor_tendon_{i}' for i in range(1, 10)]),
            ('sites', mujoco.mjtObj.mjOBJ_SITE, ['mobile_base_frame', 'executor_base_site', 'executor_tip'])):
            indices = [mujoco.mj_name2id(self.model, kind, name) for name in names]
            if min(indices) < 0:
                raise ValueError(f'单臂场景缺少 {key} 命名对象。')
            self.ids[key] = np.asarray(indices)
        self.joints = np.flatnonzero(self.model.jnt_limited)
        if not len(self.joints) or self.model.nu != 9:
            raise ValueError('单臂场景要求九个执行器和有限位的主臂关节。')
        self.qpos_indices = self.model.jnt_qposadr[self.joints]
        self.video_stride = None
        if video_fps is not None:
            stride = 1/(float(video_fps)*self.timestep)
            if not np.isfinite(stride) or stride < 1 or not np.isclose(stride, round(stride), atol=1e-9):
                raise ValueError('视频帧周期必须为物理步长整数倍。')
            self.video_stride = int(round(stride))
        self.reset()

    def reset(self):
        self.mj.mj_resetData(self.model, self.data)
        self.mj.mj_forward(self.model, self.data)
        self.neutral = self.data.ten_length[self.ids['tendons']].copy()
        self.target = np.zeros(9)
        self.data.ctrl[self.ids['actuators']] = self.neutral
        self.mj.mj_forward(self.model, self.data)
        self.physical_steps = 0
        self.base_transform = self.site_transform(0)
        self.root_transform = self.site_transform(1)

    def site_transform(self, index):
        site = self.ids['sites'][index]
        transform = np.eye(4)
        transform[:3, :3] = self.data.site_xmat[site].reshape(3, 3)
        transform[:3, 3] = self.data.site_xpos[site]
        return transform

    def read_feedback(self):
        position = self.data.qpos[self.qpos_indices]
        bounds = self.model.jnt_range[self.joints]
        margin = float(np.min(np.minimum(position-bounds[:, 0], bounds[:, 1]-position)))
        tip = self.data.site_xpos[self.ids['sites'][2]].copy()
        tendon = self.data.ten_length[self.ids['tendons']].copy()-self.neutral
        if not np.isfinite(margin) or not np.all(np.isfinite(tip)) or not np.all(np.isfinite(tendon)):
            raise FloatingPointError('MuJoCo 主臂反馈含非有限数值。')
        return ExecutorFeedback(tip, tendon, margin)

    def snapshot(self):
        return VideoState(self.physical_steps*self.timestep, self.data.qpos.copy(), self.data.qvel.copy())

    def step(self, command, dt):
        steps = int(round(dt/self.timestep))
        if steps < 1 or not np.isclose(steps*self.timestep, dt, atol=1e-12, rtol=0):
            raise ValueError('控制周期必须是物理步长整数倍。')
        expected = self.target+dt*command.tendon_rate_mps
        if not np.allclose(command.tendon_target_m, expected, atol=1e-10, rtol=0):
            raise ValueError('主臂控制与后端肌腱积分目标不一致。')
        frames = []
        increment = self.timestep*command.tendon_rate_mps
        for _ in range(steps):
            self.target += increment
            self.data.ctrl[self.ids['actuators']] = self.neutral+self.target
            # 固定基座不做基座积分，也不在物理子步前重复 mj_forward。
            self.mj.mj_step(self.model, self.data)
            self.physical_steps += 1
            if self.video_stride is not None and self.physical_steps % self.video_stride == 0:
                frames.append(self.snapshot())
        self.mj.mj_forward(self.model, self.data)
        return frames
