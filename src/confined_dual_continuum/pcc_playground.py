"""仅双臂场景的 PCC 滑块控制入口；Tk 面板与 MuJoCo 视窗并列。"""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import time
import tkinter as tk
from tkinter import ttk, messagebox
from confined_dual_continuum.adapters.robot_only_scene import robot_only_scene

import numpy as np

from confined_dual_continuum.adapters.mujoco_backend import MujocoRobotBackend
from confined_dual_continuum.actuation.compatible_tendon_control import CompatibleTendonController
from confined_dual_continuum.domain.state import ArmCommand, RobotCommand
from confined_dual_continuum.kinematics.pcc import forward_kinematics
from confined_dual_continuum.robot.configuration import load_robot_model

ARMS = ('executor', 'observer')


def pose_text(transform):
    """位置 mm；四元数 wxyz，不使用有奇异点的欧拉角。"""
    import mujoco
    quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(transform[:3, :3]).ravel())
    xyz = transform[:3, 3] * 1000
    return f'位置 mm  {xyz[0]:9.3f} {xyz[1]:9.3f} {xyz[2]:9.3f}\n四元数   ' + ' '.join(f'{x: .5f}' for x in quaternion)


class Playground:
    def __init__(self, root, backend, viewer, maximum_curvature):
        self.root, self.backend, self.viewer = root, backend, viewer
        self.robot = backend.robot
        self.dt = .01
        self.controllers = {a: CompatibleTendonController(self.robot.bending_model(a), self.robot.limits) for a in ARMS}
        self.values = {}
        self.readouts = {}
        self.paused = False
        self.closed = False
        self.frame = tk.StringVar(value='臂根')
        root.title('双臂 PCC 控制与末端位姿')
        root.geometry('1080x760')
        toolbar = ttk.Frame(root, padding=10)
        toolbar.pack(fill='x')
        ttk.Button(toolbar, text='回零目标', command=self.zero).pack(side='left')
        ttk.Button(toolbar, text='复位仿真', command=self.reset).pack(side='left', padx=8)
        self.pause_button = ttk.Button(toolbar, text='暂停', command=self.toggle_pause)
        self.pause_button.pack(side='left')
        ttk.Label(toolbar, text='位姿坐标系：').pack(side='left', padx=(20, 4))
        ttk.Combobox(toolbar, textvariable=self.frame, values=('臂根', '世界'), state='readonly', width=7).pack(side='left')
        ttk.Label(root, text='每段参数 [kx, ky]，单位 1/m；滑块范围是交互输入范围。直接更新长度参考，无额外速度节流。').pack(anchor='w', padx=12)
        panels = ttk.Frame(root)
        panels.pack(fill='both', expand=True, padx=10, pady=8)
        for arm in ARMS:
            panel = ttk.LabelFrame(panels, text=arm, padding=10)
            panel.pack(side='left', fill='both', expand=True, padx=4)
            self.values[arm] = []
            for index in range(6):
                variable = tk.DoubleVar(value=0.)
                self.values[arm].append(variable)
                tk.Scale(panel, label=f'第 {index // 2 + 1} 段  {("kx", "ky")[index % 2]}',
                         variable=variable, from_=-maximum_curvature, to=maximum_curvature,
                         resolution=.1, orient='horizontal', length=420).pack(fill='x')
            self.readouts[arm] = tk.StringVar()
            ttk.Label(panel, textvariable=self.readouts[arm], font=('Consolas', 10), justify='left').pack(anchor='w', pady=10)
        self.status = tk.StringVar()
        ttk.Label(root, textvariable=self.status, padding=8).pack(fill='x')
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.reset()

    def zero(self):
        for variables in self.values.values():
            for variable in variables:
                variable.set(0.)

    def reset(self):
        with self.viewer.lock():
            state = self.backend.reset()
        for arm in ARMS:
            self.controllers[arm].reset(getattr(state, arm).tendon_displacement_m)
        self.zero()

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.configure(text='继续' if self.paused else '暂停')

    def close(self):
        if not self.closed:
            self.closed = True
            self.backend.stop()
            self.root.destroy()

    def tick(self):
        if self.closed:
            return
        if not self.viewer.is_running():
            self.close()
            return
        started = time.perf_counter()
        try:
            with self.viewer.lock():
                state = self.backend.read_state()
                desired = {a: np.array([v.get() for v in self.values[a]]) for a in ARMS}
                limit = self.robot.limits.maximum_curvature_per_m
                if limit is not None:
                    for arm, q in desired.items():
                        pairs = q.reshape(-1, 2)
                        pairs *= np.minimum(1., limit / np.maximum(np.linalg.norm(pairs, axis=1), 1e-15))[:, None]
                        for variable, value in zip(self.values[arm], q):
                            variable.set(float(value))
                if not self.paused:
                    commands = {}
                    for arm in ARMS:
                        controller = self.controllers[arm]
                        # 以上次接受的参考为起点，单个控制周期更新到滑块目标。
                        previous = controller.model.estimate(self.backend._targets[arm] - self.backend._neutral[arm])
                        result = controller.step((desired[arm] - previous) / self.dt,
                                                 getattr(state, arm).tendon_displacement_m, self.dt)
                        commands[arm] = ArmCommand(result.bending_rate, result.tendon_rate_mps, result.target_m)
                    self.backend.apply_command(RobotCommand(np.zeros(6), **commands))
                    self.backend.step(self.dt)
                    state = self.backend.read_state()
                for arm in ARMS:
                    base = self.backend._site_transform(f'{arm}_base_site')
                    actual = self.backend._site_transform(f'{arm}_tip')
                    target = base @ forward_kinematics(desired[arm], self.robot.flexure_lengths_m, self.robot.distal_offsets_m).tip_transform
                    feedback = base @ forward_kinematics(getattr(state, arm).bending, self.robot.flexure_lengths_m, self.robot.distal_offsets_m).tip_transform
                    position_error = np.linalg.norm(feedback[:3, 3] - actual[:3, 3]) * 1000
                    angle_error = np.degrees(np.arccos(np.clip((np.trace(feedback[:3, :3].T @ actual[:3, :3]) - 1) / 2, -1, 1)))
                    reference = np.linalg.inv(base) if self.frame.get() == '臂根' else np.eye(4)
                    self.readouts[arm].set('\n'.join((
                        '滑块目标 PCC', pose_text(reference @ target),
                        '反馈反算 PCC', pose_text(reference @ feedback),
                        'MuJoCo 实际末端', pose_text(reference @ actual),
                        f'反馈 PCC / 实际误差：{position_error:.3f} mm，{angle_error:.3f}°')))
                self.status.set(f'仿真时间 {self.backend.data.time:.2f} s | {"暂停" if self.paused else "运行"} | 肌腱限制 {"开启" if self.robot.limits.enabled else "关闭"} | 当前模型保留双臂碰撞、关节限位；未接入任务避障控制器')
            self.viewer.sync()
        except Exception as error:
            messagebox.showerror('仿真停止', str(error), parent=self.root)
            self.close()
            return
        self.root.after(max(1, round((self.dt - (time.perf_counter() - started)) * 1000)), self.tick)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-config', type=Path, default=Path('configs/robot/three_segment_nine_tendon.yaml'))
    parser.add_argument('--xml', type=Path, default=Path('assets/mujoco/multi_task_patent_demo.xml'))
    parser.add_argument('--curvature-range', type=float, default=13.5, help='每个 kx/ky 滑块的正负范围，仍受机器人曲率模长限制，1/m')
    args = parser.parse_args()
    if not np.isfinite(args.curvature_range) or args.curvature_range <= 0:
        parser.error('--curvature-range 必须为有限正数')
    import mujoco.viewer
    robot = load_robot_model(args.robot_config)
    with tempfile.TemporaryDirectory(prefix='cdc-pcc-') as directory:
        scene = Path(directory) / 'dual_arm_only.xml'
        robot_only_scene(args.xml, scene)
        backend = MujocoRobotBackend(scene, robot, require_wiping_board=False)
        try:
            with mujoco.viewer.launch_passive(backend.model, backend.data) as viewer:
                with viewer.lock():
                    viewer.cam.lookat[:] = [0, 0, -.035]
                    viewer.cam.distance = .35
                    viewer.cam.azimuth = 135
                    viewer.cam.elevation = -15
                root = tk.Tk()
                app = Playground(root, backend, viewer, args.curvature_range)
                root.after(0, app.tick)
                root.mainloop()
        finally:
            backend.close()


if __name__ == '__main__':
    main()
