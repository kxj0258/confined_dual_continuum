"""相机、整臂碰撞与物理反馈适配；控制模块不导入 MuJoCo。"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from confined_dual_continuum.adapters.mujoco_backend import MujocoRobotBackend
from confined_dual_continuum.adapters.robot_only_scene import robot_only_scene
from confined_dual_continuum.domain.tracking import TrackingClearance, TrackingGeometry


def build_tracking_scene(source: Path, destination: Path, obstacles):
    """保留双臂碰撞；障碍按固定基座坐标创建，复用当前 XML 的基座初始位姿。"""
    robot_only_scene(source, destination)
    tree = ET.parse(destination)
    world = tree.getroot().find('worldbody')
    base = world.find("body[@name='mobile_base']")
    # 障碍必须静止在世界中，使用与基座相同的初始变换，不能挂到机械臂上。
    obstacle_frame = ET.SubElement(world, 'body', name='tracking_obstacles',
                                  pos=base.get('pos', '0 0 0'))
    for key in ('quat', 'euler', 'axisangle', 'xyaxes', 'zaxis'):
        if base.get(key) is not None:
            obstacle_frame.set(key, base.get(key))
    # 基座坐标原点来自 mobile_base_frame site。
    site = base.find("site[@name='mobile_base_frame']")
    if site is None:
        raise ValueError('固定场景缺少基座坐标 site。')
    frame = ET.SubElement(obstacle_frame, 'body', name='tracking_base_frame',
                          pos=site.get('pos', '0 0 0'))
    for key in ('quat', 'euler', 'axisangle', 'xyaxes', 'zaxis'):
        if site.get(key) is not None:
            frame.set(key, site.get(key))
    names = set()
    for index, obstacle in enumerate(obstacles):
        kind = obstacle['type']
        if kind not in ('sphere', 'box'):
            raise ValueError('首版障碍仅支持 sphere 和 box。')
        pos, size = np.asarray(obstacle['position_base_m'], float), np.asarray(obstacle['size_m'], float)
        if (pos.shape != (3,) or size.shape != ((1,) if kind == 'sphere' else (3,))
                or not np.all(np.isfinite(pos)) or not np.all(np.isfinite(size)) or np.any(size <= 0)):
            raise ValueError('障碍位置/尺寸无效；box 尺寸为半边长。')
        name = str(obstacle.get('name', f'obstacle_{index}'))
        if name in names:
            raise ValueError('障碍名称必须唯一。')
        names.add(name)
        ET.SubElement(frame, 'geom', name='tracking_'+name, type=kind,
            pos=' '.join(map(str, pos)), size=' '.join(map(str, size)),
            rgba='0.7 0.35 0.2 1', contype='1', conaffinity='1')
    tree.write(destination, encoding='utf-8', xml_declaration=True)


class TrackingBackend(MujocoRobotBackend):
    def __init__(self, xml_path, robot):
        super().__init__(xml_path, robot, centerline_source='mujoco',
                         tip_pose_source='mujoco', require_wiping_board=False)
        self.camera_id = self._id(self._mj.mjtObj.mjOBJ_CAMERA, 'observer_camera')
        # 沿用当前十二离散杆模型的弧长定义，拒绝不匹配模型而非静默估算。
        if not np.isclose(sum(robot.flexure_lengths_m)+sum(robot.distal_offsets_m), .12):
            raise ValueError('跟随碰撞弧长适配要求当前 120 mm / 十二杆模型。')
        self._joint_ids = {a: [int(self.model.body_jntadr[b]) for b in bodies]
                           for a, bodies in self._link_bodies.items()}
        self._excludes = set(int(v) for v in self.model.exclude_signature)

    def camera_transform(self):
        transform = np.eye(4)
        # MuJoCo 为右、上、后；控制系为右、下、前。
        transform[:3, :3] = self.data.cam_xmat[self.camera_id].reshape(3, 3)@np.diag([1., -1., -1.])
        transform[:3, 3] = self.data.cam_xpos[self.camera_id]
        return transform

    def _arc(self, arm, link, point):
        body = self._link_bodies[arm][link]
        axis = self.data.xmat[body].reshape(3, 3)[:, 2]
        return link*.01+float(np.clip((point-self.data.xpos[body])@axis, 0., .01))

    def _gap(self, arm, link, other_geom, other_arm=None, other_link=0):
        segment = np.zeros(6)
        own_geom = self._collision_geoms[arm][link]
        distance = float(self._mj.mj_geomDistance(self.model, self.data, other_geom,
                                                own_geom, .02, segment))
        if distance >= .02:
            return None
        delta = segment[3:]-segment[:3]
        norm = np.linalg.norm(delta)
        normal = delta/norm if norm > 1e-12 else np.zeros(3)
        if distance < 0:
            normal = -normal
        source = self._mj.mj_id2name(self.model, self._mj.mjtObj.mjOBJ_GEOM, other_geom)
        return TrackingClearance(arm, self._arc(arm, link, segment[3:]), distance, normal,
            other_arm, self._arc(other_arm, other_link, segment[:3]) if other_arm else 0., str(source))

    def geometry(self, target_world, target_radius_m):
        gaps = []
        for arm in ('executor', 'observer'):
            for i, geom in enumerate(self._collision_geoms[arm]):
                for environment in self._environment_geoms:
                    gap = self._gap(arm, i, environment)
                    if gap is not None:
                        gaps.append(gap)
                for j in range(i+2, 12):
                    body1, body2 = sorted((self._link_bodies[arm][i], self._link_bodies[arm][j]))
                    if (body1 << 16)+body2 in self._excludes:
                        continue
                    gap = self._gap(arm, i, self._collision_geoms[arm][j], arm, j)
                    if gap is not None:
                        gaps.append(gap)
        for i in range(12):
            for j in range(12):
                gap = self._gap('observer', i, self._collision_geoms['executor'][j], 'executor', j)
                if gap is not None:
                    gaps.append(gap)
        margins = {}
        for arm, joints in self._joint_ids.items():
            margins[arm] = min(float(min(self.data.qpos[self.model.jnt_qposadr[j]]-self.model.jnt_range[j, 0],
                self.model.jnt_range[j, 1]-self.data.qpos[self.model.jnt_qposadr[j]])) for j in joints
                if self.model.jnt_limited[j])
        camera = self.camera_transform()
        return TrackingGeometry(camera, self._site_transform('executor_base_site'),
            self._site_transform('observer_base_site'), tuple(gaps),
            self._occluded(camera, np.asarray(target_world), target_radius_m), margins)

    def _occluded(self, camera, target, radius):
        # 真实可见网格射线：透明碰撞包络不充当图像遮挡物。
        # 中心及四个区域边缘都检查，目标工具本体上的命中视为看到目标。
        origin = camera[:3, 3]
        tool_body = self._link_bodies['executor'][-1]
        for dx, dy in ((0., 0.), (radius, 0.), (-radius, 0.), (0., radius), (0., -radius)):
            endpoint = target+camera[:3, 0]*dx+camera[:3, 1]*dy
            delta = endpoint-origin
            length = float(np.linalg.norm(delta))
            if length <= 1e-10:
                return True
            geom = np.array([-1], dtype=np.int32)
            distance = float(self._mj.mj_ray(self.model, self.data, origin, delta/length,
                                           None, True, -1, geom))
            if distance >= 0 and distance < length-1e-5:
                if int(self.model.geom_bodyid[geom[0]]) != tool_body:
                    return True
        return False

    def interarm_contact_count(self):
        ex, ob = set(self._link_bodies['executor']), set(self._link_bodies['observer'])
        count = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            a, b = int(self.model.geom_bodyid[c.geom1]), int(self.model.geom_bodyid[c.geom2])
            count += int((a in ex and b in ob) or (a in ob and b in ex))
        return count
