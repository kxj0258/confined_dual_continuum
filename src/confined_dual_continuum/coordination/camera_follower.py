"""从臂相机任务：只求解从臂曲率，明确区分观察与安全降级。"""
from dataclasses import dataclass
import numpy as np
from confined_dual_continuum.kinematics.pcc import forward_kinematics
from confined_dual_continuum.kinematics.constrained_velocity import constrained_velocity


@dataclass(frozen=True)
class FollowerResult:
    rate: np.ndarray
    status: str
    view_matrix: np.ndarray
    view_lower: np.ndarray


def camera_differential(bending, lengths, offsets, actual_tip, actual_camera, target_world):
    """用实际相机修正当前位姿，PCC 仅提供局部增量导数。"""
    nominal = forward_kinematics(bending, lengths, offsets).tip_transform
    correction = actual_tip@np.linalg.inv(nominal)
    tip_camera = np.linalg.solve(actual_tip, actual_camera)
    j_point, j_position = np.zeros((3, len(bending))), np.zeros((3, len(bending)))
    for i in range(len(bending)):
        delta = np.zeros(len(bending))
        delta[i] = 1e-5
        results = []
        for sign in (1., -1.):
            camera = correction@forward_kinematics(
                bending+sign*delta, lengths, offsets).tip_transform@tip_camera
            point = camera[:3, :3].T@(target_world-camera[:3, 3])
            results.append((point, camera[:3, 3]))
        j_point[:, i] = (results[0][0]-results[1][0])/2e-5
        j_position[:, i] = (results[0][1]-results[1][1])/2e-5
    point = actual_camera[:3, :3].T@(target_world-actual_camera[:3, 3])
    return point, j_point, j_position


class CameraFollower:
    def __init__(self, camera, *, gain=2., distance_m=.04, position_weight=.15):
        self.camera, self.gain = camera, gain
        self.distance_m, self.position_weight = distance_m, position_weight

    def solve(self, point, point_jacobian, position_jacobian, camera_world,
              target_position, target_velocity, preferred_offset_world,
              safety_matrix, safety_lower, occluded, posture_rate=None):
        target_drift = camera_world[:3, :3].T@target_velocity
        m = np.vstack((point_jacobian, self.position_weight*position_jacobian))
        desired = np.concatenate((
            self.gain*(np.array([0., 0., self.distance_m])-point)-target_drift,
            self.position_weight*(target_velocity+self.gain*(
                target_position+preferred_offset_world-camera_world[:3, 3]))))
        if posture_rate is not None:
            # 样本构型只作为弱偏好，实际发送仍经过全部距离和驱动约束。
            m = np.vstack((m, .0001*np.eye(m.shape[1])))
            desired = np.concatenate((desired, .0001*posture_rate))
        planes, bounds = self.camera.planes()
        view_matrix = planes@point_jacobian
        view_lower = -self.gain*(planes@point-bounds)-planes@target_drift
        rate, feasible = constrained_velocity(m, desired,
            np.vstack((safety_matrix, view_matrix)), np.concatenate((safety_lower, view_lower)))
        if feasible:
            status = 'occluded_repositioning' if occluded else (
                'tracking' if self.camera.visible(point, False) else 'acquiring')
        else:
            # 显式安全降级：放弃视场约束，仍保持所有机械/距离约束。
            rate, feasible = constrained_velocity(m, desired, safety_matrix, safety_lower)
            status = 'view_infeasible' if feasible else 'observer_safety_infeasible'
        return FollowerResult(rate, status, view_matrix, view_lower)
