"""按真实采样姿态筛选观察偏好；不把样本当成已证明安全的运动路径。"""
import numpy as np


def select_observation_candidate(candidates, root_world, tip_camera, target_world,
                                 current_bending, camera, desired_distance):
    if not candidates:
        return None
    transforms = root_world@np.stack([c.tip_root for c in candidates])@tip_camera
    delta = target_world-transforms[:, :3, 3]
    point = np.einsum('nji,nj->ni', transforms[:, :3, :3], delta)
    a, b = camera.planes()
    allowed = np.all(point@a.T >= b, axis=1)
    if not np.any(allowed):
        return None
    indices = np.flatnonzero(allowed)
    distance = np.abs(point[indices, 2]-desired_distance)
    screen = np.linalg.norm(point[indices, :2], axis=1)
    q_delta = np.linalg.norm(np.stack([candidates[i].bending for i in indices])-current_bending, axis=1)
    # 距离/画面居中优先，并对大幅构型切换添加柔性代价。
    score = screen+distance*.3+q_delta*.0002
    return candidates[int(indices[np.argmin(score)])]
