"""从现有稳定工作空间记录提取观察构型；文件读取仅在装配边界。"""
import hashlib
import json
import numpy as np
from confined_dual_continuum.domain.tracking import ObservationCandidate


def load_observation_samples(path, minimum_joint_margin_rad, maximum_curvature_per_m):
    candidates, digest = [], hashlib.sha256()
    with path.open('rb') as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            if (row['kind'] != 'workspace' or row['arm'] != 'observer'
                    or row['status'] != 'stable' or not row['path_completed']):
                continue
            if min(leg['minimum_joint_margin_rad'] for leg in row['legs']) < minimum_joint_margin_rad:
                continue
            q = np.asarray(row['legs'][-1]['reference_bending_per_m'], float)
            tip = np.asarray(row['T_root_tip'], float)
            if (q.shape != (6,) or tip.shape != (4, 4) or not np.all(np.isfinite(q))
                    or not np.all(np.isfinite(tip))):
                raise ValueError('工作空间观察样本格式无效。')
            # 与在线约束一致的内接多边形预筛选。
            normals = np.column_stack((np.cos(np.arange(32)*2*np.pi/32), np.sin(np.arange(32)*2*np.pi/32)))
            if np.max(q.reshape(3, 2)@normals.T) > maximum_curvature_per_m*np.cos(np.pi/32)+1e-10:
                continue
            candidates.append(ObservationCandidate(int(row['sample']), q, tip))
    if not candidates:
        raise ValueError('没有满足曲率与关节余量的 Observer 稳定样本。')
    return tuple(candidates), digest.hexdigest()
