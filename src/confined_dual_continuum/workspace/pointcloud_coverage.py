"""离散点云覆盖口径，不代表动力学或逆解可达性。"""
import numpy as np


def nearest_distances(points, cloud):
    points, cloud = np.asarray(points), np.asarray(cloud)
    result = []
    for start in range(0, len(points), 32):
        delta = points[start:start+32, None, :]-cloud[None, :, :]
        result.extend(np.sqrt(np.min(np.einsum('ijk,ijk->ij', delta, delta), axis=1)))
    return np.asarray(result)


def cloud_policy(cloud):
    cloud = np.unique(np.asarray(cloud, float), axis=0)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) < 10 or not np.all(np.isfinite(cloud)):
        raise ValueError('点云要求至少十个不同的有限三维点。')
    distances = []
    for start in range(0, len(cloud), 32):
        delta = cloud[start:start+32, None, :]-cloud[None, :, :]
        squared = np.einsum('ijk,ijk->ij', delta, delta)
        squared[np.arange(len(squared)), np.arange(start, start+len(squared))] = np.inf
        distances.extend(np.sqrt(np.min(squared, axis=1)))
    tolerance = float(np.percentile(distances, 95))
    if tolerance <= 0:
        raise ValueError('点云间距不能为零。')
    low, high = cloud.min(axis=0), cloud.max(axis=0)
    return dict(tolerance_m=tolerance, rule='unique_cloud_nearest_neighbor_p95',
                minimum_m=low.tolist(), maximum_m=high.tolist(),
                inner_minimum_m=(low+.1*(high-low)).tolist(),
                inner_maximum_m=(high-.1*(high-low)).tolist(),
                check_step_m=min(.0005, tolerance/4), point_count=len(cloud))


def check_coverage(trajectory, cloud, policy):
    entry, body = trajectory.coverage_points(policy['check_step_m'])
    low, high = np.array(policy['inner_minimum_m']), np.array(policy['inner_maximum_m'])
    distances = nearest_distances(np.vstack((entry, body)), cloud)
    # 最近距离是 1-Lipschitz；预留半个最大采样弧长，覆盖离散检查点之间的曲线。
    upper_bound = float(distances.max()+policy['check_step_m']/2)
    extent = (np.sqrt((trajectory.a*trajectory.u)**2+(trajectory.a*trajectory.v)**2)
              if trajectory.kind == 'circle' else
              trajectory.a*np.abs(trajectory.u)+trajectory.b*np.abs(trajectory.v))
    in_box = bool(np.all(trajectory.center-extent >= low-1e-12)
                  and np.all(trajectory.center+extent <= high+1e-12))
    return dict(passed=in_box and upper_bound <= policy['tolerance_m'], body_in_inner_80_percent=in_box,
                entry_points=len(entry), body_points=len(body), maximum_nearest_distance_m=float(distances.max()),
                continuous_distance_upper_bound_m=upper_bound, tolerance_m=policy['tolerance_m'],
                verdict='sampled_pointcloud_coverage_only')
