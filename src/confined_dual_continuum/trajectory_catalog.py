"""轨迹目录装配、点云证据生成与覆盖检查，不启动 MuJoCo。"""
import argparse
import hashlib
import json
from collections import deque
from pathlib import Path
import numpy as np
import yaml
from confined_dual_continuum.task.catalog_trajectory import CatalogTrajectory
from confined_dual_continuum.workspace.pointcloud_coverage import cloud_policy, nearest_distances, check_coverage

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIRECTORY = ROOT/'configs/trajectories'
VERSION = 1


def digest(path):
    # Git 在 Windows 可能转换换行；文本证据统一 LF 后哈希。
    data = (path.read_text(encoding='utf-8').replace('\r\n', '\n').encode('utf-8')
            if path.suffix in ('.yaml', '.json') else path.read_bytes())
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def validate(path, directory):
    definition = yaml.safe_load(path.read_text(encoding='utf-8'))
    trajectory = CatalogTrajectory(definition)
    policy = json.loads((directory/'workspace.json').read_text(encoding='utf-8'))
    cloud_path = directory/'workspace.npz'
    if digest(cloud_path) != policy['cloud_sha256']:
        raise ValueError('点云证据文件哈希不匹配。')
    with np.load(cloud_path) as data:
        cloud = data['points']
    if not np.allclose(trajectory.entry.points[0], policy['initial_tip_base_m'], atol=1e-10):
        raise ValueError('入场段必须从采集零位起点开始。')
    result = check_coverage(trajectory, cloud, policy)
    result.update(version=VERSION, definition_sha256=digest(path), workspace_sha256=digest(directory/'workspace.json'),
                  cloud_sha256=digest(cloud_path))
    write_json(path.with_suffix('.coverage.json'), result)
    return result


def load_approved(path, directory):
    report_path = path.with_suffix('.coverage.json')
    if not report_path.exists():
        raise ValueError(f'轨迹尚未通过点云检查：{path.name}；请运行 trajectory_catalog --validate。')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if (not report.get('passed') or report.get('version') != VERSION
            or report.get('definition_sha256') != digest(path)
            or report.get('workspace_sha256') != digest(directory/'workspace.json')
            or report.get('cloud_sha256') != digest(directory/'workspace.npz')):
        raise ValueError(f'轨迹未通过检查或文件已变化：{path.name}；必须重新验证。')
    return yaml.safe_load(path.read_text(encoding='utf-8')), report


def build(source, directory):
    directory.mkdir(parents=True, exist_ok=True)
    points, initial = [], None
    with source.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if (row['kind'] != 'workspace' or row['arm'] != 'executor'
                    or row['status'] != 'stable' or not row.get('path_completed')):
                continue
            measurement = row['measurement']
            transform = np.linalg.solve(np.array(measurement['T_world_base']), np.array(row['T_world_tip']))
            points.append(transform[:3, 3])
            if np.linalg.norm(row['legs'][-1]['reference_bending_per_m']) < 1e-12:
                initial = transform[:3, 3]
    if initial is None:
        raise ValueError('采集记录缺少主臂零位。')
    cloud = np.unique(points, axis=0)
    np.savez_compressed(directory/'workspace.npz', points=cloud)
    policy = cloud_policy(cloud)
    policy.update(source=str(source), source_sha256=digest(source), frame='mobile_base_frame',
                  initial_tip_base_m=initial.tolist(), cloud_sha256=digest(directory/'workspace.npz'))
    write_json(directory/'workspace.json', policy)
    print(json.dumps(policy, ensure_ascii=False), flush=True)
    # 每条图边均短于两个覆盖球半径，保证连接段留在点云容差内。
    parents = {int(np.argmin(np.linalg.norm(cloud-initial, axis=1))): None}
    queue = deque(parents)
    edge_limit = 2*(policy['tolerance_m']-policy['check_step_m'])
    while queue:
        node = queue.popleft()
        neighbors = np.flatnonzero(np.linalg.norm(cloud-cloud[node], axis=1) <= edge_limit)
        for neighbor in neighbors:
            neighbor = int(neighbor)
            if neighbor not in parents:
                parents[neighbor] = node
                queue.append(neighbor)
    low, high = np.array(policy['inner_minimum_m']), np.array(policy['inner_maximum_m'])
    middle = (low+high)/2
    centers = [middle+np.array([x, y, z])*(high-low)
               for z in (-.2, 0, .2, .35, .45)
               for x, y in ((0,0), (.2,0), (-.2,0), (0,.2), (0,-.2), (.35,0), (-.35,0))]
    # 优先复用圆轨迹所在的中心；其余中心仍保留，避免把搜索锁死在单一点。
    circle_center = np.asarray([-.00018087399144387645, .010558443125817704,
                                .1295249627491061])
    priority_centers = sorted(centers, key=lambda center: float(np.linalg.norm(center-circle_center)))
    planes = []
    for angle in (0, 20, -20, 40, -40):
        rad = np.deg2rad(angle)
        planes.append((np.array([1.,0,0]), np.array([0,np.cos(rad),np.sin(rad)])))
        if angle:
            planes.append((np.array([np.cos(rad),0,np.sin(rad)]), np.array([0.,1,0])))
    for kind in ('circle', 'figure8', 'rectangle'):
        best, best_size = None, 0.
        # 圆形和矩形优先复用圆轨迹中心；8 字形保留原始中心顺序，
        # 避免将其限制在更适合圆形/矩形的局部点云区域。
        search_centers = priority_centers if kind in ('circle', 'rectangle') else centers
        for center in search_centers:
            for u, v in planes:
                primary_extent = np.abs(u)
                maximum_a = float(np.min(np.minimum(center-low, high-center)[primary_extent>1e-12]
                                         /primary_extent[primary_extent>1e-12]))
                # 有限候选搜索：1 mm 分辨率，非全局最优声明。
                for size in np.arange(maximum_a, max(best_size, .003), -.001):
                    if kind == 'circle':
                        candidates = [(1., size)]
                    elif kind == 'figure8':
                        # 先用 0.1 步长筛选比例；最终覆盖检查仍使用正式步长。
                        ratios = (*np.arange(.25, .951, .1), 1.)
                        candidates = [(ratio, size*ratio) for ratio in ratios]
                    else:
                        # 矩形两个半尺寸独立搜索，不绑定固定纵横比。
                        maximum_b = float(np.min(np.minimum(center-low, high-center)
                                                 [np.abs(v)>1e-12]/np.abs(v)[np.abs(v)>1e-12]))
                        # 先以 5 mm 步长筛选第二半尺寸，避免二维毫米网格爆炸。
                        candidates = [(None, b) for b in np.arange(maximum_b, .002, -.005)]
                    for ratio, secondary_size in candidates:
                        extent = np.abs(u)*size+np.abs(v)*secondary_size
                        if np.any(center-extent < low) or np.any(center+extent > high):
                            continue
                        definition = dict(name=kind, kind=kind, center_m=center.tolist(), u=u.tolist(), v=v.tolist(),
                            half_sizes_m=[float(size), float(secondary_size)], entry_s=6., motion_s=20.,
                            entry_points_m=[initial.tolist(), initial.tolist()])
                        # 先构造主体起点，粗筛后再从点云图生成入场。
                        if kind == 'circle':
                            start = center+size*u
                        elif kind == 'figure8':
                            start = center
                        else:
                            start = center-size*u-secondary_size*v
                        definition['entry_points_m'][-1] = start.tolist()
                        trajectory = CatalogTrajectory(definition)
                        # 粗筛只负责淘汰明显不可能的候选，最终仍由 check_coverage
                        # 以 0.5 mm 步长对完整曲线复查，避免在此处重复昂贵计算。
                        _, coarse_body = trajectory.coverage_points(.01)
                        if nearest_distances(coarse_body, cloud).max() > policy['tolerance_m']:
                            continue
                        nearest = int(np.argmin(np.linalg.norm(cloud-start, axis=1)))
                        if nearest not in parents:
                            continue
                        route = [start.tolist()]
                        node = nearest
                        while node is not None:
                            route.append(cloud[node].tolist())
                            node = parents[node]
                        route.reverse()
                        route[0] = initial.tolist()
                        # 贪心合并有覆盖证据的直线段，减少入场停点。
                        simplified, begin = [route[0]], 0
                        while begin < len(route)-1:
                            for end in range(len(route)-1, begin, -1):
                                a, b = np.asarray(route[begin]), np.asarray(route[end])
                                n = max(1, int(np.ceil(np.linalg.norm(b-a)/policy['check_step_m'])))
                                line = a+(b-a)*np.linspace(0, 1, n+1)[:, None]
                                if nearest_distances(line, cloud).max()+policy['check_step_m']/2 <= policy['tolerance_m']:
                                    break
                            simplified.append(route[end])
                            begin = end
                        definition['entry_points_m'] = simplified
                        trajectory = CatalogTrajectory(definition)
                        result = check_coverage(trajectory, cloud, policy)
                        if result['passed']:
                            best, best_size = definition, float(size)
                            break
        if best is None:
            raise ValueError(f'{kind} 在当前搜索范围内没有通过的候选，未放宽容差。')
        best['search'] = dict(central_span_fraction=.8, centers=len(search_centers), planes=len(planes),
                              size_resolution_m=.001, objective='largest_primary_half_size_among_tested_candidates')
        path = directory/f'{kind}.yaml'
        path.write_text(yaml.safe_dump(best, allow_unicode=True, sort_keys=False), encoding='utf-8')
        print(kind, json.dumps(validate(path, directory)), 'half_sizes_m=', best['half_sizes_m'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DEFAULT_DIRECTORY)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--build', type=Path, help='从原始 samples.jsonl 生成点云证据与三种轨迹')
    group.add_argument('--validate', help='检查目录中的轨迹名称，或 all')
    args = parser.parse_args()
    if args.build:
        build(args.build.resolve(), args.directory.resolve())
    else:
        paths = sorted(args.directory.glob('*.yaml')) if args.validate == 'all' else [args.directory/f'{args.validate}.yaml']
        if not paths:
            parser.error('没有轨迹文件。')
        failed = False
        for path in paths:
            result = validate(path, args.directory)
            print(path.stem, json.dumps(result), flush=True)
            failed |= not result['passed']
        if failed:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
