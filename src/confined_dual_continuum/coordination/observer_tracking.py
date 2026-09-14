"""单次从臂求解：机械约束投影与软避碰/朝向速度命令。"""
import numpy as np
from confined_dual_continuum.domain.segments import closest_segments
from confined_dual_continuum.domain.geometry import quaternion_rotation
from confined_dual_continuum.kinematics.pcc import direction_jacobian
from confined_dual_continuum.kinematics.fast_pcc import centerline_position_jacobians
from confined_dual_continuum.kinematics.constrained_velocity import constrained_velocity
from confined_dual_continuum.kinematics.solver import solve_velocity


def _link_points(bending, lengths, offsets, root):
    total = sum(lengths) + sum(offsets)
    arcs = np.linspace(0.0, total, 13)
    points, jacobians = centerline_position_jacobians(
        bending, lengths, offsets, arcs)
    rotation = root[:3, :3]
    return (root[:3, 3] + points @ rotation.T,
            np.einsum('ij,njk->nik', rotation, jacobians), arcs)


def _nearest_pairs(executor_points, observer_points,
                   limit=3):
    candidates = []
    for i in range(len(executor_points) - 1):
        for j in range(len(observer_points) - 1):
            ex, ob, se, so = closest_segments(
                executor_points[i], executor_points[i + 1],
                observer_points[j], observer_points[j + 1])
            delta = ob - ex
            candidates.append((float(np.linalg.norm(delta)), ex, ob, i, j, se, so))
    candidates.sort(key=lambda item: item[0])
    return candidates[:limit]


def solve_observer(robot, state, geometry, target, executor_rate, dt, config, cancelled,
                   enter_stage=None, avoidance_latched=False):
    if enter_stage is not None:
        enter_stage('drive_constraints')
    model, limits = robot.bending_model('observer'), robot.limits
    if (config.get('joint_guard_enabled', True)
            and geometry.joint_margin_rad['observer'] <= np.deg2rad(config['joint_margin_deg'])):
        return None, 'joint_guard', None
    coupling = model.coupling_matrix
    # rows/lower 只保留机械、肌腱和曲率限制；避碰与朝向前馈在速度命令层叠加。
    rows, lower = [], []
    if limits.enabled or limits.rate_enabled:
        rows.extend(coupling)
        rows.extend(-coupling)
        lower.extend(-limits.maximum_rate_mps)
        lower.extend(-limits.maximum_rate_mps)
    if limits.enabled:
        actual = state.observer.tendon_displacement_m
        lo = np.maximum(limits.minimum_m, actual-limits.maximum_lead_m)
        hi = np.minimum(limits.maximum_m, actual+limits.maximum_lead_m)
        rows.extend(dt*coupling)
        lower.extend(lo-target)
        rows.extend(-dt*coupling)
        lower.extend(target-hi)
    if limits.maximum_curvature_per_m is not None:
        bending = model.estimate(target)
        # 内接多边形保守限制下一周期曲率，最终仍经过公共相容控制器。
        bound = limits.maximum_curvature_per_m*np.cos(np.pi/32)
        for section in range(model.bending_size//2):
            for angle in np.arange(32)*2*np.pi/32:
                normal = np.array([np.cos(angle), np.sin(angle)])
                row = np.zeros(model.bending_size)
                row[2*section:2*section+2] = -dt*normal
                rows.append(row)
                lower.append(normal@bending[2*section:2*section+2]-bound)
    if enter_stage is not None:
        enter_stage('clearance_jacobians')
    executor_points, executor_jacobians, _ = _link_points(
        state.executor.bending, robot.flexure_lengths_m, robot.distal_offsets_m,
        geometry.executor_root_world)
    observer_points, observer_jacobians, _ = _link_points(
        state.observer.bending, robot.flexure_lengths_m, robot.distal_offsets_m,
        geometry.observer_root_world)
    candidates = _nearest_pairs(executor_points, observer_points,
                                int(config.get('active_constraints', 3)))
    trigger_gap = config.get('avoidance_trigger_m', config['clearance_m'])
    release_gap = config.get('avoidance_release_m', trigger_gap)
    candidate_gaps = [
        distance_center - config.get('diameter_m', 0.0125)
        for distance_center, *_ in candidates]
    if avoidance_latched and candidate_gaps and min(candidate_gaps) >= release_gap:
        avoidance_latched = False
    elif not avoidance_latched and candidate_gaps and min(candidate_gaps) <= trigger_gap:
        avoidance_latched = True
    # 先求未受避碰约束的朝向速度。朝向动作造成的距离闭合只作为软速度命令，
    # 不再加入速度求解器的硬约束。
    observer_tip_rotation = quaternion_rotation(state.observer.tip_pose.quaternion_wxyz)
    tip_axis = observer_tip_rotation[:, 2]
    direction = state.executor.tip_pose.position - state.observer.tip_pose.position
    direction /= max(np.linalg.norm(direction), 1.0e-12)
    alignment = float(np.clip(tip_axis @ direction, -1.0, 1.0))
    error = direction - alignment * tip_axis
    if alignment < 0.0 and np.linalg.norm(error) <= 1.0e-8:
        # 反向时点积明确指出 180° 错误；选择确定的正交轴脱离反向平衡点。
        reference = np.array([0.0, 0.0, 1.0])
        if abs(tip_axis @ reference) > 0.9:
            reference = np.array([1.0, 0.0, 0.0])
        error = reference - (reference @ tip_axis) * tip_axis
        error /= max(np.linalg.norm(error), 1.0e-12)
    axis_jacobian = geometry.observer_root_world[:3, :3] @ direction_jacobian(
        state.observer.bending, robot.flexure_lengths_m, robot.distal_offsets_m)
    desired = config['orientation_gain'] * error
    orientation_rate = solve_velocity(axis_jacobian, desired, damping=1.0e-6)
    avoidance_rate = np.zeros(model.bending_size)
    avoidance_rows = []
    avoidance_tasks = []
    feedforward_candidate = None
    feedforward_closing = 0.0
    candidate_data = []
    for distance_center, executor_point, observer_point, i, j, se, so in candidates:
        if cancelled():
            return None, 'deadline', None
        delta = observer_point - executor_point
        normal = delta / max(distance_center, 1.0e-12)
        observer_j = (1.0-so)*observer_jacobians[j] + so*observer_jacobians[j+1]
        executor_j = (1.0-se)*executor_jacobians[i] + se*executor_jacobians[i+1]
        drift = float(normal @ executor_j @ executor_rate)
        gap = distance_center - config.get('diameter_m', 0.0125)
        orientation_closing = max(-float(normal @ observer_j @ orientation_rate), 0.0)
        avoidance_active = avoidance_latched and gap < release_gap
        predicted_gap = gap - orientation_closing * dt
        candidate_data.append((
            predicted_gap, orientation_closing, normal, observer_j, executor_j,
            gap, drift, avoidance_active))

        if avoidance_active:
            # 速度阻尼器：在命令层抵消主臂闭合速度，并恢复距离误差。
            # normal 从主臂指向从臂，因此 drift>0 表示主臂正在逼近从臂。
            closing_speed = max(drift, 0.0)
            # 进入预警区即开始按距触发阈值恢复距离，而不是等到
            # clearance_m 内才产生距离恢复速度。
            avoidance_speed = (closing_speed
                               + config['avoidance_gain']
                               * max(trigger_gap - gap, 0.0))
            avoidance_rows.append(normal @ observer_j)
            avoidance_tasks.append((gap, normal, observer_j, avoidance_speed))

    avoidance_weights = []
    if avoidance_tasks:
        raw_weights = np.asarray([
            1.0 / max(task[0], 0.003) for task in avoidance_tasks])
        avoidance_weights = (raw_weights / np.sum(raw_weights)).tolist()
        stacked_jacobian = np.vstack([
            np.sqrt(weight) * task[2]
            for weight, task in zip(avoidance_weights, avoidance_tasks)])
        stacked_target = np.concatenate([
            np.sqrt(weight) * task[1] * task[3]
            for weight, task in zip(avoidance_weights, avoidance_tasks)])
        # 多个危险点统一最小二乘求解，避免分别逆解后直接相加。
        avoidance_rate = solve_velocity(
            stacked_jacobian, stacked_target, damping=1.0e-6)

    # 朝向前馈作用于预测距离最小的三个候选线段对。
    feedforward_rate = np.zeros(model.bending_size)
    feedforward_rows = []
    feedforward_weights = []
    if candidate_data:
        feedforward_candidates = sorted(candidate_data, key=lambda item: item[0])[:3]
        active_feedforward = []
        for (_predicted_gap, feedforward_closing, normal, observer_j, _executor_j,
             _gap, _drift, _avoidance_active) in feedforward_candidates:
            if feedforward_closing > 1.0e-10:
                active_feedforward.append(
                    (_predicted_gap, feedforward_closing, normal, observer_j))
        if active_feedforward:
            raw_weights = np.asarray([
                1.0 / max(item[0], 0.001) for item in active_feedforward])
            feedforward_weights = (raw_weights / np.sum(raw_weights)).tolist()
            stacked_jacobian = np.vstack([
                np.sqrt(weight) * item[3]
                for weight, item in zip(feedforward_weights, active_feedforward)])
            feedforward_gain = config.get('orientation_feedforward_gain', 1.0)
            stacked_target = np.concatenate([
                np.sqrt(weight) * item[2] * feedforward_gain * item[1]
                for weight, item in zip(feedforward_weights, active_feedforward)])
            # 多个危险点统一最小二乘求解，避免分别逆解后直接相加。
            feedforward_rate = solve_velocity(
                stacked_jacobian, stacked_target, damping=1.0e-6)
            feedforward_rows = [item[2] @ item[3] for item in active_feedforward]

    # 将低优先级任务投影到高优先级避碰任务的零空间，避免直接相加抵消
    # 避碰方向。零空间投影只使用矩阵运算，不调用额外的约束求解器。
    identity = np.eye(model.bending_size)
    avoidance_matrix = (np.asarray(avoidance_rows).reshape(-1, model.bending_size)
                        if avoidance_rows else np.empty((0, model.bending_size)))
    avoidance_nullspace = (identity - np.linalg.pinv(avoidance_matrix) @ avoidance_matrix
                           if len(avoidance_matrix) else identity)
    feedforward_safe = avoidance_nullspace @ feedforward_rate
    if feedforward_rows:
        priority_rows = np.vstack((avoidance_matrix, np.asarray(feedforward_rows)))
    else:
        priority_rows = avoidance_matrix
    priority_nullspace = (identity - np.linalg.pinv(priority_rows) @ priority_rows
                          if len(priority_rows) else identity)
    orientation_safe = priority_nullspace @ orientation_rate

    # 只检查少量预定义的 alpha/beta 组合，不在循环中重新求解速度问题。
    # 顺序体现优先级：先保留避碰，再逐步恢复前馈和朝向任务。
    task_candidates = ((1.0, 1.0), (1.0, 0.5), (1.0, 0.0),
                       (0.5, 0.0), (0.0, 0.0))
    selected_rate = avoidance_rate.copy()
    best_predicted_gap = -np.inf
    selected_alpha, selected_beta = task_candidates[-1]
    candidate_predicted_gaps = []
    clearance_target = config.get('clearance_m', 0.003)
    for alpha, beta in task_candidates:
        command = avoidance_rate + alpha * feedforward_safe + beta * orientation_safe
        predicted_gaps = []
        for (_predicted_gap, _closing, normal, observer_j, executor_j,
             gap, _drift, _active) in candidate_data:
            distance_rate = float(
                normal @ (observer_j @ command - executor_j @ executor_rate))
            predicted_gaps.append(gap + dt * distance_rate)
        predicted_gap = min(predicted_gaps) if predicted_gaps else np.inf
        candidate_predicted_gaps.append(float(predicted_gap))
        if predicted_gap > best_predicted_gap:
            best_predicted_gap = predicted_gap
            selected_rate = command
            selected_alpha, selected_beta = alpha, beta
        # 按优先级从上到下选择；首次达到目标距离即停止检查后续组合。
        if predicted_gap >= clearance_target:
            selected_rate = command
            selected_alpha, selected_beta = alpha, beta
            break
    if enter_stage is not None:
        enter_stage('camera_jacobian')
    a = np.asarray(rows).reshape(-1, model.bending_size)
    b = np.asarray(lower)
    if enter_stage is not None:
        enter_stage('velocity_solver')
    # 朝向、普通避碰和单个最危险段的前馈均为软命令；只有机械/肌腱/曲率
    # 限制仍作为硬约束参与最终速度投影。
    desired_rate = selected_rate
    correction, feasible = constrained_velocity(
        np.eye(model.bending_size), desired_rate, a, b,
        iterations=config['iterations'], cancelled=cancelled)
    if not feasible:
        return None, 'solve_failed', (a, b)
    if cancelled():
        return None, 'deadline', (a, b)
    final_predicted_gaps = [
        gap + dt * float(normal @ (observer_j @ correction - executor_j @ executor_rate))
        for (_predicted_gap, _closing, normal, observer_j, executor_j,
             gap, _drift, _active) in candidate_data]
    diagnostics = {
        'selected_alpha': float(selected_alpha),
        'selected_beta': float(selected_beta),
        'pre_projection_predicted_min_gap_m': float(best_predicted_gap),
        'predicted_min_gap_m': float(min(final_predicted_gaps)
                                     if final_predicted_gaps else np.inf),
        'candidate_predicted_gaps_m': candidate_predicted_gaps,
        'avoidance_rate_norm': float(np.linalg.norm(avoidance_rate)),
        'avoidance_candidate_count': len(avoidance_weights),
        'avoidance_weights': [float(weight) for weight in avoidance_weights],
        'feedforward_rate_norm': float(np.linalg.norm(feedforward_safe)),
        'orientation_rate_norm': float(np.linalg.norm(orientation_safe)),
        'feedforward_candidate_count': len(feedforward_weights),
        'feedforward_weights': [float(weight) for weight in feedforward_weights],
        'avoidance_latched': bool(avoidance_latched),
        'avoidance_trigger_m': float(trigger_gap),
        'avoidance_release_m': float(release_gap),
        'orientation_feedforward_gain': float(
            config.get('orientation_feedforward_gain', 1.0)),
    }
    return correction, 'tracking', (a, b, diagnostics)
