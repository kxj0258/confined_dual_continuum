"""小规模严格凸速度问题的有界对偶坐标迭代；失败显式返回。"""
import numpy as np


def constrained_velocity(matrix, desired, constraints, lower, *, damping=1e-4,
                         iterations=300, tolerance=1e-7, cancelled=None):
    """最小化 ||M q-v||²+λ²||q||²，约束 A q>=b。

    Hildreth 对偶坐标更新；同时要求原始可行和迭代增量收敛。
    达到迭代上限即报告失败，调用方不得发送未确认可行的结果。
    """
    m, v = np.asarray(matrix, float), np.asarray(desired, float)
    a, b = np.asarray(constraints, float), np.asarray(lower, float)
    if (m.ndim != 2 or v.shape != (m.shape[0],) or a.ndim != 2
            or a.shape[1] != m.shape[1] or b.shape != (len(a),)
            or not all(np.all(np.isfinite(x)) for x in (m, v, a, b))
            or not np.isfinite(damping) or damping <= 0
            or iterations < 1 or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError('约束求解输入无效。')
    h = m.T@m+damping*damping*np.eye(m.shape[1])
    q = np.linalg.solve(h, m.T@v)
    if not len(a):
        return q, True
    norms = np.linalg.norm(a, axis=1)
    if np.any((norms < 1e-14) & (b > tolerance)):
        return np.zeros(m.shape[1]), False
    keep = norms >= 1e-14
    a, b = a[keep]/norms[keep, None], b[keep]/norms[keep]
    if not len(a):
        return q, True
    directions = np.linalg.solve(h, a.T).T
    diagonal = np.einsum('ij,ij->i', a, directions)
    multipliers = np.zeros(len(a))
    for _ in range(iterations):
        if cancelled is not None and cancelled():
            return np.zeros(m.shape[1]), False
        previous = q.copy()
        for i in range(len(a)):
            updated = max(0., multipliers[i]+(b[i]-a[i]@q)/diagonal[i])
            q += (updated-multipliers[i])*directions[i]
            multipliers[i] = updated
        if np.max(b-a@q) <= tolerance and np.linalg.norm(q-previous) <= tolerance:
            return q, True
    return np.zeros(m.shape[1]), False
