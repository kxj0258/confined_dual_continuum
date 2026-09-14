"""仅计算 PCC 末端位置及解析 Jacobian，避免中心线采样和重复差分。"""
import numpy as np

_DW = np.array([[[0., 0., 1.], [0., 0., 0.], [-1., 0., 0.]],
                [[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]]])


def _functions(x):
    """sinc、cosc、sincc 及其对 θ² 的导数；直立附近避免消减误差。"""
    if x < 1e-3:
        return (1-x/6+x*x/120-x**3/5040+x**4/362880,
                .5-x/24+x*x/720-x**3/40320+x**4/3628800,
                1/6-x/120+x*x/5040-x**3/362880+x**4/39916800,
                -1/6+x/60-x*x/1680+x**3/90720,
                -1/24+x/360-x*x/13440+x**3/907200,
                -1/120+x/2520-x*x/120960+x**3/9979200)
    theta = np.sqrt(x)
    sine, cosine = np.sin(theta), np.cos(theta)
    return (sine/theta, (1-cosine)/x, (theta-sine)/theta**3,
            (theta*cosine-sine)/(2*theta**3),
            (theta*sine-2*(1-cosine))/(2*x*x),
            (theta*(1-cosine)-3*(theta-sine))/(2*theta**5))


def tip_position_jacobian(bending, lengths, offsets):
    q = np.asarray(bending, dtype=float)
    n = len(lengths)
    if (q.shape != (2*n,) or len(offsets) != n or not np.all(np.isfinite(q))
            or not np.all(np.isfinite(lengths)) or not np.all(np.isfinite(offsets))
            or np.any(np.asarray(lengths) <= 0) or np.any(np.asarray(offsets) < 0)):
        raise ValueError('PCC 曲率、长度或刚性偏置无效。')
    transform = np.eye(4)
    derivatives = np.zeros((2*n, 4, 4))
    for i, (length, offset) in enumerate(zip(lengths, offsets, strict=True)):
        kx, ky = q[2*i:2*i+2]
        w = np.array([[0., 0., kx], [0., 0., ky], [-kx, -ky, 0.]])
        w2 = w@w
        s, c, v, ds, dc, dv = _functions((kx*kx+ky*ky)*length*length)
        a, b, d = length*s, length**2*c, length**3*v
        rotation = np.eye(3)+a*w+b*w2
        local = np.eye(4)
        local[:3, :3] = rotation
        local[:3, 3] = (length*np.eye(3)+b*w+d*w2)[:, 2]+offset*rotation[:, 2]
        local_derivatives = np.zeros((2, 4, 4))
        for axis, value in enumerate((kx, ky)):
            dx = 2*length*length*value
            dw, dw2 = _DW[axis], _DW[axis]@w+w@_DW[axis]
            dr = length*ds*dx*w+a*dw+length**2*dc*dx*w2+b*dw2
            dp = (length**2*dc*dx*w+b*dw+length**3*dv*dx*w2+d*dw2)[:, 2]
            local_derivatives[axis, :3, :3] = dr
            local_derivatives[axis, :3, 3] = dp+offset*dr[:, 2]
        derivatives = derivatives@local
        derivatives[2*i:2*i+2] += transform@local_derivatives
        transform = transform@local
    return transform[:3, 3].copy(), derivatives[:, :3, 3].T.copy()


def centerline_position_jacobians(bending, lengths, offsets, arcs):
    """一次返回多个中心线点及其解析位置 Jacobian。"""
    lengths = tuple(float(value) for value in lengths)
    offsets = tuple(float(value) for value in offsets)
    total = sum(lengths) + sum(offsets)
    values = np.asarray(arcs, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > total):
        raise ValueError('中心线弧长必须位于有效范围内。')
    def local(length, offset, kx, ky):
        w = np.array([[0., 0., kx], [0., 0., ky], [-kx, -ky, 0.]])
        w2 = w @ w
        s, c, v, ds, dc, dv = _functions((kx*kx+ky*ky)*length*length)
        a, b, d = length*s, length**2*c, length**3*v
        rotation = np.eye(3)+a*w+b*w2
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = (length*np.eye(3)+b*w+d*w2)[:, 2]+offset*rotation[:, 2]
        derivatives = np.zeros((2, 4, 4))
        for axis, value in enumerate((kx, ky)):
            dx = 2*length*length*value
            dw, dw2 = _DW[axis], _DW[axis]@w+w@_DW[axis]
            dr = length*ds*dx*w+a*dw+length**2*dc*dx*w2+b*dw2
            dp = (length**2*dc*dx*w+b*dw+length**3*dv*dx*w2+d*dw2)[:, 2]
            derivatives[axis, :3, :3] = dr
            derivatives[axis, :3, 3] = dp+offset*dr[:, 2]
        return transform, derivatives

    q = np.asarray(bending, dtype=float)
    prefix_transforms = [np.eye(4)]
    prefix_derivatives = [np.zeros((len(q), 4, 4))]
    for index, (length, offset) in enumerate(zip(lengths, offsets, strict=True)):
        transform, local_derivatives = local(
            length, offset, q[2*index], q[2*index+1])
        previous = prefix_transforms[-1]
        derivatives = prefix_derivatives[-1] @ transform
        derivatives[2*index:2*index+2] += previous @ local_derivatives
        prefix_transforms.append(previous @ transform)
        prefix_derivatives.append(derivatives)

    points, jacobians = [], []
    boundaries = np.cumsum([0.0, *[length + offset for length, offset in zip(lengths, offsets, strict=True)]])
    for arc in values:
        segment = min(int(np.searchsorted(boundaries[1:], arc, side='right')), len(lengths)-1)
        start = boundaries[segment]
        local_arc = float(arc-start)
        length, offset = lengths[segment], offsets[segment]
        partial_length = min(local_arc, length)
        partial_offset = max(local_arc-length, 0.0)
        transform, local_derivatives = local(
            partial_length, partial_offset, q[2*segment], q[2*segment+1])
        prefix = prefix_transforms[segment]
        derivatives = prefix_derivatives[segment] @ transform
        derivatives[2*segment:2*segment+2] += prefix @ local_derivatives
        combined = prefix @ transform
        points.append(combined[:3, 3])
        jacobians.append(derivatives[:, :3, 3].T)
    return np.asarray(points), np.asarray(jacobians)
