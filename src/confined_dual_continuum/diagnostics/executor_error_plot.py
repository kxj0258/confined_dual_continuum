"""误差图统一量程和完整时段 SVG 导出；不依赖窗口或绘图库。"""
import math
import numpy as np
from confined_dual_continuum.diagnostics.executor_records import COLUMNS

ERROR_COLUMNS = tuple(COLUMNS.index(name) for name in
                      ('error_x_m', 'error_y_m', 'error_z_m', 'error_norm_m'))
COLORS = ('#d64b42', '#2d955f', '#397cc1', '#202124')


def expanded_limit(maximum_mm, previous=10.):
    """初始 ±10 mm；越界时增加余量、向上取整，运行期间不缩小。"""
    if maximum_mm <= previous:
        return previous
    return float(math.ceil(maximum_mm*1.1))


def save_error_plot(rows, path):
    if not rows:
        return False
    data = np.asarray(rows, dtype=float)
    values = data[:, ERROR_COLUMNS]*1000
    limit = 10.
    for maximum in np.max(np.abs(values), axis=1):
        limit = expanded_limit(float(maximum), limit)
    left, right, top, bottom = 85., 1160., 65., 535.
    t0, t1 = float(data[0, 0]), max(float(data[-1, 0]), float(data[0, 0])+.02)
    x = left+(data[:, 0]-t0)/(t1-t0)*(right-left)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="600" viewBox="0 0 1200 600">',
             '<rect width="1200" height="600" fill="white"/>',
             '<g font-family="sans-serif" font-size="14" fill="#333">',
             '<text x="85" y="28">主臂末端跟踪误差（完整时段）</text>',
             '<text x="12" y="50">误差 / mm</text>']
    for value in np.linspace(-limit, limit, 5):
        y = bottom-(value+limit)/(2*limit)*(bottom-top)
        color = '#888' if value == 0 else '#ddd'
        parts.append(f'<path d="M {left} {y} H {right}" stroke="{color}"/>')
        parts.append(f'<text x="75" y="{y+5}" text-anchor="end">{value:g}</text>')
    for t in np.linspace(t0, t1, 7):
        xpos = left+(t-t0)/(t1-t0)*(right-left)
        parts.append(f'<text x="{xpos}" y="560" text-anchor="middle">{t:.2f}</text>')
    parts.append('<text x="570" y="590">仿真时间 / s</text>')
    for index, (name, color) in enumerate(zip(('ex', 'ey', 'ez', '误差范数'), COLORS)):
        y = bottom-(values[:, index]+limit)/(2*limit)*(bottom-top)
        points = ' '.join(f'{px:.3f},{py:.3f}' for px, py in zip(x, y))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.6"/>')
        parts.append(f'<text x="{650+index*125}" y="28" fill="{color}">{name}</text>')
    parts.append('</g></svg>')
    path.write_text('\n'.join(parts), encoding='utf-8')
    return True


def save_interarm_plot(rows, path, trigger_m=0.007, release_m=0.010,
                       diameter_m=0.0125, centerline=False):
    """导出主从臂最小间隙或中心线距离曲线。"""
    if not rows:
        return False
    data = np.asarray(rows, dtype=float)
    times = data[:, 0]
    clearances = data[:, 1]*1000
    if centerline:
        clearances += diameter_m*1000
    left, right, top, bottom = 85., 1160., 65., 535.
    t0, t1 = float(times[0]), max(float(times[-1]), float(times[0])+.02)
    minimum = min(float(np.min(clearances)), diameter_m*1000 if centerline else 0.)
    maximum = max(float(np.max(clearances)),
                  (diameter_m+release_m)*1000 if centerline else release_m*1000)
    margin = max((maximum-minimum)*.1, 1.)
    minimum -= margin
    maximum += margin
    x = left+(times-t0)/(t1-t0)*(right-left)

    def y(value):
        return bottom-(value-minimum)/(maximum-minimum)*(bottom-top)

    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="600" viewBox="0 0 1200 600">',
             '<rect width="1200" height="600" fill="white"/>',
             '<g font-family="sans-serif" font-size="14" fill="#333">',
             f'<text x="85" y="28">主从臂{"中心线距离" if centerline else "最小间隙"}（完整时段）</text>',
             f'<text x="12" y="50">{"中心线距离" if centerline else "最小间隙"} / mm</text>']
    for value in np.linspace(minimum, maximum, 7):
        parts.append(f'<path d="M {left} {y(value)} H {right}" stroke="#ddd"/>')
        parts.append(f'<text x="75" y="{y(value)+5}" text-anchor="end">{value:.1f}</text>')
    offset = diameter_m*1000 if centerline else 0.
    for value, color, label in ((offset, '#d64b42', '碰撞边界'),
                                (offset+trigger_m*1000, '#d97706', '激活阈值'),
                                (offset+release_m*1000, '#397cc1', '释放阈值')):
        parts.append(f'<path d="M {left} {y(value)} H {right}" stroke="{color}" stroke-dasharray="6,4"/>')
        parts.append(f'<text x="{right-5}" y="{y(value)-5}" text-anchor="end" fill="{color}">{label} {value:g} mm</text>')
    for t in np.linspace(t0, t1, 7):
        xpos = left+(t-t0)/(t1-t0)*(right-left)
        parts.append(f'<text x="{xpos}" y="560" text-anchor="middle">{t:.2f}</text>')
    parts.append('<text x="570" y="590">仿真时间 / s</text>')
    points = ' '.join(f'{px:.3f},{y(value):.3f}' for px, value in zip(x, clearances))
    parts.append(f'<polyline points="{points}" fill="none" stroke="#202124" stroke-width="1.6"/>')
    parts.append('</g></svg>')
    path.write_text('\n'.join(parts), encoding='utf-8')
    return True


def save_centerline_distance_plot(rows, path, trigger_m=0.007,
                                  release_m=0.010, diameter_m=0.0125):
    """导出与运行时监视器一致的主从臂中心线最小距离曲线。"""
    return save_interarm_plot(rows, path, trigger_m, release_m, diameter_m,
                              centerline=True)
