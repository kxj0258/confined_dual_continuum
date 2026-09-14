"""连续线段最近点，不依赖机器人或仿真。"""
import numpy as np


def closest_segments(a, b, c, d):
    u, v, w = b-a, d-c, a-c
    aa, bb, cc = float(u@u), float(u@v), float(v@v)
    dd, ee = float(u@w), float(v@w)
    candidates = []
    for s in (0.0, 1.0):
        t = float(np.clip((ee+s*bb)/cc, 0, 1)) if cc > 1e-20 else 0.0
        candidates.append((s,t))
    for t in (0.0, 1.0):
        s = float(np.clip((t*bb-dd)/aa, 0, 1)) if aa > 1e-20 else 0.0
        candidates.append((s,t))
    determinant = aa*cc-bb*bb
    if determinant > 1e-12*aa*cc:
        s, t = (bb*ee-cc*dd)/determinant, (aa*ee-bb*dd)/determinant
        if 0 <= s <= 1 and 0 <= t <= 1:
            candidates.append((s,t))
    s,t = min(candidates, key=lambda pair: np.linalg.norm(w+pair[0]*u-pair[1]*v))
    return a+s*u, c+t*v, s, t
