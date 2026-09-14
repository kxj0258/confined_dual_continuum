# 轨迹目录

当前轨迹定义位于 `configs/trajectories/`，使用 `mobile_base_frame` 基座坐标，单位为米。每条轨迹包含主体曲线、入场路径和覆盖报告。

当前轨迹：

```text
circle
figure8
rectangle
```

主体运行时间为 20 秒，入场时间为 6 秒，总时间为 26 秒。运行时 `time_scale` 同时缩放两者。

## 轨迹类型

- `circle`：平面圆形轨迹。
- `figure8`：平面 8 字轨迹。
- `rectangle`：平面矩形轨迹，角点速度和加速度归零。

## 生成与验证

工作空间采集完成后，从最新采样记录重新生成定义：

```powershell
python -m confined_dual_continuum.trajectory_catalog `
  --build outputs/drive_workspace_diagnostics/<timestamp-id>/samples.jsonl
```

验证全部内置轨迹：

```powershell
python -m confined_dual_continuum.trajectory_catalog --validate all
```

覆盖报告只说明轨迹位于采样点云邻域，不证明连续空间可达、无碰撞或动力学跟踪误差达标。工作空间或轨迹定义变化后，必须重新生成覆盖报告。
