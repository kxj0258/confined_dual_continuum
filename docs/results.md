# 最新运行结果

结果日期：2026-09-14

## 轨迹覆盖验证

`python -m confined_dual_continuum.trajectory_catalog --validate all` 已通过：

| 轨迹 | 入场点 | 主体点 | 最大最近邻距离 | 连续上界 | 容差 |
|---|---:|---:|---:|---:|---:|
| circle | 96 | 577 | 3.869 mm | 4.119 mm | 4.216 mm |
| figure8 | 51 | 225 | 3.723 mm | 3.973 mm | 4.216 mm |
| rectangle | 67 | 337 | 3.530 mm | 3.780 mm | 4.216 mm |

报告 verdict 均为 `sampled_pointcloud_coverage_only`。这表示通过采样点云覆盖检查，不表示连续空间、碰撞或动力学验收已经证明。

## 工作空间采集

采集记录为：

```text
5522/5522 workspace mechanical observer step joint_limit
```

采集目录：

```text
outputs/drive_workspace_diagnostics/20260914T061705Z-125ec168/
```

报告：

```text
outputs/drive_workspace_diagnostics/20260914T061705Z-125ec168/report/report.html
```

## 单臂 Executor

最新 `circle` 运行：

```text
outputs/executor_tracking/20260914T071924Z-211a9c84
```

关键结果：

```text
completed: true
termination_reason: trajectory_elapsed
control_hz: 20
mode: realtime
simulation_time_s: 26.0
realtime_factor: 0.9996
RMSE: 0.996 mm
maximum_error: 2.852 mm
saturation_fraction: 0
maximum_compatibility_residual: 2.92e-18
```

此前同一基线下的 `figure8` 和 `rectangle` 结果也均以 `trajectory_elapsed` 完成。

## 双臂 Executor / Observer

最新 `circle` 运行：

```text
outputs/executor_observer_tracking/20260914T072036Z-a02e5a91
```

关键结果：

```text
completed: true
termination_reason: trajectory_elapsed
control_hz: 20
observer_control_hz: 20
simulation_time_s: 26.0
realtime_factor: 0.9998
RMSE: 0.996 mm
maximum_error: 2.852 mm
saturation_fraction: 0
deadline_misses: 0
observer tracking: 519
observer accepted_commands: 519
observer moving_commands: 519
```

录像结果包含 397 个丢弃快照和 396 个重复帧。该问题属于视频采集吞吐，不影响控制循环完成，但视频不应被解释为每一帧都来自独立的新物理渲染。

此前同一基线下的 `figure8` 和 `rectangle` 双臂运行也均以 `trajectory_elapsed` 完成。

## 解释边界

`completed` 只表示轨迹时钟走完。当前运行配置关闭关节保护，且轨迹覆盖是有限采样证据，因此这些结果不自动证明所有时刻满足关节安全裕量、避碰安全、观察可见性或实际机构复现条件。
