# 双臂 Executor / Observer 跟踪

双臂入口为 `confined_dual_continuum.executor_observer_tracking`。Executor 主臂独立跟踪轨迹，Observer 从臂独立求解观察方向和局部避碰约束；主臂不等待从臂成功。

当前配置使用 20 Hz 控制周期。主臂和从臂共享弯曲空间相容肌腱执行器，但求解状态和运行后端彼此独立。

运行示例：

```powershell
python -m confined_dual_continuum.executor_observer_tracking `
  --config configs/experiments/executor_observer_tracking.yaml `
  --trajectory circle
```

可将轨迹替换为 `figure8` 或 `rectangle`。结果目录重点查看：

```text
summary.json
tracking.csv
observer.jsonl
tracking-error.svg
interarm-centerline-distance.svg
interarm-clearance.svg
global.mp4
observer.mp4
```

从臂保护由 `observer.joint_guard_enabled` 控制；主臂保护由顶层 `joint_guard_enabled` 控制。两者关闭时，完成状态不代表满足关节安全裕量或避碰验收。

Observer 的距离约束是局部速度约束，不是全局路径规划。录像和窗口属于诊断输出，不参与控制决策。
