# 单臂 Executor 跟踪

单臂入口为 `confined_dual_continuum.executor_tracking`。它只推进 Executor 主臂，不包含 Observer、相机跟随或双臂避碰。

当前配置 `configs/experiments/executor_tracking.yaml`：

```text
控制频率：20 Hz
运行模式：realtime
轨迹时间：26 s
位置增益：2.0
关节保护：关闭
```

运行示例：

```powershell
python -m confined_dual_continuum.executor_tracking `
  --config configs/experiments/executor_tracking.yaml `
  --trajectory circle
```

将 `circle` 替换为 `figure8` 或 `rectangle` 可运行其他轨迹。

结果目录中的 `summary.json` 记录完成状态、终止原因、实时倍率、误差和相容性残差；`tracking.csv` 保存逐周期反馈，`tracking-error.svg` 和视频用于查看跟踪效果。

`completed` 只表示轨迹时钟走完。当前关闭关节保护是为了观察完整轨迹，不应解释为安全设置。
