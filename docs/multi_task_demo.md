# 多任务专利验证 Demo

## 场景布局

验证台覆盖约 650 × 400 × 300 mm，由四个沿 X 轴排列的工位组成：

1. **有限空间视觉探查**：基座停留在 52 mm 入口外，Observer 携带末端摄像头沿五点路径采集视频；Executor 是避让臂。
2. **曲面巡检**：Observer 沿离散弧面完成五点巡检，采集缺陷标记画面；Executor 继续承担避让。
3. **力位擦拭**：Executor 沿三条往复轨迹执行切向位置控制和法向力控制；Observer 位于自由空间侧观察。
4. **非对称障碍操作**：Executor 穿越双立柱路径，Observer 执行观察和主动避让。

## 阶段序列

```text
INITIALIZE
→ BASE_APPROACH
→ BASE_LOCK
→ CONFINED_VISUAL_INSPECTION
→ CURVED_SURFACE_TRANSIT
→ CURVED_SURFACE_INSPECTION
→ WIPING_TRANSIT
→ FORCE_POSITION_WIPING
→ OBSTACLE_TRANSIT
→ ASYMMETRIC_OBSTACLE_OPERATION
→ COMPLETE
```

任务不执行自动回撤。完成后保持最终状态，由用户关闭 viewer 或重新运行复位。

## 输出

- `observer_inspection.mp4`：Observer 末端 RGB 摄像头视频；
- `timeseries.csv`：阶段、位姿、双臂距离、接触力和肌腱相容残差；
- `summary.json`：完成状态、最小双臂距离、最大接触力和最大相容残差；
- `config.yaml`：本次运行配置快照。
