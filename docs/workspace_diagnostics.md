# 工作空间采集

工作空间采集入口为：

```powershell
python -m confined_dual_continuum.drive_workspace_probe `
  --config configs/experiments/drive_workspace_probe.yaml `
  --mode workspace `
  --workers 8
```

采集固定 XML 初始基座，不移动基座，不创建 viewer 或视频。每次只驱动一条臂，另一条臂保持零位。结果是当前场景和约束下的有限采样集合，不是无障碍完整工作空间，也不是十二维任意构型集合。

## 采集判据

每个样本经过相容肌腱控制推进，并记录目标/实际肌腱位移、PCC 与 MuJoCo 反馈、中心线、末端误差、稳定状态、关节限制、碰撞间距和失败原因。达到目标后，关节、肌腱和物理末端速度需连续满足配置的稳定窗口，样本才进入稳定集合。

采集安全监督使用原始离散中心线和 MuJoCo 环境几何；反馈来源选择不会替换物理真值通道。中断样本不计入稳定工作空间。

## 最新采集

2026-09-14 的最新记录为：

```text
5522/5522 workspace mechanical observer step joint_limit
```

结果目录：

```text
outputs/drive_workspace_diagnostics/20260914T061705Z-125ec168/
```

该目录中的 `samples.csv`、`stable_workspace.csv`、`summary.json` 和 `report/report.html` 用于后续轨迹目录生成。逐样本 trace 仅用于失败诊断，长期保存时可删除。

## 解释边界

工作空间点云不证明连续空间覆盖收敛，也不证明任意点之间存在可行路径。轨迹覆盖检查、动力学跟踪和双臂避碰需要分别验证。
