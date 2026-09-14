# confined_dual_continuum

面向有限入口复杂受限空间的分阶段非对称双连续体协同控制项目。

```text
任务层：轨迹与阶段管理
→ 协同层：Executor / Observer 独立求解
→ 执行层：弯曲空间相容肌腱控制
→ 适配层：MuJoCo 场景、记录和可视化
```

## 当前基线

当前开发基线为 `codex/clearance-optimization`。最新轨迹目录来自 2026-09-14 的工作空间采集，包含 `circle`、`figure8` 和 `rectangle`。

详细状态与结果见：[当前状态](docs/current_status.md)、[最新运行结果](docs/results.md)、[正式架构](docs/architecture.md)、[轨迹目录](docs/trajectory_catalog.md)。

## 运行环境

以下命令均从仓库根目录、`continuum_sim` 环境执行。

## 工作空间采集

```powershell
python -m confined_dual_continuum.drive_workspace_probe `
  --config configs/experiments/drive_workspace_probe.yaml `
  --mode workspace `
  --workers 8
```

## 轨迹生成与验证

```powershell
python -m confined_dual_continuum.trajectory_catalog `
  --build outputs/drive_workspace_diagnostics/<timestamp-id>/samples.jsonl
python -m confined_dual_continuum.trajectory_catalog --validate all
```

## 单臂 Executor

单臂当前使用 20 Hz realtime 模式：

```powershell
python -m confined_dual_continuum.executor_tracking --config configs/experiments/executor_tracking.yaml --trajectory circle
python -m confined_dual_continuum.executor_tracking --config configs/experiments/executor_tracking.yaml --trajectory figure8
python -m confined_dual_continuum.executor_tracking --config configs/experiments/executor_tracking.yaml --trajectory rectangle
```

## 双臂 Executor / Observer

```powershell
python -m confined_dual_continuum.executor_observer_tracking --config configs/experiments/executor_observer_tracking.yaml --trajectory circle
python -m confined_dual_continuum.executor_observer_tracking --config configs/experiments/executor_observer_tracking.yaml --trajectory figure8
python -m confined_dual_continuum.executor_observer_tracking --config configs/experiments/executor_observer_tracking.yaml --trajectory rectangle
```

## 设计约束

- 控制核心不依赖 MuJoCo、viewer、YAML 或绘图库。
- Executor 与 Observer 独立求解，只在命令层协同。
- 正式路径输出弯曲空间相容的肌腱命令。
- Observer 不使用基座自由度进行求解。
- 工作空间和轨迹覆盖是有限采样证据，不代表连续空间证明。
- `completed` 只表示轨迹时钟走完，不代表误差或安全验收通过。

响应诊断、坐标检查、PCC 交互和工作空间分析属于实验/诊断入口，说明见 `docs/` 和 `experiments/README.md`。
