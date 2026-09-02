# 专利最小实现架构

## 依赖方向

```text
domain
↑
workspace / actuation / kinematics
↑
task / coordination
↑
application / experiments
↑
adapters
```

任何底层模块不得反向导入 application、experiments、MuJoCo 或可视化模块。

## 层级职责

- `domain`：稳定状态、命令和几何数据，不包含控制算法。
- `workspace`：受限空间入口、任务区域、禁止区域、接触面及查询协议。
- `task`：阶段状态机和任务级转换条件，不求解肌腱命令。
- `coordination`：Executor、Observer、双臂安全和非对称命令合并。
- `actuation`：弯曲空间与物理肌腱映射、相容约束执行。
- `adapters`：仿真或物理设备边界。
- `experiments`：专利 Demo、消融和指标采集，不进入控制核心。

## 核心数据流

```text
WorkspaceSnapshot + OperationProgress
→ ConfinedOperationSupervisor
→ PhaseDecision
→ ExecutorController / ObserverTargetGenerator / ObserverController
→ DualArmCoordinator
→ CompatibleTendonController
→ RobotCommand
→ Backend
```

