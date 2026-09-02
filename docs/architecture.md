# 专利正式实现架构

## 依赖方向

```text
domain
↑
workspace / actuation / kinematics
↑
task / coordination
↑
application
↑
adapters
↑
cli
```

任何底层模块不得反向导入 application、experiments、MuJoCo 或可视化模块。

## 层级职责

- `domain`：稳定状态、命令和几何数据，不包含控制算法。
- `workspace`：受限空间入口、任务区域、禁止区域、接触面及查询协议。
- `task`：阶段状态机和任务级转换条件，不求解肌腱命令。
- `coordination`：Executor、Observer、双臂安全和非对称命令合并。
- `actuation`：弯曲空间与物理肌腱映射、相容约束执行。
- `application`：统一控制器、后端端口和完整任务会话。
- `adapters`：实现应用端口，不被应用层反向依赖。
- `diagnostics`：正式实验时序和汇总记录。
- `cli`：装配配置、MuJoCo 和可选 viewer，是唯一可执行入口。

## 核心数据流

```text
WorkspaceSnapshot + OperationProgress
→ ConfinedOperationSupervisor
→ PhaseDecision
→ OperationDirector
→ ExecutorController / ObserverController / InterArmSafety
→ DualArmCoordinator
→ CompatibleTendonController
→ RobotCommand
→ Backend
```
