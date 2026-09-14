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
- `robot.model`：纯机器人参数和两臂独立路由；`robot.configuration` 与 `workspace.configuration` 仅由装配层导入。
- `WorkspaceQuery`：应用层查询环境的端口，MuJoCo 后端用真实碰撞几何实现。控制层不访问 MuJoCo 对象。

## 核心数据流

任务目标的基座坐标定义、双臂采样工作空间范围及使用限制见
[双臂基座坐标工作空间基线](workspace_base_frame.md)。范围仅供采样覆盖预筛选，
不能替代姿态求解、逐段曲率约束、关节保护或碰撞检查。

```text
RobotState
→ OperationDirector（当前阶段目标及完成信号）
→ ConfinedOperationSupervisor（更新阶段，转换周期保持）
→ ExecutorController / ObserverController / InterArmSafety
→ DualArmCoordinator
→ CompatibleTendonController
→ RobotCommand
→ Backend
```

阶段激活和终态是最后的运动权限边界。环境最近点约束、双臂最近点约束分别求解，所有正式速度最终经过弯曲相容限幅；执行不可行时双臂同时撤销尚未发送的积分并停止。逐肌腱非相容基线仅供显式 `--ablation` 使用。

`RobotBackend.stop()` 清除待执行速度；`OperationSession` 在成功、超时、中止和异常时均停止并保存记录。`COMPLETE` 只表示配置的几何/力位判据达成，不表示图像缺陷识别或擦除效果识别。

## 固定基座单向跟随任务

上述双臂停止合同用于原四工位任务。新增 `fixed_base_tracking` 任务采用显式单向合同：
`TimedTrajectory → FixedTrackingController → 主臂候选命令 → CameraFollower → 相容执行`。
从臂失败只保持从臂，不复用四工位双臂停止监督，也不改变主臂时钟。
相机和多点距离由 `TrackingBackend` 提供；采样文件在适配层读入，控制层只消费纯数据。
约束求解位于 kinematics，相机数据/候选筛选位于 domain/coordination。
具体合同与物理碰撞边界见 [双臂 Executor / Observer 跟踪](executor_observer_tracking.md)。

## 主臂独立跟踪程序

`executor_tracking` 是独立装配入口，不进入双臂协调链。
`CatalogTrajectory → ExecutorTrackingController → CompatibleTendonController` 仅依赖纯参数与反馈。
`trajectory_catalog` 装配入口离线读取采样记录和 YAML，调用 `workspace/pointcloud_coverage` 生成覆盖证据；控制循环不查询点云或读取轨迹文件。
`ExecutorBackend` 在适配层负责单臂 MuJoCo 场景和步进；控制层不引用 MuJoCo 或窗口。
`fast_pcc` 提供解析位置 Jacobian；`executor_monitor` 只消费数值记录。
视频在控制结束后用独立数据实例生成。公共机器人限制直接传入，不在任务配置覆盖。
参数和手动运行命令见 [主臂独立跟踪](executor_tracking.md)。

## 异步从臂相机扩展

`executor_observer_tracking` 在独立入口装配原主臂控制器、固定基座双臂后端和后台进程。
`coordination/observer_tracking` 只消费纯状态/几何数据，单次求解从臂曲率；
进程管理与 MuJoCo 状态重建位于 adapters/cli。当前双臂提交周期为20 Hz，主臂不等待从臂。
主从物理接触响应关闭但保留几何距离查询；从臂失败只保持自身位置目标。
视频在独立进程中运行期间持续生成。详细时序和限制见
[20 Hz从臂相机跟随](executor_observer_tracking.md)。
