# TCP 作业力控与 Observer 朝向观测实施规划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 `confined_dual_continuum.tcp_wiping` 实验链路，使主臂以“力传感器 + 工具 TCP”为唯一作业控制点，以 TCP 工作空间确定作业平面，并使从臂只执行朝向观测控制，同时保持当前已完成的入口、配置和结果可复现。

**Architecture:** 新链路采用独立任务配置、工作空间配置、MuJoCo 场景和装配入口，复用现有机器人参数、弯曲空间相容肌腱执行器、几何安全查询和诊断机制。主臂链路使用 `T_world_tcp = T_world_tip · T_tip_tcp` 计算 TCP 状态、TCP Jacobian 和作业误差；从臂只通过方向 Jacobian 跟踪指向 TCP 的视轴，不要求末端位置到达作业点。现有 `executor_tracking`、`executor_observer_tracking` 及其默认配置不改写。

**Tech Stack:** Python、NumPy、MuJoCo XML、YAML、现有 `CompatibleTendonController`、PCC 位置/方向 Jacobian、现有工作空间诊断与 JSON/CSV 记录。

## Global Constraints

- 新正式入口固定命名为 `confined_dual_continuum.tcp_wiping`。
- 新链路必须以工具 TCP 作为位置、轨迹、工作空间、力控和误差记录的统一控制点。
- 主臂末端串联力传感器和工具；力传感器输出与几何接触力分开记录，正式力控优先使用传感器输出。
- Observer 只做朝向控制，不要求末端位置跟踪；Observer 仍独立求解，不使用基座自由度替代末端控制。
- 暂时不打开关节保护，但保留环境间隙、双臂避碰、接触力上限和肌腱相容约束。
- 不修改现有任务配置、默认 MuJoCo 场景、现有轨迹目录或现有入口的默认行为。
- 不新增旧仓库兼容入口、别名模块或隐式 fallback。
- 控制层不得依赖 MuJoCo、viewer、文件格式或绘图库；MuJoCo 与文件装配放在 adapters/cli。
- 文档、配置说明和重要算法注释以中文为主。

---

## 参考项目与借鉴边界

参考项目：[`kxj0258/continuum_sim`](https://github.com/kxj0258/continuum_sim)

重点借鉴内容：

1. 将工具 TCP 作为自动任务统一控制点，目标生成、Jacobian、误差记录和可视化均使用同一 TCP。
2. 在 MuJoCo 中显式建模力传感器、工具碰撞几何和 TCP，而不是直接把连续体裸末端当作作业点。
3. 参考其擦拭任务中的接触建立、法向力控制、力误差、接触稳定步数、最大接触力和安全回撤参数。
4. 只借鉴接口和实验组织方式，不复制旧项目兼容入口，不改变本项目既有分层和独立求解约束。

参考资料：

- 项目主页及 README：<https://github.com/kxj0258/continuum_sim>
- 擦拭场景配置：<https://raw.githubusercontent.com/kxj0258/continuum_sim/codex/dual-mujoco-wiping-control-migration/configs/scenarios/mujoco_wiping.yaml>
- 当前项目架构：[docs/architecture.md](../../architecture.md)
- 当前项目双臂跟踪：[docs/executor_observer_tracking.md](../../executor_observer_tracking.md)
- 当前项目结果：[docs/results.md](../../results.md)

## 文件边界

### 新增文件

- `src/confined_dual_continuum/tcp_wiping.py`：新实验入口，只负责装配新配置、后端和任务会话。
- `src/confined_dual_continuum/application/tcp_wiping.py`：TCP 作业会话与控制输入装配，不承载 MuJoCo 逻辑。
- `src/confined_dual_continuum/coordination/tcp_force_position_controller.py`：TCP 切向位置与法向力控制。
- `src/confined_dual_continuum/coordination/observer_orientation_controller.py`：Observer 纯视轴方向控制。
- `src/confined_dual_continuum/domain/tcp.py`：TCP 位姿、力传感器读数和作业点数据结构。
- `src/confined_dual_continuum/workspace/tcp_workspace.py`：TCP 工作空间变换、筛选和作业平面候选计算。
- `src/confined_dual_continuum/adapters/tcp_wiping_backend.py`：新场景反馈和 TCP/传感器适配。
- `assets/mujoco/tcp_force_tool_demo.xml`：独立的力传感器、工具和 TCP 场景。
- `configs/experiments/tcp_wiping.yaml`：新入口配置。
- `configs/workspace/tcp_wiping_bench.yaml`：基于 TCP 工作空间筛选后的新作业环境。
- `configs/trajectories/tcp_workspace.json`、`tcp_workspace.npz`：TCP 工作空间目录及采样证据。
- `docs/tcp_wiping.md`：新链路使用说明、坐标定义、结果解释和限制。

### 明确不修改的基线文件

- `configs/experiments/executor_tracking.yaml`
- `configs/experiments/executor_observer_tracking.yaml`
- `configs/workspace/multi_task_patent_bench.yaml`
- `assets/mujoco/multi_task_patent_demo.xml`
- `configs/trajectories/workspace.json`
- `configs/trajectories/workspace.npz`

如果确实需要共享类型或纯函数，优先新增小型 domain/workspace 模块；禁止为了复用而改变现有入口的默认语义。

## 实施任务

### Task 1：冻结基线和确认 TCP 坐标合同

**Files:**
- Create: `docs/tcp_wiping_coordinate_contract.md`
- Inspect only: `assets/mujoco/multi_task_patent_demo.xml`, `src/confined_dual_continuum/adapters/mujoco_backend.py`, `src/confined_dual_continuum/kinematics/pcc.py`

**Interfaces:**
- Produces: `T_tip_sensor`、`T_sensor_tool`、`T_tool_tcp`、`T_tip_tcp` 的明确坐标合同，以及 TCP 位置/姿态、法向和力的坐标系约定。

- [ ] 记录现有 `executor_tip` 的坐标轴方向、位置和姿态来源。
- [ ] 定义串联变换：`T_tip_tcp = T_tip_sensor · T_sensor_tool · T_tool_tcp`。
- [ ] 规定 `force_sensor` 输出先转换到世界坐标，再计算 `F_n = force_world · surface_normal_world`。
- [ ] 明确 TCP 轨迹、作业面和诊断误差全部使用世界坐标下 TCP。
- [ ] 在文档中列出旧的裸末端指标与新的 TCP 指标，防止混用。

### Task 2：将已有主臂采样映射为 TCP 工作空间

**Files:**
- Create: `src/confined_dual_continuum/workspace/tcp_workspace.py`
- Create: `scripts/build_tcp_workspace.py`
- Create: `docs/tcp_workspace_analysis.md`
- Create: `tests/test_tcp_workspace.py`

**Interfaces:**
- Consumes: 现有工作空间样本中的 `T_world_tip`、弯曲状态、肌腱位移和可行性字段。
- Produces: `map_tip_samples_to_tcp(samples, tip_tcp_transform) -> TcpWorkspaceSamples`；输出 TCP 点云、姿态、有效性和覆盖统计。

- [ ] 实现刚体变换：`p_world_tcp = p_world_tip + R_world_tip @ p_tip_tcp`。
- [ ] 若历史样本缺少姿态，明确判定为不可转换，禁止静默使用裸末端位置代替 TCP。
- [ ] 保留每个样本的原始弯曲、肌腱、间隙和约束字段。
- [ ] 计算 TCP 点云边界、最近邻距离、有效点比例和作业平面候选区域。
- [ ] 编写不变量测试：零偏置时 TCP 点等于末端点；纯旋转偏置会改变 TCP 位置；非有限变换被拒绝。
- [ ] 只有在已有采样无法提供姿态时，才启动针对 TCP 的重新采集任务。

### Task 3：必要时重新采集 TCP 工作空间

**Files:**
- Create: `configs/experiments/tcp_workspace_probe.yaml`
- Create: `outputs/tcp_workspace_diagnostics/` 的新产物约定
- Modify only if necessary: `src/confined_dual_continuum/drive_workspace_probe.py`
- Create: `tests/test_tcp_workspace_probe_contract.py`

**Interfaces:**
- Consumes: Task 1 的 `T_tip_tcp` 和 Task 2 的样本格式。
- Produces: 独立的 `tcp_workspace` 采样目录，不覆盖现有 workspace 采样。

- [ ] 先检查历史样本是否包含完整末端位姿；完整时不重新采集。
- [ ] 需要重采集时，记录 `T_world_tip`、`T_world_tcp`、TCP 可行性、工具间隙和传感器读数。
- [ ] 将采样结果写入 `outputs/tcp_workspace_diagnostics/<timestamp>/`。
- [ ] 生成 `configs/trajectories/tcp_workspace.json` 和 `.npz`，不覆盖旧目录。
- [ ] 测试输出字段完整性和旧工作空间文件未被写入。

### Task 4：新增 TCP 工具与力传感器 MuJoCo 场景

**Files:**
- Create: `assets/mujoco/tcp_force_tool_demo.xml`
- Create: `src/confined_dual_continuum/adapters/tcp_wiping_backend.py`
- Create: `src/confined_dual_continuum/domain/tcp.py`
- Create: `tests/test_tcp_scene_contract.py`

**Interfaces:**
- Produces: `TcpFeedback(tcp_pose, sensor_force_world, sensor_torque_world, geometric_contact_force_world, ...)`。

- [ ] 从现有场景复制并建立新 XML，不能直接修改旧 XML。
- [ ] 建立 `force_sensor_body → tool_body → executor_tcp` 的刚体层级。
- [ ] 为工具增加独立可视几何和碰撞几何。
- [ ] 读取 MuJoCo 六维力/力矩传感器，并与几何接触力分开记录。
- [ ] 为 TCP 增加独立 site 名称，后端不得继续把 `executor_tip` 自动当作 TCP。
- [ ] 记录 TCP 与裸末端的位姿差异，用于验证安装变换。

### Task 5：实现 TCP 作业位置和力控制

**Files:**
- Create: `src/confined_dual_continuum/coordination/tcp_force_position_controller.py`
- Create: `src/confined_dual_continuum/task/tcp_wiping_director.py`
- Create: `tests/test_tcp_force_position_controller.py`

**Interfaces:**
- Consumes: `TcpFeedback`、TCP 目标位姿、作业面法向和目标法向力。
- Produces: 独立的主臂弯曲速度，再交给现有 `CompatibleTendonController`。

- [ ] 将 TCP 位置误差分解为切向误差和法向误差。
- [ ] 实现阶段：`APPROACH → CONTACT_SEEK → FORCE_POSITION_TRACKING`。
- [ ] `APPROACH` 只负责接近作业面；`CONTACT_SEEK` 使用受限法向速度；`FORCE_POSITION_TRACKING` 同时控制切向位置和法向力。
- [ ] 使用传感器法向力作为正式力控输入；几何接触力只作为诊断对照。
- [ ] 增加力死区、法向速度上限、接触稳定步数、最大接触力和接触丢失处理。
- [ ] 不在该控制器内导入 MuJoCo、YAML 或绘图库。
- [ ] 测试法向归一化、力误差方向、切向误差不改变法向控制、超限进入安全停止。

### Task 6：实现 Observer 纯朝向控制

**Files:**
- Create: `src/confined_dual_continuum/coordination/observer_orientation_controller.py`
- Create: `tests/test_observer_orientation_controller.py`
- Modify only the new TCP assembly: `src/confined_dual_continuum/application/tcp_wiping.py`

**Interfaces:**
- Consumes: Observer 当前姿态、TCP 当前作业点、Observer 方向 Jacobian、双臂/环境安全速度约束。
- Produces: Observer 弯曲速度；不产生 Observer 位置目标。

- [ ] 计算从 Observer 视点指向 TCP 的期望方向。
- [ ] 只使用方向误差和方向 Jacobian 求解朝向速度。
- [ ] 不调用位置 Jacobian 生成 Observer 作业点跟踪命令。
- [ ] 将避碰和环境约束作为独立优先级覆盖，不把位置跟踪重新混入主任务。
- [ ] Observer 求解失败时保持最近安全弯曲状态，不改变 Executor 的任务时钟。
- [ ] 测试视轴完全对齐时输出接近零；反向视轴产生纠正方向；位置改变但视轴误差不变时不产生位置跟踪项。

### Task 7：建立新作业环境和入口

**Files:**
- Create: `configs/workspace/tcp_wiping_bench.yaml`
- Create: `configs/experiments/tcp_wiping.yaml`
- Create: `src/confined_dual_continuum/tcp_wiping.py`
- Create: `src/confined_dual_continuum/application/tcp_wiping.py`
- Create: `tests/test_tcp_wiping_configuration.py`

**Interfaces:**
- Produces: `python -m confined_dual_continuum.tcp_wiping --config configs/experiments/tcp_wiping.yaml`。

- [ ] 新配置引用新的 XML、TCP 工作空间、TCP 作业环境和新输出目录。
- [ ] 默认设置 `joint_guard_enabled: false`，但不关闭环境、双臂和力超限约束。
- [ ] 新入口只装配新链路，不改变 `cli.py` 默认任务行为。
- [ ] 新作业平面必须来自 TCP 工作空间分析后的有效区域，而不是直接复用旧 wiping 点位。
- [ ] 新输出目录与现有 `outputs/executor_tracking`、`outputs/executor_observer_tracking` 隔离。
- [ ] 测试配置路径、入口名称和旧配置引用不被新入口隐式替换。

### Task 8：增加 TCP 作业诊断和对照结果

**Files:**
- Create: `src/confined_dual_continuum/diagnostics/tcp_wiping_records.py`
- Create: `docs/tcp_wiping.md`
- Create: `tests/test_tcp_wiping_records.py`

**Interfaces:**
- Consumes: TCP feedback、Executor 控制量、Observer 视轴误差、安全诊断。
- Produces: `summary.json`、`tracking.csv`、TCP/力/视轴/间隙曲线及配置快照。

- [ ] 记录 `T_world_tip`、`T_world_tcp`、TCP 位置误差和作业面法向误差。
- [ ] 记录传感器力、几何接触力、法向力、力误差和力矩。
- [ ] 记录 Observer 视轴角误差，不记录或不解释 Observer 作业点位置误差为主指标。
- [ ] 记录双臂最小间隙、环境最小间隙、肌腱相容残差和命令可行性。
- [ ] 将 `completed` 解释为“作业阶段完成且指标满足稳定窗口”，不能只表示时间走完。
- [ ] 文档明确暂不打开关节保护，以及 TCP 工作空间仍属于采样证据而非连续空间证明。

### Task 9：分阶段验证并保护旧基线

**Files:**
- Modify only new tests/docs/configs created above.
- Do not modify old baseline result directories.

**Interfaces:**
- Produces: TCP 新链路的分阶段验证记录和旧基线复现对照表。

- [ ] 先验证固定作业点，再验证低速直线，再验证小矩形，最后再接入 circle/figure8。
- [ ] 对每个阶段记录 TCP 位置 RMSE、法向力误差、Observer 视轴误差、最小双臂间隙和最大力矩。
- [ ] 在新链路中维持关节保护关闭，并单独记录这一实验限制。
- [ ] 使用只读对照检查确认现有三类轨迹入口、配置和输出路径没有被新功能覆盖。
- [ ] 生成“旧末端基线 vs 新 TCP 作业链路”的结果说明，避免直接比较不同控制点的误差。

## 推荐里程碑

1. **M1：坐标合同冻结**：TCP 安装变换、力坐标、作业面坐标明确。
2. **M2：TCP 工作空间完成**：已有采样完成映射，或独立 TCP 采样完成。
3. **M3：TCP 场景反馈完成**：能读取 TCP 位姿和六维传感器数据，但尚未闭环力控。
4. **M4：TCP 位置控制完成**：固定作业点和低速直线可跟踪。
5. **M5：Observer 朝向控制完成**：只跟踪 TCP 视轴，不追踪作业点位置。
6. **M6：力位控制完成**：接触建立、法向力保持和安全停止可记录。
7. **M7：新入口交付**：`confined_dual_continuum.tcp_wiping` 可独立运行，旧基线不受影响。

## 结果判据

首个可接受版本至少应满足：

- TCP 位置控制点明确且全链路一致；
- 力传感器输出已进入正式力控数据流；
- TCP 作业平面来自 TCP 工作空间分析；
- Observer 只有朝向任务，没有位置跟踪任务；
- 关节保护保持关闭且在结果中明确说明；
- 固定点、直线和小矩形作业均能产生完整诊断；
- 现有项目入口和配置可继续独立复现；
- 所有结论区分“时间完成”“轨迹跟踪成功”“接触力控成功”和“安全约束满足”。

## 后续可扩展方向

- 从刚体固定 `T_tip_tcp` 扩展到工具柔顺或工具接触偏置标定。
- 增加力传感器零偏、低通滤波、重力补偿和坐标标定。
- 增加 Observer 的可见性评价，而不仅是视轴角误差。
- 将作业平面筛选扩展到 TCP 姿态、工具碰撞和双臂可观测性的联合优化。
- 在关节保护仍关闭的基线完成后，再建立独立的保护开启实验配置，不覆盖本方案配置。
- 在运动学力位控制稳定后，再评估导纳或阻抗控制；不提前引入动力学模型。
