# 迁移开发路线

## 已建立

- 独立仓库与中文架构约束；
- 稳定状态/命令合同；
- 通用受限空间模型和发动机 adapter；
- 统一阶段监督器；
- Executor/Observer 独立控制器；
- Observer task-assist 目标生成；
- 双臂距离安全监督；
- bending-space compatible tendon controller；
- typed diagnostics；
- 三组专利消融配置；
- 最小仿真资产集。

## 下一步

1. 从原项目提取解析三段九肌腱模型所需的最小配置加载器。
2. 迁移 PCC forward kinematics、position Jacobian 和 centerline Jacobian。
3. 实现 MuJoCo backend adapter，将新的 `RobotCommand` 映射至现有 actuator。
4. 完成统一 Demo runner，并贯通七个正常阶段及 SAFETY_STOP。
5. 实现三组实验 runner、指标聚合和专利附图。

## 删除原则

新仓库不复制旧兼容模块。旧仓库代码只有在新实现逐项验证后才决定是否删除；两个仓库之间不建立运行时路径依赖。
