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
- 三段六维常曲率 PCC 正运动学、中心线和位置雅可比；
- 独立机器人参数加载边界；
- `RobotBackend` 的 MuJoCo 实现，完成双臂肌腱与移动基座命令映射。

## 下一步

1. 增加应用装配函数，由配置创建控制器、PCC 雅可比和 MuJoCo 后端。
2. 完成统一 Demo runner，并贯通七个正常阶段及 SAFETY_STOP。
3. 对照 MuJoCo 离散关节反馈校准 PCC 弯曲估计和中心线误差。
4. 实现三组实验 runner、指标聚合和专利附图。

## 删除原则

新仓库不复制旧兼容模块。旧仓库代码只有在新实现逐项验证后才决定是否删除；两个仓库之间不建立运行时路径依赖。
