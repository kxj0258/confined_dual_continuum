# 专利实验

本目录只保存实验入口和结果说明。控制算法位于 `src/`，实验脚本不得复制控制实现。

计划固定三组对照：

1. 全自由度持续激活 vs 分阶段自由度管理；
2. 固定 Observer vs task-assist Observer；
3. 逐肌腱独立限幅 vs bending-space compatible control。

每次运行应创建独立结果目录，并保存配置快照、时序数据和指标汇总。实验结果不能写入 pytest 断言。

