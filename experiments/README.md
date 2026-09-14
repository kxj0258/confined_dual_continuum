# 专利实验

控制算法只保留在 src/，以下入口共享正式会话与记录器。

在 `continuum_sim` 环境中，分别显式启动三组消融（本轮未执行）：

```powershell
cdc-run --config configs/experiments/unified_demo.yaml --no-video --ablation configs/experiments/ablation_stage_control.yaml
cdc-run --config configs/experiments/unified_demo.yaml --no-video --ablation configs/experiments/ablation_observer_assist.yaml
cdc-run --config configs/experiments/unified_demo.yaml --no-video --ablation configs/experiments/ablation_tendon_compatibility.yaml
```

- `always_active`：非终态启用全部自由度，转场也允许机械臂追踪预设收拢目标；正式变体按阶段关闭手臂。基座仍为同一个理想平台，不能用此实验声称改善真实动态基座漂移。
- `fixed_target`：每阶段固定首次 Observer 位置目标；`task_assist` 随任务更新目标，两者都保留视轴和安全约束。
- `independent_clipping`：独立肌腱限幅，可能产生非相容命令，只作为显式消融基线；正式默认始终使用 `bending_compatible`。

每个变体保存独立 snapshot、manifest、timeseries 和 summary；批目录 comparison.json 包含全部结果，不删除失败试验。不同变体可能因超时而覆盖不同航点，不能直接把全程均方误差当作同任务完成度的胜负；应结合 target_phase、waypoint_index 和完成率比较。

尚未运行这些实验，当前没有性能提升数值、成功率或专利附图结论。
