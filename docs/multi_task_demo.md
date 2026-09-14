# 多任务专利验证 Demo

## 场景布局

验证台覆盖约 650 × 400 × 300 mm，由四个沿 X 轴排列的工位组成：

1. **有限空间视觉探查**：基座停留在 52 mm 入口外，Observer 携带末端摄像头沿五点路径采集视频；Executor 是避让臂。
2. **曲面巡检**：Observer 沿离散弧面完成五点巡检，采集缺陷标记画面；Executor 继续承担避让。
3. **力位擦拭**：Executor 沿三条往复轨迹执行切向位置控制和法向力控制；Observer 位于自由空间侧观察。
4. **非对称障碍操作**：Executor 穿越双立柱路径，Observer 执行观察和主动避让。

## 阶段序列

```text
INITIALIZE
→ BASE_APPROACH
→ BASE_LOCK
→ CONFINED_VISUAL_INSPECTION
→ CURVED_SURFACE_TRANSIT
→ CURVED_SURFACE_INSPECTION
→ WIPING_TRANSIT
→ FORCE_POSITION_WIPING
→ OBSTACLE_TRANSIT
→ ASYMMETRIC_OBSTACLE_OPERATION
→ COMPLETE
```

任务结束不执行自动回撤，终态退出运行循环并关闭 viewer。工位之间则执行竖直退出、低位横移、竖直进入，避免带着接触状态横向转场。

基座采用理想 mocap 平台。局部操作保持位姿，转场高度为 -0.10 m；参考地板没有实体碰撞，此高度仅适用于当前验证台，不代表实际设备可在地板下移动。

路径点表示末端位置。探查入口基座为 (-0.240, 0, 0.095) m，探查点最高 0.225 m；障碍工位基座从立柱实体内部移到 (0.330, -0.040, 0) m。位置改动满足臂长上界，但完整路径与接触稳定性尚未仿真确认。

巡检检查位置和视轴误差，擦拭检查切向误差及法向力误差。满足后稳定驻留 `waypoint_duration_s`，未达到时不会按时间自动成功；超时记录为 `waypoint_or_transit_timeout`。没有图像识别、污渍消失或遮挡识别算法。

## 输出

- `observer_inspection.mp4`：Observer 末端 RGB 摄像头视频；
- `global_follow.mp4`：全局跟随视频，观察中心为基座位置加世界坐标偏置 `[0, 0, 0.08]` m，距离 0.45 m，固定方位角 135°、俯仰角 -20°；
- `timeseries.csv`：阶段、位姿、双臂距离、接触力和肌腱相容残差；
- `summary.json`：完成状态、最小双臂距离、最大接触力和最大相容残差；
- `snapshot/`：主配置、机器人、空间、XML/mesh 及控制源码快照；`snapshot/config.yaml` 引用快照内资产。
- `manifest.json`：变体、代码提交、依赖版本和快照哈希。
- `comparison.json`：本批运行的全部汇总，包含失败结果。

`inter_arm_distance_m` 现指两臂胶囊表面间隙，不能直接和旧版中心线距离数值比较。CSV 同一行的状态、误差和命令属于控制前时刻；`target_phase` 标识目标所属阶段，阶段转换行不得算入正常跟踪误差。

两路视频使用相同分辨率、帧率及仿真采样时刻，保存在同一变体结果目录。全局相机只平移跟随基座，不跟随手臂姿态旋转，也不受 viewer 鼠标操作影响。两路顺序复用离屏渲染器，编码器独立管理并在退出时关闭；录制两路会增加渲染及编码开销。

`video.global_follow.enabled: false` 仅关闭全局录像；`--no-video` 同时关闭两路。需要改变构图时调整 `video.global_follow` 中的偏置、距离和角度。既有自定义视频配置需要显式补充该配置块。录像随会话停止，不延长安全停止后的仿真。
