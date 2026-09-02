# 仿真资产来源

以下资产从同级原始仓库 `continuum_sim` 迁移，用于专利统一 Demo：

- 双连续体移动基座 MuJoCo XML；
- Executor、Observer 和共享基座 STL；
- 发动机 visual/collision mesh；
- engine embodiment 场景标注。

未迁移 SolidWorks、STEP、通用调试模型、旧 viewer 资源和历史输出。

迁移资产保留原始尺寸和坐标标定。本仓库只维护正式专利实验实际使用的资产。

发动机视觉 STL 原始文件包含 253,026 个三角面，超过 MuJoCo 单个 STL 的 200,000 面限制。仓库将其按原始三角形顺序无损拆分为 `engine_visual_part_1.stl` 与 `engine_visual_part_2.stl`，场景中使用两个共位视觉 geom 组合显示；碰撞网格保持独立低面数版本。
