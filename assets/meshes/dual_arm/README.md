# 双臂 SolidWorks STL 资产

本目录保存双连续体臂的 MuJoCo 可视网格。文件路径和单位由 `configs/robots/dual_arm_meshes.yaml` 定义，模型生成入口为：

```powershell
python scripts/build_mujoco_dual_arm_model.py
```

## 坐标和单位

- STL 以毫米为单位。
- `configs/robots/dual_arm_meshes.yaml` 使用 `mesh_scale: 0.001` 转换为米。
- 所有零件使用同一个 CAD 装配坐标系。
- `assembly_preview.stl` 用于整体外观和坐标检查，不挂接到关节 body。
- 统一基座、两条臂的臂基座和每段 link 使用独立网格，以便随各自 body 运动。

这些 STL 作为 MuJoCo visual geom 使用，`contype=0`、`conaffinity=0`，不参与物理碰撞。

## 目录结构

```text
assets/meshes/dual_arm/
  assembly_preview.stl
  shared_base/
    dual_base_visual.stl
  executor/
    base_visual.stl
    segment_1_link_1_visual.stl
    ...
    segment_3_link_4_visual.stl
  observer/
    base_visual.stl
    segment_1_link_1_visual.stl
    ...
    segment_3_link_4_visual.stl
```

每条臂包含：

- 一个 `base_visual.stl`。
- 三段，每段四个 `segment_<i>_link_<j>_visual.stl`。
- `i = 1..3`，`j = 1..4`。

## 配置映射

`configs/robots/dual_arm_meshes.yaml` 分别声明：

- `shared_base.visual_mesh`：双臂统一基座。
- `arms.executor.base_visual_mesh` 和 `link_visual_meshes`：执行臂。
- `arms.observer.base_visual_mesh` 和 `link_visual_meshes`：观测臂。

文件名或目录改变时，需要同步修改该 YAML。

## CAD 导出

1. 使用统一装配坐标系导出整体预览、统一基座、两条臂基座和 24 个 link。
2. 保持毫米单位，与 `mesh_scale: 0.001` 配套。
3. 使用 ASCII 文件名，避免空格。
4. 如果网格使用零件局部坐标，需要在模型生成配置中提供相对 body 的 mesh offset。
