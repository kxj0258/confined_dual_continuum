# 坐标系与静态检查

项目采用列向量和米制，`T_A_B` 表示将 B 系坐标变换到 A 系，四元数使用 `wxyz` 顺序。

## 坐标系

- `W`：MuJoCo 世界系，由场景 XML 定义。
- `B`：`mobile_base_frame`，公共基座系。
- `E0`：`executor_base_site`，主臂根部。
- `O0`：`observer_base_site`，从臂根部。
- `Et` / `Ot`：两臂末端 site。
- `Cam`：Observer 相机坐标系。

当前安装关系为：

```text
T_W_E0 = T_W_B · Trans(0, +0.01, +0.02)
T_W_O0 = T_W_B · Trans(0, -0.01, +0.02)
T_E0_O0 = Trans(0, -0.02, 0)
```

控制器将局部 PCC Jacobian 旋转到世界系。当前两臂安装旋转为单位阵；若修改 XML 安装旋转，必须显式更新该变换，不能继续假设共用世界旋转。

## 静态坐标查看

静态查看器只调用 `mj_forward`，不推进物理、不驱动肌腱、不执行任务控制：

```powershell
python -m confined_dual_continuum.adapters.frame_inspector `
  --config configs/experiments/unified_demo.yaml `
  --station entry
```

它用于检查世界、基座、两臂根部、末端、相机和物理分段坐标，不代表对应工位已经动态可达。

## 解释边界

坐标系检查、工作空间采集、轨迹覆盖和动力学跟踪是四个独立问题。工作空间点云不能替代坐标标定，轨迹覆盖不能替代动态跟踪，仿真结果也不能直接作为实物复现保证。
