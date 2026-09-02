# confined_dual_continuum

面向有限入口复杂受限空间的分阶段非对称双连续体协同与弯曲空间相容驱动控制。

本仓库围绕一条明确的专利技术链组织：

```text
任务层：分阶段自由度管理
→ 协同层：Executor / Observer 非对称独立求解
→ 执行层：bending-space compatible tendon control
```

## 设计原则

- 发动机是受限空间实施例，不进入通用控制接口。
- 控制核心不依赖 MuJoCo、viewer、YAML 或绘图代码。
- Executor 与 Observer 分别求解，只在命令层合并。
- 肌腱限幅在弯曲相容空间内完成，不逐肌腱破坏几何约束。
- 不保留原项目的历史兼容导出和旧场景接口。

架构与迁移范围见 `docs/architecture.md` 和 `docs/migration_inventory.md`。

