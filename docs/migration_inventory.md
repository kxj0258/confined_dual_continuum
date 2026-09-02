# 原项目迁移清单

## 直接迁移思想

- `BendingSpaceModel` 与 tendon coupling 的列空间相容性定义。
- Executor/Observer 两次独立求解及 Observer 禁用 base 的边界。
- inter-arm influence / critical / release 距离语义。
- staged base approach 与 fixed-base local operation 思路。
- actual tendon feedback、rate/displacement/lead limits。

## 重新实现

- 用通用 `WorkspaceQuery` 替代 `EngineSceneQueryProtocol`。
- 用 typed diagnostics 替代跨层 metadata 字符串通信。
- 用小型 supervisor 替代两个大型 staged controller。
- 把 Observer target generation 从综合控制器中拆出。

## 不迁移

- 旧兼容导出、旧场景控制器和重复配置解析。
- viewer、实时 panel、视频 hook 与手动调试应用。
- 已删除场景的测试和针对私有函数的测试。
- CAD 源文件及与专利统一 Demo 无关的模型。

