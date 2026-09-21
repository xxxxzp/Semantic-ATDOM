# 语义链路第一性原理加固（2026-09-21）

## 验收不变量

本轮不以“流程能够运行”为正确性标准，而以以下不变量作为验收边界：

1. RGB、语义标签、有效区域和稀疏几何必须处于同一像素域；
2. ledger 中每个已接收观测都必须真实参与其 observation group；
3. `exact_final` 必须绑定训练前 preflight 接纳的同一组 RGB、标签、相机、
   overlap 和 ontology 输入；
4. 公开参数必须能够改变其声称控制的行为，否则不得暴露；
5. 在线诊断必须由真实的流式统计量生成，不能把“未计算”伪装为零。

## 关键修复

### 相机像素域

异步 `FrozenCameraSnapshot` 现在保留不可变的 `calibration` 和只读
`image_path`。RADIAL/OPENCV 标签因此与已经去畸变的 RGB、Tri_Mask 和稀疏点
使用同一映射；最终重建和在线投影不再因快照丢字段而退化成错误的原始标签投影。

### 在线 observation group

开放组最多只有四个观测。现在缓存组内全部成员，而不是只保留 anchor 与 newest；
拓扑变化后的 active-window refresh 也会刷新同一组完整证据。组满后立即关闭并释放缓存，
所以额外显存被严格限制在一个开放组内，不会随飞行历史增长。

### 在线 group disagreement

`SemanticGaussianState.replace_open_group` 通过已有一阶矩和方差重建加权二阶矩，
在替换开放组时先减旧组、再加新组。这样无需新增长期 `N×K` 张量，也能维护真实的
在线组间方差。只有局部至少两个组且历史几何未标 stale 时才声明该诊断可用。

### 最终不可变输入绑定

新增 final preflight binding。在权威最终重建前重新验证：

- preflight 报告自身 SHA-256；
- 最终 camera ID 集合；
- class spec 与 ontology 指纹；
- 最终 stage 的 `images.bin`、`cameras.bin`、名称表和 overlap matrix；
- 每张标签的 version/hash；
- 每张实际 RGB bytes 的 hash 及 metadata source binding。

绑定摘要写入最终 rebuild metrics、语义 recovery provenance 和 RGB/Semantic pair
manifest。输入在长时间训练期间被一致替换时，流程不再错误发布 `exact_final`。
底层 hook 的权威完整性默认值同时改为 false；只有训练 preflight 或显式外部验证者
可以把它提升为 true，避免未绑定的自定义调用默认声称权威。

`exact_partial` 仍是诚实描述不完整输入的诊断性最终结果；成功提交一个权威训练 run
仍要求 `exact_final`。两者不是同一验收等级。

### 参数和代码结构

- 删除四个从未参与 V1 hard-map gate 的公开阈值：margin、Top-1、normalized
  entropy 和 group disagreement。对应诊断栅格继续输出，但不能暗中改变 `semantic_id`；
- 保留 `unknown_reason` 既有位编码，避免历史栅格失去兼容性；
- 异步阶段请求从位置 tuple 改为具名 dataclass；
- AcquisitionOverlap 移到输入模块，preflight 不再依赖顶层 orchestrator；
- 原子 JSON 写入和输出预算降级策略各自收敛为单一实现；
- 可信语义 checkpoint 改用 `torch.load(weights_only=True)`。

## 性能取舍

最终精确重建的两遍投影用于 leave-one-group-out 稳健权重，不能在不保存全历史
`N×K` 证据的情况下随意合并。本轮没有以牺牲 exactness 换速度。

完整开放组相较旧实现最多多常驻两个 observation 张量，但组大小硬上限为四，组关闭后
立即释放；这换取了证据守恒和可刷新性。若未来显存基准证明该有界峰值仍不可接受，
应设计可重投影的分层充分统计量，而不是再次静默丢观测。

## 验证

- 完整测试：`231 passed`；
- CUDA 语义 observation/TDOM 集成：`22 passed`；
- 新增回归覆盖：冻结标定、三视图证据守恒、在线组间方差、RGB/标签 preflight
  篡改拒绝；
- 按当前项目约定未运行全量训练。
