# Semantic A-TDOM：逐高斯语义证据赋值的设计思路与考虑

- 状态：当前实现说明
- 适用路线：G1 Evidence V2
- 更新日期：2026-09-21
- 权威数学合同：[Semantic TDOM V1 / G1 Evidence V2](semantic-tdom-v1.md)

## 1. 核心结论

当前系统并不给每个高斯训练一个硬类别，也不把语义损失加入 3DGS 的几何或外观
优化。系统在每个高斯旁维护一个不可训练的、与几何索引一一对齐的 K 维
`Dirichlet-style evidence state`：

```text
二维闭集语义标签
    → 相机像素域统一
    → 完整高斯足迹 alpha×T 证据投影
    → 单视角类别分布与支持度
    → 相关航片 observation group
    → G0 原始证据 / G1 鲁棒证据
    → 最终冻结几何上的精确重建
    → 正射 Semantic TDOM
```

因此，“把语义赋予高斯”在本项目中的准确含义是：

> 把二维标签转换成由几何可见性加权的类别证据，并把该证据累计到高斯的软类别
> 状态；硬类别只在最终地图像素经过多高斯混合后产生。

## 2. 为什么不联合训练语义与几何

上游 SAM3、ProC-SAM3 或其他标签生成器提供的是带噪声的二维预测，不是几何真值。
如果语义 loss 可以修改高斯位置、尺度、旋转、opacity、SH、相机位姿或
densification，标签错误会反向污染 RGB/几何主链，而且难以判断失败来自几何、标签
还是联合优化。

当前设计因此建立单向依赖：

```text
RGB/几何训练 ──只读 snapshot──▶ 语义融合
RGB/几何训练 ◀────禁止反馈───── 语义状态
```

具体约束是：

- 语义状态不是 `nn.Parameter`，不进入 RGB optimizer；
- CUDA observation 只做 `torch.no_grad()` 前向统计；
- 语义异常最多使语义 sidecar 或语义产品失效，不能回滚或阻塞几何训练；
- clone、split、prune 由几何决定，语义只接收完成后的拓扑事件；
- 最终权威语义基于冻结最终几何重新建立，而不是永久相信早期在线投影。

这使语义成为可替换、可审计的下游层，也允许未来更换标签模型而不改变 A-TDOM
几何定义。

## 3. 输入语义的角色

融合核心只接受稳定的数据契约：

- `labels/<image_id>.png`：`0..K-1` 或 `ignore=255`；
- `class_spec.json`：类别 ID、名称、颜色和 background 定义；
- `metadata/<image_id>.json`：标签版本、模型、checkpoint、prompt、阈值和哈希；
- 可选类别 score：用于上游拒绝或诊断，不直接解释成独立证据量。

不直接使用 SAM score 增大 Dirichlet evidence，原因是未经真值校准的模型分数不能区分
“自信且正确”和“自信但系统性错误”。低分像素可以在适配器层变成 `ignore=255`，但
融合核心不能把高分自动理解成更多独立观测。

训练前 preflight 和最终 input binding 会重新验证标签、实际 RGB、相机、overlap、
class spec 和 ontology 的身份。`exact_final` 必须相对于同一组不可变输入成立。

## 4. 统一像素域

语义投影成立的首要条件不是类别公式，而是所有模态对同一个像素负责。对于
RADIAL/OPENCV 相机，系统将以下内容统一映射到中心主点 pinhole 域：

- RGB；
- 语义标签；
- `Tri_Mask` / 有效区域；
- 稀疏关键点与相机投影。

标签采用 nearest-neighbor 去畸变，避免产生不存在的类别 ID；映射外区域写成
`ignore=255`。冻结相机 snapshot 必须保留 calibration，不能让异步语义路径退化成
“已去畸变 RGB + 原始畸变标签”的错误组合。

## 5. 单相机如何形成逐高斯证据

### 5.1 完整足迹，而不是中心点采样

对相机 `v`、高斯 `g` 和像素 `p`，使用与 3DGS 前向合成相同的权重：

\[
\kappa_{vgp}=\alpha_{vgp}T_{vgp}
\]

其中 `alpha` 表示高斯对像素的不透明度贡献，`T` 表示前方高斯遮挡后的剩余透射率。
系统遍历高斯完整二维椭圆足迹，而不是只把高斯中心投到一个标签像素。

中心点采样会忽略：

- 高斯实际尺度；
- 遮挡顺序；
- 高斯横跨类别边界的事实；
- 与最终 TDOM alpha compositing 的一致性。

### 5.2 边界可靠度

标签边界附近使用连续权重 `b_p∈[0,1]`，`ignore=255` 的权重恒为0：

\[
e_{vgk}=\sum_{p:y_p=k}\kappa_{vgp}b_p
\]

边界降权只降低边界像素的证据强度，不腐蚀整圈边界，也不删除小目标。这样不能消除
边界误差，但可以减少单像素抖动对大高斯的污染。

### 5.3 分布和支持度分离

单相机类别分布为：

\[
q_{vgk}=\frac{e_{vgk}}{\sum_j e_{vgj}+\epsilon}
\]

同时计算全部可见质量和有效标签质量：

\[
m^{all}_{vg}=\sum_p\kappa_{vgp}
\]

\[
m^{valid}_{vg}=\sum_{p:y_p\ne255}\kappa_{vgp}b_p
\]

单相机支持度为：

\[
s_{vg}=\left(1-e^{-m^{all}_{vg}}\right)
\frac{m^{valid}_{vg}}{m^{all}_{vg}+\epsilon}
\]

`q` 回答“看起来属于什么类别”，`s` 回答“这次观测提供了多少有效支持”。两者必须
分开，否则“一次非常一致的观测”和“多次相互冲突的观测”会被压成同一种置信度。

## 6. 为什么连续航片要先分组

无人机连续影像通常高度重叠。如果把每一张相邻航片都当成独立 Dirichlet 样本，
同一片地物会因为拍摄频率高而获得虚假的高浓度。

当前系统使用 SfM overlap matrix 形成 observation group：

- 新影像与当前组 anchor 比较归一化 overlap；
- 默认阈值为 `0.4`；
- 默认最大组长为4；
- 在线开放组保留全部组内成员；
- 新帧进入开放组时替换旧 provisional 贡献，而不是重复相加。

组内聚合为：

\[
q_{hg}=\frac{\sum_{v\in h}s_{vg}q_{vg}}
{\sum_{v\in h}s_{vg}+\epsilon}
\]

\[
s_{hg}=1-\prod_{v\in h}(1-s_{vg})
\]

一个相关组对一个高斯最多贡献一个有效样本：

\[
E_{hg}=s_{hg}q_{hg}
\]

这里的最大组长是工程上的有界窗口，不是数学上“第四张以后无信息”。当前实现保留
组内全部观测，是为了满足证据守恒并支持拓扑变化后的完整 active-window refresh。

## 7. G0 与 G1：原始证据和鲁棒证据

### 7.1 G0 保留可审计原始证据

总浓度为1的对称先验为：

\[
\alpha_{0,k}=\frac{1}{K}
\]

原始状态：

\[
\alpha_g^{raw}=\alpha_0+\sum_hE_{hg}
\]

先验只保证空状态数值稳定，不代表真实 background 或任何观测类别。G0 永远保留，
用于审计输入标签、计算共识以及比较鲁棒融合到底抑制了什么。

### 7.2 G1 只软抑制跨组离群

评价组 `h` 时，必须同时排除先验和当前组自身：

\[
E_{-h,g}=\alpha_g^{raw}-\alpha_0-E_{hg}
\]

当至少有两个其他有效组且累计支持足够时，建立 leave-one-group-out 共识：

\[
d_{hg}=JS(q_{hg},m_{-h,g})
\]

\[
r_{hg}=r_{min}+(1-r_{min})\exp(-d_{hg}/\tau)
\]

正式鲁棒状态为：

\[
\alpha_g^{robust}=\alpha_0+\sum_hr_{hg}E_{hg}
\]

设计上保留正下限而不把冲突组直接删除，原因是早期少量观测不应形成不可逆多数锁定。
只有一两个组时也不做多数裁决，而是把不确定性诚实保留下来。

JS 机制只能处理“某些视角和其他视角不一致”，不能识别所有视角都稳定犯错的标签
模型系统性偏差。后者必须依赖人工真值、独立模型或小型校准集。

## 8. 在线近似与最终权威重建

### 8.1 在线阶段

每个 A-TDOM 几何阶段完成 RGB 优化和拓扑更新后，语义 worker：

1. 冻结当前投影所需几何与相机；
2. 消费尚未入账的新标签；
3. 更新开放组、G0、G1 和流式组间方差；
4. 按需发布最新 RGB/Semantic TDOM pair。

在线 G1 只使用当时已经闭合的历史组，是有因果顺序的有界近似。几何训练可以继续，
过时的地图渲染请求可以合并，但未入账的标签观测不能丢弃。

### 8.2 最终阶段

final refinement 完成后，在线状态失去权威地位。系统在冻结最终几何上：

1. 从空语义先验重新开始；
2. 对全部有效标签执行完整足迹重投影；
3. 第一遍建立全部组和完整 G0；
4. 第二遍为每个组计算真正的 leave-one-group-out 权重；
5. 重新计算组间方差和最终 G1。

两遍投影不是无意义重复。如果不保存全部历史 `group×Gaussian×class` 张量，就需要
第二遍才能得到每个组相对于“所有其他组”的精确权重。当前设计选择用最终计算时间换取
在线显存有界和最终结果可审计。

## 9. 高斯拓扑变化

语义 sidecar 必须跟随高斯索引变化，但不能把 clone/split 误认为新观测：

- 新增且没有父节点的高斯：只初始化对称弱先验；
- clone/split：继承父高斯的软类别结构和已有统计，标记 approximate/dirty；
- prune：删除对应语义状态；
- 活动窗口：对 dirty 高斯重新投影当前组；
- 最终阶段：从原始标签在最终几何上彻底重建。

继承的目标是保持拓扑前后渲染场尽量连续，不是因为 densification 凭空获得了新语义
证据。任何无法通过活动窗口修正的历史近似最终都由 exact rebuild 清除。

## 10. 从高斯证据生成 Semantic TDOM

正式渲染使用 `alpha_robust`，首先扣除先验：

\[
E_{gk}=\max(\alpha^{robust}_{gk}-\alpha_{0,k},0)
\]

无真实 evidence 的高斯不向任何类别 numerator 提供质量。随后使用与 RGB TDOM 相同
的正射相机、范围、GSD、排序和 alpha compositing，把多个高斯的软分布混合到地图
像素。

高斯不会先变成一个硬 `semantic_id`。硬类别只在像素级产生，并且只有三个门控：

1. 几何 coverage 足够；
2. 绝对 semantic evidence support 足够；
3. evidence support / geometry coverage 比例足够。

不满足时输出 `ignore=255`。Top-1、margin、entropy、Gaussian mixture
disagreement、observation-group disagreement 和 variance proxy 都是独立诊断，不能
通过隐藏阈值改变 V1 `semantic_id`。

渲染前还会排除异常膨胀的语义高斯支持域，防止 RGB 中接近黑色但 opacity 很高的超大
高斯把类别扩散到地图外缘。该过滤只作用于 Semantic TDOM 产品，不删除高斯、不修改
训练，也不定义通用 no-data 区域。

## 11. 文献思想与当前实现的关系

当前路线不是复现某一篇 Semantic GS，而是组合少量与约束相符的思想：

| 来源 | 吸收的思想 | 当前实现中的改造或边界 |
|---|---|---|
| A-TDOM | 渐进式 3DGS、阶段边界、正射 TDOM、在线 SfM/训练工作流 | 唯一代码基线；语义作为其下游 sidecar |
| [Continuous Semantic Splatting](https://arxiv.org/abs/2411.02547) | `alpha×T` 完整足迹、逐高斯 Dirichlet 证据、无语义反向训练、不确定性前向渲染 | 增加相关航片分组、G0/G1、ledger、渐进拓扑、在线/最终双路径 |
| [Semantic Gaussians](https://arxiv.org/abs/2403.15624) | learning-free 的二维语义到三维高斯投影方向 | 不采用其开放词汇特征平均，权威状态保持闭集 K 维证据 |
| [ReSemGS-SLAM](https://www.sciencedirect.com/science/article/pii/S0950705126007811) | 多帧类别分布和 JS 一致性 | JS 被改造成 leave-one-group-out 证据软权重；不让语义参与 tracking |
| SGS-SLAM、SemGauss-SLAM、GS3LAM | 在线语义地图、关键帧和活动窗口经验 | 只吸收调度思想，拒绝语义参与位姿、BA 和几何管理 |
| [NRGS](https://arxiv.org/abs/2604.22439) | 跨视角方差应作为独立可靠性信号 | 实现流式/最终 group disagreement，不采用其条件 MLP |
| NG-GS、COB-GS、Trace3D | 大高斯横跨对象边界是基础误差源 | 采用软分布、边界降权和 unknown；不采用语义引导 split/prune |
| [EvoPropGS](https://ojs.aaai.org/index.php/AAAI/article/view/42418) | 航拍场景的尺度、重复结构和域差异 | 当前只作为未来稀疏标签传播参考，不进入逐高斯状态主链 |

Feature 3DGS、LEGaussians、SAGA、Gaussian Grouping 等学习式 embedding
方法已经调研，但没有进入当前权威路径。它们面向开放词汇、实例查询或编辑，需要额外
优化和更高维状态；当前产品是闭集地物 Semantic TDOM，K 维证据状态更轻、更容易审计。

完整文献比较见
[轻量 Semantic 3DGS 文献深读](../research/lightweight-semantic-3dgs-literature-review-2026.md)。

## 12. 当前设计明确不解决的问题

- 多视角一致但全部错误的 SAM 系统性偏差；
- 几何本身没有表达出的真实细边界；
- 开放词汇查询、实例身份和对象编辑；
- 在没有人工真值时证明语义准确率提升；
- 把 Dirichlet-style state 或 variance proxy 解释成校准后的错误概率。

因此当前系统能够证明的是输入身份、证据守恒、跨视角一致性、几何隔离和最终重建
完整性；要证明类别精度提升，仍需要独立真值集、人工抽检或可靠的外部评价基准。

## 13. 代码对应关系

| 设计部分 | 当前实现 |
|---|---|
| 相机去畸变和统一像素域 | `src/semantic_atdom/utils/camera_calibration.py` |
| 单相机 CUDA evidence projection | `src/semantic_atdom/semantics/cuda_observation.py` |
| observation group、在线 G0/G1 | `src/semantic_atdom/semantics/orchestrator.py` |
| sidecar、拓扑和流式 moments | `src/semantic_atdom/semantics/state.py` |
| 最终两遍精确重建 | `src/semantic_atdom/semantics/rebuild.py` |
| evidence-conditioned TDOM | `src/semantic_atdom/semantics/tdom.py` |
| preflight/final input binding | `src/semantic_atdom/semantics/preflight.py`、`input_binding.py` |
