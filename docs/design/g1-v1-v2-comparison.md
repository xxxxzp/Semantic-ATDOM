# Semantic A-TDOM 语义路线总结：G1 V1、V1.1 与 V2

## 文档定位

- 状态：当前实现总结，依据截至2026-09-14已经合入并通过 Phantom3 全量运行的代码。
- 当前语义方法：`g1-evidence-v2`。
- 历史对照：`g1-prior-v1`。
- 过渡版本：`g1-evidence-v1.1`。
- 本文解释设计演进和工程差异，不替代
  [Semantic TDOM V1 权威数学与产品合同](semantic-tdom-v1.md)。

## 首先澄清两个“版本”维度

项目中有两个独立的版本概念：

1. **Semantic TDOM V1** 是当前俯视语义地图的产品合同：输出类别图、unknown、
   evidence support 和独立不确定性诊断，并与 RGB TDOM 对齐。
2. **G1 Evidence V2** 是构建该产品的逐高斯语义方法版本。

因此，“Semantic TDOM V1：G1 Evidence V2”不是自相矛盾。输入标签仍遵循
[Phantom3 标签交付规范 V1](phantom3-semantic-label-contract-v1.md)，输出产品仍遵循
Semantic TDOM V1 合同，但内部证据方法已经从 G1 Prior V1 演进为 G1 Evidence V2。
状态、恢复文件、semantic manifest、pair manifest 和标签 schema 各自独立版本化，
不能用单个“V2”概括所有文件格式。

## 当前研究问题

输入是连续、高重叠、基本下视的无人机影像。每张 RGB 影像已经具有离线生成的
八分类 SAM 标签。目标不是训练一个新的二维分割器，而是在 A-TDOM 渐进重建过程中
同步建立逐高斯语义场，并最终输出像素严格配准的：

- RGB TDOM；
- Semantic TDOM；
- 语义证据支持和彼此独立的不确定性诊断。

“离线语义”只表示标签在任务开始前已经生成。语义融合发生在几何训练期间，最终
refinement 后再基于冻结几何执行一次权威重建；它不是等几何全部结束后才开始的
单次二维后处理。

## 所有版本都不改变的核心原则

### G 路线是主线

语义属于 Gaussian primitive，而不是仅对最终 RGB TDOM 做二维分割，也不是将源标签
直接镶嵌到地图网格。D、M 路线只作为未来对照实验。

### 几何防火墙

几何单向影响语义，语义绝不反向影响几何：

- 高斯位置、尺度、旋转、opacity 和相机只以 detached snapshot 被读取；
- 语义状态不加入 RGB optimizer，也不产生几何梯度；
- 语义标签不参与 RGB loss、densify、clone、split 或 prune 决策；
- 语义 worker、标签或地图发布失败不能中断、回滚或改变几何训练。

这样做是因为当前 SAM 标签本身存在噪声，不能被当作几何真值。

### 一张标签不能被重复计票

语义 ledger 以 `image_id + label_version` 标识观测。RGB 训练重复采样同一影像时，
不会把同一张标签反复累加为独立证据。

### 相关影像先分组

连续高重叠航片不是独立样本。系统按真实采集顺序和 anchor overlap 将影像组成有限
观测组；一个组对一个高斯最多贡献一个 Dirichlet-style 有效样本。默认 overlap
阈值为0.4，最大组长为4，但二者都是待评测的配置，不是理论常数。

### 训练期同步加最终重建

每个渐进阶段在几何更新完成后冻结语义投影 snapshot，异步吸收新标签。在线结果服务
准实时预览；最终冻结几何上的 exact rebuild 从标签重新投影全部有效观测，消除在线
历史证据随几何变化而产生的不可避免陈旧性。

## 主链路

```text
预生成 RGB label
       │
       ▼
A-TDOM 完成当前阶段几何优化和拓扑更新
       │
       ▼
冻结 detached geometry/camera snapshot
       │
       ├── 标签账本去重
       ├── 完整 Gaussian footprint 的 alpha-T 投影
       ├── 相关航片分组和组内聚合
       ├── G0 raw evidence
       └── G1 robust evidence
       │
       ▼
按发布策略从同一冻结 snapshot 原子生成 RGB/Semantic TDOM pair
       │
       ▼
最终 refinement 后在最终几何上执行两遍 exact semantic rebuild
```

单相机投影使用 Gaussian 完整足迹，而不是只将高斯中心投到一个标签像素。对相机
`v`、高斯 `g` 和源像素 `p`：

\[
\kappa_{vgp}=\alpha_{vgp}T_{vgp}
\]

边界可靠度降低硬标签边缘的影响，`ignore=255` 权重恒为零。单相机证据被压缩为
类别分布与有效支持，再在观测组内聚合。G0 保存未做冲突抑制的 raw evidence；G1
只对满足条件的跨组冲突做有正下限的软降权，不硬删除观测。

## V1 到 V2 的核心差异

| 维度 | G1 Prior V1 | G1 Evidence V2 |
|---|---|---|
| 方法身份 | 历史分支 `semantic/g1-prior-v1`；state schema 2 | `semantic_method_version=g1-evidence-v2`；state schema 4、recovery schema 2 |
| Dirichlet 先验 | 先验参与正式类别均值 | 先验只稳定空状态，正式类别质量严格来自 `alpha-alpha0` |
| background | 一旦存在证据，对称先验会向所有类别（包括 background）泄漏少量质量 | background 只能来自真实类别0证据；ignore、无几何和先验都不能生成 background |
| LOO 共识 | 扣除当前组但保留先验 | 同时扣除当前组和先验，只使用其余观测的真实 evidence |
| LOO 资格 | 主要按有效组数判断 | 其余有效组至少2个且总支持达到 `S_min`，否则可靠度固定为1 |
| clone/split | 将父 evidence 按后代权重分配，强调张量总量守恒 | 后代复制父局部语义状态，优先保持渲染场连续，并标记 dirty/stale 等待重投影 |
| Semantic TDOM 支持域 | 默认排除最大尺度超过场景中位最大尺度20倍的异常高斯 | 保留同一只读尺度异常过滤，并按完整 Phantom3 冻结快照将默认比值校准为14；它不改变 evidence 数学或几何训练，只防止近黑巨型高斯给地图边缘着色 |
| hard-map gate | coverage、absolute support、support ratio、margin；还允许额外诊断 gate 参与拒绝 | 只允许 coverage、absolute evidence support、support ratio 三个结构 gate |
| 不确定性 | margin/entropy/Top-1/group disagreement 与类别拒绝存在耦合 | Top-1、margin、entropy、mixture/group disagreement、variance proxy 全部独立输出 |
| 无证据语义 | 部分量容易受先验解释影响 | 类别分布和不确定性为未定义/NaN，类别输出为 `ignore=255` |
| 组间分歧 | 有组统计，但有效支持语义不够严格 | 只由至少两个有效闭合组贡献，并使用独立分母；无有效支持为NaN而不是0 |
| 在线状态 | 主要报告 lag 与同步状态 | 明确区分 `approx_online`、`stale_online`、`online_fallback`、`exact_partial`、`exact_final`、`failed` |
| 地图坐标 | 各阶段依据当时几何生成相机/范围，跨阶段身份较弱 | 一次任务固定方向、origin、GSD 和 grid ID，只通过 tile index 扩张范围 |
| RGB/Semantic 配对 | 依赖 product key 和相近触发时机 | 从同一冻结 snapshot 原子发布，`latest.json -> pair_manifest.json` 提供完整哈希链 |
| 调度 | 语义同步和地图生成耦合较紧 | 几何阶段、标签消费和地图发布解耦；地图 latest-wins，但标签绝不丢弃 |
| provenance | 单一 geometry hash 和 sidecar 校验为主 | 分离 projection geometry、完整 RGB model、semantic state、alignment、ontology 和 palette hash |
| 数据访问 | semantic root 可隐式回退到 RGB/SfM root，派生产物边界不够严格 | 启用语义时必须显式传 root；训练前只读 preflight，派生 PLY/index 全写入 output cache |
| 资源控制 | semantic 支持分片和显存检查，但 RGB publication 仍可能全图分配 | 轻量 projection snapshot、共同 pair snapshot、显式 stream/headroom/time/output budget，RGB 与 semantic 均可精确分片 |
| 最终恢复 | 有最终重建，但依赖在线运行状态较多 | final rebuilder 可在 online hook 失败后独立从最终几何和原始标签从零启动 |

## 最关键的数学修正

### V1：先验进入正式类别分布

令总浓度 `A_g=sum(alpha_g)`，V1 使用：

\[
\mu_g(k)=\frac{\alpha_{gk}}{A_g},\qquad
\rho_g=\frac{A_g-1}{A_g}
\]

并将 `rho_g * mu_g(k)` 作为类别质量。`rho_g` 可以阻止完全空状态直接产生支持，
但只要存在任意真实证据，`mu_g` 中的对称先验仍会分配到所有类别。这意味着：即使
上游从未预测 background，先验也会给 background 一个小的正式类别份额。

### V2：先扣除先验，再形成类别分布

V2 显式提取真实证据：

\[
E_{gk}=\max(\alpha_{gk}-\alpha_{0,k},0)
\]

\[
E_g=\sum_kE_{gk},\qquad
\tilde q_g(k)=\frac{E_{gk}}{E_g}
\]

\[
\rho_g=\frac{E_g}{1+E_g},\qquad
B_g(k)=\rho_g\tilde q_g(k)=\frac{E_{gk}}{1+E_g}
\]

无真实证据时 `rho_g=0`，该高斯不为任何类别提供 numerator。先验仍保留在
Dirichlet state 中用于数值稳定和未校准 variance proxy，但不再被解释成观测类别。

### V2：真正的 leave-one-group-out

评价观测组 `h` 时：

\[
E_{-h,g}=\alpha^{raw}_g-\alpha_0-E_{hg}
=\sum_{j\ne h}E_{jg}
\]

只有其余有效组数量不少于2且 `sum(E_-h,g) >= S_min` 时，才用其形成共识并计算
JS divergence。否则 `r_hg=1`，不在证据不足时凭先验制造“共识”去惩罚真实观测。
最终 exact rebuild 使用两遍计算，每个组都与排除自身后的其他组比较。

## V2 的 Semantic TDOM 定义

令 `kappa_pg` 为高斯对 TDOM 像素的 alpha-T 几何权重：

\[
C_p=\sum_g\kappa_{pg}
\]

\[
U_p=\sum_g\kappa_{pg}\rho_g
\]

\[
N_p(k)=\sum_g\kappa_{pg}B_g(k),\qquad
P_p(k)=\frac{N_p(k)}{U_p+\epsilon}
\]

- `C_p` 是几何覆盖；
- `U_p` 是绝对真实语义证据支持；
- `U_p/(C_p+epsilon)` 是有几何区域中的语义支持比例；
- 剩余透射率表示无几何，不属于 background；
- 先混合所有可见高斯的软类别分布，再做像素 argmax。

只有以下三个条件同时成立时，`semantic_id` 才输出类别，否则输出255：

\[
C_p\ge\theta_C,\qquad
U_p\ge\theta_U,\qquad
\frac{U_p}{C_p+\epsilon}\ge\theta_R
\]

margin、entropy、Top-1、mixture disagreement、group disagreement 和
prior-regularized variance proxy 都是独立诊断，不得暗中改变主类别图。

## 为什么 V1.1 不是当前主线

`g1-evidence-v1.1` 已经采用 evidence-conditioned belief、prior-free LOO、三条件主
产品 gate 和拓扑局部连续性，解决了最关键的数学问题。但它有意保留 V1 的工程管线：

- 没有 V2 固定网格和原子 pair publication；
- 没有 latest-wins 地图调度；
- 没有完整的低优先级 CUDA stream、snapshot 和 provenance 重构；
- 仍保留 semantic-only Gaussian scale outlier filter；
- 主类别图之外另有 conservative/ambiguity 产品。

因此 V1.1 适合作为隔离数学变化的 A/B 过渡版本，而不是当前端到端发布基线。

## V2 在线与最终结果的关系

在线历史证据由当时的 geometry/opacity/scale/camera 投影得到。几何继续优化后，只重投影
活动窗口无法让所有历史证据完全精确。因此 V2 不把在线地图伪装成最终真值：

- `approx_online`：正常在线预览；
- `stale_online`：lag、历史陈旧、deferred 或错误使其只适合作诊断；
- `online_fallback`：请求 final 后 exact rebuild 失败，保留在线结果诊断；
- `exact_partial`：在最终几何上重建了全部可用标签，但输入或 source binding 不完整；
- `exact_final`：预期标签、来源绑定和最终重建均完整；
- `failed`：不发布类别产品。

异步出现临时 lag 是几何训练速度、标签投影、checkpoint 和大图 publication 吞吐不同导致
的队列现象，不表示标签被丢弃。final rebuild 会消费所有可用标签；输入完整时应将 lag
收敛为0，输入确有缺失时则必须通过 `exact_partial` 和缺失列表诚实表达。

## V2 地图发布与资源控制

同一任务首次发布时固定本地地图网格的方向、origin、GSD、像素约定和 grid ID。未知
完整范围时只扩张512像素 tile 索引，不移动旧地图坐标。输出尺寸因此表示“固定 GSD 下
覆盖多少世界范围”，不是输入影像尺寸。

每次正式发布使用同一冻结 snapshot 生成 RGB 与 Semantic TDOM，并按以下路径提交：

```text
Map_Products/
├── map_grid.json
├── latest.json
└── <product_key>/
    ├── rgb/tdom.png
    ├── rgb/manifest.json
    ├── semantic/...
    ├── semantic/manifest.json
    └── pair_manifest.json
```

消费者只读取 hash 校验通过的 `latest.json -> pair_manifest.json`。临时 publishing
目录、单独 sidecar slot 或仅有某一侧产品均不是已提交 pair。

在线调度允许合并已经过时的地图请求，但标签账本独立推进。投影 snapshot 不复制 SH；
只有确实发布 RGB/Semantic pair 时才冻结完整 RGB appearance。显存不足时先减少可选
诊断，再跳过过时地图，不能阻塞几何或丢弃标签。

V2 全局实验进一步修复了两个端到端问题：

1. `Use_Tri_Mask=False` 不能再关闭渐进新高斯所需的 `point3D_ids/xys/tri` 元数据；
   该参数现在只控制训练 mask。
2. 大范围 RGB TDOM 与 semantic TDOM 均使用保持全局投影和深度顺序的精确纵向分片；
   解决9216×11264最终 pair 的全图 CUDA OOM。

完整运行后还识别出一个分支继承回归：V2 从早于 `eb57e4d` 的提交点开发，没有带入
V1 已验证的异常尺度语义支持域过滤，导致纯黑 RGB 边缘仍被赋予类别。最新 V2 恢复
该过滤；它发生在语义光栅化前，不增加第四个 hard-map gate，见 ADR 0007。

## 兼容性与迁移规则

- V1、V1.1 和 V2 的 sidecar 数学含义不同，禁止跨版本静默 resume。
- 旧状态只能作为冻结 A/B 输入；要得到 V2 结果，必须在对应几何上从原始标签重建。
- label contract V1 仍有效，不需要因为 method 升级而重命名全部标签。
- palette 只控制显示颜色；修改 palette 不重建 evidence，也不改变 ontology identity。
- ontology、类别顺序、ignore/background 语义变化必须产生新版本并重建。
- V2 正式产品应通过 pair manifest 消费，不能用目录编号或文件修改时间自行配对。

## 已验证事实

当前 V2 在 Phantom3 完整399张影像上已经跑通：

- 942,438个高斯；
- 最终 RGB/Semantic TDOM 为9216×11264；
- 399/399个相机中心位于最终地图范围；
- 最终语义使用399张标签、180个观测组；
- `semantic_through_image=000402`、`lag=0`、无 error/deferred；
- RGB 指标为 L1 0.0174595、PSNR 31.3079、SSIM 0.937277、LPIPS 0.184639；
- 143项 CPU/CUDA/schema/orchestration/publication 测试通过；
- 15个发布阶段的产品文件、子 manifest、pair manifest 和 latest 哈希链复算通过。

最终状态为 `exact_partial`，原因只是现有 metadata 未完整绑定实际加载的 RGB bytes，
不是标签缺失、final rebuild 失败或在线 fallback。完整运行见
[Phantom3 V2 实验记录](../project/phantom3-semantic-atom-v2-result.md)。

## 尚未由 V2 解决的问题

- G1 的跨组一致性只能抑制视角间离群，不能发现所有视角共同出现的 SAM 系统性错误。
- Dirichlet-style state 和 variance proxy 未经过概率校准，不能解释为分类错误概率。
- 三个 hard-map 阈值及 JS 参数尚无人工真值支持其最优性。
- 异常尺度比14是在 Phantom3 冻结最终快照上按 RGB 无内容区支持域回归校准，
  尚未通过语义真值或其他数据集校准；它不能直接解释为通用最优阈值。
- 当前尚未完成正式语义准确率评测；“成功生成地图”不等于“语义质量已经充分”。
- 当前固定网格没有 GIS CRS，不能宣称地理坐标已经校准。
- 24 GB单卡目标硬件上的峰值显存和准实时预算仍需单独实测。
- `lag<=1` 是目标而非硬保证；final `lag=0` 才是当前正式结果要求。

## 后续开发基线

后续 G 路线工作应从 `g1-evidence-v2` 继续，而不是重新启用 prior-weighted V1：

1. 先补齐 source-image binding，使完整数据具备 `exact_final` 资格；
2. 再建立语义质量评测和人工抽检，校准 support/JS 参数；
3. 在24 GB目标硬件上 profile 在线 lag、显存和 publication cadence；
4. G3 轻量学习残差只能作为可选研究扩展，不能破坏几何防火墙或取代 V2 闭式基线；
5. D、M 路线保留为对比/消融，不与 G 路线主产品混名。

## 代码与历史锚点

- 原始 A-TDOM 只读基线：`e50a5f7`。
- G1 Prior V1 baseline：`91e0ec3`；稳定修复：`eb57e4d`。
- G1 Evidence V1.1：`023617f`。
- G1 Evidence V2 完成基线：`d1daa4b`。
- 渐进全局范围修复：`f6bfe42`。
- paired RGB TDOM 分片修复：`b8635fc`。
