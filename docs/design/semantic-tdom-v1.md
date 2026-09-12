# Semantic TDOM V1：G1 Evidence V2 合同

## 文档状态与权威范围

- 状态：设计已确认并实现；Phantom3 完整399张端到端运行已验证，最终质量状态为 `exact_partial`。
- 确认日期：2026-09-12。
- 语义方法版本：`g1-evidence-v2`。
- 本文是 Semantic TDOM V1 数学、状态和产品行为的唯一权威合同。

[全局认知](../project/global-context.md) 只维护稳定的项目入口、数据/环境事实和本文摘要；ADR 只解释为什么作出决定。其他文档中与本文冲突的公式或产品约定均由本文取代，尤其包括：

完整运行的命令、指标、产物身份和未验证边界见 [Phantom3 Semantic A-TDOM V2 实验记录](../project/phantom3-semantic-atom-v2-result.md)。

- 使用 `rho * (alpha / A)` 作为类别质量；
- leave-one-group-out 共识仍包含 Dirichlet 先验；
- 使用 margin、entropy、Top-1 或 disagreement 改变 V1 hard-map gate；
- clone/split 必须机械守恒所有后代的 Dirichlet evidence 总量；
- 每个几何阶段必然发布一对 RGB/Semantic TDOM。

相关决策依据见：

- [渲染真实语义证据，而非先验类别质量](../decisions/0005-evidence-conditioned-semantic-rendering.md)；
- [在线语义产品是有界近似](../decisions/0006-bounded-online-semantic-products.md)；
- [语义严格位于几何下游](../decisions/0001-one-way-semantic-dependency.md)；
- [语义随渐进阶段同步](../decisions/0003-stage-synchronous-semantics.md)。

## 目标与非目标

### 目标

1. 在 A-TDOM 渐进几何训练期间同步构建逐高斯闭集语义证据场。
2. 利用连续、高重叠无人机下视影像降低单帧 SAM 标签噪声。
3. 在不改变几何的前提下，生成与 RGB 正射图严格成对对齐的俯视 Semantic TDOM。
4. 使在线预览保持准实时可用，并在冻结最终几何上提供可复现的精确重建。
5. 将观测不足、类别混合和跨观测组冲突分别表达，而不是压缩成单一置信度。

### 非目标

- 不用语义 loss、标签或不确定性更新位置、尺度、旋转、opacity、SH、densification、prune 或 RGB 优化。
- 不把最终 RGB TDOM 的二维分割或二维标签镶嵌称为 G 路线结果；它们仅是未来 D/M 对照。
- 不在 V1 中训练逐高斯 logits、任意特征 embedding 或语义 MLP。
- 不承诺纠正同一 SAM 模型在所有视角上的系统性错误。
- 不承诺任意透视视角的语义产品，也不在没有 CRS 合同时声称输出是 GeoTIFF。

## 符号与语义

| 符号 | 含义 |
|---|---|
| `g` | 一个 Gaussian primitive |
| `v` | 一张源影像/相机观测 |
| `h` | 一组高度相关的相机观测 |
| `p` | 一个源影像或 TDOM 像素 |
| `k` | 一个闭集类别，`0 <= k < K` |
| `alpha0` | 总浓度为 1 的对称 Dirichlet 先验 |
| `E_gk` | 高斯 `g` 对类别 `k` 的真实数据证据 |
| `A_g` | `sum(alpha_g)`，等于 `1 + sum(E_g)` |
| `rho_g` | 高斯 Dirichlet 总质量中真实证据所占比例 |
| `kappa_vgp` | 高斯对像素的标准 alpha-T 可见性贡献 |
| `C_p` | TDOM 像素的几何覆盖 |
| `U_p` | TDOM 像素的真实证据支持 |

本文中的 Dirichlet 是一个轻量、可合并的 `Dirichlet-style evidence state`。由于相邻影像、SAM 错误、高斯足迹、拓扑后代和鲁棒权重并不独立，它不能被解释成经过校准的标准贝叶斯后验或分类错误概率。

## 数据与标签前提

运行时语义根目录包含：

```text
semantic/
├── class_spec.json
├── labels/<image-id>.png
├── metadata/<image-id>.json
└── scores/<image-id>.npy       # 可选；G1 V2 不依赖
```

- `class_spec.json` 是类别本体的机器可读真源。
- label 是单通道整数图，值域为 `0..K-1` 或 `ignore=255`。
- `background=0` 只表示上游模型明确产生的 background 证据。
- ignore、缺失、低置信或无效像素不得转换成 background。
- 同一 `image_id + label_version` 只能被 ledger 计入一次。
- 标签必须与运行时 RGB/camera tensor 同尺寸、同方向、同像素坐标；运行时不得偷偷 resize、rotate 或 flip。
- 启用语义时必须显式提供 semantic data root，不得隐式回退到 RGB/SfM source root。

当前 Phantom3 使用八类 schema，但 stage 100 的前100张标签没有任何 background 像素；全部399张中也只有 `000112.png` 含6038个 background 像素。因此当前事实是“八类 schema，前100张实质观测到类别1–7和ignore”，不是“八个类别均已得到验证”。是否长期保留 background，需要在检查上游 ProC-SAM3 的类别语义后另行决定。

详细交付格式见 [Phantom3 语义标签交付规范](phantom3-semantic-label-contract-v1.md)。当前早期 metadata 的 source binding 技术债和本机路径见 [全局认知](../project/global-context.md)。

## 几何防火墙与阶段时序

语义严格位于几何下游：

- 所有高斯属性以 detached snapshot 读取；
- 所有语义更新在无 autograd 的状态下执行；
- 语义状态不注册进 RGB optimizer；
- 语义失败只改变语义状态/产品资格，不能阻塞、回滚或改变几何训练。

每个渐进阶段的逻辑顺序是：

1. 新影像加入并完成当前阶段的几何优化；
2. densify/prune 等几何拓扑更新完成；
3. 冻结语义投影所需的当前几何与相机 snapshot；
4. 消费尚未记账且已经就绪的标签；
5. 更新 G0/G1 sidecar 与在线诊断；
6. 地图发布调度器按需生成最新的 RGB/Semantic 产品对。

在线流水线可以合并过时的几何/地图请求，但不能丢弃未计账的标签观测。`lag <= 1` 是待 profile 验证的 SLO，不是当前实现已经保证的硬事实。

## 单相机观测

对相机 `v`、高斯 `g` 和像素 `p`，使用与前向 Gaussian rasterization 相同的完整足迹和 alpha compositing：

\[
\kappa_{vgp}=\alpha_{vgp}T_{vgp}
\]

标签边界可靠度 `b_p` 位于 `[0,1]`；`ignore=255` 的类别权重恒为零。单相机类别证据为：

\[
e_{vgk}=\sum_{p:y_p=k}\kappa_{vgp}b_p
\]

\[
q_{vgk}=\frac{e_{vgk}}{\sum_j e_{vgj}+\epsilon}
\]

同时区分总可见质量和有效标签质量：

\[
m^{all}_{vg}=\sum_p\kappa_{vgp}
\]

\[
m^{valid}_{vg}=\sum_{p:y_p\ne255}\kappa_{vgp}b_p
\]

单相机有效支持采用：

\[
s_{vg}=\left(1-e^{-m^{all}_{vg}}\right)
\frac{m^{valid}_{vg}}{m^{all}_{vg}+\epsilon}
\]

`projected_valid_mass[g]` 另行累计原始有效投影质量，用于观测量诊断；它不是 TDOM 像素的 `U_p`，也不是独立 Dirichlet 样本数。

## 相关观测分组

连续无人机影像高度相关，不能把每张相邻影像视为完全独立证据。当前分组规则为：

- 使用六位数字 image stem 表示当前采集顺序；
- `imagesNames.txt` 只用于将 ID 映射到 match matrix 的行列；
- 新影像始终与当前组 anchor 比较归一化 overlap；
- overlap 低于阈值或达到当前最大组长时关闭当前组；
- 一个组对一个高斯最多贡献一个 Dirichlet-style 有效样本；
- 开放组的暂定贡献在新帧到达时替换，不能重复累加。

组分布和组支持为：

\[
q_{hg}=\frac{\sum_{v\in h}s_{vg}q_{vg}}
{\sum_{v\in h}s_{vg}+\epsilon}
\]

\[
s_{hg}=1-\prod_{v\in h}(1-s_{vg})
\]

组证据向量为：

\[
E_{hg}=s_{hg}q_{hg}
\]

在线活动窗口至少保留开放组，并可保留有限数量的最近闭合组用于几何变化后的重投影。最终 exact rebuild 使用每组全部有效影像。默认 overlap threshold 为 0.4，默认最大组长为4；4不是数学硬上限。

## G0：原始证据状态

总浓度为1的对称先验为：

\[
\alpha_{0,k}=\frac{1}{K}
\]

原始数据证据和 Dirichlet-style 状态为：

\[
E^{raw}_g=\sum_hE_{hg}
\]

\[
\alpha^{raw}_g=\alpha_0+E^{raw}_g
\]

`alpha_raw` 不得被 robust 结果覆盖。它用于诊断标签原始证据、计算 leave-one-group-out 共识，并与 robust 结果比较。

## G1：排除先验和自身的鲁棒证据

评价当前组 `h` 时，从 G0 中同时扣除对称先验和当前组自身：

\[
E_{-h,g}=\alpha^{raw}_g-\alpha_0-E_{hg}
=\sum_{j\ne h}E_{jg}
\]

只有满足以下两个条件才建立 leave-one-group-out 共识：

\[
N_{-h,g}\ge2
\]

\[
S_{-h,g}=\sum_kE_{-h,g,k}\ge S_{min}
\]

此时：

\[
m_{-h,g}=\frac{E_{-h,g}}
{\sum_kE_{-h,g,k}+\epsilon}
\]

\[
d_{hg}=JS(q_{hg},m_{-h,g})
\]

\[
r_{hg}=r_{min}+(1-r_{min})\exp(-d_{hg}/\tau)
\]

若其他有效组数量或总支持不足，则 `r_hg=1`。最终鲁棒证据为：

\[
E^{robust}_g=\sum_hr_{hg}E_{hg}
\]

\[
\alpha^{robust}_g=\alpha_0+E^{robust}_g
\]

禁止使用包含当前组自身的全局均值评价当前组。先验只保证空状态数值稳定，不参与“其他观测形成的共识”。JS 只降低新增组证据且保留正下限；它只处理跨视角不一致，不能识别所有视角一致的 SAM 系统性错误。

在线 G1 使用当时已经闭合的其他组形成有因果顺序的近似权重；最终冻结几何上的 exact rebuild 使用两遍计算，对每组执行真正的 leave-one-group-out。产物必须记录 raw/robust 的证据衰减和类别差异统计。

## 在线组间分歧

在线路径对闭合组的原始 `q_hg` 和 `s_hg` 维护逐高斯加权流式一阶/二阶矩：

- 使用 raw 组分布，不使用已经经过 JS 抑制的 robust evidence；
- 先验不参与组间分歧；
- 开放组只形成可替换的 provisional 统计；
- 少于两个达到有效支持的闭合组时，分歧为 NaN，不得以0表示“完全一致”；
- final exact rebuild 使用全部有效组重新计算权威统计。

像素级组间分歧必须有独立于总语义支持的有效支持分母。令
`m_g=1` 表示高斯 `g` 至少有两个有效闭合组，否则为0，并令
`b^group_pg=b_pg m_g`，则：

\[
D^{group}_p=
\frac{\sum_g b^{group}_{pg}D^{group}_g}
{\sum_g b^{group}_{pg}+\epsilon}
\]

当分母为0时输出 NaN。单组高斯既不能用伪零稀释已有分歧，也不能
因为局部不满足条件而使整张地图的组间分歧失效。

manifest 必须将该层标记为 `online_estimate` 或 `exact_final`。

## 高斯拓扑变化

语义不决定 clone、split 或 prune。几何模块只在操作完成后发送显式 topology delta。

没有新语义观测时，拓扑重参数化的核心不变量是：在固定审计相机上，类别分布、evidence support ratio、entropy 和 hard class 应尽量连续。Dirichlet concentration 不是物理质量，不能仅以“所有后代 evidence 张量之和不变”替代渲染场连续性。

V2 的初始近似为：

- clone：父高斯保留状态，子高斯复制父高斯的类别 evidence distribution、concentration、group moments 和 group count；
- split：每个子高斯复制父状态，父高斯删除；
- prune：删除对应语义状态；
- 没有明确父节点的新高斯：只初始化对称先验；
- `projected_valid_mass` 等物理累计量与 Dirichlet confidence 分开处理，可按 topology weight 分配；
- 所有 clone/split 后代均标记为 approximate/dirty，并由活动窗口重新投影。

复制策略不是“产生新统计证据”，而是保持局部原语对已有知识的继承。它必须通过拓扑前后像素级渲染测试验证；若不能满足容差，则产品标记 `topology_approximate`。最终 exact rebuild 从原始标签重新建立状态。

## Evidence-conditioned Semantic TDOM

正式产品使用 `alpha_robust`。先扣除先验，得到真实类别证据：

\[
E_{gk}=\max(\alpha^{robust}_{gk}-\alpha_{0,k},0)
\]

\[
E_g=\sum_kE_{gk},\qquad A_g=1+E_g
\]

有真实证据时，证据条件类别分布为：

\[
\tilde q_g(k)=\frac{E_{gk}}{E_g}
\]

高斯 evidence ratio 和逐类别 belief mass 为：

\[
\rho_g=\frac{E_g}{A_g}=\frac{A_g-1}{A_g}
\]

\[
B_g(k)=\rho_g\tilde q_g(k)=\frac{E_{gk}}{A_g}
\]

没有真实证据时 `rho_g=0`，该高斯不向任何类别 numerator 提供质量；`tilde q_g` 视为未定义。均匀先验不能生成 background 或其他类别。

令 `kappa_pg` 为与 RGB TDOM 相同排序和 alpha compositing 得到的几何权重：

\[
C_p=\sum_g\kappa_{pg}
\]

\[
U_p=\sum_g\kappa_{pg}\rho_g
\]

\[
N_p(k)=\sum_g\kappa_{pg}B_g(k)
\]

\[
P_p(k)=\frac{N_p(k)}{U_p+\epsilon}
\]

- `geometry_coverage[p] = C_p`；
- `pixel_evidence_support[p] = U_p`；
- `pixel_support_ratio[p] = U_p/(C_p+epsilon)`；
- 剩余透射率不计入 background；
- background 只能来自 `E_g,0 > 0` 的真实类别0证据。

`alpha/A` 可以作为包含先验的内部 predictive 诊断，但不能再作为正式类别 numerator，也不能被描述为已经排除先验的类别分布。

## Hard-map gate

V1 只有三个条件能够决定是否输出类别：

\[
C_p\ge\theta_C,\qquad
U_p\ge\theta_U,\qquad
\frac{U_p}{C_p+\epsilon}\ge\theta_R
\]

三个条件同时满足时输出 `argmax_k P_p(k)`，否则输出 `ignore=255`。

Top-1、margin、entropy、mixture disagreement、group disagreement 和 variance proxy 只能作为独立诊断，禁止通过默认值或可选参数改变 V1 `semantic_id`。需要研究额外拒绝条件时，应使用新的明确产品/method version，而不是静默改变 V1。

当前未校准默认值为：

| 参数 | 默认值 |
|---|---:|
| `min_coverage` | 0.05 |
| `min_absolute_support` | 0.01 |
| `min_support_ratio` | 0.05 |

实际值必须写入 manifest；在没有评测前不得描述为最优阈值。

## 独立诊断量

### 类别决策诊断

Top-1 probability、Top-1/Top-2 margin、categorical entropy 和 normalized entropy 均从最终 evidence-conditioned `P_p` 计算。无真实 evidence 时它们为 NaN。

### 高斯混合分歧

令 `b_pg = kappa_pg * rho_g`，则像素内不同高斯类别意见的混合分歧为：

\[
D^{mix}_p=
\frac{\sum_gb_{pg}\lVert\tilde q_g\rVert_2^2}{U_p+\epsilon}
-\lVert P_p\rVert_2^2
\]

该值与 observation-group disagreement 分开输出，不参与 hard-map gate。

### Prior-regularized variance proxy

原始 `alpha_robust` 的逐类别 Dirichlet 方差为：

\[
V_{gk}=\frac{\alpha_{gk}(A_g-\alpha_{gk})}
{A_g^2(A_g+1)}
\]

它包含先验，用于表达 evidence concentration 下的启发式不确定性，不是 `P_p` 的校准错误概率。令：

\[
\bar b_{pg}=\frac{b_{pg}}{U_p+\epsilon}
\]

像素 trace proxy 按独立近似进行平方权重传播：

\[
V^{proxy}_p=\sum_g\bar b_{pg}^{\,2}\sum_kV_{gk}
\]

manifest 必须将其定义为 `independence-approximation, prior-regularized Dirichlet trace-variance proxy`，并注明未校准。无真实 evidence 时为 NaN。`1-rho_g` 另行表达逐高斯先验/未知质量。

## 在线一致性与产品状态

历史证据由当时的几何、opacity、scale 和相机姿态投影产生；只刷新有限活动窗口不能使全部历史 evidence 与最新几何严格一致。因此在线结果一律视为有界近似。

至少区分：

| 状态 | 含义 | 产品资格 |
|---|---|---|
| `approx_online` | 正常在线近似，陈旧度与 lag 未超当前 SLO | 实时预览 |
| `stale_online` | lag、历史证据陈旧、deferred 或 error 超限 | 诊断 |
| `online_fallback` | 已请求 final，但 exact rebuild 失败 | 诊断 |
| `exact_partial` | 最终几何上重投影全部可用标签，但输入不完整或 source 未完全绑定 | 有质量声明的最终结果 |
| `exact_final` | 预期标签、来源和最终重建全部完整 | 唯一完整权威结果 |
| `failed` | 未达到最低产品要求 | 不发布类别产品 |

在线 readiness 使用连续前缀 watermark 计算 lag；最终冻结几何上的重建可以使用缺口后的所有有效标签。final manifest 必须记录预期、有效、缺失、失败和 source-unverified 的影像 ID 与比例。

最终 exact rebuilder 必须能够在 online hook 已失效时，从最终几何、相机、overlap、class spec 和原始标签独立从零启动。在线 ledger 可以加速查找，但不能成为正确性的必要条件。

## 固定地图网格与成对发布

同一任务在开始时固定局部地图网格的：

- 投影平面法向和二维轴方向；
- origin；
- GSD；
- pixel-axis convention；
- grid ID。

未知完整范围时，通过固定 origin/GSD 的 tile index 向外扩展，不能移动或缩放已经发布的网格。尚未实现固定网格时，产品只能标为 `pair-aligned`，不能声称不同 stage 之间已经对齐。

地图发布事件与几何阶段分开。latest-wins 调度可以用更新的发布事件替换旧事件，但普通非发布 stage 不能取消尚未开始的 cadence 发布意图；该意图会迁移到最新 snapshot。标签观测始终独立累计，不能因地图请求合并而丢弃。每个发布事件冻结一个共同 snapshot，并原子发布：

- RGB TDOM 与 sidecar；
- Semantic TDOM 与 manifest；
- 引用两侧 manifest/hash 的 `pair_manifest.json`。

两侧必须共享 `product_key`、`projection_geometry_hash`、相机、grid、extent、GSD、尺寸和实际渲染所用几何阶段。`projection_geometry_hash` 只覆盖语义投影相关的 `xyz/opacity/scale/rotation`；相机、grid 和 window 的完整 JSON 另由共同的 `alignment_sha256` 绑定。`rgb_model_hash` 另含 SH/appearance。Semantic manifest 另记录由语义张量与 recovery context 计算的不可变 `semantic_state_hash`、method/schema version、label version 和所有产物 hash；pair 不引用会被双槽 checkpoint 后续覆盖的 state 文件。

当前 pair 的提交布局为：

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

消费者只接受 hash 校验通过的 `latest.json -> pair_manifest.json`。`pair_manifest.json` 直接记录 projection/RGB/state hash、两侧 manifest hash 及两侧所有产品文件 hash；任何 `.publishing` 或 `.previous` 目录都不是已提交产品。

## 产物分级

`approx_online` 的核心产物为：

```text
semantic_id.png
semantic_color.png
unknown_mask.png
unknown_reason.png
pixel_evidence_support.tif
semantic_entropy.tif
class_spec.json
manifest.json
```

`exact_final`/`exact_partial` 默认额外输出：

```text
geometry_coverage.tif
pixel_support_ratio.tif
top1_probability.tif
margin.tif
mixture_disagreement.tif
group_disagreement.tif
prior_regularized_variance_proxy.tif
```

完整 `K x H x W` evidence-conditioned 概率栅格始终是可选的 `semantic_probabilities.npy`。manifest 必须列出实际文件、hash、encoding 和缺省原因，消费者不能假设可选诊断一定存在。

`unknown_reason.png` 只编码低几何覆盖、低绝对 evidence support、低 support ratio 和无 evidence。其他诊断不得暗中增加拒绝原因。旧名称 `semantic_support.tif` 和 `posterior_variance.tif` 最多保留一个 schema 版本的 deprecated alias。

## 状态、身份与恢复

每高斯 sidecar 至少保存：

```text
alpha_raw[N, K]
alpha_robust[N, K]
group weighted moments/count/support
projected_valid_mass[N]
gaussian_ids[N]
topology version and dirty/staleness state
open/recent group caches
label ledger and provenance
semantic method/config versions
```

身份拆分为：

- `ontology_hash`：ignore ID、类别 ID、名称、顺序和 background 语义；改变时发布新标签/方法版本并重建 evidence；
- `palette_hash`：显示颜色；改变时只需重渲染 color 产品；
- 完整 `class_spec_sha256`：兼容与审计用途；
- `semantic_projection_hash`：投影相关几何和相机；
- `rgb_model_hash`：包含 SH 的完整 RGB 模型；
- `semantic_state_hash`：sidecar 内容与 recovery context。

仅修改 palette 不得强迫重建 alpha。恢复必须校验 method/schema、ontology、label provenance、投影几何、高斯 ID/数量、拓扑版本、配置、ledger 和开放组。任何关键项不匹配时显式拒绝，不得静默继续。

`g1-evidence-v2` 必须使用新的 state/recovery schema 和 product/manifest schema。旧 sidecar 不允许直接 resume；必须从对应标签和几何显式重建。旧产物可以保留用于比较，但必须标记为 superseded，不能在原目录无提示覆盖。

## 准实时资源合同

- 正式开发硬件基线为单张 24 GB RTX 4090。
- 非发布 stage 的 semantic projection snapshot 只复制 `xyz/opacity/scale/rotation` 和必要相机数据，不复制无关 SH；paired publication 的共同 snapshot 为渲染 RGB 必须额外包含 SH。
- 语义 GPU 工作使用显式低优先级 CUDA stream 和 snapshot-ready event；CPU worker 异步本身不代表 GPU 隔离。
- 必须配置显存 headroom、单次预算和 checkpoint cadence。
- 超预算时依次停止中间诊断写盘、跳过过时 TDOM、合并到最新 snapshot；不得丢弃未计账标签或阻塞几何。
- 不默认每个阶段保存完整 RGB geometry checkpoint；语义 sidecar 使用周期 checkpoint 与最终原子 checkpoint。sidecar checkpoint 只保存语义投影所需四个几何字段，`rgb_model_hash` 允许为空；只有 paired publication 的冻结快照包含 SH。
- full RGB snapshot 复制前先按模型 tensor 字节数检查剩余显存，并按未压缩上界预检输出预算；先关闭可选 diagnostics/probabilities，仍超限才跳过地图。在线时间预算在 semantic raster 后、RGB raster 前再次检查；单个不可中断 CUDA 调用本身不宣称硬 deadline。
- final exact rebuild 的耗时单独统计，不计入在线 SLO。
- 在干净环境完成 stage 30 到 100 profile 前，不固定具体在线耗时百分比。

性能记录至少拆分标签读取、边界权重、projection/fusion、group update、TDOM render、写盘、checkpoint、CPU/GPU 峰值内存和 lag。

## 当前未校准默认值

| 参数 | 默认值 | 说明 |
|---|---:|---|
| overlap threshold | 0.4 | anchor-based 分组 |
| max group size | 4 | 延迟/相关性默认，不是数学上限 |
| boundary bandwidth | 2 | label pixel |
| boundary floor | 0.25 | `[0,1]` |
| `S_min` | 1.0 | LOO 其他组累计有效支持候选值 |
| reliability floor | 0.1 | JS 权重正下限 |
| JS temperature | 0.1 | JS 软权重温度 |
| min geometry coverage | 0.05 | hard-map gate |
| min pixel evidence support | 0.01 | hard-map gate |
| min pixel support ratio | 0.05 | hard-map gate |

这些值全部必须配置化并写入 manifest；当前没有人工真值或正式评测支持其最优性。

## 实施与验证门槛

本版本按以下顺序完成实现与验证：

1. CPU/reference 数学与边界条件测试；
2. state/schema/version 与旧状态拒绝测试；
3. CUDA 与 CPU reference 数值一致性；
4. 拓扑前后渲染场连续性测试；
5. online staleness/group disagreement/final recovery；
6. 固定网格与原子 paired publication；
7. 只读 dataset preflight/cache；
8. 干净 `semantic-atdom` 环境中的真实 CUDA import；
9. Phantom3 stage 30 到 31 smoke；
10. Phantom3 完整399张运行与资源 profile。

CPU 数学合同先于 CUDA 实现稳定。单元测试不替代端到端运行；每项仍未完成的验证必须显式标为“未验证”。

当前实现已经覆盖 evidence-conditioned numerator、prior-free LOO、三条件 gate、新 schema、投影专用 snapshot、固定本地网格、原子 RGB/Semantic pair、独立 final rebuild 和只读 dataset preflight/cache。正式环境、真实 CUDA smoke 与 Phantom3 完整399张渐进运行均已验证；剩余边界以[全局认知](../project/global-context.md)和[完整实验记录](../project/phantom3-semantic-atom-v2-result.md)为准。
