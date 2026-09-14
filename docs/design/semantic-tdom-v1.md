# Semantic TDOM V1：已确认设计

> **V1.1 分支说明（2026-09-14）**：本文保留为 V1 工程基线。
> `semantic/g1-evidence-v1.1` 中有关正式类别概率、G1 共识资格、
> clone/split 继承、hard-map gate、状态/产品版本与质量状态的定义，已由
> [ADR 0005：V1.1 Evidence-lite](../decisions/0005-v1-1-evidence-lite.md)
> 覆盖。未被 ADR 覆盖的训练主链、几何防火墙、标签契约、全分辨率渲染和
> 异常尺度安全阀继续沿用本文。

## 文档状态

本文记录截至 2026-09-09 已确认的 Semantic TDOM V1 实施基线。研究比较和取舍依据见 [Semantic A-TDOM 推荐技术路线](../research/semantic-atdom-recommended-route.md)。仍未校准的数值参数集中列在文末，必须保持配置化，不应在实现中伪装成经验最优值。

## 项目边界

- 开发仓库只以 `A-TDOM` 为代码基线。
- `A-TDOM_Online` 只用于说明课题背景和曾经开展的工作，不是算法思路、代码结构或实现细节的参考来源，也不得从中复制实现。
- 第一阶段使用类似 `/data3/xzp/data/Onthefly/phantom3_test100` 的连续无人机下视影像，先取前 100 张构成小规模验证场景。
- 产品输出只要求俯视 RGB TDOM 与对应的 Semantic TDOM；任意视角语义渲染最多作为内部诊断能力。
- 正式部署目标为单张 24 GB RTX 4090。第一阶段语义标签离线生成，不把 SAM 推理时间计入 mapping latency。

## 目标与硬约束

1. 语义随 A-TDOM 渐进训练同步生长，而不是等全部几何训练完成后再统一后处理。
2. “离线语义”表示每张影像的标签已经预先生成，不表示语义构建发生在几何训练结束之后。
3. 语义只能读取当前高斯的几何、可见性和外观属性，绝不向位置、尺度、旋转、opacity、SH、densification 或 RGB 损失传播梯度。
4. 第一版语义更新不做反向训练，采用闭式概率更新；同一张标签不能因 RGB 训练被重复采样而被重复计为独立语义证据。
5. G（高斯语义场）是研究主线。直接分割最终 RGB TDOM 的 D 路线、直接将源标签镶嵌到地图网格的 M 路线，仅作为以后可能的比较方案。

## 数据契约

每张 RGB 影像必须存在同名的单通道语义标签 PNG，并推荐同时提供完整类别 score。语义结果在数据集根目录全局保存一次，阶段目录只表达渐进 SfM 快照：

`phantom3_test100` 的具体目录、文件内容和交付验收要求见 [Phantom3 渐进数据语义标签交付规范 V1](phantom3-semantic-label-contract-v1.md)。

```text
dataset/
├── semantic/
│   ├── class_spec.json
│   ├── labels/000001.png
│   ├── scores/000001.npy
│   └── metadata/000001.json
├── 3/
├── ...
└── 100/
```

语义账本的唯一键为 `image_id + label_version`。`label_version` 至少绑定模型/checkpoint、类别本体、prompt、阈值和标签内容哈希；发布文件必须采用临时文件加原子重命名，避免读取半文件。

类别在一次任务开始前固定，但数量 \(K\)、ID、名称、颜色和 background 策略全部来自 `class_spec.json`，不得在融合核心中写死。推荐首个任务采用以下八分类：

| ID | 类别 |
|---:|---|
| 0 | background |
| 1 | bare_soil |
| 2 | building |
| 3 | vegetation |
| 4 | paved_area |
| 5 | vehicle |
| 6 | water |
| 7 | road |
| 255 | ignore / unknown observation |

`0` 只表示明确预测的 background。SAM 分数低于阈值、图像无效区或没有标签的像素必须写为 `255`，不得作为 background 证据。适配器的标准输出为：

- `labels/<id>.png`：单通道整数图，值域为 `0..K-1` 或 `255`；
- `scores/<id>.npy`：可选的 `[K,H,W]` 浮点分数；
- `metadata/<id>.json`：模型、checkpoint、类别顺序、prompt、阈值、分辨率和哈希；
- `class_spec.json`：schema 版本、类别定义和颜色。

硬标签是必需输入，完整 scores 是推荐输入且必须被框架支持；只有 PNG 时仍可运行。由于当前没有人工真值，未校准的 SAM score 只用于阈值拒绝和诊断，不直接转换为 Dirichlet 伪计数。

## 渐进同步时序

语义更新复用 A-TDOM 已有的阶段边界，而不是插入每个 RGB iteration：

```text
新 RGB + 预生成 semantic label 到达
                │
                ▼
A-TDOM 添加相机与新高斯，并完成当前渐进批次的 RGB 训练
                │
                ▼
Semantic synchronization point（无语义反向传播）
                │
                ├── 同步高斯生命周期
                ├── 吸收尚未计入的观测组
                └── 更新 Dirichlet 语义状态
                │
                ▼
同时渲染 RGB TDOM 与 Semantic TDOM
```

确认的同步点包括：

- 前 30 张影像的初始训练、densify 和 prune 结束；
- 每次 `Progressive_Train` 批次结束；
- 最终 refinement 结束。

初始化同步时融合前 30 张观测并生成第一张 Semantic TDOM；第 31 张开始渐进同步。若标签尚未就绪，几何训练不得等待，语义最多允许落后一个 progressive batch。输出 manifest 必须记录 `geometry_stage`、`semantic_through_image` 和 `label_version`。

每张影像在在线路径中只投影一次。只有 dirty 高斯和活动窗口影像允许局部重投影；最终 refinement 后基于最终几何执行全量精确语义重建。

## 每高斯概率语义状态

第一版使用闭集 categorical–Dirichlet 语义场。每个高斯 `g` 保存 \(K\) 维 Dirichlet 浓度参数：

\[
\boldsymbol{\alpha}_g=[\alpha_{g0},\ldots,\alpha_{g,K-1}]
\]

采用总浓度为 1 的对称弱先验：

\[
\alpha_{0,k}=\frac{1}{K}
\]

不使用从 SAM 标签频率估计出的类别先验，避免先验偏向 background 或面积较大的类别。

每高斯还需要维护或派生：

```text
dirichlet_alpha_raw[N, K]
dirichlet_alpha_robust[N, K]
group_variance[N, K]
group_count[N]
semantic_support[N]
dirty[N]
```

类别概率、类别 ID、Dirichlet 方差、熵和 margin 均从这些状态派生，而不是作为独立真值维护。

这些状态属于非参数化的 `SemanticGaussianState`，只能读取高斯属性的 detached 视图，并在 `torch.no_grad()` 下更新；不得注册进 RGB optimizer。

## 从二维标签形成单相机观测

对相机 `v`、高斯 `g` 和像素 `p`，采用标准前向 alpha compositing 的可见性贡献：

\[
\kappa_{vgp}=\alpha_{vgp}T_{vgp}
\]

必须累计高斯的完整投影足迹，不使用仅投影高斯中心并读取单像素标签的近似。

在硬标签边界附近使用轻量距离权重 `b_p` 降低不可靠标签的影响；边界像素不做整带删除，小目标也不做额外腐蚀。`ignore=255` 的权重恒为零。

类别证据和单相机类别分布为：

\[
e_{vgk}=\sum_{p:y_p=k}\kappa_{vgp}b_p
\]

\[
q_{vgk}=\frac{e_{vgk}}{\sum_j e_{vgj}+\epsilon}
\]

## 观测支持度

定义总可见质量与有效标签质量：

\[
m^{all}_{vg}=\sum_p\kappa_{vgp}
\]

\[
m^{valid}_{vg}=\sum_{p:y_p\neq255}\kappa_{vgp}b_p
\]

单相机支持度采用无硬可见阈值的自然饱和形式：

\[
s_{vg}=
\left(1-e^{-m^{all}_{vg}}\right)
\frac{m^{valid}_{vg}}{m^{all}_{vg}+\epsilon}
\]

## 相关观测分组

连续无人机影像高度相关，因此相机不能全部作为独立 Dirichlet 样本。首个实验采用按采集顺序的锚点重叠率分组：

- 当前数据优先使用 COLMAP `imageMatchMatrix.txt`；
- 必须先根据真实采集编号重排 `imagesNames.txt` 与匹配矩阵，不能把矩阵行号误当成采集顺序；
- 新帧始终与当前组锚点比较，而不是只与前一帧比较，避免传递式重叠令分组无限增长；
- 首轮归一化重叠阈值设为 `0.4`；
- 每组最多 4 张影像；
- 一个观测组对一个高斯最多贡献一个 Dirichlet 有效样本。

组内分布和支持度为：

\[
\mathbf q_{hg}=
\frac{\sum_{v\in h}s_{vg}\mathbf q_{vg}}
{\sum_{v\in h}s_{vg}+\epsilon}
\]

\[
s_{hg}=1-\prod_{v\in h}(1-s_{vg})
\]

未压缩的投影贡献另外累计到 `semantic_support`，不直接作为 Dirichlet 样本数。

开放组必须立即提供暂定贡献，避免等待 2–4 帧才更新；新影像加入同一组时，以重算后的聚合结果**替换**旧暂定值，不得再次累加。新帧与锚点重叠低于 `0.4`、组大小达到 4、final refinement 开始或数据流结束时关闭当前组。在线每组最多保留 1–2 张代表性关键帧；最终精确重建使用组内全部有效影像。

## G0：原始 Dirichlet 后验

对已经关闭的独立观测组进行更新：

\[
\boldsymbol{\alpha}^{raw}_g=
\boldsymbol{\alpha}_0+
\sum_h s_{hg}\mathbf q_{hg}
\]

`raw` 表示给定几何版本和观测集合下未经鲁棒抑制的概率证据，不能被 G1 或后续正则结果覆盖。

## G1：两遍鲁棒 Dirichlet 后验

G1 采用确定性两遍构建。第一遍形成 `raw`；第二遍重新计算观测组贡献，并使用 leave-one-group-out 共识评价当前组。

\[
\boldsymbol{\alpha}_{-h,g}=
\boldsymbol{\alpha}^{raw}_g-s_{hg}\mathbf q_{hg}
\]

\[
\mathbf m_{-h,g}=
\frac{\boldsymbol{\alpha}_{-h,g}}
{\sum_k\alpha_{-h,g,k}}
\]

\[
d_{hg}=JS(\mathbf q_{hg},\mathbf m_{-h,g})
\]

最终鲁棒后验为：

\[
\boldsymbol{\alpha}^{robust}_g=
\boldsymbol{\alpha}_0+
\sum_h r_{hg}s_{hg}\mathbf q_{hg}
\]

`r_hg` 是随 JS divergence 单调下降且具有正下限的软权重，首版采用可配置形式：

\[
r(d)=r_{min}+(1-r_{min})\exp(-d/\tau)
\]

不得硬删除冲突组。只有一个有效组时不做冲突降权；两个组冲突时保留双方并维持不确定性；三个以上观测组中，少数冲突组由其余组形成的共识降权。

在线路径不重放全部历史组：前两个有效组权重为 1，从第三个组开始与先前闭合组共识计算 JS 软权重，并只保留必要聚合统计。最终 refinement 后，从空先验对最终几何执行全量投影，并用上述 leave-one-group-out 公式两遍精确重建 G0/G1。\(r_{min}\) 与 \(\tau\) 在没有真值前保持配置化。

## 高斯生命周期

densification API 必须在几何变更完成后向已注册的语义 observer 发布显式 `TopologyDelta`，包括保留/删除索引、父子映射与子权重；未注册 observer 时不构造事件，语义只响应拓扑结果，不参与 clone、split 或 prune 的决策。

observer 必须绑定一个具有 `num_gaussians`、`is_valid` 和 `invalidate()` 的明确 owner；closure 只有显式传入 owner 后才能注册。observer 失败不得中断或回滚几何训练。几何模块必须禁用失败的 observer、记录 `requires_rebuild` 状态并继续原几何流程；对应语义状态同时失效，只有在当前几何上显式重建并重新注册后才能继续使用。`restore`、`load_ply`、`create_from_pcd` 等整模型替换入口也必须先解绑并使已有语义状态失效。

| 几何事件 | 语义处理 |
|---|---|
| 初始点/新点扩张 | 初始化为对称弱先验，不从邻居猜类别 |
| clone/split | 子高斯继承父节点的数据证据类别比例，证据质量在后代间守恒分配 |
| prune | 删除对应语义状态和活动统计 |
| 小幅参数变化 | 在线保留近似状态 |
| 显著投影变化 | 标记 dirty，在活动窗口内重新观测 |
| final refinement 完成 | 在最终几何上从空先验全量重建 |

令父高斯的数据证据为

\[
\mathbf E_g=\boldsymbol\alpha_g-\boldsymbol\alpha_0.
\]

若产生 \(m\) 个后代，则

\[
\boldsymbol\alpha_{c_i}=\boldsymbol\alpha_0+\lambda_i\mathbf E_g,
\qquad
\lambda_i\ge0,\quad\sum_{i=1}^{m}\lambda_i=1.
\]

\(\lambda_i\) 优先按子高斯 opacity×体积归一化，无法稳定计算时均分。该规则保持的是数据证据的类别比例；由于每个子高斯重新拥有一份弱先验，其后验均值会保守地向先验收缩。不得给每个子高斯复制父节点全部浓度，否则 densification 会在没有新观测时凭空增加语义置信度。

## Semantic TDOM 概率渲染

Semantic TDOM 与同阶段 RGB TDOM 共享正射相机、范围、GSD、几何版本和像素坐标系，但语义产品使用一份只读、无梯度的几何视图。为避免异常膨胀的高斯用很暗的 SH 外观渲染成近黑 RGB、却凭高 opacity 将某一语义类别扩散到地图外缘，V1 默认仅让满足下式的高斯参与 Semantic TDOM 光栅化：

\[
\max(\mathbf s_g) \le 20\,\operatorname{median}_h\max(\mathbf s_h)
\]

这里的 `20` 是尚未经过真值校准的默认尺度比，可通过 `semantic_tdom_max_gaussian_scale_ratio` 配置，设为 `0` 可关闭。筛选只影响当前 Semantic TDOM 的几何支持域和覆盖率计算，不删除或修改高斯，不改变 RGB TDOM，不参与位置、尺度、旋转、opacity、SH、densification 或任何优化器更新。输出 manifest 必须记录输入、保留、排除高斯数量、中位尺度和实际尺度上限，以便复现实验。

对选定的 Dirichlet 后验，定义：

\[
A_g=\sum_k\alpha_{gk},\qquad
\mu_{gk}=\frac{\alpha_{gk}}{A_g}
\]

由于先验总浓度为 1，高斯的数据支持比例为：

\[
\rho_g=\frac{A_g-1}{A_g}
\]

对俯视像素 `p`：

\[
C_p=\sum_g\kappa_{pg}
\]

\[
U_p=\sum_g\kappa_{pg}\rho_g
\]

\[
N_{pk}=\sum_g\kappa_{pg}\rho_g\mu_{gk}
\]

\[
P^{TDOM}_{pk}=\frac{N_{pk}}{U_p+\epsilon}
\]

其中：

- `C_p` 是几何覆盖；
- `U_p` 是绝对语义支持；
- `U_p / C_p` 是几何覆盖区域中的语义支持比例；
- 剩余透射率 `1-C_p` 表示没有几何，不能写入 background；
- 只有几何覆盖、绝对语义支持、语义支持比例和 Top-1/Top-2 margin 同时达到阈值时才输出 `0..K-1`，否则输出 `255`。

一个跨越建筑和植被的大高斯必须保留软概率；渲染器先在 TDOM 像素级混合所有可见高斯的概率，再执行 argmax/unknown gate，不得先把每个高斯硬化成置信度更高的类别。

默认产品为与 RGB TDOM 同正射相机、范围、GSD 和渲染节奏的 `semantic_id`、`semantic_color`、`unknown_mask`、`unknown_reason`、`geometry_valid_mask`、`semantic_support` 和 `semantic_entropy`。`geometry_valid_mask` 由完成异常尺度筛选后的覆盖率 gate 生成：有效为 255，无有效几何支持为 0；无效像素在 `semantic_id` 中必须为 255、在 `semantic_color` 中必须为黑色。其余独立诊断栅格通过 `--semantic_tdom_save_diagnostics` 显式启用，完整 \(K\) 通道概率栅格仅作为可选调试/科研输出。具体 gate 阈值尚未校准，必须由配置提供。当前 RGB TDOM 没有 CRS，因此这里只保证像素级配准，不能宣称 GIS 地理参考已经完备。

`semantic_color` 只是一项派生可视化产品。运行时可从语义数据根目录的 `semantic/tdom_palette.json` 按类别名称覆盖颜色；它不得修改 `class_spec.json`、类别 ID、类别概率、unknown gate、证据融合或类别规范哈希。每次产品发布都写出完整的 `render_palette.json` 及其 SHA256，未覆盖类别继续使用 `class_spec.json` 中的原颜色。调色板文件读取失败时只产生语义告警并回退到类别规范颜色，不能中断 RGB 几何或核心语义同步。

WP4 使用独立的 forward-only CUDA kernel，在一次 TDOM 几何预处理和一次 alpha-T 遍历中渲染 `K+4` 个通道：类别分子、绝对语义支持、类别均值二阶矩、航片组分歧和 Dirichlet 方差传播项；方差传播项由 kernel 额外按平方后的 alpha-T 语义权重累积。不创建 autograd 图，也不写入几何参数。当前 kernel 支持最多 20 个 feature channels，因此本项目 8 类对应的 12 通道可直接运行。初始化、每个渐进阶段及最终 refinement 的现有 RGB TDOM 触发后，对应语义产品写入 `<Model_Path_Dir>/Semantic_TDOM/stage_<geometry_stage>/<product_key>/`。Schema V2 包含 hard map、unknown 原因、核心标量层、可选完整诊断层、完整 `class_spec.json` 和最后原子发布的 `manifest.json`；`semantic_probabilities.npy` 仅在显式启用时保存。诊断层默认量化为带显式 scale/nodata 元数据的 uint16 tiled lossless-compressed TIFF；量化误差不超过对应指标定义域跨度的 \(1/65534\)，也可配置为 float32。RGB sidecar 与语义 manifest 共享唯一 `product_key`，并记录正射相机矩阵、投影平面原点/法向/基向量/角点、视场角、范围、请求/实际 GSD、尺寸和显式 CRS；语义 manifest 另记录几何阶段、语义追踪到的影像、标签版本、几何哈希、类别规范、gate 参数、质量状态、渲染耗时及各文件 SHA256。消费者必须按 `product_key` 配对，不能仅依赖文件序号。

未校准 gate 的暂定默认值为 `coverage=0.05`、`absolute_support=0.01`、`support_ratio=0.05`、`margin=0.05`。新增的 Top-1、归一化 entropy 和 group disagreement gate 分别以 `0`、`1`、`2` 处于兼容模式，不在无真值条件下武断改变 V1 覆盖率；异常尺度筛选比暂定为 `20`。它们全部可由命令行配置，后续必须依据真实场景诊断结果校准。

渲染前按实际 feature channels、行带输出、tile finalize 工作集和每高斯状态保守估算显存，并同时检查实时空闲 CUDA 显存和默认 512 MiB headroom；CUDA 预处理得到当前行带真实的 Gaussian–tile 交集数后，还必须在分配 radix-sort/binning buffer 前校验剩余硬预算。默认优先一次性渲染；预算不足或显式设置 `render_tile_rows` 时，保持完整相机投影和全局深度顺序，以 16 像素对齐行带精确输出并逐带转移到 CPU。该降级路径会重复轻量 projection/preprocess，但每带只排序与本带相交的实例，优先保证全分辨率结果可生成和逐像素一致性，不使用半分辨率替代。

## 不确定性输出

高斯的 Dirichlet 后验方差为：

\[
V_{gk}=
\frac{\alpha_{gk}(A_g-\alpha_{gk})}
{A_g^2(A_g+1)}
\]

像素级后验方差使用归一化语义权重的平方传播。以下量必须分别输出，不能混合为单一“confidence”：

- Dirichlet 后验方差：观测数量不足带来的不确定性；
- 高斯间分歧：同一像素的可见高斯具有不同类别均值；
- 最终类别熵：像素概率分布的混合程度；
- Top-1/Top-2 margin：最终分类决策的稳定程度；
- 几何覆盖、绝对语义支持和语义支持比例。

令 \(w_{pg}=\kappa_{pg}\rho_g\)，实现中的标量分歧与方差为：

\[
D_p=\frac{\sum_g w_{pg}\lVert\mu_g\rVert_2^2}{U_p+\epsilon}
-\lVert P^{TDOM}_p\rVert_2^2
\]

\[
V_p=\frac{\sum_g w_{pg}^2\sum_kV_{gk}}{U_p^2+\epsilon}
\]

航片组分歧将每高斯 `group_variance` 的类别维 trace 按 \(w_{pg}\) 加权平均；两个相反 one-hot 分布的平方距离决定其理论范围为 `[0, 2]`，不能截断成 `[0, 1]`。三者分别描述像素内高斯意见不同、证据数量导致的后验方差和相关航片组之间的不一致，不能互相冒充。没有语义证据时，这三类不确定性与熵均为未定义值 `NaN`，不能用数值零伪装成确定。

## 状态持久化与版本校验

每高斯概率必须存在于运行时，并推荐保存为内部 `semantic_state.pt` sidecar，以支持中断恢复；它不是用户产品。sidecar 至少保存 raw/robust 浓度、group count/variance、support、dirty 状态、开放组暂定增量、label ledger、类别 schema 和配置。

恢复时必须验证高斯数量、稳定 ID 的内部完整性、几何 checkpoint 哈希、类别本体哈希，以及由严格校验后的 ledger、label provenance、开放组、类别 schema 和语义配置共同形成的 recovery-context 哈希。几何哈希由 `GaussianModel.geometry_sha256()` 对全部有序、逐行对齐的高斯属性及其形状和 dtype 计算，作为几何与 sidecar 的独立可信绑定，不能只散列文件名。任一不匹配都必须显式拒绝静默续算，并选择局部修复或从空先验重建。WP3 sidecar schema V2 保存开放组的成员元数据、暂定 G0/G1 增量和组内观测统计，恢复后可以继续以替换语义吸收同组影像。

WP3 通过 `tools/train_progressive.py --enable_semantics` 启用。同步状态写入 `<Model_Path_Dir>/semantic/`：两个轮换 slot 分别保存 `semantic_state.<slot>.pt`、`recovery_context.<slot>.json` 与对应的只读 `geometry_checkpoint.<slot>.pt`。几何 checkpoint 保存全部有序高斯属性；`latest.json` 同时绑定其文件 SHA256、属性哈希、语义 slot 和 recovery-context 哈希，并且只在整组写完后原子切换，避免中断写盘破坏上一个可恢复状态或发布一个没有对应持久化几何的语义状态。`ledger.json` 提供逐影像幂等记录，`sync_status.json` 记录 `geometry_stage`、`semantic_through_image`、`label_version` 和滞后状态。标签尚未发布或语义处理失败时只延后语义，不能中断 RGB 几何流程。

当语义交付不位于 RGB 渐进数据根目录时，通过 `--semantic_data_root` 指向包含 `semantic/class_spec.json`、`semantic/labels/` 和 `semantic/metadata/` 的独立数据根；COLMAP stage 与 RGB images 仍由 `Source_Path_Dir` 提供。该参数正好覆盖当前 `/home/xuzipeng/code/Semantic-A-TDOM/datas/semantic_labels` 的交付方式，无需复制 102 张标签。

WP5 在最终几何冻结后仅在所有已排序影像标签均就绪且无同步错误时启动：第一遍从空先验投影全部影像并形成使用组内全部视图的精确 G0；第二遍再次投影全部影像，以 leave-one-group-out 共识计算精确 G1。最终状态替换在线近似状态，最终 Semantic TDOM manifest 标记 `posterior=robust_exact_final`；耗时、峰值 CUDA 显存、组数以及在线/最终 JS 偏差和类别一致率写入 `<Model_Path_Dir>/semantic/final_rebuild_metrics.json`。若全量重建失败，保留在线状态并将错误反馈为语义告警，几何结果不受影响。

V1 的语义状态、融合和渲染计算默认使用 `float32`；产品诊断栅格可按 manifest 中的比例量化为 uint16，并由环境中明确声明的 `tifffile` 写成 tiled TIFF。scores 使用 mmap/分块读取，只保留当前影像或小批次；不得将全数据 score tensor 一次加载到显存。

## 准实时指标

- `mapping latency`：标签已可用时，从几何阶段结束到 semantic sync 与 Semantic TDOM 写盘完成；
- `end-to-end latency`：从新影像到标签、几何同步和最终产品完成，留作后续在线 SAM 阶段评价。

首版目标是在单张 24 GB RTX 4090 上，在线 semantic sync 的摊销耗时不超过对应 progressive 几何阶段耗时的 20%。final exact rebuild 单独统计，不纳入在线预算。标签默认使用原始分辨率；半分辨率只能作为 profiling/ablation 开关。

## 当前实现优先级

第一版只实现以下主链路：

```text
预生成 labels + scores（scores 可缺省）
  → 与 A-TDOM 阶段同步
  → 完整足迹 alpha-T 观测
  → 相关帧分组
  → 在线 G0 + 近似 G1
  → 最终几何上的全量精确 G0/G1
  → 概率 Semantic TDOM 与独立不确定性图层
```

实施工作包按以下顺序推进：

1. WP0：类别/标签数据契约、ledger、纯 PyTorch/CPU 参考实现和微型测试；
2. WP1：`SemanticGaussianState`、`TopologyDelta`、证据守恒和几何隔离测试；
3. WP2：无梯度 CUDA observation kernel；
4. WP3：阶段 hook、分组、开放组替换和在线 orchestrator；
5. WP4：Semantic TDOM renderer 与产品输出；
6. WP5：最终精确重建和单卡性能基准。

以下内容只记录为以后可能的扩展，不阻塞第一版：

- G2：受 NG-GS 启发的边界歧义检测与局部连续语义场；
- G3：受 NRGS 启发、冻结高斯属性的轻量语义正则网络；
- D：直接分割最终 RGB TDOM；
- M：源标签直接融合到地图网格。

## 保持配置化、等待校准的参数

以下内容不是架构悬案，不阻塞实现：

1. JS 软权重的 `r_min` 和 `tau`；
2. `C_min`、`U_min`、语义支持比例和 Top-1/Top-2 margin 阈值；
3. 标签边界距离权重的带宽和下限；
4. dirty 判定所需的位置、投影协方差和 opacity 变化阈值；
5. CUDA 的最佳 tile、分块和常用 \(K\) 模板配置。
6. Semantic TDOM 异常高斯尺度比阈值。

第一阶段不制作人工真值，因此只验收几何隔离、幂等、证据守恒、顺序稳定、unknown 行为、遮挡/足迹正确性、恢复一致性、RGB/Semantic TDOM 配准和资源预算。SAM 模型选择、阈值精度校准、正式评价指标与消融设计留到后续阶段。

## 论文依据

- Wilson et al., *Modeling Uncertainty in 3D Gaussian Splatting through Continuous Semantic Splatting*, ICRA 2025: <https://arxiv.org/abs/2411.02547>
- Zhao et al., *ReSemGS-SLAM: Real-time semantic Gaussian Splatting SLAM based on semantic consistency perception*, 2026: <https://www.sciencedirect.com/science/article/pii/S0950705126007811>
- He et al., *NG-GS: NeRF-Guided 3D Gaussian Splatting Segmentation*, CVPR 2026 Highlight: <https://arxiv.org/abs/2604.14706>
- Yang et al., *NRGS: Neural Regularization for Robust 3D Semantic Gaussian Splatting*, ICME 2026: <https://arxiv.org/abs/2604.22439>
- Gao et al., *Evolving Semantic Propagation for Aerial Semantic 3D Gaussian Splatting*, AAAI 2026: <https://ojs.aaai.org/index.php/AAAI/article/view/42418>
