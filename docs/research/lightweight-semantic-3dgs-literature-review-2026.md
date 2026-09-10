# 面向 Semantic A-TDOM 的轻量 Semantic 3DGS 文献深读（2026）

## 文档目的

本文回答的不是“哪篇 Semantic 3DGS 论文指标最高”，而是：在连续无人机下视影像、已有 A-TDOM 渐进式几何、离线 SAM3 类别标签、单张 24 GB RTX 4090，以及**语义不得影响任何几何优化**的约束下，哪些论文思想能够组成一条合理的准实时路线。

检索截至 2026-09-10。2026 年工作尚不具备长期引用统计，因此“有影响力”按顶级会议/期刊、后续方法采用情况和与本项目问题的直接相关性共同判断，而不把“最新”等同于“已经充分验证”。正文只引用论文、会议论文页和作者代码等一手来源。

当前已实现的 V1 基线见 [Semantic TDOM V1 设计](../design/semantic-tdom-v1.md)，本文重点审视其后续 G2/G3 语义优化方向，不修改已经确认的几何防火墙。

## 结论先行

1. **当前 categorical–Dirichlet 主线是正确的权威状态。** Continuous Semantic Splatting（CSS）证明了把 `alpha × transmittance` 当作概率核、对每个高斯累计 Dirichlet 证据，可以在不训练语义特征的情况下完成多视角融合并显式表达不确定性。[CSS](https://arxiv.org/abs/2411.02547)
2. **“轻量语义场”应定义为任务维度 $K$，而不是经过压缩的视觉 embedding。** 对当前八分类任务，权威状态应是 $K$ 维类别浓度/概率及少量可靠性统计；16D、128D、512D 特征仍分别服务于开放词汇、实例检索或语言对齐，不是本项目的必要成本。
3. **学习模块不应取代概率证据层。** 最合适的第一种可学习能力是小型“观测可靠度校准器”；第二种才是只在不确定局部运行的 $K$ 维残差图正则器。二者都只能读取 `detach` 后的几何属性，只更新语义 sidecar，并必须保留 G0/G1 以便审计和回退。
4. **跨视角一致并不等于语义正确。** 多视角优化能抑制随机错误、遮挡错误和边界抖动，却无法从内部识别“所有视角都稳定预测错误”的 SAM 系统性偏差。没有人工真值或独立先验时，学习模块不能被宣称提高了真实精度。
5. **大高斯跨类别边界是表示上限，不是简单投票问题。** 保留 `building=0.55, vegetation=0.45` 比强制选择 0.55 更诚实。NG-GS、COB-GS、Trace3D 和 ExtrinSplat 都从不同角度确认了边界高斯的基础性困难；在禁止语义修改几何的前提下，我们只能降低污染、标记不确定和优化 TDOM 决策，不能彻底恢复几何没有表达出的边界。[NG-GS](https://arxiv.org/abs/2604.14706) · [COB-GS](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_COB-GS_Clear_Object_Boundaries_in_3DGS_Segmentation_Based_on_Boundary-Adaptive_CVPR_2025_paper.html) · [Trace3D](https://openaccess.thecvf.com/content/ICCV2025/html/Shen_Trace3D_Consistent_Segmentation_Lifting_via_Gaussian_Instance_Tracing_ICCV_2025_paper.html) · [ExtrinSplat](https://arxiv.org/abs/2509.22225)
6. **准实时必须依靠有界增量状态，而不是反复全局训练。** S2GS 的严格因果、历史不重处理原则与本项目方向一致；但其大模型双分支和实例查询表示并不适合我们的闭集制图任务。[S2GS](https://arxiv.org/abs/2603.14232)

综合判断：V1 的 G0/G1 不是临时占位，而应长期作为系统的概率底座。未来学习式优化应增加一个独立的 G2/G3 结果层，而不是把系统改造成 Feature 3DGS、LangSplat 或 NRGS 式的高维 embedding 场。

## 从第一性原理描述问题

对一个高斯 $g$，不同相机 $v$ 提供的二维类别观测并不是独立真值，而是以下因素的混合结果：

\[
y_{v,p}=\mathcal{A}(z_g,\;\text{projection}_{v,g,p},\;\text{occlusion},\;\text{SAM error},\;\text{geometry error})
\]

其中 $z_g$ 是我们想估计的类别状态，$\mathcal{A}$ 包含标签模型、视角、遮挡和当前 3DGS 表示造成的误差。由此至少要区分四类不一致：

| 误差类型 | 典型表现 | 语义优化能否处理 |
|---|---|---|
| 随机观测噪声 | 少数视角把同一区域分错 | 多视角概率融合、鲁棒降权可以处理 |
| 相关帧重复 | 连续航片重复同一个错误 | 必须先做观测分组；普通累加会制造“高置信错误” |
| 边界/遮挡歧义 | 大高斯跨建筑和植被，或前后景混合 | 可保留软分布、降低边界权重、局部正则；不能凭语义完全恢复缺失的几何边界 |
| 系统性标签偏差 | SAM 在所有视角都稳定把某类分错 | 仅靠跨视角一致性不可识别，需要真值、类别先验或另一独立模型 |

所以“解决多视角不一致”不能泛化成“让结果更平滑”。我们的目标应是：在不丢失冲突证据的前提下，提高**可解释的一致性**；对于无法判定的区域保持不确定，而不是让网络自信地补一个类别。

## 文献脉络与适用性

### 1. 高维特征蒸馏：能力强，但任务错配

[Feature 3DGS（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Feature_3DGS_Supercharging_3D_Gaussian_Splatting_to_Enable_Distilled_Feature_CVPR_2024_paper.pdf)把任意维度特征作为高斯属性参与可微渲染，为后续语言场和实例场奠定了通用接口。[LEGaussians（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Shi_Language_Embedded_3D_Gaussians_for_Open-Vocabulary_Scene_Understanding_CVPR_2024_paper.html)使用量化语义潜变量、轻量解码器和不确定性建模；[SAGA](https://arxiv.org/abs/2312.00860)学习可按尺度查询的 affinity feature；[Gaussian Grouping（ECCV 2024）](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04195.pdf)为高斯学习 identity encoding 和 3D 空间一致性。

这些方法优化的是开放词汇查询、实例选择、编辑或通用特征蒸馏。它们证明了“语义可以随高斯渲染”，却没有证明固定八分类地图需要一个高维潜空间。对本项目而言，可借鉴的是语义 rasterization、局部空间一致性和不确定性门控；应拒绝的是每场景长迭代、高维 feature sidecar，以及语义损失影响几何或 densification。

### 2. 闭式投影与概率融合：最接近我们的主干

[CSS（ICRA 2025）](https://arxiv.org/html/2411.02547)把高斯对像素的前向合成贡献作为核，对 one-hot 类别观测执行 Dirichlet 更新，并能渲染类别期望与方差。它的重要性在于：

- 语义融合不需要 optimizer；
- 状态维度直接等于类别数 $K$；
- 证据不足和类别冲突不会被压成同一个硬标签；
- 几何可以完全冻结。

CSS 的不足也恰好定义了本项目的研究增量：原论文作用于训练完成的 3DGS，没有处理渐进式 split/prune、连续航片相关性、标签版本与幂等、在线近似和最终重建两条路径。当前 V1 的 evidence group、ledger、TopologyDelta、G0/G1 和 final exact rebuild，正是在补这些系统缺口。

[FlashSplat（ECCV 2024）](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03300.pdf)在冻结 3DGS 上把二值 mask 到高斯标签的分配写成固定 alpha 权重下的优化问题，并给出快速求解。这进一步说明语义赋值不必与几何训练绑定。但它面向单对象二值硬分配，缺少我们需要的多类别概率、不确定性和渐进状态。

[LUDVIG（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/papers/Marrie_LUDVIG_Learning-Free_Uplifting_of_2D_Visual_Features_to_Gaussian_Splatting_ICCV_2025_paper.pdf)给出了重要的工程补充：把所有视图中的完整渲染权重写成归一化反向聚合 `D⁻¹WᵀF`，无需场景训练，CUDA 实现报告约为每视图、每特征维 2 ms。它仍服务于任意维视觉特征，并可选使用 DINOv2 与 3D KNN 图扩散；我们应借鉴其完整 alpha-T 反向规约和增量统计实现，不引入其高维扩散场。

### 3. 多视角噪声修正：应借思想，不应照搬表示

[NRGS（2026）](https://arxiv.org/html/2604.22439)先把多视角视觉特征聚合到高斯，计算跨视角逐维方差，再用位置、opacity、旋转、尺度和 RGB 驱动一个条件 MLP；低方差高斯在损失中权重更高。其最终输出仍是 512D 特征，且单场景正则需要数分钟。论文最值得吸收的是两个原则：

1. 跨视角方差应成为独立的可靠性输入；
2. 低不确定区域可以充当高质量锚点，不能让边界/高方差样本平等地主导训练。

不应吸收的是 512D 输出、对所有高斯执行同一网络以及以“拟合初始特征”为主要目标。对闭集制图，网络最多输出一个标量可靠度、$K$ 个类别校准量或 $K$ 维残差。

[CAGS（2025）](https://arxiv.org/abs/2504.11893)使用局部高斯图处理跨视角粒度不一致，并报告过深图传播会过度平滑。这支持我们未来只做 1–2 层、边缘感知、残差式消息传递，而不是在整场景反复扩散类别。

[ReSemGS-SLAM（Knowledge-Based Systems 2026）](https://www.sciencedirect.com/science/article/pii/S0950705126007811)采用加权多帧类别分布、语义金字塔和跨尺度 JS divergence 正则，并报告在 Jetson AGX Orin 上运行。它说明分布级一致性适合在线语义，但其 semantic-aware tracking 已让语义参与位姿估计，不能直接作为严格几何隔离的实现依据。当前 G1 将 JS 改造成 leave-one-group-out 观测冲突软权重，是本项目自己的改造，并非复现该论文公式。

[PLGS（IEEE TIP 2025）](https://arxiv.org/abs/2410.17505)更直接处理机器标签错误和跨视图不一致：先用深度 warp 筛选相邻视图一致标签，再通过 3D 聚合、区域融合和自训练产生较稳定伪标签。它同时处理实例 ID，并用语义点云初始化和训练结构化高斯，因此违反几何防火墙；可借鉴的是观测级一致性筛选，而不是其场表示或联合训练。

### 4. 边界问题：有些误差不能在语义层“优化掉”

[COB-GS（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_COB-GS_Clear_Object_Boundaries_in_3DGS_Segmentation_Based_on_Boundary-Adaptive_CVPR_2025_paper.html)通过语义梯度指导高斯 split；[Trace3D（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Shen_Trace3D_Consistent_Segmentation_Lifting_via_Gaussian_Instance_Tracing_ICCV_2025_paper.html)追踪跨视角实例并对歧义高斯进行密度控制；[NG-GS（CVPR 2026 Highlight）](https://arxiv.org/abs/2604.14706)用 mask variance 找边界歧义高斯，再以 RBF 连续场和轻量 NeRF 进行空间连续性优化。

三者共同说明，边界误差部分来自“一个离散高斯覆盖了多个语义区域”。前两者通过改变高斯拓扑解决，NG-GS 另建连续场，但仍采用联合对齐/连续性优化。这些方案越过了本项目“语义不影响几何”的边界。

[ExtrinSplat（CVPR 2026）](https://arxiv.org/html/2509.22225)进一步提出“neutral points”：有些边界高斯本来就是为了外观混合而存在，强制给每个点唯一语义会制造错误。它采用对象分组和外部文本索引，适合开放词汇/多义对象；本项目不需要其 VLM 管线，但应接受其核心提醒：**高斯是几何/外观原语，不天然是语义原子。** 我们应允许低 margin、高熵或低 opacity 的边界高斯保持 uncertain，而不是硬取最高置信类。

### 5. 在线 Semantic GS：速度数字不能脱离系统边界

[SGS-SLAM（ECCV 2024）](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04516.pdf)、[SemGauss-SLAM](https://arxiv.org/abs/2403.07494)和 [GS3LAM（ACM MM 2024）](https://dl.acm.org/doi/10.1145/3664647.3680739)都展示了在线语义高斯地图，但语义分别进入 tracking、bundle adjustment、mapping loss、几何正则或高斯管理。因此，它们证明“可以运行”，却不满足我们的因果隔离要求。

[GSFF-SLAM](https://arxiv.org/abs/2504.19409)更接近本项目：tracking 使用 RGB-D，并显式阻断语义 embedding 对高斯均值和协方差的梯度。它仍维护 128D 特征，语义处理代价对闭集任务过高，但其梯度防火墙应该成为我们的自动化不变量测试。

[S2GS（2026）](https://arxiv.org/html/2603.14232)强调严格因果、历史不重处理和持久状态，能处理 1,000+ 帧；这与我们的在线账本和有界活动窗口方向一致。但它是从无位姿 RGB 流同时回归几何、实例 query 和开放词汇语义的 feed-forward 系统，使用 foundation backbone、query decoder 和 instance memory，不能用来推断我们的轻量融合也需要同等复杂度。

[SAGOnline（2026）](https://arxiv.org/html/2508.08219)提供了另一种有用的分层思想：先快速得到离散高斯标签，再后台训练二维 RGB+粗语义细化器。它的初始聚合很快，但后续每场景仍需分钟级蒸馏，且细化结果主要存在于渲染图像空间。对我们而言，这可成为未来 Semantic TDOM 的非权威后处理对照，不能替代高斯上的权威概率场。

### 6. 航拍语义 3DGS：场景相似不代表目标相同

[EvoPropGS（AAAI 2026）](https://ojs.aaai.org/index.php/AAAI/article/view/42418)专门处理航拍 3DGS：用 SAM mask、DINOv2 prompt library、演化选择和 3D/2D 一致性从 2% 标注像素传播语义。它确认了航拍场景的尺度变化、结构重复和域差异不能按室内数据处理。

但 EvoPropGS 解决的是“绝大多数影像没有标签”，而当前 Semantic A-TDOM 计划为每张源影像提供 SAM3 类别标签。因此其演化搜索和高维 prompt feature 不应进入 V1；如果未来减少标签生成频率，它适合作为独立上游标签传播器，而不是高斯语义状态本身。

## 表示与资源量级

以当前完整实验约 658,036 个高斯为例，仅比较每高斯主向量的 float32 理论存储：

| 表示 | 每高斯主向量 | 约占用 |
|---|---:|---:|
| 当前八分类单个概率/浓度向量 | 8D | 20.1 MiB |
| raw + robust 两个八分类向量 | 16D | 40.2 MiB |
| 16D 紧凑特征 | 16D | 40.2 MiB |
| GSFF-SLAM 类 128D 特征 | 128D | 321.3 MiB |
| NRGS 输出的 512D 特征 | 512D | 1.26 GiB |

实际状态还包括方差、支持度、账本和临时统计，但数量级结论不变：固定类别任务没有理由先扩张到 128/512D，再用 decoder 压回 $K$ 类。轻量的判断标准不只是 MLP 参数少，而是**持久化场状态、渲染通道和每阶段工作量都随 $K$ 而非视觉 embedding 维度增长**。

## 推荐的分层语义路线

```text
2D SAM3 observations
        │
        ▼
G0 raw Dirichlet evidence       可审计、不可被后续覆盖
        │
        ▼
G1 robust Dirichlet posterior   分组 + JS/方差软降权，在线主结果
        │
        ├──────────────▶ Semantic TDOM（稳定基线）
        │
        ▼
G2 lightweight calibration      可选，小标量/K维，先局部后全局
        │
        ▼
G3 local categorical residual   可选，1–2层图/极小MLP，仅不确定区域
        │
        └──────────────▶ learned Semantic TDOM（独立版本）
```

### G0/G1：长期保留的概率底座

G0 保存原始分组证据，G1 保存确定性鲁棒后验。即使未来有学习网络，它们也不能被覆盖，因为：

- 网络训练失败或换版本时可以立即回退；
- 可以区分“输入证据变了”和“优化器改变了答案”；
- 能为学习模块提供 support、entropy、margin、group variance 和 JS 等可靠性特征；
- final exact rebuild 可以在最终几何上重新产生稳定教师。

### 最先值得实现的学习模块：观测可靠度校准器

对观测组 $h$ 和高斯 $g$，令一个很小的网络输出标量门控：

\[
w_{hg}=w_{\min}+(1-w_{\min})\,\sigma(f_\theta(x_{hg}))
\]

输入可包括：SAM score 的统计量、边界距离、`alpha × T` 支持、有效覆盖比例、投影尺度、入射角、观测组重叠、当前 leave-one-group-out JS、历史 entropy，以及 `detach` 后的 RGB/opacity/scale。输出只修正该组进入 Dirichlet 的权重；设正下限避免网络把冲突证据完全删除。

这是首选，因为它仍然回答一个可解释问题：“这条二维观测有多可信？”而不是让网络直接凭外观猜类别。网络可以是每类 temperature/confusion matrix，或两层、少量隐藏单元的 MLP；运行成本与新观测数量成正比，可以异步执行。

重要限制：没有 GT 时，网络只能以跨视角一致性做代理目标，容易把“多数但错误”的标签强化。因此当前阶段应只预留接口、记录训练样本和做自监督可行性实验；不能把它升级为默认权威输出。

### 第二种学习模块：局部 $K$ 类残差正则器

如果观测校准仍无法消除空间碎片，可在冻结高斯图上学习：

\[
\hat{\ell}_g=\log(\mu^{G1}_g+\epsilon)+a_g\,\Delta\ell_g,
\qquad \mu^{G3}_g=\operatorname{softmax}(\hat{\ell}_g)
\]

其中网络只输出 $K$ 维残差和一个门控 $a_g$。建议约束为：

- 只对高 entropy/低 margin/高 group variance 的 dirty 高斯运行；
- 1–2 次局部消息传递，避免 CAGS 所示的过平滑；
- 邻边同时考虑空间距离、RGB/SH 外观差、尺度/法向近似和 TDOM 邻接，不能只按欧氏距离；
- 低方差、高支持的 G1 高斯作为锚点，参考 NRGS 的 variance weighting；
- 使用 residual + trust region/KL 约束，避免远离真实观测；
- 所有几何输入 `detach`，网络参数与 RGB optimizer 完全分离；
- G3 另存版本和输出，不覆盖 G0/G1。

它比“位置到类别”的纯 MLP 更安全，因为绝对位置容易记住单场景，空间平滑容易跨越道路/屋顶边界，而残差形式明确把 G1 作为证据中心。

### 不建议作为权威场的方案

- 每高斯 CLIP/DINO/SAM embedding，即使压到 16D；
- 直接训练 RGB+semantic U-Net 只修最终 TDOM；它可做产品后处理对照，但结果不再来自权威 3D 场；
- 语义引导 split、clone、prune 或相机优化；
- 对整场景每个阶段从头训练 MLP/NeRF；
- 仅以邻居一致性为目标的无限扩散；
- 把最高类别概率直接写成不可逆硬标签。

## 准实时实现原则

1. 在线路径每张标签只吸收一次，以 `image_id + label_version` 幂等。
2. 新帧只更新可见/dirty 高斯和开放观测组；历史由固定大小统计状态表示。
3. 学习式模块必须异步且可跳过；来不及完成时 G1 仍可发布 Semantic TDOM。
4. G2/G3 的在线耗时单独计量，不混入 SAM 离线时间，也不以“渲染 FPS”冒充端到端吞吐。
5. 最终几何冻结后允许一次全量 G0/G1 重建和可选 G3 精修；两者分别报告。
6. 任何语义异常只能令语义 sidecar 失效或降级，不能阻塞、回滚或改变 A-TDOM 几何。

## 如何证明学习模块确实有用

当前没有人工真值时，只能验证：几何隔离、幂等、资源上界、顺序稳定、跨视角 JS 降低、在线/最终一致性和不确定性行为。这些能证明系统正确运行，不能证明类别更准确。

后续一旦进入 G2/G3 研究，最低限度需要一小套**从未参与训练和调参**的人工校验区，不必为全部影像制作密集真值。评价至少拆成：

- 全图 mIoU / per-class IoU；
- 建筑–植被、道路–裸土等边界带 IoU；
- expected calibration error 或可靠性曲线；
- selective risk：拒绝高不确定像素后，剩余区域错误率是否下降；
- 多视角一致性与真实精度是否同时改善；
- 额外显存、同步延迟和总完成时间。

必须包含 `G0 → G1 → G2 → G3` 消融。若一致性提高但独立真值精度下降，就说明模型只是在制造平滑或强化系统性偏差。

## 研究结论对现有设计的影响

现有 V1 主干无需推翻，但后续路线应重新精确定义：

| 现有概念 | 深读后的更准确含义 |
|---|---|
| 轻量语义场 | $K$ 维 categorical/Dirichlet 状态和少量统计，不是压缩 embedding |
| G2 边界优化 | 优先做歧义检测、unknown/margin 决策和局部类别残差；不承诺恢复几何未表达的边界 |
| G3 神经正则 | 先校准观测可靠度，再考虑 \(K\) 维局部残差；不照搬 NRGS 512D MLP |
| 多视角一致性 | 抑制可识别的随机/边界冲突，不等价于纠正多视角一致的系统错误 |
| 准实时 | G1 始终可发布，G2/G3 异步可跳过，计算量不随完整历史线性增长 |
| 语义–几何解耦 | 不仅不把语义参数加入 RGB optimizer，还要自动验证语义开关不改变几何轨迹和拓扑决策 |

## 下一轮讨论必须先确定的根问题

文献中“semantic”实际混合了三种不同产品：闭集地物类别图、实例/对象身份图、开放词汇特征场。它们需要完全不同的状态、损失和评价。当前八分类标签、静态车辆和 Semantic TDOM 输出天然对应第一种；如果同时要求车辆实例 ID、建筑实例或自然语言查询，轻量 $K$ 维场就不再是完整答案。

因此，下一步应先确认 V1 的语义原子究竟是**地物类别**还是还包括**实例身份**，再讨论 G2/G3 的具体网络与损失。否则我们会在类别平滑、实例关联和开放词汇三条互不相同的路线之间反复摇摆。

## 主要一手资料

- Wilson et al., [Modeling Uncertainty in 3D Gaussian Splatting through Continuous Semantic Splatting](https://arxiv.org/abs/2411.02547), ICRA 2025.
- Marrie et al., [LUDVIG](https://openaccess.thecvf.com/content/ICCV2025/papers/Marrie_LUDVIG_Learning-Free_Uplifting_of_2D_Visual_Features_to_Gaussian_Splatting_ICCV_2025_paper.pdf), ICCV 2025.
- Zhou et al., [Feature 3DGS](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Feature_3DGS_Supercharging_3D_Gaussian_Splatting_to_Enable_Distilled_Feature_CVPR_2024_paper.pdf), CVPR 2024.
- Shi et al., [Language Embedded 3D Gaussians](https://openaccess.thecvf.com/content/CVPR2024/html/Shi_Language_Embedded_3D_Gaussians_for_Open-Vocabulary_Scene_Understanding_CVPR_2024_paper.html), CVPR 2024.
- Cen et al., [SAGA](https://arxiv.org/abs/2312.00860), AAAI 2025.
- Ye et al., [Gaussian Grouping](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04195.pdf), ECCV 2024.
- He et al., [NG-GS](https://arxiv.org/abs/2604.14706), CVPR 2026 Highlight.
- Zhang et al., [COB-GS](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_COB-GS_Clear_Object_Boundaries_in_3DGS_Segmentation_Based_on_Boundary-Adaptive_CVPR_2025_paper.html), CVPR 2025.
- Shen et al., [Trace3D](https://openaccess.thecvf.com/content/ICCV2025/html/Shen_Trace3D_Consistent_Segmentation_Lifting_via_Gaussian_Instance_Tracing_ICCV_2025_paper.html), ICCV 2025.
- Yang et al., [NRGS](https://arxiv.org/abs/2604.22439), 2026.
- Wang et al., [PLGS](https://arxiv.org/abs/2410.17505), IEEE TIP 2025.
- Ding et al., [ExtrinSplat](https://arxiv.org/abs/2509.22225), CVPR 2026.
- Zhang et al., [S2GS](https://arxiv.org/abs/2603.14232), 2026.
- Sun et al., [SAGOnline](https://arxiv.org/abs/2508.08219), 2026.
- Gao et al., [Evolving Semantic Propagation for Aerial Semantic 3D Gaussian Splatting](https://ojs.aaai.org/index.php/AAAI/article/view/42418), AAAI 2026.
- Li et al., [SGS-SLAM](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04516.pdf), ECCV 2024.
- Ji et al., [SemGauss-SLAM](https://arxiv.org/abs/2403.07494), IROS 2025.
- Jiang et al., [GS3LAM](https://dl.acm.org/doi/10.1145/3664647.3680739), ACM MM 2024.
- Yan et al., [GSFF-SLAM](https://arxiv.org/abs/2504.19409), 2025/2026.
