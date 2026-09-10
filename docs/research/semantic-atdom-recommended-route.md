# Semantic A-TDOM 推荐技术路线

## 结论

Semantic A-TDOM V1 应采用一种**与几何优化严格单向解耦、阶段同步、闭式更新的概率语义高斯场**：A-TDOM 继续独立负责相机、几何、外观和 densification；语义模块在每个训练阶段结束后读取冻结的高斯状态，把已经生成的二维语义观测融合为每高斯类别分布，再使用与 RGB TDOM 完全相同的正射相机、范围、GSD 和输出节奏生成 Semantic TDOM。

推荐保留现有 categorical–Dirichlet 主线，但将系统拆成在线与最终两条计算路径：

```text
离线标签生产者（当前阶段）
        │ 标准标签、分数与元数据
        ▼
A-TDOM RGB/几何训练 ──阶段结束──▶ Semantic Sync
        │                              │
        │                              ├─ 在线：活动窗口、观测组、近似鲁棒更新
        │                              └─ 最终：最终几何、全部关键帧、精确两遍更新
        ▼                              ▼
      RGB TDOM                  Semantic TDOM + 诊断图层
```

这条路线满足四个核心条件：语义随渐进重建生长；不向几何或 RGB 优化传播梯度；能够吸收噪声标签并表达不确定性；在线计算量有明确上界。它也明确接受一个边界：V1 不借助语义修改高斯几何，因此语义边界的上限受 RGB 高斯结构约束。

## 已确认需求与范围

- 开发基线只使用 A-TDOM，不从 A-TDOM_Online 复制算法或代码。
- 第一阶段标签由 SAM3 系列管线离线生成；具体选择 SAM3、SAM3.1、ProC-SAM3 或其他模型不阻塞语义融合框架。
- 语义生产者与 Semantic A-TDOM 通过稳定文件契约解耦。
- V1 使用任务开始前固定、但由配置提供的闭集类别，不在代码中写死类别数量。
- `0..K-1` 是真实类别；`255` 统一表示 unknown、低置信度或无效观测。只有模型明确预测 background 时，background 才能作为一个真实类别。
- 默认消费硬标签 PNG 和完整 `[K,H,W]` 类别分数；只有 PNG 时也能运行。
- SAM3 分数在无真值阶段只用于阈值拒绝和诊断，不直接解释为 Dirichlet 样本强度。
- 第一阶段不制作人工真值，只搭建框架、验证确定性和测量效率；模型选择、阈值校准和精度评价延后。
- Phantom3 前 30 张影像用于初始化，第 31 张开始渐进处理。
- 标签投影默认使用原始分辨率；降采样仅作为后续加速开关和消融项。
- 正式部署目标为单张 24 GB RTX 4090。当前阶段 SAM 离线运行，不处理与 A-TDOM 的单卡并发调度。
- 产品主输出是 Semantic TDOM。每高斯概率状态只作为运行时状态和可恢复 sidecar，不作为用户产品。
- Semantic TDOM 与 RGB TDOM 使用相同正射相机、空间范围、GSD、渲染时机和文件组织。

## 文献路线比较

### 闭式概率融合

Continuous Semantic Splatting（CSS）直接把二维分类观测通过高斯渲染核更新到每个椭球的 Dirichlet 浓度，并传播期望与方差到像素空间。这是与本项目约束最一致的文献基础：无需学习语义特征、天然支持多视角平滑和不确定性表达。其论文实验基于已经训练完成的 3DGS，并把所有像素作为观测；本项目需要进一步解决渐进几何、相关航片、重复计数和生命周期问题。[^1]

Semantic Gaussians 也提供无需额外场景训练的二维到三维投影，但主要借助深度表面对应和跨视角特征平均，不提供 Dirichlet 式证据量与冲突不确定性。它支持“投影优先，而非语义反向训练”的总方向，但不能直接替代本项目的概率更新。[^2]

### 学习式语义特征场

Feature 3DGS、SAGA 和 Gaussian Grouping 都在高斯上增加可学习特征或 identity encoding，再用二维 foundation model 输出监督语义渲染。它们适合开放词汇、可提示实例分割或编辑，但需要额外优化，且目标并不是闭集航拍 TDOM。[^3][^4][^5]

SGS-SLAM、SemGauss-SLAM 和 GS3LAM 展示了 Semantic GS 在线运行的可行性，但都让语义参与联合损失、跟踪、bundle adjustment、几何正则或高斯管理。它们可以贡献关键帧选择、选择性语义渲染和在线缓存经验，不能作为本项目的主算法，因为本项目明确禁止语义影响几何。[^6][^7][^8]

GSFF-SLAM 把语义 embedding 的梯度与位置、尺度等属性隔离，这比联合优化更接近本项目；但它依然需要迭代优化语义 feature field。其可借鉴点是按需启用语义通道、共视关键帧和稀疏/噪声先验处理，而不是其训练式语义表示。[^9]

### 噪声、多视角与边界

ReSemGS-SLAM 使用加权多帧语义状态、语义金字塔和跨尺度 JS divergence 一致性正则，并报告实时语义映射。它支持本项目使用多帧分布、JS 冲突度量和层级处理的判断；当前项目将 JS 改造成观测可靠性软权重是自己的设计。该方法的语义参与 tracking，不符合本项目的几何防火墙。[^10]

NRGS 使用多视角方差识别不可靠的三维语义，并通过条件 MLP 修正语义场。它说明“跨视角方差”应作为单独信号保存，但整场景分钟级正则更适合作为未来最终精修，而不是准实时 V1。[^11]

NG-GS、COB-GS 和 Trace3D 都指出高斯跨越真实对象边界是 Semantic GS 的基础误差源，并分别通过连续场、语义引导 split/prune 或实例追踪改善边界。[^12][^13][^14] 这些结果说明当前设计中的轻量边界降权是必要的，也说明 V1 必须诚实呈现混合高斯的不确定性；但语义引导 densification 明确越过了本项目边界。

### 航拍与在线系统

EvoPropGS 针对航拍视角中的尺度、域差异和重复地物，用 SAM mask、DINOv2 特征和演化搜索从少量标注传播语义。它证明航拍场景不应机械沿用室内 Semantic GS，同时也表明特征传播适合“标签稀疏”问题。当前项目计划为每张图生成标签；该方法包含 SAM/DINO 和演化搜索且未报告在线性能，因此不应进入 V1 准实时主链。若未来只标注少量影像，可把它作为独立上游标签增强器。[^15]

VISTA 则证明在线 semantic 3DGS 可以服务于机器人实时任务，但它面向开放词汇主动探索，而非固定类别正射制图。它对本项目最有价值的启示是把任务相关计算限制在有效视角和局部地图，而不是反复全局处理。[^16]

## 为什么选择 Dirichlet Sidecar

### 与需求匹配

每高斯只需保存 `K` 维非负浓度，即可同时得到类别均值、观测稀缺带来的方差和总证据量。更新是加法，不需要 optimizer、学习率、反向训练或 replay loss。与几十至数百维的语言/视觉 embedding 相比，闭集八分类的状态更小，也更容易审计。

语义状态必须与 `GaussianModel` 一一对齐，但不应成为 `nn.Parameter`，也不能加入 RGB optimizer。建议单独建立 `SemanticGaussianState`，所有输入均从高斯属性的 `.detach()` 视图读取，并在 `torch.no_grad()` 作用域内更新。测试必须证明语义开关不改变 RGB 梯度、densification 决策和同随机种子下的 RGB 训练轨迹。

### 不把硬类别作为内部真值

对高斯 \(g\)，内部维护：

\[
\boldsymbol{\alpha}^{raw}_g,\quad
\boldsymbol{\alpha}^{robust}_g,\quad
\boldsymbol{\mu}_g=\frac{\boldsymbol{\alpha}_g}{\sum_k\alpha_{gk}}
\]

可以派生 `semantic_id = argmax(mu)` 用于检查，但它不是权威状态。跨边界高斯可能是 `building=0.55, vegetation=0.45`；如果提前硬化为 building，其整个足迹都会污染植被侧。正确做法是把软分布保留到 TDOM 像素级混合后再决策。

## 标准语义数据契约

建议每张原始影像只保存一份语义结果：

```text
phantom3_test100/
├── semantic/
│   ├── class_spec.json
│   ├── labels/000001.png
│   ├── scores/000001.npy
│   └── metadata/000001.json
├── 3/
├── ...
└── 100/
```

阶段目录继续表达渐进 SfM 快照，不复制历史语义文件。语义账本的唯一键应为 `image_id + label_version`；`label_version` 至少绑定模型/checkpoint 哈希、类别本体哈希、prompt 配置、阈值和标签内容哈希。文件应通过临时路径写完后原子发布，避免消费者读到半文件。

统一适配器应输出：

- `labels/<id>.png`：单通道整数图，值域为 `0..K-1` 或 `255`；
- `scores/<id>.npy`：可选的 `[K,H,W]` 浮点分数；
- `metadata/<id>.json`：模型、checkpoint、类别顺序、prompt、阈值、分辨率和哈希；
- `class_spec.json`：类别 ID、名称、颜色、background 策略和 schema 版本。

当前原生 SAM3 实验把 `0` 定义成 unknown，而 ProC-SAM3 五分类把 `0` 定义成显式 background。两者必须在适配器层统一；融合核心不能猜测 `0` 的含义。

## 单相机完整足迹观测

对图像像素 \(p\) 和按深度排列的高斯 \(g\)，继续使用 RGB rasterizer 相同的前向合成贡献：

\[
\kappa_{vgp}=\alpha_{vgp}T_{vgp},\qquad
T_{vgp}=\prod_{j<g}(1-\alpha_{vjp})
\]

必须遍历高斯的完整二维椭圆足迹。只投影中心并读取一个标签像素会忽略高斯尺度、遮挡和边界混合，也无法与最终 TDOM 的 alpha compositing 保持一致。

V1 对每个有效硬标签像素累计：

\[
e_{vgk}=\sum_{p:y_p=k}\kappa_{vgp}b_p
\]

\[
m^{all}_{vg}=\sum_p\kappa_{vgp},\qquad
m^{valid}_{vg}=\sum_{p:y_p\ne255}\kappa_{vgp}b_p
\]

其中 \(b_p\) 是标签边界距离权重。它在边界附近连续降低，但不删除整圈边界、不腐蚀小目标；unknown 像素的权重恒为零。

单相机分布和支持度为：

\[
q_{vgk}=\frac{e_{vgk}}{\sum_j e_{vgj}+\epsilon}
\]

\[
s_{vg}=\left(1-e^{-m^{all}_{vg}}\right)
\frac{m^{valid}_{vg}}{m^{all}_{vg}+\epsilon}
\]

SAM3 分数不直接乘进 \(s_{vg}\)。在缺少真值校准时，分数只用于把低于阈值的像素变为 `255`，并作为诊断数据保留。这避免将“自信但错误”的 prompt 预测误当成更多独立证据。

### CUDA 实现建议

现有 rasterizer 已在 tile 内计算每个像素的 `alpha * T`，但颜色通道目前固定为三通道。Semantic TDOM 需要两种新能力：

1. observation kernel：在前向遍历中对 `e[N,K]`、`m_all[N]` 和 `m_valid[N]` 做分块规约/原子累积；
2. semantic render kernel：支持小型动态 `K` 或编译期模板通道，并输出概率分子、覆盖、支持和不确定性矩。

生产实现不应通过语义 loss 调用 autograd。可以在 CPU/PyTorch 参考实现中利用现成运算验证公式，但 CUDA 主路径应是无梯度前向统计核，从接口上切断所有通往位置、尺度、旋转、opacity 和 SH 的梯度。

## 相关帧分组与在线更新

### 分组

连续航片高度相关，不能把每张图都当成独立 Dirichlet 样本。优先读取当前 SfM 快照的 `imagesNames.txt` 和 `imageMatchMatrix.txt`，先按真实采集编号重排矩阵，再计算归一化重叠。例如：

\[
o(i,j)=\frac{M_{ij}}{\min(M_{ii},M_{jj})}
\]

新帧只与当前组锚点比较。首轮使用 `0.4` 阈值，并增加最大组长 4。对 Phantom3 前 100 张的现有矩阵，按采集编号重排后，该配置通常得到 2–4 张一组；这比仅和前一帧比较更能避免传递式无限分组。

组内统计为：

\[
\mathbf q_{hg}=
\frac{\sum_{v\in h}s_{vg}\mathbf q_{vg}}
{\sum_{v\in h}s_{vg}+\epsilon},\qquad
s_{hg}=1-\prod_{v\in h}(1-s_{vg})
\]

一个组对一个高斯最多贡献一个有效样本。原始投影质量另计入 `semantic_support`，不能直接当成独立样本数。

### 开放组

等待组关闭才更新会带来 2–4 帧延迟，因此开放组应产生暂定贡献。每次新影像加入时重新计算当前组的聚合结果，并以“替换旧暂定值”而非“再次相加”的方式更新。组在以下任一条件满足时关闭：

- 新帧与锚点的归一化重叠低于 `0.4`；
- 组大小达到 4；
- final refinement 开始或数据流结束。

每组在线最多保留 1–2 张代表性语义关键帧；最终精确阶段可使用组内全部有效影像。

### 在线 G0/G1

G0 保留原始、可审计的证据：

\[
\boldsymbol\alpha^{raw}_g=\boldsymbol\alpha_0+
\sum_h s_{hg}\mathbf q_{hg}
\]

在线 G1 不在每个同步点重放所有历史组。前两个有效组保持权重为 1；从第三个组开始，将当前组与先前闭合组的共识计算 JS divergence，并使用有正下限的单调软权重，例如参数化的：

\[
r(d)=r_{min}+(1-r_{min})\exp(-d/\tau)
\]

具体 \(r_{min}\) 和 \(\tau\) 保持配置化，在没有真值时不能宣称已经校准。正下限、两组 warm-up 和开放组替换共同防止早期错误形成不可逆锁定。

### 最终精确 G1

最终 refinement 完成后，丢弃在线近似产生的权威地位，针对最终几何重新投影全部有效语义关键帧，重建分组和 G0，然后执行精确 leave-one-group-out：

\[
\boldsymbol\alpha_{-h,g}=\boldsymbol\alpha^{raw}_g-s_{hg}\mathbf q_{hg}
\]

\[
d_{hg}=JS\left(\mathbf q_{hg},
\frac{\boldsymbol\alpha_{-h,g}}{\sum_k\alpha_{-h,g,k}}\right)
\]

\[
\boldsymbol\alpha^{robust}_g=\boldsymbol\alpha_0+
\sum_h r(d_{hg})s_{hg}\mathbf q_{hg}
\]

一个有效组时不降权；两个冲突组时保留双方并体现不确定性；至少三个组后才允许多数共识软抑制少数组。

## 高斯生命周期

语义状态不能仅靠数组拼接被动跟随高斯数量，densification API 应在几何变更完成后向已注册的语义 observer 发布显式拓扑变化描述，例如 `TopologyDelta`：保留索引、删除索引、父子映射和子权重。未注册 observer 时不构造事件。

observer 必须绑定可失效的明确 owner；closure 需要显式声明 owner。observer 异常必须与几何主链路隔离：立即禁用语义同步、记录需要重建的显式状态并让几何阶段继续；失效的语义 sidecar 不得继续融合或保存。所有整模型替换入口同样必须使已绑定 sidecar 失效。

推荐规则如下：

| 几何事件 | 语义处理 |
|---|---|
| 初始点/新点扩张 | 初始化为对称弱先验，不从邻居猜类别 |
| clone | 父与新子共享数据证据类别比例，并在父子间分配证据质量 |
| split | 子高斯继承父节点的数据证据类别比例，证据在子高斯间守恒分配 |
| prune | 删除对应语义状态和活动统计 |
| 小幅位置/尺度/opacity 变化 | 在线保留近似状态 |
| 显著投影变化 | 标记 dirty，在活动窗口内重新观测 |
| final refinement 完成 | 全部语义状态基于最终几何重建 |

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

\(\lambda_i\) 可先按子高斯的 opacity×体积归一化，无法稳定计算时均分。这样所有后代的数据证据之和等于 \(\mathbf E_g\)，且数据证据的类别比例保持不变；每个后代因重新拥有弱先验，后验均值会保守地向先验收缩。完整复制会让 densification 在没有新影像的情况下凭空增加语义置信度。

移动显著性的准确阈值暂不在无真值条件下拍脑袋固定。实现上同时记录中心位移、投影协方差变化和 opacity 相对变化，并把阈值配置化；首轮性能测试后再确定哪些变化值得触发活动窗口重投影。

## 阶段同步协议

语义与渐进几何不是逐迭代交织，而是在已有 A-TDOM 阶段边界上同步：

1. **初始化阶段**：前 30 张影像完成 RGB/几何训练、densify 和 prune；随后冻结高斯，融合前 30 张语义观测，生成第一对 RGB TDOM 与 Semantic TDOM。
2. **渐进阶段**：从第 31 张开始，每个已有 progressive batch 完成几何更新后触发一次 semantic sync；同步消费尚未入账的新影像和活动组替换，随后生成配对产品。
3. **最终阶段**：final refinement 完成后冻结最终高斯；从空语义先验出发，对全部有效影像执行完整足迹重投影和精确两遍 G1，生成最终 Semantic TDOM。

若异步语义 worker 在某个阶段尚未完成，几何训练继续运行，Semantic TDOM 可以落后一个 progressive batch。输出清单必须分别记录 `geometry_stage`、`semantic_through_image` 和 `label_version`，不能把滞后的语义图伪装成当前阶段产品。

每张影像在在线路径中只进入账本一次。几何后续变化不会触发无界历史 replay：只有 dirty 高斯与活动窗口影像允许局部重投影；最终一致性由全量重建保证。

## Semantic TDOM 渲染与输出

渲染严格复用 RGB TDOM 的正射相机、tile、范围、GSD 和 alpha compositing 顺序。对正射像素 \(p\)，定义：

\[
A_p=\sum_g \kappa_{gp},\qquad
\mathbf C_p=\sum_g \kappa_{gp}\boldsymbol\mu_g
\]

\[
\mathbf P_p=\frac{\mathbf C_p}{A_p+\epsilon}
\]

其中 \(\boldsymbol\mu_g\) 使用 robust Dirichlet 后验均值。这样，一个同时跨越建筑与植被的大高斯会保留软分布；像素还会结合沿光线其他高斯的分布，最终才做类别判定，而不是先给该高斯强制分配“置信度更高的一类”。

令 \(c_g=1-\exp(-support_g)\) 表示高斯语义支持的饱和映射。覆盖度和语义支持必须分开：

\[
\rho_p=A_p,\qquad
S_p=\frac{\sum_g\kappa_{gp}c_g}{A_p+\epsilon}
\]

仅当覆盖度、支持度以及第一名与第二名的概率间隔均达到配置阈值时，输出

\[
\hat y_p=\arg\max_k P_{pk};
\]

否则输出 `255`。这让混合边界、未观测区域和多视角强冲突自然进入 unknown，而不是制造看似精确的硬边界。

熵只表达类别混合：

\[
H_p=-\sum_k P_{pk}\log(P_{pk}+\epsilon)
\]

证据不足则由 `support` 和 Dirichlet 总浓度表达。两者不能合并为一张“置信度”，否则“意见一致但只看过一次”和“看过很多次但相互冲突”会不可区分。

默认产品包括：

- `semantic_id`：`0..K-1` 或 `255`；
- `semantic_color`：按 `class_spec.json` 着色；
- `unknown_mask`；
- `semantic_support`；
- `semantic_entropy`；
- 与 RGB TDOM 对齐的地理参考和 manifest。

完整 \(K\) 通道概率栅格作为可选调试/科研输出，默认不发布，以免显著增加 I/O。文件名、坐标系和目录层级直接跟随当前 RGB TDOM 输出逻辑。

## 状态、恢复与版本一致性

运行时状态至少包含：

- `alpha_raw[N,K]` 与 `alpha_robust[N,K]`；
- 有效组数、累计投影支持、冲突/JS 统计；
- dirty 标志和最后消费的 `image_id + label_version`；
- 开放组的暂定聚合及其可替换增量；
- 类别 schema、语义配置和最终几何标识。

建议将它们保存为内部 `semantic_state.pt` sidecar。恢复时必须验证高斯数量、稳定 ID 的内部完整性、类别本体哈希、严格校验并交叉散列的 label ledger/provenance/config，以及覆盖全部有序高斯属性的 geometry checkpoint 哈希；任何一项不一致都应拒绝静默续算，并选择局部修复或从空先验重建。

V1 优先使用 `float32`，先保证数值和可审计性。以 \(K=8\) 计，raw 与 robust 两份浓度约为每高斯 64 字节，另加少量统计；百万高斯仍在百 MB 量级。若显存测量成为瓶颈，再评估只常驻 robust、raw 分块落盘或 `float16/bfloat16`，不能一开始用低精度掩盖累加误差。

上游完整 score tensor 的体积反而可能更大。例如 \(7\times1600\times897\) 的 `float32` 单图约 40 MB。框架需支持 `.npy` mmap/分块读取；在线融合只保留当前影像或小批次，不能把全数据 scores 一次载入显存。

## 准实时边界与测量方法

第一阶段的“准实时”限定为：离线标签已就绪时，语义融合不会阻塞 RGB/几何训练，并能在每个几何阶段完成后较快追平并输出 Semantic TDOM。它不是当前阶段对 SAM3 推理速度的承诺。

统一报告两种延迟：

- **mapping latency**：标签已经可用，从几何阶段结束到 semantic sync 与 Semantic TDOM 写盘完成；
- **end-to-end latency**：从新影像到标签、几何同步和最终产品完成。第一阶段只记录接口占位，不用离线标签成绩冒充端到端实时性。

单张 24 GB RTX 4090 是正式性能基线。首版目标是在线 semantic sync 的摊销耗时不超过对应 progressive 几何阶段耗时的 20%，同时保证显存峰值可与 A-TDOM 顺序复用；final exact rebuild 单独统计，不纳入在线 SLA。

默认保持原始 label 分辨率。性能优化按以下顺序进行：

1. 只处理新影像、活动组和可见高斯；
2. 将投影统计融合进现有 tile rasterization，避免导出稠密 per-Gaussian/per-pixel 中间量；
3. 分块规约、减少全局 atomic 冲突，并按 `K` 专门化常用 kernel；
4. semantic render 只在当前 RGB TDOM 本就需要输出时触发；
5. score 使用 mmap、pinned memory 与异步传输；
6. 最后才评估半分辨率，并必须作为精度—延迟消融，不得成为默认值。

性能日志至少分解为标签读取、边界权重、projection/fusion、group update、TDOM render、写盘和峰值显存，避免只给一个不可诊断的总耗时。

## 代码落点与实施顺序

### WP0：数据契约与参考实现

- 新增 `SemanticClassSpec`、label adapter、manifest/ledger 校验；
- 用纯 PyTorch/CPU 小场景实现完整足迹统计、Dirichlet 更新和 TDOM 概率渲染；
- 构造手工微型标签，不构建真实数据人工真值；
- 先固定 `255`、background、边界权重和空观测行为。

### WP1：语义状态与几何防火墙

- 在 `src/semantic_atdom/scene/` 增加非参数化的 `SemanticGaussianState`；
- 为 `GaussianModel` 的 clone、split、prune、append 暴露明确 `TopologyDelta`；
- 实现证据守恒继承和 `semantic_state.pt`；
- 添加语义开/关训练轨迹一致性测试。

### WP2：观测 CUDA 核

- 在 `third_party/src/diff-gaussian-rasterization/` 增加无梯度 observation forward；
- 复用 `alpha*T` 和可见性/深度排序，输出 \(e\)、\(m^{all}\)、\(m^{valid}\)；
- 与参考实现逐项对齐，再做 tile 与原子规约优化；
- 不修改 RGB loss 或 backward。

### WP3：在线 orchestrator

- 在 `tools/train_progressive.py` 的初始训练结束、每个 progressive stage 结束和 final refinement 结束处加入显式 semantic hook；
- 实现 acquisition-order 重排、重叠分组、开放组替换、账本幂等和异步落后一阶段；
- 在线维护 G0 与近似 G1，禁止历史影像重复投影。

### WP4：Semantic TDOM renderer

- 扩展 rasterizer 以支持小型动态/模板化 \(K\) 通道；
- 复用当前 RGB TDOM 相机与输出配置；
- 输出 class/color/unknown/support/entropy，概率栅格可选；
- 做像素级概率归一化与 unknown gate。

### WP5：最终精确重建与基准

- 最终几何冻结后从空先验全量重投影；
- 实现 G0 + leave-one-group-out G1 两遍计算；
- 对比在线近似与最终精确结果；
- 在单张 4090 上报告 mapping latency、显存和分项耗时。

实施顺序为 `WP0 → WP1 → WP2 → WP3 → WP4 → WP5`。WP0/WP1 先建立语义隔离与可验证基线，避免一开始把正确性问题埋进 CUDA。

## 验收与测试

在没有人工真值的第一阶段，验收针对系统不变量而不是语义精度：

1. **几何隔离**：同种子下开启/关闭语义后，RGB checkpoint、densification/prune 决策和 RGB TDOM 在既定数值容差内一致。
2. **幂等性**：相同 `image_id + label_version` 重放不改变状态；开放组更新是替换，不是重复相加。
3. **证据守恒**：clone/split 前后数据证据总量不凭空增长。
4. **顺序稳定性**：组内影像顺序变化不改变闭合组结果；final exact 在固定输入下确定。
5. **unknown 语义**：`255` 不进入任何真实类别；无支持、低覆盖和强混合像素输出 `255`。
6. **遮挡与足迹**：合成前后景、跨边界大高斯与小目标用例和参考公式一致。
7. **恢复一致性**：中断恢复结果与连续运行一致；版本/hash 不一致会显式失败。
8. **阶段对齐**：RGB/Semantic TDOM 的相机、尺寸、地理范围和 manifest 可逐像素配准。
9. **资源约束**：原始分辨率下峰值显存适配单张 24 GB 4090，在线摊销满足阶段预算。

即便没有 GT，也应保存以下诊断：覆盖率、unknown 比例、每类面积、跨视角 JS 分布、Dirichlet 总证据分布、边界区域熵、在线—最终差异和每阶段变化率。这些不能证明准确，但能发现重复计数、类别坍塌和异常跳变。

## 主要风险与控制

| 风险 | 表现 | V1 控制 |
|---|---|---|
| SAM3 系统性错误 | 多视角一致但整体错误 | 不把分数当证据量；保留 label 版本；未来用小型校准集验证 |
| 相邻航片过度计数 | 后验异常尖锐 | overlap 分组；每组每高斯最多一个样本；支持与样本数分离 |
| 大高斯跨类别边界 | TDOM 边界污染 | 完整足迹、轻量边界降权、软分布、像素级 unknown gate |
| 渐进几何漂移 | 老语义与新几何错位 | dirty 活动窗口修复；final 全量重建 |
| densification 复制置信度 | 无新观测却越来越确定 | clone/split 证据守恒 |
| 标签/类别版本混用 | 类别 ID 语义悄然变化 | class/schema/hash 强校验，账本以版本为键 |
| CUDA 动态通道开销 | sync 超时或爆显存 | 小 \(K\) 模板化、分块处理、分项 profiling |
| 无真值过早调参 | 指标好看但不可解释 | 当前只验系统不变量；阈值配置化且不宣称最优 |

## V1 明确不做

- 不让语义 loss、mask、边界或类别影响位置、尺度、旋转、opacity、SH、相机、densification 或 prune；
- 不学习高维语义/语言 embedding，不做开放词汇查询；
- 不做语义引导 Gaussian split 或 boundary reconstruction；
- 不把车辆当动态目标；当前数据中的车辆按普通静态类别处理；
- 不在阶段目录重复保存历史 labels/scores；
- 不默认半分辨率；
- 不在没有人工真值时宣称 SAM 模型、阈值或总体路线已经达到最优精度；
- 不把每高斯 sidecar 当成最终用户产品；
- 不要求在线路径每阶段精确重放全部历史观测。

## 对现有设计文档的修订结论

现有 [semantic-tdom-v1.md](../design/semantic-tdom-v1.md) 的“Dirichlet + 完整足迹 + JS 鲁棒融合 + 不确定性 TDOM”主干成立，但实施前应做以下实质修订：

| 原始倾向 | 采用后的定义 |
|---|---|
| 固定八类实现 | V1 闭集，但 \(K\)、类别 ID、颜色和 background 全部配置化 |
| 标签作为主要输入 | 标准契约同时支持硬标签与完整 scores；scores 未校准时不充当证据强度 |
| 全历史精确 G1 | 在线活动窗口近似 G1；最终几何上全量两遍精确 G1 |
| 相关帧分组 | 锚点 overlap `0.4`、最多 4 帧、开放组暂定贡献可替换 |
| densification 跟随数组变化 | 显式拓扑事件；clone/split 保持概率均值并守恒分配证据 |
| 每阶段语义目录 | 数据集根目录全局保存一次，阶段以 ledger 引用 |
| 语义输出 | 与 RGB TDOM 同相机/范围/GSD/节奏的附属产品 |
| 边界高斯硬取高置信类 | 高斯保留软分布，TDOM 像素混合后用支持与 margin 决定类别或 `255` |
| 未明确默认分辨率 | 原始 label 分辨率默认；半分辨率仅 profiling/ablation |
| 未明确初始化边界 | 前 30 张初始化，第 31 张开始 progressive |

因此，最合适的路线不是复制某篇 Semantic GS 的完整方法，而是以 CSS 的闭式概率更新为数学核心，吸收 Semantic Gaussians 的无训练投影、ReSemGS 的多帧一致性和在线 Semantic SLAM 的关键帧/活动窗口思想，同时明确拒绝会反向改变几何的联合优化与语义引导 densification。

## 参考文献

[^1]: Wilson et al., [Modeling Uncertainty in 3D Gaussian Splatting through Continuous Semantic Splatting](https://arxiv.org/abs/2411.02547), ICRA 2025.
[^2]: Guo et al., [Semantic Gaussians: Open-Vocabulary Scene Understanding with 3D Gaussian Splatting](https://arxiv.org/abs/2403.15624), 2024.
[^3]: Zhou et al., [Feature 3DGS: Supercharging 3D Gaussian Splatting to Enable Distilled Feature Fields](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Feature_3DGS_Supercharging_3D_Gaussian_Splatting_to_Enable_Distilled_Feature_CVPR_2024_paper.pdf), CVPR 2024.
[^4]: Cen et al., [SAGA: Segment Any 3D Gaussians](https://arxiv.org/abs/2312.00860), 2023.
[^5]: Ye et al., [Gaussian Grouping: Segment and Edit Anything in 3D Scenes](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/4195_ECCV_2024_paper.php), ECCV 2024.
[^6]: Matsuki et al., [SGS-SLAM: Semantic Gaussian Splatting for Neural Dense SLAM](https://arxiv.org/abs/2402.03246), 2024.
[^7]: Ji et al., [SemGauss-SLAM: Dense Semantic Gaussian Splatting SLAM](https://arxiv.org/abs/2403.07494), 2024.
[^8]: Jiang et al., [GS3LAM: Gaussian Semantic Splatting for SLAM](https://lif314.github.io/projects/gs3lam/), 2024.
[^9]: Yan et al., [GSFF-SLAM: Gaussian Splatting Feature Field SLAM](https://arxiv.org/abs/2504.19409), 2025.
[^10]: Zhao et al., [ReSemGS-SLAM: Real-time semantic Gaussian Splatting SLAM based on semantic consistency perception](https://www.sciencedirect.com/science/article/pii/S0950705126007811), Knowledge-Based Systems 2026.
[^11]: Yang et al., [NRGS: Neural Regularization for Robust 3D Semantic Gaussian Splatting](https://arxiv.org/abs/2604.22439), 2026.
[^12]: He et al., [NG-GS: NeRF-guided 3D Gaussian Splatting Segmentation](https://openaccess.thecvf.com/content/CVPR2026/html/He_NG-GS_NeRF-guided_3D_Gaussian_Splatting_Segmentation_CVPR_2026_paper.html), CVPR 2026.
[^13]: Zhang et al., [COB-GS: Clear Object Boundaries in 3DGS Segmentation Based on Boundary-Adaptive Gaussian Splitting](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_COB-GS_Clear_Object_Boundaries_in_3DGS_Segmentation_Based_on_Boundary-Adaptive_CVPR_2025_paper.html), CVPR 2025.
[^14]: Shen et al., [Trace3D: Consistent Segmentation Lifting via Gaussian Instance Tracing](https://openaccess.thecvf.com/content/ICCV2025/html/Shen_Trace3D_Consistent_Segmentation_Lifting_via_Gaussian_Instance_Tracing_ICCV_2025_paper.html), ICCV 2025.
[^15]: Gao et al., [Evolving Semantic Propagation for Aerial Semantic 3D Gaussian Splatting](https://ojs.aaai.org/index.php/AAAI/article/view/42418), AAAI 2026.
[^16]: [VISTA: Open-Vocabulary Online Semantic Mapping for Vision-Based Autonomous Exploration](https://arxiv.org/abs/2507.01125), 2025.
