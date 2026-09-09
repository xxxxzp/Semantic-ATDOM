# Semantic TDOM V1：已确认设计

## 文档状态

本文记录截至 2026-09-08 已由讨论明确确认的 Semantic TDOM V1 设计。尚未确认的内容集中列在文末，不应在实现时被默认补全。

## 项目边界

- 开发仓库只以 `A-TDOM` 为代码基线。
- `A-TDOM_Online` 只用于说明课题背景和曾经开展的工作，不是算法思路、代码结构或实现细节的参考来源，也不得从中复制实现。
- 第一阶段使用类似 `/data3/xzp/data/Onthefly/phantom3` 的连续无人机下视影像，先取前 100 张构成小规模验证场景。
- 产品输出只要求俯视 RGB TDOM 与对应的 Semantic TDOM；任意视角语义渲染最多作为内部诊断能力。

## 目标与硬约束

1. 语义随 A-TDOM 渐进训练同步生长，而不是等全部几何训练完成后再统一后处理。
2. “离线语义”表示每张影像的标签已经预先生成，不表示语义构建发生在几何训练结束之后。
3. 语义只能读取当前高斯的几何、可见性和外观属性，绝不向位置、尺度、旋转、opacity、SH、densification 或 RGB 损失传播梯度。
4. 第一版语义更新不做反向训练，采用闭式概率更新；同一张标签不能因 RGB 训练被重复采样而被重复计为独立语义证据。
5. G（高斯语义场）是研究主线。直接分割最终 RGB TDOM 的 D 路线、直接将源标签镶嵌到地图网格的 M 路线，仅作为以后可能的比较方案。

## 数据契约

每张 RGB 影像必须存在同名的单通道语义标签 PNG。建议数据集结构为：

```text
dataset/
├── images/
│   ├── 000001.jpg
│   └── ...
├── semantic_labels/
│   ├── 000001.png
│   └── ...
└── semantic_scores/        # 可选，不是主链路依赖
```

类别在一次任务开始前固定，首个任务使用以下八分类：

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

`0` 只表示明确预测的 background。SAM 分数低于阈值、图像无效区或没有标签的像素必须写为 `255`，不得作为 background 证据。主链路必须只依赖硬标签 PNG；SAM score 数组可以作为以后附加的可靠性信息，但不能成为必需输入。

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

- 初始训练阶段结束；
- 每次 `Progressive_Train` 批次结束；
- 最终 refinement 结束。

语义原始证据只在同一几何版本内不可变。高斯拓扑或参数变化后，相关证据必须通过后续确定的生命周期规则失效或重算。

## 每高斯概率语义状态

第一版使用闭集 categorical–Dirichlet 语义场。每个高斯 `g` 保存八维 Dirichlet 浓度参数：

\[
\boldsymbol{\alpha}_g=[\alpha_{g0},\ldots,\alpha_{g7}]
\]

采用总浓度为 1 的对称弱先验：

\[
\alpha_{0,k}=\frac{1}{8}=0.125
\]

不使用从 SAM 标签频率估计出的类别先验，避免先验偏向 background 或面积较大的类别。

每高斯还需要维护或派生：

```text
dirichlet_alpha_raw[N, 8]
dirichlet_alpha_robust[N, 8]
group_variance[N, 8]
group_count[N]
semantic_support[N]
```

类别概率、类别 ID、Dirichlet 方差、熵和 margin 均从这些状态派生，而不是作为独立真值维护。

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
- 新帧始终与当前组锚点比较，而不是只与前一帧比较，避免传递式重叠令分组无限增长；
- 首轮归一化重叠阈值设为 `0.4`；
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

`r_hg` 是随 JS divergence 单调下降且具有正下限的软权重。不得硬删除冲突组。只有一个有效组时不做冲突降权；两个组冲突时保留双方并维持不确定性；三个以上观测组中，少数冲突组由其余组形成的共识降权。

两遍精确 G1 如何与每个渐进同步点高效结合，仍是待确认问题。

## Semantic TDOM 概率渲染

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
- 只有几何覆盖、绝对语义支持及语义支持比例同时达到阈值时才输出 `0..7`，否则输出 `255`。

渲染器需要输出以上连续结果。具体阈值尚未确认，应由后续配置生成标签图。

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

## 当前实现优先级

第一版只实现以下主链路：

```text
预生成硬标签
  → 与 A-TDOM 阶段同步
  → 完整足迹 alpha-T 观测
  → 相关帧分组
  → G0 Dirichlet
  → G1 leave-one-group-out 鲁棒后验
  → 概率 Semantic TDOM 与独立不确定性图层
```

以下内容只记录为以后可能的扩展，不阻塞第一版：

- G2：受 NG-GS 启发的边界歧义检测与局部连续语义场；
- G3：受 NRGS 启发、冻结高斯属性的轻量语义正则网络；
- D：直接分割最终 RGB TDOM；
- M：源标签直接融合到地图网格。

## 尚未确认

1. densification、clone、split、prune、新点扩张和高斯参数移动时，Dirichlet 状态的继承、守恒、失效与局部重算规则；
2. 渐进同步阶段使用快速近似 G1，还是保存足够信息执行局部精确两遍 G1；
3. 观测组在连续数据流中何时关闭；
4. JS 软权重函数的具体形式、温度和正下限；
5. `C_min`、`U_min` 和语义支持比例阈值的初值；
6. 标签边界距离权重的确切带宽和下限；
7. GPU 数据类型、持久化格式及 CUDA 核函数划分；
8. 第一批 100 张影像中初始阶段与渐进阶段的具体切分；
9. 运行时间和显存预算；
10. 评价数据、指标和正式消融设计。当前明确暂缓评测讨论。

## 论文依据

- Wilson et al., *Modeling Uncertainty in 3D Gaussian Splatting through Continuous Semantic Splatting*, ICRA 2025: <https://arxiv.org/abs/2411.02547>
- Zhao et al., *ReSemGS-SLAM: Real-time semantic Gaussian Splatting SLAM based on semantic consistency perception*, 2026: <https://www.sciencedirect.com/science/article/pii/S0950705126007811>
- He et al., *NG-GS: NeRF-Guided 3D Gaussian Splatting Segmentation*, CVPR 2026 Highlight: <https://arxiv.org/abs/2604.14706>
- Yang et al., *NRGS: Neural Regularization for Robust 3D Semantic Gaussian Splatting*, ICME 2026: <https://arxiv.org/abs/2604.22439>
- Gao et al., *Evolving Semantic Propagation for Aerial Semantic 3D Gaussian Splatting*, AAAI 2026: <https://ojs.aaai.org/index.php/AAAI/article/view/42418>
