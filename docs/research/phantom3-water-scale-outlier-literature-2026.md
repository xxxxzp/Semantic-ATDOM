# Phantom3 水体尺度异常：文献与场景适配结论（2026）

## 研究问题与结论

本文针对 Phantom3 渐进式 3DGS 中水面附近异常大、高度各向异性 Gaussian 的形成与治理，交叉检查近年一手论文、当前训练代码及真实运行数据。目标不是追求通用新视角合成榜单，而是在保持严格在线因果、语义不反向控制几何、单卡资源和现有发布安全边界的条件下，选择可证伪、可落地的修复路径。

结论如下：

1. **联合邻域初始化必要，但不可能单独根治。** 当前新增点只在当前批内计算 3-NN 尺度，违反表示尺度应由完整已知局部支撑决定的原则；但最终异常尺度与最终局部中心密度严重脱钩，说明训练期增长和未触发 split 同样关键。
2. **第一种生命周期操作应是 split，不是 hard clamp 或无差别 prune。** 异常点通常不是无贡献垃圾，而是高支持、高覆盖的极端长轴 Gaussian；删除或硬缩会直接制造 RGB/语义覆盖缺口。
3. **触发信号必须组合世界尺度与屏幕贡献。** Pixel-GS、AbsGS、Revising Densification 和 Taming 3DGS 从像素覆盖、梯度抵消、逐像素误差归因和贡献评分四个角度说明：仅用平均位置梯度会让大 Gaussian 冻结。
4. **水面非朗伯性是放大器，不是首轮应专门建模的根因。** 鲁棒光度损失可作为后续类无关增强，但在缺少独立几何监督时，过强降权可能把整片水的训练信号一起删掉。
5. **首轮不采用单目深度、水面专用网络或反射/折射双场。** 本地没有已验证对齐的深度/LiDAR；水体深度预测本身不可靠，而专用水面模型会大幅改变表示、训练时延与产品边界。

## 本场景的关键证据

- 最终 677,723 个 Gaussian 中，4,848 个最大轴尺度超过 1.949 m；最大值 1188.62 m。
- 4,848 个异常点中，4,733 个在最终中心集上的局部 3-NN 尺度不超过 1.949 m；`stored_scale/local_3NN` 中位数 18.35，最大约 9986。
- 160 个 water 异常点中 157 个轴比不小于 4，`max/min` 轴比中位数约 86.3；它们的语义观测组和支持度中位数分别约 42 和 20,227，高于全部 water Gaussian 的 25 和 701。
- 建议的 59 个水域帧中，water 的有效 SfM 观测密度约为其他主要地物的十分之一；其灰度 Sobel 纹理约为非背景加权均值的 21.4%。这验证了“弱纹理 + 稀疏几何支撑”，但不能单独证明异常出生阶段。
- 三次全量结果的极端尾不稳定，而主体分位数接近，说明极端点受随机优化和生命周期影响；现有最终 PLY 没有 birth identity，不能从终态倒推出形成时刻。

## 文献机制与可迁移性

### 原始 3DGS：基线机制而非可直接复用答案

[3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) 使用 SfM 点云的邻域距离初始化尺度，并以位置梯度驱动 clone/split、以 opacity 和可选的屏幕/世界尺寸执行 prune。当前项目保留了初始化期 ADC 的一部分，却在 progressive 与 final refinement 中关闭常规 densification，且初始化 2000 步短于尺寸 prune 的 3000 步门槛。因此当前行为不能被视作原论文生命周期的等价实现。

**可迁移部分：** 联合已知点集定义局部尺度、split 大原语、透明点 prune。  
**风险：** 原始平均梯度准则对大覆盖 Gaussian 会失效；直接恢复原逻辑仍可能漏掉水面大原语。

### Pixel-GS（ECCV 2024）：把像素覆盖纳入增长条件

[Pixel-GS](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf) 指出，初始化稀疏区域的大 Gaussian 往往可被许多视角看到，但平均梯度会稀释其增长信号；方法用覆盖像素数加权梯度，并按相机距离缩放以抑制近相机 floater。

**本场景适配：高。** water 区域初始化稀疏，异常 Gaussian 又具有高观测支持。累计屏幕覆盖与贡献应进入 split 触发。  
**风险：** 仅按像素数可能偏爱大面积低误差水面；必须同时要求局部尺度/轴比异常及多视角贡献。

### AbsGS（ACM MM 2024）：梯度碰撞使大 Gaussian 不 split

[AbsGS](https://arxiv.org/abs/2404.10484) 将过度重建归因于不同像素位置梯度的方向抵消，并用同向绝对梯度识别应拆分的大 Gaussian。

**本场景适配：高。** 当前异常多为极端长轴点，而现有 split 依赖平均位置梯度；这与“有高覆盖但不触发 split”的机制直接对应。  
**风险：** 需要 rasterizer 暴露更细粒度梯度统计；首轮可先用现有可得的屏幕半径、贡献、残差和轴比建立近似触发，再决定是否改 CUDA。

### Revising Densification（ECCV 2024）：按逐像素误差贡献归因

[Revising Densification in Gaussian Splatting](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08041.pdf) 把逐像素误差按各 Gaussian 的渲染贡献反传到密度控制决策，并修正 clone 时 opacity 复制导致的合成偏差，同时控制总原语数。

**本场景适配：高。** 它比“scale 超阈值即 split”更接近第一性原理：只有既异常又确实承担难解释像素误差/贡献的原语才应被细化。  
**风险：** 额外的 per-pixel 归因有计算与实现成本，应先测是否超过每 stage 10% 预算。

### Taming 3DGS（SIGGRAPH Asia 2024）：多信号贡献评分与预算

[Taming 3DGS](https://humansensinglab.github.io/taming-3dgs/) 用梯度、显著性、覆盖像素、到投影中心的距离、混合权重、深度、opacity 和 scale 等多信号评分，并按可预测增长曲线控制 Gaussian 数量。

**本场景适配：中高。** 支持用“局部尺度比 + 轴比 + 屏幕覆盖 + alpha 贡献 + 多视角次数”联合触发，而非单阈值。  
**风险：** 完整移植过重；应只借鉴评分变量和固定预算。

### RAIN-GS（2024）：大尺度原语并非天然错误

[RAIN-GS](https://arxiv.org/abs/2403.09413) 通过稀疏大 Gaussian 初始化和渐进低频到高频优化改善随机初始化收敛。它提醒我们，大尺度 Gaussian 可承担早期低频建模功能。

**本场景含义：** 不应对所有大尺度做全局硬上限；治理对象应是相对局部邻域失配、长期高轴比且已经获得足够观测的冻结原语。

### 2DGS（SIGGRAPH 2024）：薄片形状本身可能合法

[2D Gaussian Splatting](https://surfsplatting.github.io/) 用二维 surfel 表示表面并通过几何正则提升一致性。它说明大的 `max/min` 轴比并不能单独证明错误：表面原语天然可以很薄。

**本场景含义：** 不能只用 `max/min` 轴比 prune；更应关注切向最大轴相对局部支撑的失配、屏幕覆盖和多视角行为。当前 water 异常点 `max/mid` 也明显偏大，因此更像针/长条退化，而非普通薄片。

### GaussianPro（ICML 2024）：纹理缺失区域需要几何传播

[GaussianPro](https://arxiv.org/abs/2402.14650) 针对大场景纹理缺失导致的 SfM 初始化不足，使用已有几何和 patch matching 渐进传播新 Gaussian。

**本场景适配：中。** 它准确命中“水域 SfM 极稀疏”问题，但水面反射和波纹会破坏 patch matching，且工程改造大。适合作为 E1–E3 失败后的补点路线，不是首选。

### 3DGS-MCMC（NeurIPS 2024）：从启发式 ADC 转向受控再定位

[3D Gaussian Splatting as Markov Chain Monte Carlo](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4f3b30f3b6cd90b7867a0d3b2c9972ff-Abstract-Conference.html) 将 Gaussian 优化重述为 MCMC，并以 relocate 处理低贡献原语，在固定原语预算下改善表示。

**本场景适配：中低。** 固定预算和贡献感知值得借鉴，但替换优化范式超出首轮最小修复范围。

### RobustNeRF 与 WildGaussians：非静态外观的鲁棒估计

[RobustNeRF](https://arxiv.org/abs/2302.00833) 把跨视图不一致的干扰物作为鲁棒估计中的离群观测；[WildGaussians](https://arxiv.org/abs/2407.08447) 将外观变化与不确定性建模引入 3DGS。

**本场景适配：中低。** 水面反射、波纹和曝光变化确会产生不一致梯度，类无关的残差/不确定性权重可降低其把 geometry 推向大尺度的动力。  
**风险：** 水面在大量像素上系统性不一致，模型可能把整个有效区域降权。必须设置权重下限，并以覆盖不退化为硬验收。

### GaussianShader、Ref-Gaussian 与水面专用模型

[GaussianShader](https://openaccess.thecvf.com/content/CVPR2024/html/Jiang_GaussianShader_3D_Gaussian_Splatting_with_Shading_Functions_for_Reflective_Surfaces_CVPR_2024_paper.html) 与 [Reflective Gaussian Splatting](https://arxiv.org/abs/2412.19282) 显式建模反射材质；[WSGS](https://arxiv.org/abs/2503.19481) 面向平静水面建模反射/折射及双组 Gaussian。

**本场景适配：首轮低。** 这些方法改变渲染方程或表示，通常要求可靠法向、平静平面水假设或额外 mask，并显著增加训练资源。它们解决的是逼真水面外观，不是当前通用尺度生命周期缺失；只有 E1–E3 和鲁棒损失均失败且产品确实要求水面新视角逼真度时才应考虑。

## 推荐的可证伪路线

### P0：先补生命周期遥测

为每个 Gaussian 保留稳定 identity、出生 stage、出生 3-NN 尺度、各轴尺度、每阶段优化后尺度、累计可见次数、屏幕半径/覆盖、alpha 贡献、opacity 及 split/prune 原因。必须单独记录 final refinement 前后。

要回答的首个问题是：最终异常点中多少是“出生即异常”、多少在在线 200 步中增长、多少在 final 4000 步中增长。没有这个分解，不应选择正则强度或 split 频率。

### P1：联合鲁棒 kNN 初始化

- 在当前新点和因果可见的已有可信中心联合参考集上计算至少 3-NN；
- 去重、排除 self，显式处理 0–3 个参考点和 `inf`；
- 对明显 SfM 空间孤立点结合 track length 与重投影误差做门控；仅重投影误差不足以识别孤立点；
- 测试只要求同一 stage 内实现分块不改变结果，不要求早到点使用未来点。

### P1：恢复贡献感知的 progressive/final split 生命周期

对满足下列组合条件并持续多个观测窗口的原语优先 split：

- 最大轴相对局部 robust 尺度异常；
- `max/mid` 轴比异常，排除普通薄 surfel；
- 多视角累计屏幕覆盖或 alpha 贡献高；
- opacity 不低，且确实承担像素误差或覆盖。

只有低 opacity、低可见、低贡献且局部失配的点才 prune。split 时需保持密度/opacity 近似守恒并验证瞬时渲染差异。现有 `max_radii2D` 在每次 append 后被全量清零，必须先新增不会丢失的跨 stage 生命周期统计。

### P2：必要时接入 AbsGS/逐像素误差信号

若 P1 的代理信号仍漏掉高贡献冻结点，再接同向绝对梯度或逐像素误差贡献。两者分别检验“梯度抵消”和“低误差大覆盖是否无需拆分”，不应同时首次启用。

### P2：软尺度正则是后备，不是默认第二步

正则只惩罚超过 detached 局部参考尺度的 log 比值及异常 `max/mid` 长轴；不使用全局固定米制上限，不惩罚合法薄片，不对所有点每步重算邻域。由于缩小会立即丢覆盖，先与 split 单独消融，再决定是否组合。

### P3：类无关鲁棒光度权重

仅在生命周期治理后仍观察到水面持续放大时，引入带下限的跨视角残差/不确定性权重。它不能读取 water 标签，也不能让水区有效权重整体接近零。

## 实验设计要求

1. 先从 stage 14 跑一次只增加遥测的完整基线，在 188、228、273、325、375 保存 Gaussian、optimizer、累计相机与 RNG 的完整 checkpoint。
2. 五个水域窗口必须分别从对应 checkpoint replay；不能拼成一个 59 相机伪序列。
3. 每个实验只改变一个机制：E0 遥测、E1 联合 kNN、E1-LR 新生尺度短时冻结诊断、E2 生命周期 split、E3 软正则、E4 鲁棒光度。
4. 先用单 checkpoint/单连续窗口红绿验证，再扩到五窗；只将胜出方案完整跑到 stage 414。
5. 除 RGB PSNR/SSIM/LPIPS 外，必须评价主孔洞、边界连通 unknown、局部尺度比/轴比、split 后瞬时渲染差、Gaussian 数、每 stage 延迟和显存。

## 推荐优先级

| 优先级 | 方案 | 原因 |
|---|---|---|
| P0 | identity + 生命周期遥测 + checkpoint replay | 关闭当前最关键的因果缺口 |
| P1 | 联合鲁棒 3-NN 初始化 | 修复分批依赖和重复/小批边界 |
| P1 | 多视角贡献感知的 split，低贡献才 prune | 最贴合真实异常点的高支持、长轴特征 |
| P2 | AbsGS 或逐像素误差触发 | 处理平均梯度抵消/冻结原语 |
| P2 | 局部软尺度与长轴正则 | 防止再增长，但必须避免覆盖损失 |
| P3 | 类无关鲁棒光度权重 | 处理水面非朗伯性，风险是丢失全部水监督 |
| P4 | 深度、反射材质或水面专用表示 | 依赖额外先验且改变系统边界 |

这一路线保留现有 ratio=14 发布过滤和平面多视角 fallback，直到至少一条完整 14–414 训练在关闭 fallback 时通过孔洞、外部 unknown、RGB 与性能验收。
