# Semantic A-TDOM 全局认知

## 文档状态

- 状态：G1 Evidence V2 主链路已实现，有历史全量成功运行；2026-10-09 审查确认仍有恢复、进程收尾、几何采样和产品声明缺口。
- 核对日期：2026-10-09；当前代码基线 `0665f4b`。
- 当前开发分支：`semantic/g1-evidence-v2`。
- 适用范围：Semantic A-TDOM 的仓库治理、数据、环境、训练期语义主链路和产品状态。

本文是开发者和代码代理开始工作前的项目入口。它保存跨实验仍应成立的共同认知，不复制完整数学合同，也不把源码中的在建行为描述成已完成能力。

关键词：

- **MUST**：必须满足；违反时显式失败。
- **FORBIDDEN**：禁止，不能通过降级路径绕过。
- **CURRENT DECISION**：当前确认的设计决定。
- **DEFAULT**：可由显式配置覆盖，实际值必须进入 manifest。
- **CURRENT STATUS**：截至文档日期观察到的事实，不代表目标状态。
- **OPEN**：尚未通过实现、真实运行或实验确定。

## 权威边界

各类事实只有一个主要权威来源：

| 内容 | 权威来源 |
|---|---|
| Semantic TDOM 数学、状态和产品合同 | [Semantic TDOM V1：G1 Evidence V2](../design/semantic-tdom-v1.md) |
| G1 V1、V1.1 与 V2 的设计演进 | [G1 V1→V2 差异总结](../design/g1-v1-v2-comparison.md) |
| 决策原因与取舍 | `docs/decisions/` 中的 ADR |
| Phantom3 label/metadata 的文件 schema 与类别 ID | [Phantom3 语义标签交付规范](../design/phantom3-semantic-label-contract-v1.md) |
| 类别本体 | 当前数据版本的 `semantic/class_spec.json` |
| 单次运行参数 | 保存的实验配置和显式 CLI |
| 当前本机数据与实验进展 | [当前数据与实验核对](current-data-experiment-status.md)及其中引用的 run/预检/产品记录 |
| 当前代码与合同的偏差 | [第一性原理审查](../problem/first-principles-code-review-2026-10-09.md)，R01–R09 均待处理 |
| 环境依赖 | 正式验证后的 `environment.yml` 与 fingerprint |
| 标签来源 | 每影像 metadata |
| 产品身份、对齐和质量状态 | 产品及 pair manifest |

若旧研究文档、历史源码默认或注释与 G1 Evidence V2 合同冲突，以当前合同为准，并将旧内容标为 superseded，而不是维持两套“已确认”定义。

设计合同描述必须满足的要求，不能单凭合同推断实现已经满足。实验结论绑定特定 run、
输入版本和代码状态；dirty worktree 的成功不能外推为当前干净提交的验收。
Phantom3 交付规范中的 `phantom3_test100`、102 张示例，以及
[399 张实验](phantom3-semantic-atom-v2-result.md)均为历史记录；当前交付与进展以
[数据与实验核对](current-data-experiment-status.md)为准。

## 项目与代码边界

### CURRENT DECISION：研究范围

- Semantic A-TDOM 只以原始 A-TDOM 为代码基础。
- `ref/A-TDOM` 是只读参考基线。
- A-TDOM_Online 只说明此前研究课题，**FORBIDDEN** 将其作为算法、架构、实现思路或复制代码的来源。
- G（逐高斯语义场）是研究主线。
- D（直接分割 RGB TDOM）和 M（二维标签直接融合到地图网格）只作为未来对比/消融。
- G3 学习式残差是低优先级扩展，不能成为当前权威产品的必要依赖。
- 当前产品是闭集地物类别的俯视 RGB/Semantic TDOM；任意透视视角语义只作内部诊断。

### MUST：几何防火墙

几何影响语义，语义绝不反向影响几何：

- 语义只能读取 detached 的高斯几何、可见性和相机信息；
- 不参与反向传播或 RGB optimizer；
- 不改变位置、尺度、旋转、opacity、SH、loss、densify、clone、split 或 prune；
- 标签、语义 worker 或 Semantic TDOM 失败不能阻断、回滚或改变几何训练。

### CURRENT DECISION：仓库治理

- `main` 保存历史/重构基线，**FORBIDDEN** 直接进行当前语义开发。
- G1 Prior V1 已退役，不再保留开发分支或 worktree；历史代码由
  `g1-prior-v1-baseline` 和 `g1-prior-v1-stable-eb57e4d` 标签冻结，仅用于复现退化对照。
- `semantic/g1-evidence-v1.1` 由 `g1-evidence-v1.1-baseline` 标签冻结，作为当前最小算法
  baseline；它只用于复现和对照，不承接新功能。
- `semantic/g1-evidence-v2` 是唯一活跃的 G 路线开发与端到端产品主线。
- 重大算法、数据合同、环境大版本或产品 schema 变更使用独立分支与 ADR。
- `upstream` 保持只读，`origin` 用于个人仓库推送。
- MUST 识别并保留工作区中不属于当前任务的修改。

## 本机数据与首个任务

### 规范本地路径

```text
Phantom3 场景根：/data3/xzp/data/Onthefly/phantom3
Phantom3 渐进输入：/data3/xzp/data/Onthefly/phantom3/GaussianSplatting
Phantom3 语义交付父目录：/data3/xzp/data/Onthefly/phantom3/semantic_labels
Phantom3 实际语义目录：/data3/xzp/data/Onthefly/phantom3/semantic_labels/semantic

AMtown01 场景根：/data3/xzp/data/UAVScenes_Onthefly/interval5_AMtown01
AMtown01 语义交付父目录：/data3/xzp/data/UAVScenes_Onthefly/interval5_AMtown01/outputs/semantic_inputs/uavscenes_amtown01_observed_v1
```

这些是本机已存在的路径，不是算法默认值。启用语义时必须显式提供 non-empty
semantic data root，即包含 `semantic/` 的父目录。历史文档中不含 `phantom3/` 的
旧语义路径不能直接用于当前交付。

### HISTORICAL STATUS：早期验证切片

- 初始化 stage 30。
- stage 30 到 31 smoke 已通过。
- 2026-09-14 的历史运行采用 `StartFromImageNo=30`、`EndAtImageNo=400`，完成最终集合399张；stage 编号不是已消费影像数。
- “前100张”指 stage 100 注册集合，不指任意目录排序的前100个文件。
- stage 30 到 100 是早期验证目标，已被当时的完整399张实验覆盖。

### CURRENT STATUS：当前数据事实

- Phantom3 当前为 stage 14–414，共 401 个目录；最近全量预检/最终相机清单包含
  414 个身份，交付 414 份 label/metadata，尺寸记录为 1870×1049，没有 scores。
- AMtown01 当前为 stage 13–487，共 475 个目录；交付 490 份 label/metadata，
  尺寸记录为 2448×2048；所选全量阶段的最终有效相机数为 487，不能写成 490。
- 两个当前交付的 metadata 分别有 414/490 份 source hash；相应历史成功运行预检
  记录源绑定完整。本次盘点未重新执行全部输入内容哈希验证。
- Phantom3 是八类、AMtown01 是十五类，类别名称/ID 从各自 class spec 读取。
- 399 张、1600×897、缺少 `000384..000386` 和 6038 个 background 像素等结论
  只描述旧数据版本，不用于当前分布；当前各类别准确率尚未独立验证。

## 语义数据合同摘要

### MUST：身份与目录

```text
semantic/
├── class_spec.json
├── labels/<image-id>.png
├── metadata/<image-id>.json
└── scores/<image-id>.npy        # 可选
```

- RGB、camera、label 和 metadata 共享规范化 image ID；Phantom3 为六位编号，
  AMtown01 为小数时间戳。`normalize_image_identity()` 只剥离已知图像后缀，保留小数。
- 当前消费顺序按数值身份排序；overlap 名称表用于矩阵寻址，不等于采集顺序。
- 真实采集顺序若不同，必须新增显式 manifest，不能静默改排序。
- label 必须与运行时 RGB/camera tensor 严格同尺寸、同方向、同坐标。
- 运行时禁止隐式 resize/rotate/flip label；启用语义须使用 `--resolution 1`。
  对有畸变/主点偏移的相机，当前代码显式使用同一标定映射整流 RGB 和标签，
  分别采用线性/最近邻插值；这与任意改变标签像素域不同。
- stage 内遗留三分类 `labels/` 不是权威输入，禁止消费。

### MUST：类别与 ignore

下表是 Phantom3 当前交付的类别顺序；AMtown01 的十五类见其 class spec，不能复用此表：

| ID | 名称 |
|---:|---|
| 0 | `background` |
| 1 | `bare_soil` |
| 2 | `building` |
| 3 | `vegetation` |
| 4 | `paved_area` |
| 5 | `vehicle` |
| 6 | `water` |
| 7 | `road` |
| 255 | ignore |

- background 是上游明确产生时才成立的真实类别证据。
- ignore、缺失、低置信或无效像素不得解释为 background。
- 是否长期保留 background 是待核查 ProC-SAM3 语义后的 OPEN 项；当前不破坏既有标签版本。

### MUST：来源、版本和缺口

- metadata 必须绑定 label/version/model/checkpoint/prompt/threshold/尺寸/ontology/label hash。
- 新生成 metadata 必须包含 source image 和 SHA-256。
- 早期 102/399 张交付有 source binding 缺口，这是历史事实；当前两份交付和
  相应运行预检已记录完整绑定，不能继续笼统描述为“当前标签来源未验证”。
- 正式 `exact_final` 要求 source hash 与当前 RGB 一致，并验证方向/坐标约定。
- 仅补齐正确 source hash且像素/生成配置不变时，可以保留 label version。
- 标签、本体、模型或生成配置变化时必须发布新版本。
- 在线遇到首个缺口后须维持连续 readiness watermark；最终重建应利用缺口后的
  有效标签。但默认训练 preflight 会先要求所选集合标签完整，独立 partial API
  对损坏 metadata 的隔离仍有 R06 缺口。
- 缺失不得跳过后当作 background。

### MUST：原始数据只读

RGB、SfM 和 semantic roots 必须只读。当前 progressive 训练的派生 PLY/cache 位于
`<run>/.work/cache/<dataset-id>/`，预检副本位于 `provenance/dataset_preflight.json`；
成功后工作缓存清理。独立 `tools/render.py` 仍可能写 source PLY（R03），不能声明
所有入口都已满足只读要求。

运行前的只读 preflight 必须核验 stage/camera/RGB/overlap 集合、矩阵、累积性、label/schema/metadata/hash 和显式 root，并报告类别分布、ignore、overlap 分组、source provenance 与后期异常。报告 hash 进入训练和最终产品 manifest。

## 环境合同

- `ATDOM_XZP` 只是环境迁移来源，不是正式运行环境。
- 已验证组合为 Python 3.10、PyTorch 2.4.0+cu121、CUDA 12.1、RTX 4090。
- 正式且唯一受支持的 Conda 环境名为 `semantic-atdom`；
  `semantic-atom` 不是环境名。历史文件名和输出路径中的
  `semantic_atom` 只是早期产物 slug，不能据此推导 Conda 环境。
- 不能复制历史 editable package 或 CUDA 二进制；本地扩展必须从当前 `third_party/src` 构建。
- `environment.yml` 在验证后成为可执行真源；重大依赖升级使用独立分支。

环境通过以下检查后从 candidate 晋级：

1. 从 manifest 新建环境；
2. editable install 与所有扩展均指向当前仓库；
3. 真实 import 和 semantic CUDA symbols 验证；
4. CPU/schema 测试；
5. CUDA observation 和 Semantic TDOM 最小测试；
6. Phantom3 stage 30 到 31 smoke；
7. 保存 Python/Torch/CUDA/compiler/GPU/package source/git fingerprint。

### CURRENT STATUS

- 正式 `semantic-atdom` 已从当前 `environment.yml` 建立；editable package 和全部本地扩展指向当前仓库源码。
- 真实 import、semantic CUDA symbols、CPU/schema/CUDA 测试和 stage 30→31 smoke 均已通过。
- 当前审查环境的完整测试为 245 项通过，包括 CUDA；143 项是 2026-09-14 的历史计数。
- 当前 output schema v2 的环境记录为 `provenance/environment_start.json` 和
  `provenance/environment_end.json`；旧 fingerprint 文件名仅用于历史结果。

## G1 Evidence V2 摘要

完整公式、边界条件和状态定义只见 [权威数学合同](../design/semantic-tdom-v1.md)。本文只记录不可丢失的原则：

- 状态是不可训练的 `Dirichlet-style evidence state`，不是校准的错误概率模型。
- 先验总浓度为1，只稳定空状态；正式类别质量、LOO 共识和 background 均不能来自先验。
- 正式类别分布从 `alpha - alpha0` 的非负真实 evidence 条件化得到。
- G0 保存 raw evidence；G1 使用排除当前组和先验的 leave-one-group-out JS。
- JS 至少需要两个其他闭合组和足够累计支持；只处理跨视角离群，不解决系统性 SAM 错误。
- 当前未校准默认包括 `S_min=1.0`、reliability floor 0.1、temperature 0.1。
- hard map 只由 geometry coverage、pixel evidence support 和 support ratio 三个 gate 决定。
- entropy、margin、mixture/group disagreement 和 variance proxy 分开输出，不参与 V1 gate。
- variance 是包含先验、基于独立近似的未校准 proxy，不是分类错误概率。
- Semantic TDOM 光栅化前默认排除最大尺度超过场景中位最大尺度14倍的异常高斯；
- 对明确配置的平面类别，最终精确发布可用有界连通域、局部平面和多观测组原始标签一致性恢复被尺度过滤误伤的内部孔洞；默认关闭，边界连通未知区永不回填，并单独发布类别来源图；
  这只是只读产品支持域清理，不是第四个 hard-map gate，也不影响 RGB 或几何训练。
- 上述尺度过滤只是 Phantom3 上的临时回归抑制：它会使 RGB 与 Semantic 使用不同的
  高斯子集，且不能定义通用 no-data footprint；后续须另行设计与 RGB 颜色、SAM 标签
  均无关的权威像素有效域。
- clone/split 的硬不变量是拓扑前后渲染语义场连续；后代复制父状态只是待数值测试的 V2 初始近似。
- online group disagreement 必须由至少两个有效闭合组支持。实现仍会计入开放组，
  且将可用在线诊断标为 `exact_rebuild`（R05）；不能把这一合同当作已完成的验证。

## 在线、最终与地图产品

- 在线状态随几何阶段更新，但历史投影不会天然变成精确；在线产品一律是有界近似。
- 活动窗口保存开放组和有限最近闭合组；manifest 记录证据 geometry version/staleness。
- 几何阶段与地图发布事件分开；latest-wins 可以用更新发布事件替换旧事件，但非发布 stage 不得取消尚未开始的 cadence 发布意图；标签观测独立累计。
- 一旦发布产品对，RGB 与 semantic 必须来自同一冻结 snapshot。
- final exact rebuilder 必须能在 online hook 已失败时独立从零启动。

产品状态为：

| 状态 | 资格 |
|---|---|
| `approx_online` | 实时预览 |
| `stale_online` | 诊断 |
| `online_fallback` | final 失败后的诊断 |
| `exact_partial` | 最终几何上的有质量声明结果 |
| `exact_final` | 唯一完整权威结果 |
| `failed` | 不发布类别产品 |

同一任务首次发布时固定局部 grid direction/origin/GSD/pixel convention，并将 `map_grid.json` 写入输出根；未知范围只通过固定 tile index 扩展。当前是无 GIS CRS 的本地固定网格，不能声称具有世界坐标系。

RGB/Semantic 产品使用共同 `projection_geometry_hash`，完整 RGB 另有 `rgb_model_hash`，语义另有由状态内容计算的不可变 `semantic_state_hash`。pair 不引用会被轮换覆盖的 sidecar slot；两侧 manifest 由最后原子发布的 `pair_manifest.json` 引用。

当前 CLI 默认只保存核心 hard/color/unknown/evidence-support/entropy/schema/manifest；
`semantic_tdom_save_diagnostics=False`，最终发布也不默认保存全部诊断栅格。
诊断和概率栅格需分别显式启用。合同的理想诊断集合与代码默认产物不得混用。

## 术语

- `projected_valid_mass[g]`：逐高斯累计的二维有效投影质量。
- `dirichlet_evidence[g,k]`：`alpha-alpha0` 的真实类别证据。
- `evidence_ratio[g]`：逐高斯真实证据比例。
- `geometry_coverage[p]`：TDOM 像素几何覆盖。
- `pixel_evidence_support[p]`：TDOM 像素绝对 evidence support。
- `pixel_support_ratio[p]`：evidence support 与几何覆盖之比。

旧的泛称 `semantic_support` 不再同时指逐高斯观测量和像素 evidence support。旧文件名最多保留一个 schema 版本的 deprecated alias。

类别身份拆分为 `ontology_hash` 与 `palette_hash`。颜色变化只重渲染 color 产品，不重建 evidence；完整 `class_spec_sha256` 继续用于兼容与审计。

## 准实时资源原则

- 正式硬件基线是单张 24 GB RTX 4090。
- 非发布 stage 的 semantic snapshot 只包含投影需要的几何与相机，不复制无关 SH；paired publication 的共同 snapshot 为渲染 RGB 必须额外包含 SH。
- 使用显式 CUDA stream/priority、ready event、snapshot 复制前显存预检、输出预算和时间预算；CLI 默认 priority 为 0，不能笼统声明默认是低优先级。
- 超预算时依次停止中间诊断写盘、跳过过时地图渲染并合并到最新 snapshot；不阻塞几何、不丢标签。
- sidecar checkpoint 只保存 `xyz/opacity/scale/rotation`，不复制 SH；其 `rgb_model_hash` 可以为空。只有 paired publication 的共同冻结快照包含 RGB appearance。
- 当前未校准 CLI 默认：`semantic_tdom_final_only=True`、语义 checkpoint 间隔 0
  （关闭过程恢复槽）、512 像素 tile、512 MiB headroom、8 GiB pair 上限、30 秒
  在线预算、stream priority 0。地图 cadence 参数为 1，但不表示默认逐阶段发布。
  无在线消费者时语义只在最终边界重建；训练视频会要求在线语义。final 模型仍归档。
  时间预算是调用边界的 best-effort gate，不是不可中断 kernel 的硬 deadline。
- final exact rebuild 单独统计，不计入在线 SLO。
- `lag<=1` 和具体耗时比例在真实 profile 前只是目标，不能写成已保证事实。
- 2026-09-14 的历史399张运行中，final exact rebuild 用时111.338秒，峰值 CUDA allocation 约3.15 GB；该数据来自可用显存高于单卡24 GB基线的机器，不替代24 GB目标硬件验证。

## 版本与实施状态

- 新方法版本：`g1-evidence-v2`。
- state/recovery 和 product/manifest schema 必须升级。
- 旧 sidecar 不允许静默 resume，必须从对应几何和标签重建。
- 旧产物保留时必须标为 superseded，不在原目录无提示覆盖。

截至 2026-10-09，区分已存在的实现、历史运行和未关闭缺口：

| 项目 | 当前状态 |
|---|---|
| G1 Evidence V2 主链路 | 已实现；存在历史全量训练、最终重建与 paired publication 成功记录 |
| 数学/产品合同 | 已确认，不能据此断言所有代码边界已满足 |
| CPU/CUDA 测试 | 当前审查 245 passed；R01–R09 仍待处理 |
| 数据与身份 | 当前 Phantom3 414、AMtown01 最终相机 487；输入与实验细节见核对记录 |
| 恢复与进程收尾 | checkpoint 接口存在；语义恢复 hash 不稳定（R01）、输出退出可能挂起（R02） |
| 只读输入与完整来源 | progressive cache 已实现；独立渲染仍写 source（R03），完整生成配置绑定有 R08 缺口 |
| 在线与独立 final API | 已实现；闭合组资格/诊断状态有 R05 缺口，无效标签隔离有 R06 缺口 |
| 几何采样与产品补洞 | 投影插值有 R07 问题，补洞可信尺度集合有 R09 偏差 |
| 固定网格/配对发布 | Phantom3 414 张和 AMtown01 487 张各有已复核的历史完整地图发布 |
| 当前 HEAD 全量验收 | 本次未重跑；历史 dirty run 不等于当前 `0665f4b` 的全量验收 |
| 独立语义精度/GIS/资源 SLO | 尚未建立独立语义真值评价、GIS CRS 或固定 24 GB/准实时保证 |

当前实验事实以 [当前数据与实验核对](current-data-experiment-status.md)为准。
Phantom3 的 `outputs/latest` 已指向 dirty 尺度生命周期实验分支的恢复运行，且没有
请求地图发布，不能将该指针当作当前主线 paired map 的验收。AMtown01 最新成功
运行有 160 帧视频；159 帧文档是计划参数。两者都不能用日期或目录 SHA 替代完整
实验身份。旧 [399 张实验](phantom3-semantic-atom-v2-result.md)保留原来的
`exact_partial` 结论，当前 414 张交付及历史成功运行则有完整 source binding。

已完成的历史实施任务见 [G1 Evidence V2 completed plan](../../plans/completed/2026-09-11-g1-evidence-v2.md)。
最新审查与修复验收要求见 [第一性原理问题记录](../problem/first-principles-code-review-2026-10-09.md)；
文档更新不会关闭任何代码问题。

## 开始工作前检查

1. 当前分支不是 `main`；
2. 已识别并保留既有 worktree 修改；
3. 当前任务属于 active plan 的哪一项已明确；
4. RGB root、semantic root、stage 范围和 label version 已明确；
5. 输入、cache 和 output 路径没有混淆；
6. 环境、editable package 和扩展实际来源已打印；
7. 所需 semantic symbols 真实存在；
8. 产品状态和未验证项将被诚实记录。

## 关联文档

- [G1 V1→V2 差异总结](../design/g1-v1-v2-comparison.md)
- [Semantic TDOM V1：G1 Evidence V2](../design/semantic-tdom-v1.md)
- [Evidence-conditioned 渲染 ADR](../decisions/0005-evidence-conditioned-semantic-rendering.md)
- [异常尺度语义支持域 ADR](../decisions/0007-bound-semantic-render-support.md)
- [有界在线产品 ADR](../decisions/0006-bounded-online-semantic-products.md)
- [几何防火墙 ADR](../decisions/0001-one-way-semantic-dependency.md)
- [阶段同步 ADR](../decisions/0003-stage-synchronous-semantics.md)
- [A-TDOM 谱系 ADR](../decisions/0004-clean-atdom-lineage.md)
- [Phantom3 标签交付规范](../design/phantom3-semantic-label-contract-v1.md)
- [推荐技术路线](../research/semantic-atdom-recommended-route.md)
- [2026 文献综述](../research/lightweight-semantic-3dgs-literature-review-2026.md)
- [Phantom3 Semantic A-TDOM V2 完整实验记录](phantom3-semantic-atom-v2-result.md)
