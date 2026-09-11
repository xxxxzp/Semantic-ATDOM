# Semantic A-TDOM 全局认知

## 文档状态

- 状态：项目边界与 G1 Evidence V2 设计已确认；实现和端到端运行尚未验证。
- 确认日期：2026-09-11。
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
| 决策原因与取舍 | `docs/decisions/` 中的 ADR |
| Phantom3 label/metadata 的文件 schema 与类别 ID | [Phantom3 语义标签交付规范](../design/phantom3-semantic-label-contract-v1.md) |
| 类别本体 | 当前数据版本的 `semantic/class_spec.json` |
| 单次运行参数 | 保存的实验配置和显式 CLI |
| 环境依赖 | 正式验证后的 `environment.yml` 与 fingerprint |
| 标签来源 | 每影像 metadata |
| 产品身份、对齐和质量状态 | 产品及 pair manifest |

若旧研究文档、历史源码默认或注释与 G1 Evidence V2 合同冲突，以当前合同为准，并将旧内容标为 superseded，而不是维持两套“已确认”定义。

Phantom3 交付规范仍含早期 `phantom3_test100`、102张和 semantic 同置布局示例；这些历史路径/计数不是当前运行默认。当前本机路径、stage 30→100 切片和外置 semantic root 以本文为准，交付规范需在独立文档维护任务中收敛。

## 项目与代码边界

### CURRENT DECISION：研究范围

- Semantic A-TDOM 只以原始 A-TDOM 为代码基础。
- `ref/A-TDOM` 是只读参考基线。
- A-TDOM_Online 只说明此前研究课题，**FORBIDDEN** 将其作为算法、架构、实现思路或复制代码的来源。
- G（逐高斯语义场）是研究主线。
- D（直接分割 RGB TDOM）和 M（二维标签直接融合到地图网格）只作为未来对比/消融。
- G3 学习式残差是低优先级扩展，不能成为 V1 权威产品的必要依赖。
- 当前产品是闭集地物类别的俯视 RGB/Semantic TDOM；任意透视视角语义只作内部诊断。

### MUST：几何防火墙

几何影响语义，语义绝不反向影响几何：

- 语义只能读取 detached 的高斯几何、可见性和相机信息；
- 不参与反向传播或 RGB optimizer；
- 不改变位置、尺度、旋转、opacity、SH、loss、densify、clone、split 或 prune；
- 标签、语义 worker 或 Semantic TDOM 失败不能阻断、回滚或改变几何训练。

### CURRENT DECISION：仓库治理

- `main` 保存历史/重构基线，**FORBIDDEN** 直接进行当前语义开发。
- `semantic/xzp` 是上一阶段语义开发基线。
- `semantic/g1-evidence-v2` 用于本次重大合同修订和实施。
- 重大算法、数据合同、环境大版本或产品 schema 变更使用独立分支与 ADR。
- `upstream` 保持只读，`origin` 用于个人仓库推送。
- MUST 识别并保留工作区中不属于当前任务的修改。

## 本机数据与首个任务

### 规范本地路径

```text
RGB/SfM 数据根目录：/data3/xzp/data/Onthefly/phantom3
语义交付根目录：   /data3/xzp/data/Onthefly/semantic_labels
实际语义目录：     /data3/xzp/data/Onthefly/semantic_labels/semantic
```

这些是当前机器默认值，不允许硬编码进算法。启用语义时必须显式提供 non-empty semantic data root；当前 CLI 语义是传入“包含 `semantic/` 的父目录”。

### CURRENT DECISION：验证切片

- 初始化 stage 30。
- 首个完整目标是 stage 30 到 stage 100。
- “前100张”指 stage 100 注册集合，不指任意目录排序的前100个文件。
- 首个 smoke 是 stage 30 到 stage 31；通过后才运行至 stage 100。
- 完整399张数据暂不宣称受支持。

### CURRENT STATUS：实际数据事实

- `phantom3` 有数字阶段 3 到 400。
- stage 100 的 ID 为 `000001..000100`。
- 最终集合有399张，编号 `000001..000402`，缺少 `000384..000386`。
- 后期阶段存在计数和累积性异常，当前入口没有完整防护。
- 存储 RGB/label 为 1600×897；`cam_params.json` 原始 SfM 坐标为 1870×1049，两套坐标不能混用。
- 当前语义交付有399份 label/metadata，没有 `scores/`。
- 前100张 background 像素为0；全部399张只有 `000112.png` 含6038个 background 像素，约占总像素0.001054%。
- 当前应描述为“八类 schema，前100张实质观测到类别1–7和ignore”，不能声称八类均已验证。

## 语义数据合同摘要

### MUST：身份与目录

```text
semantic/
├── class_spec.json
├── labels/<six-digit-image-id>.png
├── metadata/<six-digit-image-id>.json
└── scores/<six-digit-image-id>.npy        # 可选
```

- 六位数字 filename stem 是 RGB、camera、label 和 metadata 的共同身份。
- 当前消费顺序是数字 stem；`imagesNames.txt` 只用于 match matrix 行列寻址。
- 真实采集顺序若不同，必须新增显式 manifest，不能静默改排序。
- label 必须与运行时 RGB/camera tensor 严格同尺寸、同方向、同坐标。
- 运行时禁止 resize/rotate/flip label；转换必须作为新数据准备版本完成。
- stage 内遗留三分类 `labels/` 不是权威输入，禁止消费。

### MUST：类别与 ignore

当前数据版本的类别顺序是：

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
- 当前最早102份缺少 source binding，允许用于标为 `source_binding=unverified` 的开发 smoke。
- 正式 `exact_final` 要求 source hash 与当前 RGB 一致，并验证方向/坐标约定。
- 仅补齐正确 source hash且像素/生成配置不变时，可以保留 label version。
- 标签、本体、模型或生成配置变化时必须发布新版本。
- 在线遇到首个缺口后维持连续 readiness watermark；最终重建可以利用缺口后的所有有效标签。
- 缺失不得跳过后当作 background。

### MUST：原始数据只读

RGB、SfM 和 semantic roots 是只读输入。派生 PLY、索引和转换结果写入 `OUTPUT_ROOT/cache/<dataset-id>/<stage>/`，模型、日志、sidecar 和地图写入 output root。在缓存架构完成前，只能对明确的工作副本执行会写 source 的入口。

运行前的只读 preflight 必须核验 stage/camera/RGB/overlap 集合、矩阵、累积性、label/schema/metadata/hash 和显式 root，并报告类别分布、ignore、overlap 分组、source provenance 与后期异常。报告 hash 进入训练和最终产品 manifest。

## 环境合同

- `ATDOM_XZP` 只是 candidate migration baseline，不是正式环境。
- 候选组合为 Python 3.10、PyTorch 2.4.0+cu121、CUDA 12.1、RTX 4090。
- 正式环境名为 `semantic-atdom`。
- 不能复制历史 editable package 或 CUDA 二进制；本地扩展必须从当前 `third_party/src` 构建。
- `environment.yml` 在验证后成为可执行真源；重大依赖升级使用独立分支。

环境只有在以下检查通过后才从 candidate 晋级：

1. 从 manifest 新建环境；
2. editable install 与所有扩展均指向当前仓库；
3. 真实 import 和 semantic CUDA symbols 验证；
4. CPU/schema 测试；
5. CUDA observation 和 Semantic TDOM 最小测试；
6. Phantom3 stage 30 到 31 smoke；
7. 保存 Python/Torch/CUDA/compiler/GPU/package source/git fingerprint。

### CURRENT STATUS

- 当前 `environment.yml` 仍描述旧 ATDOM/Python3.7/Torch1.12/CUDA11.6，不是正式合同。
- 当前工作树含在建语义源码，但现有本地扩展尚未证明来自当前源码并具有所需 symbols。
- 正式 `semantic-atdom` 尚未建立；端到端能力未验证。

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
- clone/split 的硬不变量是拓扑前后渲染语义场连续；后代复制父状态只是待数值测试的 V2 初始近似。
- online group disagreement 必须真实更新；支持不足时是 NaN，不是0。

## 在线、最终与地图产品

- 在线状态随几何阶段更新，但历史投影不会天然变成精确；在线产品一律是有界近似。
- 活动窗口保存开放组和有限最近闭合组；manifest 记录证据 geometry version/staleness。
- 几何阶段与地图发布事件分开；latest-wins 可以跳过过时地图请求，不能丢弃标签观测。
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

同一任务固定局部 grid direction/origin/GSD/pixel convention；未知范围通过固定 tile index 扩展。尚未实现时只能声称同一产品对 `pair-aligned`，不能声称跨 stage 对齐或具有 GIS CRS。

RGB/Semantic 产品使用共同 `projection_geometry_hash`，完整 RGB 另有 `rgb_model_hash`，语义另有 `semantic_state_hash`。两侧 manifest 由最后原子发布的 `pair_manifest.json` 引用。

在线核心产物只保存 hard/color/unknown/evidence-support/entropy/schema/manifest；final 默认增加 coverage、support ratio、Top-1、margin、mixture/group disagreement 和 variance proxy。完整概率栅格始终可选。文件精确定义见权威数学合同。

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
- semantic snapshot 只包含投影需要的几何与相机，不复制无关 SH。
- 使用显式低优先级 CUDA stream、ready event、显存 headroom、时间预算和 checkpoint cadence。
- 超预算时依次停止中间诊断写盘、跳过过时地图渲染并合并到最新 snapshot；不阻塞几何、不丢标签。
- final exact rebuild 单独统计，不计入在线 SLO。
- `lag<=1` 和具体耗时比例在真实 profile 前只是目标，不能写成已保证事实。

## 版本与实施状态

- 新方法版本：`g1-evidence-v2`。
- state/recovery 和 product/manifest schema 必须升级。
- 旧 sidecar 不允许静默 resume，必须从对应几何和标签重建。
- 旧产物保留时必须标为 superseded，不在原目录无提示覆盖。

截至 2026-09-11：

| 项目 | 状态 |
|---|---|
| 项目范围和几何防火墙 | 已确认 |
| G1 Evidence V2 数学/产品合同 | 已确认 |
| 文档权威收敛 | 已完成 |
| CPU reference/tests | 已完成 |
| G1 state/recovery/product schema | 已完成 |
| CUDA V2 实现 | 已完成 |
| online 状态与独立 final recovery | 已完成 |
| paired publication | 未开始 |
| read-only preflight/cache | 未开始 |
| 正式 `semantic-atdom` 环境 | 未建立 |
| stage 30→31 / 30→100 | 未验证 |

当前 product/scheduling 源码的已知差距包括动态跨阶段网格、非原子 RGB/Semantic 配对、仍复制无关 SH 的 snapshot，以及尚未确定 GPU 资源/过载策略。CPU/CUDA evidence renderer、prior-free LOO、在线陈旧度/状态机、独立 final recovery 和新版本 schema 已完成对齐；online group disagreement 在 exact rebuild 前明确为不可用而不是伪零。

实施任务与顺序见 [G1 Evidence V2 active plan](../../plans/active/2026-09-11-g1-evidence-v2.md)。在 CPU 数学合同稳定前不修改 CUDA。

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

- [Semantic TDOM V1：G1 Evidence V2](../design/semantic-tdom-v1.md)
- [Evidence-conditioned 渲染 ADR](../decisions/0005-evidence-conditioned-semantic-rendering.md)
- [有界在线产品 ADR](../decisions/0006-bounded-online-semantic-products.md)
- [几何防火墙 ADR](../decisions/0001-one-way-semantic-dependency.md)
- [阶段同步 ADR](../decisions/0003-stage-synchronous-semantics.md)
- [A-TDOM 谱系 ADR](../decisions/0004-clean-atdom-lineage.md)
- [Phantom3 标签交付规范](../design/phantom3-semantic-label-contract-v1.md)
- [推荐技术路线](../research/semantic-atdom-recommended-route.md)
- [2026 文献综述](../research/lightweight-semantic-3dgs-literature-review-2026.md)
