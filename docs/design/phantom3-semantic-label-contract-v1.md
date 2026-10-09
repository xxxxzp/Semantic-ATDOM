# Phantom3 渐进数据语义标签交付规范 V1

## 目的与适用范围

2026-10-09 范围说明：本文保留早期 102 张 `phantom3_test100` 的接口示例，不能把
它当作当前数据盘点。当前 Phantom3 交付为 414 份标签和 metadata，放在场景根的
`semantic_labels/semantic/` 下；具体路径、尺寸及绑定证据见
[当前数据与实验核对](../project/current-data-experiment-status.md)。
文件内容约束仍适用，但 semantic root 可独立于 RGB/SfM root。AMtown01 是独立
十五类 schema，使用小数时间戳身份，不能复用本例的六位编号与八类定义。

本文规定 `datas/phantom3_test100` 在 Semantic A-TDOM 第一阶段需要提供的离线语义结果。它是 SAM3、ProC-SAM3 或后续其他标签生成器与 Semantic A-TDOM 之间的文件接口，不规定必须使用哪一个模型。

`phantom3_test100` 的数字阶段目录是累积式 SfM 快照：例如 `30/images` 包含前 30 张影像，`31/images` 包含前 31 张影像。相同影像不会因为出现在多个阶段中而成为新的语义观测。因此，每张原始影像的语义结果在数据集根目录只保存一份，不在各阶段目录中复制。

## 最小交付内容

第一阶段应提供：

- 一份数据集级 `class_spec.json`；
- 每张 RGB 影像对应的一张单通道类别 ID PNG；
- 每张类别 ID PNG 对应的一份 provenance metadata JSON。

完整类别 scores 不是第一阶段运行的必需输入。`confidence map`、实例 mask、检测框和彩色预览图也不是融合输入。

对于本历史示例的 102 张影像，最小交付物共包含 1 份类别规范、102 张标签 PNG 和 102 份 metadata JSON。

## 规范目录结构

```text
phantom3_test100/
├── 3/
├── 4/
├── ...
├── 102/
│   └── images/
│       ├── 000001.jpg
│       ├── ...
│       └── 000102.jpg
└── semantic/
    ├── class_spec.json
    ├── labels/
    │   ├── 000001.png
    │   ├── ...
    │   └── 000102.png
    └── metadata/
        ├── 000001.json
        ├── ...
        └── 000102.json
```

标签生成器可直接以最终阶段 `102/images` 作为 102 张唯一 RGB 影像的离线输入。阶段目录中不应再创建 `labels/`；这样可避免重复存储以及同一影像被错误计为多次独立证据。

## 标签 PNG

以 `000001.jpg` 为例，对应文件必须是 `semantic/labels/000001.png`。

标签文件必须满足：

- 与 RGB 影像使用相同文件主名；
- 与 RGB 影像逐像素对齐，本历史示例为宽 1600、高 897；当前交付尺寸见数据核对记录；
- 单通道 8-bit grayscale PNG，即 PIL `L` 模式、NumPy `uint8`；
- 每个像素只保存一个闭集类别 ID；
- 像素值只能是 `0..7` 或 `255`；
- 不做抗锯齿，不保存 RGB 颜色，不保存实例 ID。

V1 类别映射固定为：

| ID | 类别 | 含义 |
|---:|---|---|
| 0 | `background` | 模型明确判定的背景 |
| 1 | `bare_soil` | 裸土、未铺装土面 |
| 2 | `building` | 建筑物 |
| 3 | `vegetation` | 树木、草地及其他植被 |
| 4 | `paved_area` | 广场、停车区等铺装面 |
| 5 | `vehicle` | 车辆；当前数据中的车辆按静态物体处理 |
| 6 | `water` | 水体 |
| 7 | `road` | 道路 |
| 255 | ignore | 未知、不可靠或无有效观测，不属于真实类别集合 |

不同实例共享语义类别。例如所有建筑实例都写为 `2`，所有树木都写为 `3`。若多个类别候选覆盖同一像素，标签适配器应采用已记录的确定性冲突策略，例如选择最高类别分数；若没有类别达到有效阈值，则写为 `255`。

### Background 与 ignore

`background=0` 和 `ignore=255` 不得混用：

- `0` 会作为 background 的正语义证据参与高斯融合；
- `255` 只参与总可见质量统计，不产生任何类别证据；
- SAM 分数低于阈值、标签生成失败、黑边或其他图像无效区域默认写为 `255`；
- 只有标签生成器明确支持并判定为 background 时才写 `0`。

这一规则用于避免把“模型没有识别出来”错误地解释成“模型确认它是背景”。

## 类别规范

`semantic/class_spec.json` 在一次语义任务开始前固定。建议内容如下；颜色只用于 Semantic TDOM 和诊断图可视化，不参与概率融合。

```json
{
  "schema_version": 1,
  "ignore_id": 255,
  "classes": [
    {"id": 0, "name": "background", "color": [0, 0, 0], "is_background": true},
    {"id": 1, "name": "bare_soil", "color": [166, 97, 26], "is_background": false},
    {"id": 2, "name": "building", "color": [230, 25, 75], "is_background": false},
    {"id": 3, "name": "vegetation", "color": [77, 175, 74], "is_background": false},
    {"id": 4, "name": "paved_area", "color": [255, 127, 0], "is_background": false},
    {"id": 5, "name": "vehicle", "color": [152, 78, 163], "is_background": false},
    {"id": 6, "name": "water", "color": [55, 126, 184], "is_background": false},
    {"id": 7, "name": "road", "color": [228, 26, 28], "is_background": false}
  ]
}
```

### Semantic TDOM 可视化调色板

`class_spec.json` 是标签类别契约的一部分，其 fingerprint 会写入每张
metadata；不能为了改显示颜色而修改它。可在同一 `semantic/` 目录放置可选的
`tdom_palette.json`，只覆盖 Semantic TDOM 的可视化颜色：

```json
{
  "schema_version": 1,
  "overrides": {
    "building": [255, 0, 0],
    "road": [128, 128, 128]
  }
}
```

未列出的类别沿用 `class_spec.json` 的颜色。该文件不改变类别 ID、label、score、
逐高斯概率、多视角融合或 unknown 判定，也不参与标签契约 fingerprint。渲染产物
会另外写出完整的 `render_palette.json` 和哈希，确保地图配色可追溯。
训练入口会自动读取 `<semantic_data_root>/semantic/tdom_palette.json`；也可通过
`--semantic_tdom_palette_path` 显式指定其他文件。

类别 ID 必须连续、从零开始，并与这里的顺序一致。修改类别、顺序、名称或颜色会改变类别规范哈希，必须作为新的标签版本发布。

## 每影像 metadata

每张标签必须具有同名 metadata。例如 `semantic/metadata/000001.json`：

```json
{
  "schema_version": 1,
  "image_id": "000001",
  "label_version": "phantom3-proc-sam3-v1",
  "model": "ProC-SAM3",
  "checkpoint_sha256": "<64位小写十六进制SHA256>",
  "prompt_config": {
    "classes": [
      "background",
      "bare_soil",
      "building",
      "vegetation",
      "paved_area",
      "vehicle",
      "water",
      "road"
    ],
    "prompt_pool_version": "v1",
    "class_aggregation": "max",
    "conflict_policy": "highest_score_wins"
  },
  "thresholds": {
    "confidence": 0.5,
    "probability": 0.1
  },
  "width": 1600,
  "height": 897,
  "class_spec_sha256": "<规范化class_spec内容的SHA256>",
  "label_sha256": "<标签PNG文件的SHA256>"
}
```

尖括号字段是说明占位符，不能直接作为交付值。标签生成器必须写入真实模型标识、checkpoint 哈希、有效 prompt 配置、阈值和文件哈希。使用同一模型配置生成的一批标签可以共享 `label_version`；任何配置或标签内容变化都必须发布新版本，不能用同一 `image_id + label_version` 静默覆盖不同内容。

## 可选完整 scores

若后续决定保留完整类别分数，则添加：

```text
semantic/scores/000001.npy
```

其要求为：

- shape 为 `[K,H,W]`，当前即 `[8,897,1600]`；
- 类别维顺序与 `class_spec.json` 完全一致；
- dtype 为浮点类型，推荐 `float16` 或 `float32`；
- 所有数值有限；
- 对应 metadata 增加真实的 `score_sha256`。

单通道 `[H,W]` confidence map 不等同于完整类别 scores，不能放入 `scores/<id>.npy`。如需保留，只能作为标签生成器的诊断产物，Semantic A-TDOM V1 不读取它。

未校准 scores 只用于阈值拒绝和诊断，不直接转换为 Dirichlet 伪计数。

## 渐进消费顺序

文件存储是全局唯一的，消费仍然是渐进的：

1. 前 30 张影像几何初始化完成后，一次性吸收 `000001..000030`；
2. 第 31 阶段同步时只吸收尚未处理的 `000031`；
3. 后续阶段继续只吸收新到达且未记入 ledger 的影像；
4. 同一 `image_id + label_version` 不得因 RGB 训练重复迭代而重复计数；
5. 最终 refinement 后，使用最终几何对全部已登记标签执行一次精确重建。

## 发布与验收要求

标签生成器应先写临时文件，完成后通过原子重命名发布，避免 WP3 读取半写入文件。每张影像交付前至少验证：

- RGB、标签和 metadata 的文件主名一致；
- PNG 是 `[897,1600]` 的单通道 `uint8`；
- 标签值域仅包含 `0..7` 和 `255`；
- metadata 分辨率与 PNG 一致；
- `image_id`、`label_version`、模型配置和阈值存在；
- checkpoint、类别规范、PNG 以及可选 scores 的 SHA256 正确；
- 所有 102 张 RGB 都有且只有一份对应标签和 metadata；
- 不把未识别像素、低置信像素或无效图像区域写成 background。

运行时校验实现位于 `src/semantic_atdom/semantics/schema.py`。不满足本规范的文件必须显式拒绝，不能静默修正类别 ID、尺寸或版本信息。
