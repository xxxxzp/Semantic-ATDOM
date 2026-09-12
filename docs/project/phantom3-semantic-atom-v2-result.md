# Phantom3 Semantic A-TDOM V2 完整实验记录

## 实验身份

- 日期：2026-09-12
- 数据：`/home/xuzipeng/code/Semantic-A-TDOM/datas/phantom3`
  （指向 `/data3/xzp/data/Onthefly/phantom3`）
- 语义标签：`/data3/xzp/data/Onthefly/semantic_labels`
- 输出根：`/data3/xzp/data/Onthefly/outputs/semantic_atom_v2`
- 环境：`semantic-atdom`
- 训练开始代码：`c6ca96b98b454710297c386f84dc4d7ecce779b9`
- 最终发布修复代码：`847525cb96e3c7f7eaee9b32b31e40c03fd18019`
- 原始 A-TDOM 参考：`e50a5f7986664ca97b135c066b4cd462eb0bb4bf`
  （tracked 内容未修改）

正式运行使用：

```shell
CUDA_VISIBLE_DEVICES=1 python tools/train_progressive.py \
  --Source_Path_Dir /home/xuzipeng/code/Semantic-A-TDOM/datas/phantom3 \
  --Model_Path_Dir /data3/xzp/data/Onthefly/outputs/semantic_atom_v2 \
  --StartFromImageNo 30 --EndAtImageNo 400 \
  --IterationFirstScene 2000 \
  --IterationPerMergeScene 100 \
  --GlobalOptimizationInterval 1 \
  --GlobalOptimizationIteration 100 \
  --FinalOptimizationIterations 4000 \
  --OriginImageHeight 1049 --OriginImageWidth 1870 \
  --points_per_triangle 8 --LogMaskThreshold 0.1 \
  --render_TDOM --enable_semantics \
  --semantic_data_root /data3/xzp/data/Onthefly/semantic_labels \
  --semantic_tdom_save_diagnostics \
  --semantic_checkpoint_interval 5 \
  --semantic_map_publication_interval 10 \
  --NoDebug
```

## 完成结果

- 进程正常退出，日志终止标记为 `All Done!!!`。
- 最终训练迭代：79,800；高斯数：538,792。
- 399/399 张相机完成完整 RGB 评测和最终渲染。
- RGB：L1 0.0668768，PSNR 21.7285 dB，SSIM 0.7553775，
  LPIPS 0.4240095。
- 最终精确语义重建使用399张标签、180个观测组，状态为
  `exact_partial`，`lag=0`，无 error、无 deferred image。
- `exact_partial` 的唯一已知原因是现有标签 metadata 缺少完整 RGB
  source-image hash binding；不是标签缺失、重建失败或在线 fallback。
- 最终地图为 4608×6144；RGB/Semantic 使用同一 `grid_id`、
  `projection_geometry_hash` 和冻结几何快照。
- 最终 RGB TDOM 非空：uint8 范围 0–255，均值 25.5115。
- 最终 hard semantic map 输出类别1–7，unknown 像素为0。各类像素数：
  bare soil 4,989,302；building 6,094,787；vegetation 6,898,206；
  paved area 8,558,344；vehicle 103,257；water 688,406；road 979,250。
- group disagreement 使用独立的多组有效支持，状态为 `exact_rebuild`；
  解码范围 0–0.473617，均值 0.138917。

最终权威入口是：

```text
/data3/xzp/data/Onthefly/outputs/semantic_atom_v2/Map_Products/latest.json
  -> stage_000399_final_refinement/pair_manifest.json
```

最终 `pair_manifest.json` SHA-256 为
`feba6e634168eb1ac83a4f8f26203e3b8d20ba2731087ac676d4cdf73966b7ab`。
清单中18项产品/子清单 hash 全部复算通过。输出根约6.0 GiB；其中包含
准备阶段的 smoke/debug 子目录，消费者必须以 `Map_Products/latest.json`
为准，不应按目录名猜测最终产品。

## 时间与资源

- 初始几何训练：13.750 s。
- 增量几何训练：614.912 s，记录的均摊时间为 1.541 s/image。
- 最终几何精修：35.130 s。
- 冻结几何全量语义重建：307.408 s，其中投影 298.087 s、读取标签
  8.689 s、融合 0.426 s；语义重建峰值 CUDA allocation 约2.06 GiB。
- 最终评测约7分36秒；最终399张渲染约22秒。
- 本次机器报告为 NVIDIA GeForce RTX 4090、CUDA 12.1、PyTorch 2.4.0，
  可见显存约48 GiB。该实测不能替代24 GiB目标硬件上的资源验证。

## 运行中发现并关闭的问题

1. RGB rasterizer 原先把部分 kernel/CUB/memcpy 固定到默认 CUDA stream，
   而 paired publisher 在非默认低优先级 stream 上运行，导致 RGB TDOM
   间歇性全黑。修复后整个 RGB forward 使用 PyTorch current stream；新增
   非默认 stream CUDA 回归测试，并从同一最终 checkpoint 原子重发 pair。
2. 仅有单组观测的少量高斯曾使整张 group disagreement 被标为不可用。
   修复后分歧有独立有效支持分母：单组高斯不贡献伪零，仅无多组支持的
   像素为 NaN。几何、类别 posterior 和 hard semantic map 均未改变。
3. 仓库结构测试的支持工具白名单遗漏 `capture_environment.py`，已补齐
   显式登记，没有放宽目录约束。

最终代码验证为141项测试全部通过，覆盖 CPU/schema、CUDA observation、
Semantic TDOM、非默认 stream、拓扑同步、final rebuild、原子 pair 发布和
仓库结构。

## 仍未验证

- 标签 source-image binding 尚不完整，因此不能升级为 `exact_final`。
- 当前地图是无 GIS CRS 的局部固定网格，不能宣称地理坐标已校准。
- 24 GiB RTX 4090 目标硬件上的峰值显存和准实时预算尚未实测。
- 本次只证明训练、重建、配对渲染与产品完整性；语义准确率、阈值优劣和
  SAM 标签质量尚未评测。unknown=0 也不是语义质量充分的证据，后续仍需
  结合标注或人工抽检校准 support gate。
