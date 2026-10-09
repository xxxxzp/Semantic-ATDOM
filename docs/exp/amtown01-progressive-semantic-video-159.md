# AMtown01 渐进式语义训练与 159 帧视频实验参考


状态：159 帧计划参数与复现模板；已有独立的 160 帧全量成功运行，不代表本方案已按原参数验收
数据集：`UAVScenes_Onthefly/interval5_AMtown01`  
更新日期：2026-10-09

现存 `amtown01-full-progress160-fixed_20260930-134152_bf6a7c9` 已成功完成：
stage 13–487、487 个最终相机、100,800 次优化、最终 `exact_final`，视频经本次
ffprobe 核对为 160 帧、4 fps。该运行来自 dirty worktree，不是当前干净 HEAD 的
全量验收。本文件保留 159 帧计划；实际运行身份和数据见
[当前数据与实验核对](../project/current-data-experiment-status.md)。

## 1. 实验目标

基于当前仓库最新代码，对 AMtown01 数据集执行一次完整的渐进式 Gaussian
训练，并发布：

- 最终 Gaussian 模型；
- 最终 RGB/Semantic TDOM 语义地图；
- 全部相机的最终渲染与指标；
- 展示训练过程中地图逐步扩展的 RGB + Semantic TDOM 视频。

视频只用于展示训练过程，不改变训练覆盖范围。训练仍然处理全部 stage；视频只对
stage 边界进行均匀采样。

## 2. 代码与输入

### 代码版本

- 仓库：`/home/xuzipeng/code/Semantic-A-TDOM/Semantic-A-TDOM`
- 分支：`semantic/g1-evidence-v2`
- 原方案基线：`bf6a7c9`；本次文档核对的当前代码为 `0665f4b`，没有重新全量训练
- 关键语义空洞修复：`550408c`，即 bounded planar semantic fallback

正式运行前应将以下信息写入 run 的 `provenance/`，并确认代码改动状态：

```bash
git status --short
git rev-parse HEAD
```

### 数据集

```text
Source_Path_Dir:
/data3/xzp/data/UAVScenes_Onthefly/interval5_AMtown01

Progressive Gaussian input:
/data3/xzp/data/UAVScenes_Onthefly/interval5_AMtown01/GaussianSplatting

Semantic input:
/data3/xzp/data/UAVScenes_Onthefly/interval5_AMtown01/outputs/semantic_inputs/uavscenes_amtown01_observed_v1
```

已确认的数据契约：

- progressive stage：`13..487`，共 475 个连续 stage；
- RGB：`interval5_CAM`；
- 原始影像尺寸：约 `2448 × 2048`；
- 语义交付：490 张；所选阶段最终预检及累计相机清单是 487 个身份，不能将两种计数等同；
- 语义类别：15 类，ignore id 为 `255`；
- full preflight 应报告：`labels_complete=True`、`source_binding_complete=True`。

## 3. 分辨率与训练策略

本实验必须使用：

```text
--resolution 1
```

这表示保持输入影像原始分辨率。不要使用 `--resolution -1`：该选项会对宽度大于
1600 的影像自动缩放到约 1600，但当前语义投影代码要求 RGB 与标签共享全分辨率
像素网格，`enable_semantics` 与非 1 的 resolution 组合会直接拒绝。

`--data_device cpu` 用于让历史 RGB/mask 常驻 CPU，只将当前训练视图搬到 GPU，避免
全量影像随 progressive stage 累积占满显存。

## 4. 正式全量训练参数

以下命令是本实验的正式参数基线。`GPU_ID` 应替换为当前 GPU-enabled 环境中空闲且
显存足够的 GPU；先确认 `nvidia-smi` 和 PyTorch CUDA 均可用。

```bash
GPU_ID=1

CUDA_VISIBLE_DEVICES=$GPU_ID conda run --no-capture-output -n semantic-atdom \
  python tools/train_progressive.py \
  --Source_Path_Dir /data3/xzp/data/UAVScenes_Onthefly/interval5_AMtown01 \
  --Model_Path_Dir /data3/xzp/data/UAVScenes_Onthefly/interval5_AMtown01/outputs \
  --run_name amtown01-full-semantic-video-159 \
  --StartFromImageNo 13 \
  --EndAtImageNo 487 \
  --IterationFirstScene 2000 \
  --IterationPerMergeScene 100 \
  --GlobalOptimizationInterval 1 \
  --GlobalOptimizationIteration 100 \
  --FinalOptimizationIterations 4000 \
  --OriginImageHeight 2048 \
  --OriginImageWidth 2448 \
  --resolution 1 \
  --points_per_triangle 8 \
  --LogMaskThreshold 0.1 \
  --GaussiansNumMax 1000000 \
  --GaussianBudgetSlack 0.05 \
  --data_device cpu \
  --progressive_identity_policy persistent_union \
  --render_TDOM \
  --enable_semantics \
  --semantic_data_root /data3/xzp/data/UAVScenes_Onthefly/interval5_AMtown01/outputs/semantic_inputs/uavscenes_amtown01_observed_v1 \
  --semantic_tdom_render_tile_rows 256 \
  --render_all_cameras \
  --render_training_progress \
  --training_progress_target_frames 159 \
  --training_progress_fps 4 \
  --NoDebug
```

如需中断恢复，可额外增加例如：

```text
--recovery_checkpoint_every_stages 20
```

这只保留原子替换的 recovery checkpoint，不会为每个 stage 累积完整 Gaussian
模型。默认不保存完成态 optimizer state；只有确实需要时才增加
`--save_final_training_state`。

## 5. 视频采样约定

本次视频参数为：

```text
target frames: 159
fps: 4
expected duration: approximately 39.75 seconds
codec: H.264 / libx264
```

`159` 是视频目标帧数，不是训练 stage 数。全量训练仍处理 475 个 stage。当前实现
会在 progressive stage 边界中均匀、确定性地选择约 158 个边界，并额外保留最终
refinement 帧；选择不是随机抽样。

视频帧使用当前训练状态渲染 RGB TDOM 和 Semantic TDOM，然后交给 DataSaver 保存
图像与元数据。它不会为每一个视频帧保存一个独立的 Gaussian checkpoint。视频专用
TDOM 相机使用约 8 倍更粗的地图采样率，以控制渲染和进程间传输开销；这不改变训练
影像的输入分辨率。

视频输出：

```text
<run_root>/videos/training_progress.mp4
<run_root>/metrics/training_progress.json
```

159 相较于 120 的目标帧数增加 32.5%，不等于总渲染开销已实测增加同样比例。
Gaussian 优化迭代次数不变；墙钟时间、渲染、队列传输和存储成本须通过同配置
实验测量，不能从帧数直接推出性能结论。

## 6. 小范围 smoke test

在全量运行前，可使用以下参数验证语义同步、视频编码和输出 schema：

```bash
GPU_ID=1

CUDA_VISIBLE_DEVICES=$GPU_ID conda run --no-capture-output -n semantic-atdom \
  python tools/train_progressive.py \
  --Source_Path_Dir /data3/xzp/data/UAVScenes_Onthefly/interval5_AMtown01 \
  --Model_Path_Dir /data3/xzp/data/UAVScenes_Onthefly/interval5_AMtown01/outputs \
  --run_name amtown01-smoke-stage13-60-video24 \
  --StartFromImageNo 13 \
  --EndAtImageNo 60 \
  --IterationFirstScene 2000 \
  --IterationPerMergeScene 100 \
  --GlobalOptimizationInterval 1 \
  --GlobalOptimizationIteration 100 \
  --FinalOptimizationIterations 400 \
  --OriginImageHeight 2048 \
  --OriginImageWidth 2448 \
  --resolution 1 \
  --points_per_triangle 8 \
  --LogMaskThreshold 0.1 \
  --GaussiansNumMax 1000000 \
  --GaussianBudgetSlack 0.05 \
  --data_device cpu \
  --progressive_identity_policy persistent_union \
  --render_TDOM \
  --enable_semantics \
  --semantic_data_root /data3/xzp/data/UAVScenes_Onthefly/interval5_AMtown01/outputs/semantic_inputs/uavscenes_amtown01_observed_v1 \
  --semantic_tdom_render_tile_rows 256 \
  --render_all_cameras \
  --render_training_progress \
  --training_progress_target_frames 24 \
  --training_progress_fps 4 \
  --NoDebug
```

该 smoke test 的 24 帧约为 6 秒视频，目标是检查流程，不用于最终质量评价。它仍
保持 `resolution=1`，以验证真实语义投影路径。

## 7. 启动前检查

必须在实际 GPU-enabled shell 中执行：

```bash
nvidia-smi

conda run --no-capture-output -n semantic-atdom python -c \
  "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

df -h /dev/shm /data3
test -w /data3/xzp/data/UAVScenes_Onthefly/interval5_AMtown01/outputs
```

特别注意：`/dev/shm` 是多进程 PyTorch 队列使用的共享内存，不是 `/data3` 磁盘。
此前曾有其他实验目录长期占用约 182 GB `/dev/shm`，导致训练在 stage 约 106
以退出码 135 失败，并在切换 `file_system` 策略时触发
`torch_shm_manager` socket 创建失败。正式运行前应确认共享内存已经释放，不能擅自
删除其他用户的任务目录。

历史隔离 shell 曾出现宿主 GPU 可见但设备未透传的情况。2026-10-09 审查环境的
PyTorch CUDA 可用，CUDA 测试通过；不能继续笼统描述当前环境没有 GPU。
每次实际运行仍应检查该 shell 的设备可用性。

## 8. 成功验收标准

完成后检查：

1. `manifest.json` 的 `status` 为 `succeeded`；
2. `final_stage=487`，且没有未处理的 failure；
3. `metrics/training_progress.json` 存在，`frame_count` 为 159，`fps` 为 4；
4. `videos/training_progress.mp4` 可被 `ffprobe` 验证为 H.264，帧数与 metadata 一致；
5. `products/maps/` 中存在最终 RGB/Semantic 地图及 pair manifest；
6. `renders/cameras/` 中存在全部最终相机渲染；
7. 最终语义状态必须为 `exact_final`，地图发布为 `complete`，且 error/deferred 为空；
   `exact_partial`、`online_fallback` 不是这个完整方案的等价成功状态；
8. 在线语义状态只用于体现真实渐进过程，最终地图质量以 final exact rebuild 为准。

## 9. 相关设计记录

- 数据集适配与历史验收：`docs/project/uavscenes-interval5-amtown01-v2.md`
- 输出目录契约：`docs/decisions/0008-run-output-contract.md`
- 训练视频契约：`docs/decisions/0009-published-renders-and-map-training-video.md`
- 语义空洞修复：`docs/decisions/0010-bounded-planar-semantic-fallback.md`
- 训练规模与异常诊断：`docs/problem/phantom3-water-scale-outlier-training-guidance.md`
