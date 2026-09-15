# UAVScenes interval5_AMtown01 / V2 适配与运行记录

## 输入契约

- 场景根目录：`/data3/xzp/data/UAVScenes_Onthefly/interval5_AMtown01`
- 固定渐进重建目录：`GaussianSplatting/<stage>/bin`
- 累计匹配输入：`match_matrix_image_name.txt`、`match_matrix_value.txt`
- 共享 RGB 目录：`interval5_CAM`
- 当前原始 ID 标签：`semantic_label/interval5_CAM_label_id`

训练入口可从场景根目录自动发现 `GaussianSplatting` 和唯一的 `*_CAM`
共享 RGB 目录。相机和语义身份保留完整十进制时间戳。

语义内容不属于固定目录契约。当前数据通过
`configs/semantic/uavscenes_amtown01_observed_v1.json` 显式映射；该配置只接受
本场景实际出现的 15 个源 ID，新 ID 会使适配失败，避免静默误分类。

## 已验证结果

- 适配标签：490/490，逐图绑定标签 SHA-256 与源 RGB SHA-256。
- 全量预检：13–487 共 475 个连续阶段，最终 487/487 张注册影像标签完整、
  RGB 源绑定完整。
- 全分辨率阶段 13–30 冒烟：30/30 张语义观测完成，lag=0，最终成对 RGB /
  Semantic TDOM 发布状态为 `complete`。
- 正式全量运行：13–487 共 475 个增量阶段和 4,000 次最终优化均完成；最终
  精确语义重建覆盖 487/487 张影像，`geometry_stage=487`、`lag_batches=0`、
  `status=exact_final`、地图发布状态为 `complete`，错误与延迟队列均为空。
- 最终语义重建耗时 489.779 秒（其中投影 459.980 秒），得到 314 个融合组；
  在线/最终类别一致率为 0.861679，平均 JS 散度为 0.051481。
- 487 张训练视图的最终 RGB 指标：L1 0.046980、PSNR 23.566551 dB、
  SSIM 0.800672、LPIPS 0.508867；最终模型包含 1,044,417 个高斯。
- `OutputImages` 中 487/487 个相机目录均包含非空的
  `PredictionImages100800.jpg` 和 `PSNR.txt`。首次收尾渲染暴露了未使用列表持有
  历史 CUDA 图像导致的线性显存增长；修复张量生命周期后，从最终点云和各阶段
  累计相机集合恢复渲染，完整 487 帧通过此前故障点。
- 完整测试发现：159/159 通过；另行复核的 63 个 unit 和 22 个语义 CUDA
  integration 也全部通过，完整发现中包含 9 个高斯拓扑 CUDA 测试。
- 正式结果目录：
  `/data3/xzp/data/UAVScenes_Onthefly/interval5_AMtown01/outputs/semantic_atdom_v2_amtown01_observed_v1_r8`。
  其中最终 checkpoint 为 `487/chkpnt487.pth`，点云为
  `487/point_cloud/iteration_487/point_cloud.ply`，最终地图位于
  `Map_Products/stage_000487_final_refinement`。

`r2`–`r7` 是定位显存驻留、预算裁剪和裁剪吞吐问题时保留的诊断运行，不应作为
正式结果引用；`r8` 是当前唯一权威全量结果。

## 正式运行命令

```shell
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n semantic-atdom \
  python tools/train_progressive.py \
  --Source_Path_Dir /data3/xzp/data/UAVScenes_Onthefly/interval5_AMtown01 \
  --Model_Path_Dir /data3/xzp/data/UAVScenes_Onthefly/interval5_AMtown01/outputs/semantic_atdom_v2_amtown01_observed_v1_r8 \
  --StartFromImageNo 13 --EndAtImageNo 487 \
  --IterationFirstScene 2000 \
  --IterationPerMergeScene 100 \
  --GlobalOptimizationInterval 1 \
  --GlobalOptimizationIteration 100 \
  --FinalOptimizationIterations 4000 \
  --OriginImageHeight 2048 --OriginImageWidth 2448 \
  --resolution 1 \
  --points_per_triangle 8 --LogMaskThreshold 0.1 \
  --GaussiansNumMax 1000000 --GaussianBudgetSlack 0.05 \
  --data_device cpu \
  --render_TDOM --enable_semantics \
  --semantic_data_root /data3/xzp/data/UAVScenes_Onthefly/interval5_AMtown01/outputs/semantic_inputs/uavscenes_amtown01_observed_v1 \
  --semantic_tdom_save_diagnostics \
  --semantic_tdom_render_tile_rows 256 \
  --semantic_checkpoint_interval 5 \
  --semantic_map_publication_interval 10 \
  --NoDebug
```

必须使用 `--resolution 1`：RGB 与人工标签原始分辨率均为 2448×2048；默认的
1.6K 自动缩放会破坏逐像素语义对应关系。

全量渐进训练必须使用 `--data_device cpu`，使历史 RGB 和 mask 常驻主存、仅把
当前训练视图搬到 GPU；否则 487 张全分辨率影像会随阶段累积占满显存。
`GaussiansNumMax=1000000` 配合 5% 滞回区间将模型控制在 100–105 万高斯，并将
昂贵的优化器/语义状态压缩由逐阶段执行改为批量执行。
