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
- 全分辨率阶段 13 冒烟：13/13 张语义观测完成，lag=0，最终成对 RGB / Semantic
  TDOM 发布状态为 `complete`。
- 测试：61 unit、71 核心、22 CUDA integration，全部通过。

## 正式运行命令

```shell
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n semantic-atdom \
  python tools/train_progressive.py \
  --Source_Path_Dir /data3/xzp/data/UAVScenes_Onthefly/interval5_AMtown01 \
  --Model_Path_Dir /data3/xzp/data/UAVScenes_Onthefly/interval5_AMtown01/outputs/semantic_atdom_v2_amtown01_observed_v1_r2 \
  --StartFromImageNo 13 --EndAtImageNo 487 \
  --IterationFirstScene 2000 \
  --IterationPerMergeScene 100 \
  --GlobalOptimizationInterval 1 \
  --GlobalOptimizationIteration 100 \
  --FinalOptimizationIterations 4000 \
  --OriginImageHeight 2048 --OriginImageWidth 2448 \
  --resolution 1 \
  --points_per_triangle 8 --LogMaskThreshold 0.1 \
  --render_TDOM --enable_semantics \
  --semantic_data_root /data3/xzp/data/UAVScenes_Onthefly/interval5_AMtown01/outputs/semantic_inputs/uavscenes_amtown01_observed_v1 \
  --semantic_tdom_save_diagnostics \
  --semantic_tdom_render_tile_rows 512 \
  --semantic_checkpoint_interval 5 \
  --semantic_map_publication_interval 10 \
  --NoDebug
```

必须使用 `--resolution 1`：RGB 与人工标签原始分辨率均为 2448×2048；默认的
1.6K 自动缩放会破坏逐像素语义对应关系。
