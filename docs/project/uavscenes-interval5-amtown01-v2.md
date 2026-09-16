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
- 最终语义重建耗时 658.432 秒（其中投影 627.705 秒），得到 314 个融合组；
  在线/最终类别一致率为 0.933635，平均 JS 散度为 0.009075。
- 修正后 487 帧全量指标为 L1 0.040385、PSNR 24.805448 dB、SSIM
  0.625038、LPIPS 0.509187；最终模型包含 1,034,636 个高斯。旧 r8 的
  SSIM 0.800672、LPIPS 0.508867 因张量布局和重复归一化错误已经撤销，不得引用。
- `OutputImages` 中 487/487 个相机目录均包含非空的
  `PredictionImages100800.jpg` 和 `PSNR.txt`。首次收尾渲染暴露了未使用列表持有
  历史 CUDA 图像导致的线性显存增长；修复张量生命周期后，从最终点云和各阶段
  累计相机集合恢复渲染，完整 487 帧通过此前故障点。
- final-only 输出策略加入后完整测试发现：175/175 通过。
- 正式结果目录：
  `/data3/xzp/data/UAVScenes_Onthefly/interval5_AMtown01/outputs/semantic_atdom_v2_amtown01_observed_v1_r9`。
  已清除 474 个中间 stage 目录，结果由 86 GiB 收敛至 2.6 GiB；最终 checkpoint
  使用可恢复 schema v2，点云为
  `487/point_cloud/iteration_487/point_cloud.ply`，最终地图位于
  `Map_Products/stage_000487_final_refinement`。

`r2`–`r7` 是定位显存驻留、预算裁剪和裁剪吞吐问题时保留的诊断运行，不应作为
正式结果引用。`r8` 完整跑通了旧投影链路，但审计确认该链路忽略 COLMAP RADIAL
畸变，且旧版 SSIM/LPIPS 实现有误，因此 `r8` 只能作为工程诊断结果，不能继续作为
几何、语义或指标的权威 baseline。`r9` 是修复后的正式最终结果；其在线语义线程
在 stage 20 后因拓扑失配降级，但最终 exact rebuild 完整覆盖 487/487 张观测，
因此只将最终语义产品视为权威，不将该运行用于证明中间在线语义发布稳定性。

## 后续正式运行命令（final-only）

```shell
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n semantic-atdom \
  python tools/train_progressive.py \
  --Source_Path_Dir /data3/xzp/data/UAVScenes_Onthefly/interval5_AMtown01 \
  --Model_Path_Dir /data3/xzp/data/UAVScenes_Onthefly/interval5_AMtown01/outputs/semantic_atdom_v2_amtown01_observed_v1_r10 \
  --StartFromImageNo 13 --EndAtImageNo 487 \
  --IterationFirstScene 2000 \
  --IterationPerMergeScene 100 \
  --OutputModelInterval 0 \
  --GlobalOptimizationInterval 1 \
  --GlobalOptimizationIteration 100 \
  --FinalOptimizationIterations 4000 \
  --OriginImageHeight 2048 --OriginImageWidth 2448 \
  --resolution 1 \
  --points_per_triangle 8 --LogMaskThreshold 0.1 \
  --GaussiansNumMax 1000000 --GaussianBudgetSlack 0.05 \
  --data_device cpu \
  --progressive_identity_policy persistent_union \
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

AMtown01 的 stage 18 会暂时省略两个已经出现过的 identity，stage 19/20 再次出现。
`persistent_union` 明确声明这些临时省略不代表删除；未提供该参数时预检采用
`strict_cumulative` 并拒绝这一数据。训练输入的 RADIAL/主点偏移会在读取时统一
转换为同尺寸中心主点 pinhole，RGB、语义标签、稀疏关键点和有效区域共享同一
rectification。正式运行默认使用 `OutputModelInterval=0`：渐进阶段不周期性保存
完整模型，但进入最终优化前仍强制保存 stage 487，训练完成后再以最终 checkpoint
覆盖它。成功结束后会删除除最终 stage 以外的数字阶段目录，避免空壳目录和历史
模型占满输出盘。若显式设置正数间隔，每个对应阶段的 checkpoint 包含累计相机、
optimizer、每图训练计数和 RNG 状态，并保留这些阶段目录用于恢复。

修复验收包括：`pytest -q` 175/175 通过；真实 13–487 persistent-union 预检仍为
487/487；stage 13–20 极小训练穿过焦距变化和非累积阶段；stage 14 checkpoint 在
新进程恢复后得到与连续训练相同的最终 L1/PSNR；真实 stage 13 全分辨率标签完成
CUDA 投影和 final exact rebuild，状态为 `exact_final`、lag=0、无 error/deferred。
