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
- 当时的 final-only 输出策略测试为 175/175 通过；本轮发布式输出重构后为
  190/190 通过。
- 正式结果目录：
  `/data3/xzp/data/UAVScenes_Onthefly/interval5_AMtown01/outputs/semantic_atdom_v2_amtown01_observed_v1_r9`。
  已清除 474 个中间 stage 目录，并迁移到自描述输出 schema v1；最终点云和训练
  状态位于 `model/`，最终地图位于 `products/maps/releases/`，指标、日志、配置与
  迁移清单分别位于 `metrics/`、`logs/` 和 `provenance/`。根目录
  `manifest.json` 是结果入口。为维持旧地图清单的哈希链，仅保留其引用的预检缓存；
  其余旧式数字 stage、`Map_Products`、`OutputImages`、`Intermediate_Results` 和双份
  语义恢复槽均已移除。

`r2`–`r7` 是定位显存驻留、预算裁剪和裁剪吞吐问题时保留的诊断运行，不应作为
正式结果引用。`r8` 完整跑通了旧投影链路，但审计确认该链路忽略 COLMAP RADIAL
畸变，且旧版 SSIM/LPIPS 实现有误，因此 `r8` 只能作为工程诊断结果，不能继续作为
几何、语义或指标的权威 baseline。`r9` 是修复后的正式最终结果；其在线语义线程
在 stage 20 后因拓扑失配降级，但最终 exact rebuild 完整覆盖 487/487 张观测，
因此只将最终语义产品视为权威，不将该运行用于证明中间在线语义发布稳定性。

## 后续正式运行命令（发布式输出）

```shell
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n semantic-atdom \
  python tools/train_progressive.py \
  --Source_Path_Dir /data3/xzp/data/UAVScenes_Onthefly/interval5_AMtown01 \
  --Model_Path_Dir /data3/xzp/data/UAVScenes_Onthefly/interval5_AMtown01/outputs \
  --run_name amtown01-v2 \
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
  --progressive_identity_policy persistent_union \
  --render_TDOM --enable_semantics \
  --semantic_data_root /data3/xzp/data/UAVScenes_Onthefly/interval5_AMtown01/outputs/semantic_inputs/uavscenes_amtown01_observed_v1 \
  --semantic_tdom_render_tile_rows 256 \
  --render_all_cameras \
  --progress_video both \
  --progress_video_content rgb \
  --progress_video_stride 10 \
  --progress_video_fps 4 \
  --NoDebug
```

运行目录自动创建为 `outputs/runs/<run_name>_<timestamp>_<git-sha>/`。只有完整成功
发布的运行才会原子更新 `outputs/latest`；失败或中断的运行会保留诊断信息，但不会
成为 latest。逐相机渲染由 `--render_all_cameras` 显式开启。渐进视频可以选择
`camera`、`tdom` 或 `both`，内容可以选择 `rgb`、`semantic` 或 `side_by_side`；
这些选项会产生额外渲染开销，因此默认关闭。

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
rectification。正式运行默认不保存过程恢复点；成功后只发布一次最终 RGB 训练状态、
点云和最终语义模型。若确有中断恢复需求，可显式设置
`--recovery_checkpoint_every_stages N`，它只原子替换 `.work/recovery/latest.pt`，
不会累积每个 stage 的完整模型；成功发布后该工作区会被删除。

当前输出重构验收包括：`pytest -q` 190/190 通过；真实 13–487 persistent-union 预检仍为
487/487；stage 13–20 极小训练穿过焦距变化和非累积阶段；stage 14 checkpoint 在
新进程恢复后得到与连续训练相同的最终 L1/PSNR；真实 stage 13 全分辨率标签完成
CUDA 投影和 final exact rebuild，状态为 `exact_final`、lag=0、无 error/deferred。

输出目录的设计依据、保留策略、迁移边界和视频决策见
`docs/decisions/0008-run-output-contract.md`。
