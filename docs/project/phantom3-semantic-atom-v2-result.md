# Phantom3 Semantic A-TDOM V2 完整实验记录

## 实验身份

- 日期：2026-09-14
- 数据：`/home/xuzipeng/code/Semantic-A-TDOM/datas/phantom3`
  （指向 `/data3/xzp/data/Onthefly/phantom3`）
- 语义标签：`/data3/xzp/data/Onthefly/semantic_labels`
- 正式输出根：`/data3/xzp/data/Onthefly/outputs/semantic_atom_v2`
- 环境：`semantic-atdom`
- 采样/地图范围修复：`f6bfe42f`（`fix: decouple progressive sampling from training masks`）
- 大图 RGB 分片修复：`b8635fcf`（`fix: tile paired RGB TDOM publication`）
- 原始 A-TDOM 参考：`e50a5f7986664ca97b135c066b4cd462eb0bb4bf`
  （tracked 内容未修改）

正式运行使用：

```shell
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n semantic-atdom \
  python tools/train_progressive.py \
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
  --semantic_tdom_render_tile_rows 512 \
  --semantic_checkpoint_interval 5 \
  --semantic_map_publication_interval 10 \
  --NoDebug
```

`Use_Tri_Mask` 保持关闭。修复后的实现无论是否使用三角形训练 mask，都会生成
增量采样需要的 `point3D_ids/xys/tri` 元数据；该参数只控制训练 mask，不再错误地
控制新高斯初始化。

## 完成结果

- 进程正常退出，日志终止标记为 `All Done!!!`。
- 最终训练迭代：79,800；高斯数：942,438。
- 399/399 张相机完成完整 RGB 评测和最终渲染。
- RGB：L1 0.0174595，PSNR 31.3079 dB，SSIM 0.937277，
  LPIPS 0.184639。
- 最终精确语义重建使用399张标签、180个观测组，状态为
  `exact_partial`，`semantic_through_image=000402`，`lag=0`，无 error、
  无 deferred image；posterior 为 `robust_exact_final`。
- `exact_partial` 的唯一已知原因是现有标签 metadata 缺少完整 RGB
  source-image hash binding；不是标签缺失、重建失败或在线 fallback。
- 最终地图为 9216×11264，固定网格世界范围约 221.092×270.224，
  有效 GSD 约 0.023990×0.023990。全部399个相机中心均落入地图范围。
- RGB/Semantic 使用同一 `grid_id`、`projection_geometry_hash` 和冻结几何快照。
- 最终产品包含 RGB TDOM、hard/color/unknown 语义结果，以及 coverage、
  evidence support、support ratio、entropy、Top-1、margin、mixture/group
  disagreement 和 variance proxy 诊断。
- 下采样人工查看确认 RGB 与 semantic 均覆盖完整航测区域，不再是早期错误运行的
  局部 4608×6144 区域。

最终权威入口是：

```text
/data3/xzp/data/Onthefly/outputs/semantic_atom_v2/Map_Products/latest.json
  -> stage_000399_final_refinement/pair_manifest.json
```

最终 `pair_manifest.json` SHA-256 为
`352c42b750b6082b7eb0f22894d4d30d7f2f4926da3edba53b85c20e92debefb`。
正式输出包含15个历史发布阶段；逐阶段复算16项产品 hash、两份子 manifest
hash、alignment hash、字节数以及最终 latest 指针，结果均通过。最终产品数据约
1.28 GB，整个正式输出根约12 GiB。消费者必须以
`Map_Products/latest.json` 为准，不应按目录名猜测最终产品。

旧的局部错误结果及本轮产生的 rebuild/smoke/debug 输出已于2026-09-14清理，
不再保留对应目录。其根因记录仍保留：`Use_Tri_Mask=False` 曾同时关闭增量采样
元数据，导致渐进阶段无法加入新高斯；旧图只有144/399个相机中心落入范围。

## 时间与资源

- 初始几何训练：16.192 s。
- 增量几何训练：859.108 s，记录的均摊时间为 2.153 s/image。
- 最终几何精修：47.353 s。
- 冻结几何全量语义重建：111.338 s，其中投影 101.603 s、读取标签
  8.759 s、融合 0.749 s；峰值 CUDA allocation 约3.15 GB。
- 最终399张评测约7分33秒。
- 该实测所在机器向任务暴露的可用显存高于单卡24 GB基线，不能替代24 GB
  RTX 4090上的资源验证。

## 修复和验证

1. 渐进采样元数据已与 `Use_Tri_Mask` 解耦；新增回归测试保证关闭训练 mask 时
   仍生成 `point3D_ids/xys/tri`，而 `Tri_Mask` 保持 `None`。
2. paired RGB TDOM 改为与 semantic TDOM 相同的纵向分片渲染路径，避免
   9216×11264 全图一次性 rasterize 导致 CUDA OOM。CUDA 集成测试比较分片与
   全图 RGB（含非黑背景），结果在容差内一致；调度单测覆盖分片路由。
3. 小范围 stage 30→31 smoke 在相同提交和环境下正常结束，最终产品
   4608×5632、`lag=0`、无 error/deferred，并成功保存 checkpoint 和点云。
4. 完整399张训练在同一代码上正常结束；stage 280 已成功发布
   7680×11264 中间 pair，最终发布 9216×11264，证明大图 RGB 分片修复实际生效。
5. 代码测试共143项全部通过；`python -m compileall -q src`、
   `git diff --check` 和正式产物完整性检查均通过。

## 仍未验证

- 标签 source-image binding 尚不完整，因此不能升级为 `exact_final`。
- 当前地图是无 GIS CRS 的局部固定网格，不能宣称地理坐标已校准。
- 24 GB RTX 4090目标硬件上的峰值显存和准实时预算尚未实测。
- 本次只证明训练、重建、配对渲染与产品完整性；语义准确率、阈值优劣和
  SAM 标签质量尚未评测，后续仍需结合标注或人工抽检校准 support gate。
