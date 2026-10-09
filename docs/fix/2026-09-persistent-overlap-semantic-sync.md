# 局部 overlap 快照与累计相机的语义同步修复

日期：2026-09-30。

## 根因与复现

UAVScenes interval5_AMtown01 的 stage 18 overlap 导出暂时省略两张已观测影像，
stage 19 仍省略其中一张。`persistent_union` 几何训练保留这些相机、RGB 和历史
语义证据，这是该 identity policy 的正常行为。

旧语义同步却要求当前 overlap 名单包含所有累计相机；只要省略一个旧 identity，
就禁止消费全部新标签。视频采样等待 worker 空闲后仍得到未追上的语义状态，
因此 stage 19 的请求显式失败。这个错误发生于全分辨率训练，与 GPU 透传及
降分辨率无关。

## 第一性原理与修复

- 相机、标签和 Gaussian 几何决定一张影像能否产生语义投影。
- overlap 只用于判断新观测与开放组 anchor 的相关性；它不是相机存活清单。
- 在线接收集合和连续记账边界必须由实际累计相机决定，不能随局部导出缩小。
- 删除对所有历史 identity 的整体 overlap 完整性要求，仍使用真实矩阵查询
  分组所需的 anchor/new-image 关系。所需关系未知时显式延后该观测，不补造
  零重叠、不丢失 ledger 记录，也不使用未来阶段数据。
- 最终精确重建继续使用最终权威 overlap 图与 preflight 输入绑定；视频继续
  要求同一训练边界，在线历史近似仍如实标记为 `stale_online`。

## 验证

`tests/unit/test_semantic_persistent_overlap.py` 直接覆盖真实 hook 同步入口：局部
名单省略无关旧相机、无新增观测时保留累计边界、必要 anchor 关系缺失后补齐、
已满组无需额外查询 overlap。未知必要关系的失败路径保持不变。

全套测试在 `semantic-atdom` 环境及 GPU 上通过：243 passed。原数据集 475 个
阶段使用真实 overlap 和 metadata 做 CPU orchestration 回放，全部 487 个
identity 恰好记账一次且 lag 为 0；每隔三个阶段消费一次累计相机的回放也通过。
该回放使用固定的单 Gaussian 投影，只验证时序、分组、记账和边界，不替代真实
CUDA 投影、训练质量或最终视频验收。

真实全量训练已重新启动，160 帧、4 FPS，独立输出目录为：

```text
/data3/xzp/data/UAVScenes_Onthefly/interval5_AMtown01/outputs/runs/amtown01-full-progress160-syncfix_20260930-121920_bf6a7c9
```

该运行记录 `git.dirty=true`，本次实现差异保存在
`provenance/source_changes.diff`；本记录不宣称全量训练或最终视频已经完成。

## 最终保存接口的第二个独立错误

上述全量运行完成全部 100800 次优化、475 个阶段、487 个相机和最终精确语义
重建，但在最终相机清单序列化时失败。旧 `camera_to_JSON` 读取 CameraInfo 的
`width/height/FovX/FovY`，实际训练相机 Camera 使用
`image_width/image_height/FoVx/FoVy`。已有测试伪造的是前一种字段结构，未覆盖
真实训练对象。

新增真实 Camera 构造回归测试复现 AttributeError；另一个测试复现元数据失败
使模型根本没有保存的顺序问题。先观察两项测试失败，再修复为按真实对象类型
适配字段，并用其实际像素网格计算焦距。先保存完整 Gaussian PLY，再写最终
相机清单；模型持久化提前至最终优化完成后，位于衍生地图、语义重建和评估
之前。仍严格验证所有最终产品，不将收尾异常伪装成成功。

GPU 全套测试为 245 passed。第二次运行的最终地图、精确语义状态和 160 帧真实
训练视频保留；视频已通过 ffprobe 验证为 H.264、1630×1080、4 FPS、40 秒。
该次失败运行没有完整 RGB Gaussian 模型，语义几何 checkpoint 不包含颜色 SH
参数，不能替代 RGB 模型；因此完整成功运行需要重新训练，不能只修改 manifest。
真实 GPU stage 13–22 小规模端到端验证成功：真实相机清单、完整 RGB 模型、
22 张相机渲染、SSIM/LPIPS 评估、exact_final 语义地图、5 帧视频和最终 manifest
发布均完成，状态为 succeeded。新的完整运行开启每 50 阶段恢复检查点及最终
训练状态保存，目录为：

```text
/data3/xzp/data/UAVScenes_Onthefly/interval5_AMtown01/outputs/runs/amtown01-full-progress160-fixed_20260930-134152_bf6a7c9
```

该运行在 2026-09-30 15:02:55（Asia/Shanghai）成功完成，manifest 为 succeeded、
进程退出码为 0，outputs/latest 已指向该运行。475 个渐进阶段、487 张相机和
100800 次优化全部完成，最终模型含 1030412 个 Gaussian。

独立验收重新加载 RGB PLY 与最终训练状态，模型完整 RGB 哈希和投影几何哈希
均与最终语义状态及地图绑定一致；16 个 manifest 记录文件、9 个地图产品的
哈希验证通过，487 张相机渲染及 per-camera 记录齐全。视频首帧、阶段中点和
最后一帧目视检查通过，全部 160 帧使用同一地图网格、边界滞后为 0、无语义
错误，末帧是 exact_final。最终 H.264 视频为 1540×1080、4 FPS、40 秒。

训练集完整评估：PSNR 24.79724、SSIM 0.62491、LPIPS 0.50910、L1 0.040431。
这些是训练相机的重建指标，不代表独立测试集泛化能力。最终地图为 5632×9728。
验收记录保存于该成功运行的 provenance/acceptance.json；修复代码与测试差异
保存于 provenance/source_changes.diff。旧失败运行如实保留，未改写其失败状态。

## 2026-10-09 提交前复核

当前实现及回归测试的完整 diff 与上述成功全量训练保存的
`provenance/source_changes.diff` 完全一致。提交前重跑完整测试集合：
212 passed、33 skipped、3 warnings；当前执行环境无法初始化 CUDA，GPU 相关
测试被跳过，不将本次结果宣称为新的 GPU 验证。此前全套 245 项通过及真实
GPU 全量训练的验收记录仍保留，且对应相同的代码差异。
