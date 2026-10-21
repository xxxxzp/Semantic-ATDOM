# 语义处理耗时优化与完整验证，2026-10-21

完整原始 XinCao 实验已成功完成：229 张累计影像、stage 30–229、45800 次优化、最终 `exact_final`、120 帧 H.264 视频。代码运行提交为 `d6df0c179c5b3c9a30127cd71fd3e224a1fafa30`，分支 `fix/semantic-first-principles-20261020`。主 checkout 保持 `cc99f13`。

增量端到端平均耗时由修复后的 **8.300 秒/帧变为 4.543 秒/帧**，每帧差值 -3.757 秒，耗时下降 45.26%。相对更早的 6.280 秒/帧基线，差值为 -1.736 秒/帧。下表保留三组实测值，避免只选择较慢的对照。

## 改动及正确性边界

这次生产代码只增加两项行为：

1. 加速 `label_boundary_weights` 的精确欧氏距离计算。在带宽不超过 128、栅格尺寸不超过 32768 时使用 OpenCV 的精确 L2 变换；像素格点到边界的距离平方为整数，在有效带内从 float32 距离恢复整数平方，再按原来的 float64 sqrt/ramp 计算最终 float32 权重。超过范围或特殊参数继续走原 SciPy 路径。
2. 在 GaussianTrainer 进程启动、语义 executor 创建之前，将 OpenCV 线程数设为已有 Torch/OMP CPU 预算。本次为 4；投影调用过程中不修改全局线程设置。

原始 RGB、SfM、标签、掩膜及相机映射不变，类别 ID 继续最近邻选取。没有新增持久缓存、削弱输入身份校验、修改 CUDA 扩展、减少训练迭代或视频采样。此前的透视采样、薄三角形数值稳定性、有界 worker/快照、闭组诊断、严格输入绑定等正确性修复全部保留。状态仍为 schema 5。

[OpenCV 官方说明](https://docs.opencv.org/4.10.0/d7/d1b/group__imgproc__misc.html)明确区分精确 L2 距离与近似 chamfer 距离；本次使用 `DIST_MASK_PRECISE`，并非近似的 3×3/5×5 距离。

## 可重复的性能与权重检查

- 实际 4032×3024 原始标签上的公开函数预算检查，改动前中位数 2.533833 秒，超过 0.75 秒预算而失败；改动后 0.468115 秒，通过预算检查。原型中的约 0.25 秒只计算核心部分，不与完整函数混用。
- 对同一份 229 个最终相机的标定与掩膜，逐帧交替执行旧、新实现；**全部原始网格和训练网格权重逐位一致**，类别 ID 的最近邻映射未改。两种实现的逐帧计时中位数分别为 1.433322 / 0.168191 秒；该批验证用了两个并发验证线程，属于成对微基准，与整场墙钟分开解释。
- 测试全集由两个不重叠分区执行：224 项测试和 68 个子测试、另 53 项根目录测试，合计 **277 项测试及 68 个子测试通过**，包含现有 CUDA 集成检查。
- 独立数值用例覆盖斜边、稀疏边界、未知区域、只读/非连续数组、单行/单列、不同带宽、整数 ID 超过 255、特殊参数及 SciPy 退回路径。

证据：[全部权重比较](/data3/xzp/experiments/Onthefly/XinCao/semantic-performance-20261021/all-boundary-weights.json)、[改动前预算](/data3/xzp/experiments/Onthefly/XinCao/semantic-performance-20261021/before-boundary-budget.json)、[改动后预算](/data3/xzp/experiments/Onthefly/XinCao/semantic-performance-20261021/after-boundary-budget.json)、`all-tests.log` 和 `remaining-root-tests.log`。

## 完整运行的耗时对比

| 项目 | 原始降采样代码 | 正确性修复后 | 本次耗时优化后 |
|---|---:|---:|---:|
| 几何训练 (s/新增影像) | 3.316 | 3.313 | 3.255 |
| 增量端到端 (s/新增影像) | 6.280 | 8.300 | 4.543 |
| 视频采样前语义等待 (s/新增影像) | 2.368 | 4.342 | 0.674 |
| 最终语义重建 (s) | 730.136 | 1523.899 | 173.096 |
| 其中投影环节 (s) | 692.739 | 1469.773 | 142.483 |
| 其中读取标签 (s) | 36.135 | 52.933 | 29.791 |
| 完整运行墙钟 (min) | 39.156 | 61.072 | 24.087 |

语义等待是 `_capture_training_progress_tdom` 在视频采样阶段调用 `semantic_hook.wait_for_idle()` 的实测阻塞时间；平均值分母均为全部 199 个增量阶段，包含没有视频请求而等待为零的阶段。它不是标签模型分割推理时间，也不代表每一帧固定等待同样久。上次正确性修复后，该项从 2.368 增加到 4.342 秒/新增影像；增量总耗时增加约 2.020 秒，其中约 1.974 秒来自这项等待。几何优化后的最终全量语义重建另行计时，不计入这里的增量平均值。

这些日志证明上次整场运行出现了语义等待退化，但不能单凭先后关系确定某一处修复是唯一原因。此前同 GPU 的成对三视图回放没有复现两倍投影差距，旧、新版本当时的边界权重函数体也相同；原始记录与诊断见前一次实验的 `runtime-comparison-diagnostics.json`。本次处理的是受控回放中稳定测到的 CPU 边界权重瓶颈，再通过完整运行检验改善，没有把未证实的历史原因写成结论。

原始降采样运行使用物理 GPU 3；正确性修复与本次优化运行均使用物理 GPU 2。GPU 均为 RTX 4090。三次运行的原始数据和训练参数相同，本次已重新核对相机/匹配文件、原始 RGB/标签/分数/生成配置及实际 trainer 参数。完整训练期间没有并行启动额外的模型评估，也没有调试器暂停；所有离线 RGB 评估在训练结束后执行。

这是各次完整运行的实测值，共享 CPU、存储负载及不同日期仍可能影响数值。受控函数测试证明本次算子本身更快且输出一致；整场改善幅度不能被当成所有场景的固定收益。199 个新增影像阶段的耗时记录完整保留在 `full-original-1612/metrics/timing.json`，P50/P95/P99 及视频语义等待时间可在结构化结果中核查。

![逐阶段耗时与资源](/data3/xzp/experiments/Onthefly/XinCao/semantic-performance-20261021/comparison/runtime.png)

## 重建结果对比

RGB 使用相同 229 个相机、原始 RGB、1612×1209 像素网格、标定与掩膜评估；相机清单 SHA256 已验证完全相同。原始/正确性修复参考指标取自之前同一 GPU 4、同一清单的已保存评估；本次也在物理 GPU 4 使用相同评估实现。复制参考值的来源和 SHA256 已留档，未改写旧实验。

| 指标 | 原始降采样代码 | 正确性修复后 | 本次耗时优化后 | 相对正确性修复后的差值 |
|---|---:|---:|---:|---:|
| PSNR ↑ (dB) | 23.307311 | 23.302038 | 23.305209 | +0.003171 |
| SSIM ↑ | 0.800479 | 0.800365 | 0.800562 | +0.000197 |
| LPIPS-VGG ↓ | 0.281875 | 0.281768 | 0.282076 | +0.000307 |
| L1 ↓ | 0.042301 | 0.042316 | 0.042306 | -0.000010 |

这组均值接近：PSNR 增加约 0.00317 dB，SSIM 增加约 0.000197，LPIPS 增加约 0.000307（略差），不能据此宣称质量全面提高。固定视图与地图总览已人工查看，主要结构一致，差异集中在纹理细节、类别边缘和地图外围。

这些是训练视图重建指标，不是留出视图泛化指标。逐影像成对差值及范围见 [summary.json](/data3/xzp/experiments/Onthefly/XinCao/semantic-performance-20261021/comparison/summary.json) 和 [rgb-paired-views.csv](/data3/xzp/experiments/Onthefly/XinCao/semantic-performance-20261021/comparison/rgb-paired-views.csv)。输入权重逐位一致证明这次优化保留了语义计算定义；独立完整训练仍可能受 GPU 浮点累计及优化轨迹影响，不能承诺最终模型逐位一致。

![训练视图指标](/data3/xzp/experiments/Onthefly/XinCao/semantic-performance-20261021/comparison/rgb-metrics.png)

![固定训练视图](/data3/xzp/experiments/Onthefly/XinCao/semantic-performance-20261021/comparison/rgb-views.png)

最终地图与正确性修复结果按同一原始 SfM 世界坐标和实际投影矩阵对齐，类别 ID 只选取最近像素。共同已知区域的类别一致率为 **97.22%**，至少一方已知范围的一致率为 **94.65%**；计入未知缺失的 mIoU 为 88.53%。未知比例为 27.07% / 27.17%。这些衡量新旧一致性，不代表独立人工语义真值上的准确率。

![最终地图与差异](/data3/xzp/experiments/Onthefly/XinCao/semantic-performance-20261021/comparison/map-overview.png)

此前发现的地图 grid 列方向元数据与实际投影方向相反的问题仍独立保留，本次没有混入修复或改写旧地图。本次空间对齐使用实际投影矩阵，不使用该错误轴向字段。其复核结果见 `validation/grid-metadata-crosscheck.json`。

## 完整性与保护验证

- 最终 stage 229 / 45800 次优化，229 张相机，无 errors/deferred，lag 为 0；最终语义重建覆盖 229 张影像、183 组。
- schema-5 状态通过公开 loader 重载，归档类别指纹、输入绑定、几何/上下文哈希一致，诊断来源为 `exact_final`，无 open group 残留。
- 最终地图 [5632, 5120]，发布指针及全部 9 项产品哈希核验通过。
- 视频 2562×1080、120 帧，已全片解码；最后一帧对应 stage 229 / iteration 45800 / `final_refinement` / `exact_final`。
- 原始数据、两份已有实验和主 checkout 的 3406 个受保护文件哈希前后相同。训练/检查进程把原始数据、既有实验、共享 Conda 和扩展缓存挂为只读，仅本次新实验目录可写；物理 GPU 1 未参与。

完整执行命令见 [launch.json](/data3/xzp/experiments/Onthefly/XinCao/semantic-performance-20261021/launch.json)，输入及参数验证见 [protocol-verification.json](/data3/xzp/experiments/Onthefly/XinCao/semantic-performance-20261021/validation/protocol-verification.json)，产品验收见 [full-artifact-verification.json](/data3/xzp/experiments/Onthefly/XinCao/semantic-performance-20261021/validation/full-artifact-verification.json)，受保护文件核验见 [completion.json](/data3/xzp/experiments/Onthefly/XinCao/semantic-performance-20261021/completion.json)。
