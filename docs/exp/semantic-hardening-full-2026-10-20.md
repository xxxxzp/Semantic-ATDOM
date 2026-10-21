# 语义模块修复后的完整 XinCao 实验，2026-10-20

本次完整运行已结束：229 张累计影像，第 30–229 阶段，45,800 次优化；最终语义状态为 `exact_final`。初始化、199 个增量阶段、最终精炼、229 视图的最终语义重建，以及视频发布均完成。以下比较使用新旧两次实际完整运行。

**结论：完整生命周期通过，RGB 重建质量基本保持，最终语义主体一致；但本次增量端到端和最终语义重建更慢，并额外确认了新旧地图共同存在的栅格列方向元数据错误。不能将这次实验概括成“全部问题已消除”或“整体没有回归”。**

## 代码、数据与控制变量

- 新代码：`fix/semantic-first-principles-20261020`，运行提交 `e2c530152788b7678424eb9b9e155a84be22cb1c`。本报告在运行结束后加入仓库，未改变运行代码。
- 基线：`/data3/xzp/experiments/Onthefly/XinCao/xincao-downsample-1612-20261016/original-to-1612-standardalloc`；旧运行提交 `3266655f9b3cda54eafddd464411143249cd44bc`。
- 新结果：`/data3/xzp/experiments/Onthefly/XinCao/semantic-hardening-full-20261020/full-original-1612`。
- 唯一数据源：`/data3/xzp/data/Onthefly/XinCao`。原始 RGB 为 4032×3024；新旧训练均为 `--resolution 1612`，实际 1612×1209。这与已有基线精确匹配，无需再跑多个分辨率。
- 原始 RGB、SfM、标签、分数、标签版本、生成模型/权重/提示词/阈值和语义类别绑定核验一致，全部 200 个阶段的相机/匹配快照一致；训练超参数核验一致。只允许新输出路径、缓存路径及新增 worker 关闭超时等运行字段不同。
- 初始化 2000 次，每新增影像局部 100 次加全局 100 次，最终 4000 次，总计 45800 次。随机种子及采样、优化参数沿用旧实验。未使用小分辨率数据集的 SfM 或标签。
- 标签保持最近邻采样；类别 ID 的比较同样只直接选取最近像素 ID。RGB 比较中的双线性只用于 RGB。
- 训练使用物理 GPU 2，离线新旧 RGB 评估都使用物理 GPU 4；均为 RTX 4090。两者均按 UUID 限定可见设备，进程内 `gpu=0` 为映射后的设备。未使用物理 GPU 1。
- 运行通过只读挂载保护原始数据、全部既有实验、主 checkout、共享 Conda 和 CUDA 扩展缓存，只开放新实验目录写入。运行前后 3365 个受保护文件哈希全部一致。

原始主 checkout 保留在 `semantic/g1-evidence-v2@cc99f13`，本次训练在独立 worktree 完成。输入及参数证据见 [protocol-verification.json](/data3/xzp/experiments/Onthefly/XinCao/semantic-hardening-full-20261020/validation/protocol-verification.json)；完整命令见 [launch.json](/data3/xzp/experiments/Onthefly/XinCao/semantic-hardening-full-20261020/launch.json)。

## RGB 重建质量

两模型均使用旧实验最终 checkpoint 的同一份 229 相机清单、姿态、内参、1612×1209 网格及掩膜，在同一物理 GPU 上离线渲染。PSNR/L1 只计相同有效像素，SSIM/LPIPS 按仓库的最终评估定义对整幅掩膜后影像计算，LPIPS 使用已有 VGG 权重。新实验最终 checkpoint 的相机、标定和逐像素掩膜也已验证与这份清单完全相同。

新模型评估读取全部 45800 次优化完成后保存的固定 RGB 模型快照，与最终语义重建并行执行；快照哈希已在实验结束后验证与最终发布 `model/gaussians.ply` 完全一致。评估及快照均保存在新实验目录中。

| 指标 | 旧实验均值 | 修复后均值 | 新−旧 | 改善的视图数 |
|---|---:|---:|---:|---:|
| PSNR ↑ (dB) | 23.307311 | 23.302038 | -0.005273 | 104/229 |
| SSIM ↑ | 0.800479 | 0.800365 | -0.000114 | 102/229 |
| LPIPS-VGG ↓ | 0.281875 | 0.281768 | -0.000107 | 120/229 |
| L1 ↓ | 0.042301 | 0.042316 | +0.000015 | 108/229 |

这些是**训练视图**的 RGB 重建指标，并非留出视图泛化指标，也不直接证明语义准确率。逐影像成对差值见 [rgb-paired-views.csv](/data3/xzp/experiments/Onthefly/XinCao/semantic-hardening-full-20261020/comparison/rgb-paired-views.csv)，全部均值、中位数和分位数见 [summary.json](/data3/xzp/experiments/Onthefly/XinCao/semantic-hardening-full-20261020/comparison/summary.json)。

未只检查平均值：逐视图 PSNR 的新−旧范围为 [-0.155283, +0.090628] dB；最小差值对应影像 `000021`。成对差值的 P05/P95 为 [-0.062272, +0.047893] dB。

![229 个相机的 RGB 指标](/data3/xzp/experiments/Onthefly/XinCao/semantic-hardening-full-20261020/comparison/rgb-metrics.png)

![固定相机的原始 RGB 与两次重建](/data3/xzp/experiments/Onthefly/XinCao/semantic-hardening-full-20261020/comparison/rgb-views.png)

## 最终语义及正射图比较

两次运行使用同一原始 SfM 世界坐标系。比较把旧图像素中心反投影到其元数据定义的世界平面，再通过新图投影矩阵映射到新图；不拟合图像配准或 Sim(3)。语义采用最近像素中心的整数类别选取，RGB 采用双线性。新图尺寸 [5632, 5120]，旧图 [5632, 5120]；旧图范围内共同栅格覆盖 99.89%。所有下述统计均限定共同栅格范围。

- 双方均已知的像素类别一致率：**97.17%**；双方均已知条件下 mIoU：**90.30%**。
- 至少一方已知的像素类别一致率：**94.63%**；将未知造成的缺失纳入后的 mIoU：**87.62%**。
- 未知比例：旧 26.95%，新 27.27%。
- 旧已知→新未知 324,920 像素；旧未知→新已知 230,457 像素。两者分别报告，不将覆盖变化直接当作准确率改善。
- 双方语义已知区域的 RGB 正射图相互 PSNR 为 18.914 dB，MAE 为 18.100/255；有效窗口 SSIM 为 0.720357。这里参照旧图，属于结果差异指标，不是 RGB 真值质量。

| 类别 | 双方已知条件下 IoU | 计入未知缺失的 IoU | 旧像素数 | 新像素数 |
|---|---:|---:|---:|---:|
| bare_soil | 93.59% | 90.78% | 1407386 | 1420265 |
| building | 94.87% | 91.42% | 4880733 | 4867123 |
| vegetation | 95.32% | 93.26% | 8503446 | 8496674 |
| paved_area | 94.49% | 92.57% | 5228151 | 5180105 |
| vehicle | 83.85% | 82.79% | 173195 | 178402 |
| water | 81.01% | 75.35% | 106873 | 89121 |
| road | 88.94% | 87.17% | 743106 | 716737 |

没有独立人工标注的正射语义真值，以上数值衡量**新旧结果一致性**。旧代码的屏幕重心坐标直接用于世界坐标会造成错误初始化，此次修复有意改变几何；新旧结果不应要求逐位一致。修复后的重投影约束此前已通过独立几何 oracle 和回归测试验证。本次单组完整实验比较的是整组修复，无法独立归因到某一项改动，也不能排除不同 GPU 浮点轨迹的影响。

![同一世界坐标下的最终 RGB、语义及差异](/data3/xzp/experiments/Onthefly/XinCao/semantic-hardening-full-20261020/comparison/map-overview.png)

图中类别变化、旧已知/新未知、旧未知/新已知分别着色；共同栅格外另行标记。可视化采用一致裁剪，类别显示不混合 ID。完整混淆矩阵、对齐矩阵及原尺寸差异图见 [map-comparison.json](/data3/xzp/experiments/Onthefly/XinCao/semantic-hardening-full-20261020/comparison/map-comparison.json) 和 [maps](/data3/xzp/experiments/Onthefly/XinCao/semantic-hardening-full-20261020/comparison/maps)。

已经查看生成的对比图：主体区域与训练视图外观相近，语义变化集中在类别边界及覆盖边缘；RGB 正射图仍有可见的局部纹理和边缘差异，不能据训练视图指标近似就声称正射图逐像素一致。图中的未知区外拖影两次均存在。水类的计入未知 IoU 约 75.35%，比主体大类低，也不能忽略少数类变化。

## 额外发现：历史栅格元数据方向错误，尚未修复

将实际投影矩阵独立反投影，得到栅格每增加一列的世界方向，再与 `grid.pixel_x_axis_world` 声明比较：新旧图的方向余弦都为 **−1**，行方向为 +1。进一步把元数据声称的 window 首像素中心投影回实际图像，得到列 **5631**、行约 0，而非列 0、行 0（图宽 5632）。这是明确的列方向矛盾，**旧基线也存在**。

按这些轴向字段直接做地理定位的外部程序可能把列方向镜像；这不改变输入 RGB、SfM、标签或掩膜的训练坐标。报告中的地图对齐只使用实际 `full_projection_transform` 和世界平面，未使用相矛盾的列方向字段，所以不受该字段影响。结构化生命周期验收通过不代表这个额外空间元数据检查也通过。

本次任务执行完整实验与比较，未在运行后改写训练实现或偷偷修补已发布产品；该问题作为后续必须关注的已有缺陷保留。独立复现脚本为实验目录 `check_grid_metadata.py`，数值证据见 [grid-metadata-crosscheck.json](/data3/xzp/experiments/Onthefly/XinCao/semantic-hardening-full-20261020/validation/grid-metadata-crosscheck.json)。

## 时间与资源

| 项目 | 旧实验 | 修复后 |
|---|---:|---:|
| 初始化几何训练 (s) | 30.096 | 30.492 |
| 增量几何平均 (s/新影像) | 3.316 | 3.313 |
| 增量端到端平均 (s/新影像) | 6.280 | 8.300 |
| 最终几何精炼 (s) | 88.612 | 92.532 |
| 完整运行墙钟 (s) | 2349.362 | 3664.327 |
| 最终语义重建 (s) | 730.136 | 1523.899 |
| 其中投影 (s) | 692.739 | 1469.773 |
| 其中读取标签 (s) | 36.135 | 52.933 |
| 最终 Gaussian 数 | 1011973.000 | 1010568.000 |
| 增量 CUDA 峰值 allocated (GiB) | 3.094 | 2.838 |
| 增量 CUDA 峰值 reserved (GiB) | 7.549 | 7.832 |
| 增量端到端 p50 (s/新影像) | 6.899 | 9.041 |
| 增量端到端 p95 (s/新影像) | 9.452 | 14.058 |
| 增量端到端 p99 (s/新影像) | 11.724 | 16.600 |

完整 199 个增量阶段的记录均已落盘，stage 31–229 每次新增一张影像；包括几何训练、语义等待、投影入队、视频等待/编码入队和 checkpoint 时间。记录见新实验 `metrics/timing.json` 和汇总 JSON。阶段时间不含异步最终重建的全部成本，故最终语义重建与完整墙钟另外列出。单阶段 CUDA 峰值以实际记录定义为准，不等同于整台机器所有进程的显存。

旧训练使用物理 GPU 3，新训练使用物理 GPU 2；CPU/存储负载、文件缓存和调度也可能不同。这些数字说明本次运行更慢，尚不能据此把差异归因于某项代码改动，或认定存在稳定的性能变化。

本次观察到的增量端到端平均耗时增加 **32.2%**，最终语义重建增加 **108.7%**；其中投影累计从约 693 秒变为 1470 秒，是记录中的主要差异。几何训练平均值基本不变。对质量的结论不能抵消这个实测耗时问题。

已核对两次归档的软件包版本一致；`CudaSemanticProjector` 与 `label_boundary_weights` 的语法树也一致。这只说明这些函数本体未改变，不能排除调用方式、几何或运行资源对耗时的影响。收尾期间进行了三次 GDB 调用栈采样，造成短暂调试停顿，因此本次墙钟也不是无干扰的性能基准。最后一次 Python 栈处于重建后的输入绑定校验/PIL 标签读取；这个采样不能独立解释投影时间增长，未据此声称已定位根因。证据见 [runtime-comparison-diagnostics.json](/data3/xzp/experiments/Onthefly/XinCao/semantic-hardening-full-20261020/validation/runtime-comparison-diagnostics.json) 和实验目录 `validation/trainer-python-stack.log`。

![逐阶段时间、Gaussian 数和 CUDA 峰值](/data3/xzp/experiments/Onthefly/XinCao/semantic-hardening-full-20261020/comparison/runtime.png)

## 完整性验收

- manifest 成功：最终 stage 229，45,800 次优化，229 累计相机。
- 最终 sync 为 `exact_final`，无错误、无 deferred 影像、lag 为 0。
- 最终重建 229 张影像，183 个组；所有影像恰好覆盖一次。
- 语义状态 schema 5 可通过公开 loader 重载，几何/上下文哈希一致；闭组诊断来源 `exact_final`，无残留 open group。
- 最终归档 schema 与实际融合类别指纹和输入绑定一致；preflight 内容哈希与最终绑定一致。
- 成对 RGB/语义 map 发布完整，指针及 9 个产品哈希核验通过，尺寸 [5632, 5120]。
- H.264 视频 2562×1080，120 帧；最终帧为 stage 229 / iteration 45800 / `final_refinement` / `exact_final`，全片使用同一 grid。
- 新旧两份视频均已通过 ffmpeg 全片解码；旧基线的最终成对产品文件哈希也逐项核验通过。
- 最终相机姿态、内参、尺寸、标定及掩膜与基线一致；输入及既有结果哈希保持不变。

以上是完成状态、输入身份、模型状态及文件完整性检查。**栅格轴向的额外几何一致性检查发现历史问题，详见上节，不纳入“全部空间契约通过”的表述。**

结构化验收见 [full-artifact-verification.json](/data3/xzp/experiments/Onthefly/XinCao/semantic-hardening-full-20261020/validation/full-artifact-verification.json)，受保护文件核验见 [completion.json](/data3/xzp/experiments/Onthefly/XinCao/semantic-hardening-full-20261020/completion.json)。视频位于 [training_progress.mp4](/data3/xzp/experiments/Onthefly/XinCao/semantic-hardening-full-20261020/full-original-1612/videos/training_progress.mp4)。

## 复现与证据

实验目录中的 `run_full.py` 是实际全程执行器；`verify_protocol.py` 核验数据和训练参数；`evaluate_rgb.py`、`run_rgb_eval.py` 在相同 GPU 和相机清单上评价两模型；`compare_maps.py` 执行世界坐标对齐；`verify_full.py` 验收最终状态与产品；`summarize_comparison.py` 汇总 JSON/CSV/PNG/PDF，`write_report.py` 生成本报告。

`comparison-selftest.json` 保存比较器的恒等/已知平移/离散标签/混淆矩阵自检。分析脚本与命令均保存在独立实验目录，不修改训练实现、原始数据、共享环境或旧结果。

## 2026-10-21：性能重放补充

对相同原始标签、相同相机/标定/掩膜和同一物理 GPU 4，分别载入旧、新最终 Gaussian 模型，选取影像 000001、000115、000229；排除首次初始化，交替顺序各重复两次。该诊断不修改原始数据、已有训练结果、Conda 或 CUDA 扩展。

- 旧模型投影中位数 1.530201 秒，新模型 1.550867 秒，新/旧约 1.0135；这个三视图重放未复现完整实验约两倍的差距，不足以证明任何全局稳定的速度结论。
- CPU 原始分辨率边界权重计算中位数约 1.380 / 1.388 秒，约占投影总时间 90%；CUDA wrapper/kernel 约 0.0143 / 0.0140 秒。该 CPU 处理存在重复计算，属于后续优化目标，但其函数本体在新旧版本之间没有改变。
- 增量端到端总时间增加 402.015 秒，其中等待语义结果增加 392.844 秒，约占新增耗时 97.7%。视频帧入队耗时两次均约 0.004 秒，不能将变慢直接归咎于新增的有界视频队列。
- 这缩小了耗时范围，尚未确定完整运行中投影慢约两倍的触发条件；三视图重放不能排除其余视图、worker 调度、资源负载或其他调用上下文的作用。新增严格校验会产生开销，但不能独自解释投影计时内的全部增加。
- 本次未启用平面语义回填（`semantic_tdom_planar_fallback_class` 为空），因此平面回填筛选一致性修复不能解释本次画质变化。

结构化数据与执行脚本：[projection-replay.json](/data3/xzp/experiments/Onthefly/XinCao/semantic-hardening-full-20261020/validation/projection-replay.json)、[replay_projection.py](/data3/xzp/experiments/Onthefly/XinCao/semantic-hardening-full-20261020/validation/replay_projection.py)。这是归因诊断，未修改生产代码，也未将性能问题标记为已解决。
