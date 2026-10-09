# 当前数据与实验状态核对

核对日期：2026-10-09

当前代码基线：`semantic/g1-evidence-v2` / `0665f4ba911ddb09001aa63618c729309011d90d`

范围：本机输入目录、现存 run manifest、预检记录、最终模型与地图、视频元数据

## 1. 如何解读本记录

本文件维护当前可核验的数据和实验事实；数学要求以
[语义合同](../design/semantic-tdom-v1.md)为准，代码缺陷以
[第一性原理审查](../problem/first-principles-code-review-2026-10-09.md)为准。

本次只读盘点重新统计交付文件，读取运行与预检记录，并复算下述两次地图发布的
pair、RGB/Semantic 子 manifest 和各 9 项产品文件的 SHA-256；视频通过 ffprobe
核对编码、尺寸、帧数和帧率。未重新执行全量训练，也未重新对全部源 RGB/标签做
完整 dataset preflight。source binding 的完整性结论来自相应运行的预检记录，当前
metadata 的绑定字段存在性另行检查。

运行目录名中的短 SHA 只表示当时的 Git HEAD。以下三个运行的 manifest 均记载
`git.dirty=true`，不能把目录名当作可完全复现的干净源码身份；也不能将历史成功
外推为当前 `0665f4b` 已完成相同配置的全量验收。

## 2. 当前输入与计数

| 项目 | Phantom3 | UAVScenes AMtown01 |
|---|---|---|
| 场景根目录 | `/data3/xzp/data/Onthefly/phantom3` | `/data3/xzp/data/UAVScenes_Onthefly/interval5_AMtown01` |
| 渐进几何目录 | `GaussianSplatting/` | `GaussianSplatting/` |
| 当前数字阶段范围 | 14–414，共 401 个目录 | 13–487，共 475 个目录 |
| RGB 目录 | `images/` | `interval5_CAM/` |
| 语义交付父目录 | 场景根下 `semantic_labels/` | 场景根下 `outputs/semantic_inputs/uavscenes_amtown01_observed_v1/` |
| 标签 PNG / metadata JSON 数 | 414 / 414 | 490 / 490 |
| metadata 中有 source hash 的数量 | 414 | 490 |
| metadata 记录的影像尺寸 | 1870×1049 | 2448×2048 |
| 类别数 / ignore ID | 8 / 255 | 15 / 255 |
| label version | `phantom3-proc-sam3-v1` | `uavscenes-amtown01-observed-v1` |
| 下述全量运行的最终有效身份数 | 414 | 487 |

阶段编号、阶段目录数、标签交付数、最终累计相机数、视频帧数是不同计数。
AMtown01 的 490 份标签并不表示 13–487 训练消费了 490 个相机；最终预检与相机
清单是 487。Phantom3 的 399 张、1600×897、早期缺失 source binding 记录对应
2026-09-14 的旧输入与运行，不能继续用于描述当前交付。

Phantom3 使用六位编号；AMtown01 使用带小数的时间戳身份。代码只剥离已知图像
后缀，保留时间戳小数部分；不得把六位编号作为整个项目的通用身份要求。
`persistent_union` 用于保留暂时从阶段快照省略的历史相机；阶段本地快照不等于
最终累计相机清单。

## 3. Phantom3：带地图和视频的已核验运行

```text
/data3/xzp/data/Onthefly/phantom3/outputs/runs/
  phantom3-v2-progress240-fps1_20260921-180740_c886ce6
```

- manifest：`succeeded`；分支 `semantic/g1-evidence-v2`，HEAD `c886ce6`，dirty 为 true。
- 最终阶段 414，累计优化 86,000 次，高斯数 677,723。
- 最终重建 414 张影像、189 个组；状态 `exact_final`、lag=0、errors/deferred 为空。
- 训练集 RGB 指标：L1 0.01835777、PSNR 31.057830 dB、SSIM 0.890348、LPIPS 0.267047。
- 最终地图宽×高 9728×12288；pair、两份子 manifest、9/9 产品文件 hash 本次复核通过。
- pair manifest SHA-256：
  `a90e8a3812ad75e93fc88adae763d77eec456fa650eab53e2ccd355493d7cba4`。
- `model/cameras.json` 有 414 个身份，`renders/cameras/` 有 414 张 JPEG。
- 视频为 H.264、1958×1080、240 帧、1 fps。
- 预检报告 SHA-256：
  `38538a2c84fea2c8036a30a4119cbe3ea0d2174410a719742ad0c3786682b6f4`；
  记录 `labels_complete=True`、`source_binding_complete=True`。

这证明此运行的记录与现存发布文件一致，不证明独立语义真值精度，也不消除审查
发现的几何采样、诊断声明或恢复问题。局部网格没有 GIS CRS，GSD/尺度字段不应
自动解释为已验证的地理坐标或米制精度。

## 4. Phantom3：当前 outputs/latest 指向实验分支

```text
/data3/xzp/data/Onthefly/phantom3/outputs/latest
  -> runs/phantom3-scale-lifecycle-e0-resume273-v2_20260922-182208_bf6a7c9
```

该运行来自 `fix/phantom3-scale-lifecycle-v1` 的 dirty worktree，最终阶段 414、
86,000 次优化、高斯数 683,318，语义重建状态 `exact_final`。
RGB 指标为 L1 0.01835491、PSNR 31.071894 dB、SSIM 0.890553、LPIPS 0.267132。

命令包含尺度生命周期 telemetry、诊断 checkpoint 和 stage 273 恢复。
它没有请求 `render_TDOM`、逐相机渲染或训练视频；现存结果没有地图 latest 指针，
也没有逐相机 JPEG。不能仅凭 `outputs/latest` 就把它当作主线成对地图发布的验收。
该 dirty 实验分支的恢复成功也不证明当前主线 R01 已修复。

## 5. AMtown01：最新成功运行与 160 帧视频

```text
/data3/xzp/data/UAVScenes_Onthefly/interval5_AMtown01/outputs/latest
  -> runs/amtown01-full-progress160-fixed_20260930-134152_bf6a7c9
```

- manifest：`succeeded`；分支 `semantic/g1-evidence-v2`，HEAD `bf6a7c9`，dirty 为 true。
- 最终阶段 487，累计优化 100,800 次，高斯数 1,030,412。
- 最终重建 487 张影像、314 个组，用时约 784.707 秒；状态 `exact_final`、lag=0，
  errors/deferred 为空。在线/最终类别一致率约 0.856493，是两种状态的比较，非真值准确率。
- 训练集 RGB 指标：L1 0.04043133、PSNR 24.797242 dB、SSIM 0.624908、LPIPS 0.509098。
- 最终地图宽×高 5632×9728；pair、两份子 manifest、9/9 产品文件 hash 本次复核通过。
- pair manifest SHA-256：
  `9ddcf352b888158f6cc0f0fba904d63dda7973b1df0d786a20c41acb72eee014`。
- `model/cameras.json` 有 487 个身份，`renders/cameras/` 有 487 张 JPEG。
- timeline 记录 160 帧；ffprobe 确认 H.264、1540×1080、160 帧、4 fps。
- 预检报告 SHA-256：
  `f88b62ff13b0f2e5e4383f35307d395d32de3a7788093decde5f2179e4e411de`；
  记录 `labels_complete=True`、`source_binding_complete=True`。

[159 帧实验参考](../exp/amtown01-progressive-semantic-video-159.md)保留其计划参数；
它不能作为上述 160 帧运行的准确命令记录，更不能说明 AMtown01 尚无全量成功结果。
[AMtown01 适配记录](uavscenes-interval5-amtown01-v2.md)中的 r8/r9 和 2026-09-16
指标继续作为各自的历史实验记录，不覆盖本次盘点的最新运行。

## 6. 实验状态与实现状态的边界

- 当前代码审查的 245 项测试通过，仍有 R01–R09 待处理；没有本次新全量训练。
- `exact_final` 表示相应版本的最终证据重建与输入资格，不表示模型准确率已校准，
  也不证明 R08 的完整生成配置绑定已经实现。
- scale-lifecycle worktree 的实验参数不属于当前主线公开接口，不能照搬到主线。
- 若一个历史目录仍记载 `running`，本次不据此断言进程正在运行或训练已完成；
  需另外核对进程、日志及产物。本次未修改任何实验 manifest 或 latest 指针。
- 未重新验证所有历史输出、当前输入全部内容哈希、独立语义真值、米制/GIS 精度，
  或固定 24 GB 单卡资源与准实时 SLO。成功运行不能替代这些验收。

## 7. 后续维护

新增或修改实验结论时必须同时记录 run 路径、代码身份和 dirty 状态、实际相机/标签
集合、输入版本、请求的产物和验证方法。`latest` 是可变指针，引用实验结果须保留
不可变 run 名；计划命令与已执行命令分开记录。文档补全不得把待修复问题标为已完成。
