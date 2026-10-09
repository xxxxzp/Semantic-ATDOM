# 官方位姿与 LiDAR 驱动的渐进式 3DGS 适配指导

状态：设计与开发指导；尚未实现，尚未声明质量提升  
记录日期：2026-09-17  
目标分支：`feature/uavscenes-interval5-amtown01`  
记录基线提交：`d497459de73290015989ff3fcb789f7f2d3d0dd1`  
关联文档：`docs/problem/progressive-training-performance-bottlenecks.md`

## 1. 执行摘要

当前项目的核心是让 3D Gaussian Splatting 模型随影像到达持续增长和优化，而不是在
3DGS 内部估计相机位姿。现有实现把
`GaussianSplatting/<stage>` 中的 COLMAP/SfM 快照同时当作阶段时序、相机参数、
稀疏几何和影像重叠关系的载体，因此数据加载层与上游 SfM 绑定，但 Gaussian
优化器本身并不要求位姿和初始化点必须由 SfM 产生。

本指导建议保留现有 `GaussianSplatting/13...487` 阶段结构和影像到达顺序，通过
可配置输入 provider 覆盖其中的几何权威来源：

- 阶段时序、影像身份和第一轮语义 overlap 继续来自现有目录；
- 相机位姿改由 `sampleinfos_interpolated.json` 提供；
- 相机模型改由 UAVScenes 官方标定提供，并对 RGB、语义标签和 LiDAR 投影执行同一
  套整流；
- 新增几何改由与新影像同步的 LiDAR 扫描提供；
- 现有 SfM 特征轨迹可以在 pose-only 实验中继续用于重新三角化，但不再作为位姿
  真值；
- Terra 全局模型只作为离线质量上界，不进入严格在线主实验。

这一方案保持以下渐进式不变量：模型状态跨阶段延续、每阶段只接收截至当前时刻的
观测、相机身份稳定、新增几何按阶段注入、最终仍执行一致的 RGB/语义优化和发布。
它不能在没有说明的情况下继续被称为“从原始 RGB 端到端在线估计 pose”，因为官方
位姿是数据集提供的离线标定结果。建议将这条实验线明确命名为
`GT-pose-assisted progressive 3DGS` 或“官方位姿辅助的渐进式 3DGS”。

## 2. 问题边界与非目标

### 2.1 本文解决的问题

1. 当前渐进训练能否在不丢失阶段语义的前提下使用更可靠的官方 pose 和 LiDAR；
2. 哪些现有目录内容是渐进式训练的必要契约，哪些只是 SfM 实现细节；
3. 官方坐标、LiDAR 局部坐标、相机坐标和训练坐标应如何形成唯一变换链；
4. 如何限制 LiDAR 初始化点数量，避免几何质量提升反而造成训练速度和显存回归；
5. 如何设计逐步实验，分别验证 pose、点云和额外影像的影响；
6. 哪些结果可以用于严格在线结论，哪些只能作为带先验或离线上界。

### 2.2 本文不直接解决的问题

- 不在本阶段实现 FAST-LIVO2、GNSS/INS 或其他在线位姿估计器；
- 不把完整 Terra 模型包装成没有未来信息的在线输入；
- 不覆盖或原地修改现有只读 `GaussianSplatting` 数据；
- 不假设官方位姿必然改善最终图像指标，必须通过同帧、同迭代 A/B 实验验证；
- 不以降低高斯上限、降低图像分辨率或减少训练迭代冒充等价的性能优化；
- 不在尚未完成小范围验证前启动 13--487 或 13--490 全量训练。

## 3. 当前输入契约为什么绑定 SfM

### 3.1 路径解析

`tools/train_progressive.py` 调用
`resolve_progressive_dataset_layout()`。当传入场景根目录且存在
`GaussianSplatting` 子目录时，解析器会把该子目录设为 `progressive_root`，共享
`*_CAM` 目录仅用来提供 RGB。

相关代码：

- `tools/train_progressive.py:350-357`
- `src/semantic_atdom/scene/progressive_layout.py:73-113`

随后训练入口把数字子目录转换成 `source_path_list`，所以当前阶段序列由
`GaussianSplatting/<number>` 决定，而不是由 JSON、LiDAR 时间戳或 RGB 数量决定。

### 3.2 每阶段实际读取的内容

当前 `SingleImage_readColmapSceneInfo_Part1()` 在新格式下固定读取：

```text
<stage>/bin/images.bin
<stage>/bin/cameras.bin
```

它从 `images.bin` 获得相机 pose、二维特征、`point3D_id` 和影像身份，从
`cameras.bin` 获得相机模型和内参。相关代码位于
`src/semantic_atdom/scene/dataset_readers.py:893-961`。

`SingleImage_readColmapSceneInfo_Part2()` 固定读取：

```text
<stage>/bin/points3D.bin
```

然后依赖二维特征、Delaunay 三角网和二维—三维 ID 对应生成新增初始化点。相关代码
位于 `src/semantic_atdom/scene/dataset_readers.py:963-1144`。

因此，任意 LiDAR 坐标不能直接覆盖 `points3D.bin`。如果三维点 ID 不再与
`images.bin` 的二维观测一致，当前采样链会得到错误关联或空点集。

### 3.3 语义预检的额外要求

启用语义时，当前预检要求每阶段同时存在：

```text
images.bin
cameras.bin
points3D.bin
match_matrix_image_name.txt
match_matrix_value.txt
```

相关代码位于 `src/semantic_atdom/semantics/preflight.py:185-220`。其中 overlap
名称集合还承担累计影像身份契约。AMtown01 存在临时省略历史身份的阶段，正式训练
使用 `persistent_union` 保留已经出现的相机，不能退回严格快照删除语义。

### 3.4 训练器本身没有 SfM 硬依赖

预处理完成后，训练器接收的是：

- 新相机对象；
- 更新后的累计相机矩阵；
- `BasicPointCloud`；
- 场景 extent 和新增影像数量。

消息构造位于 `tools/train_progressive.py:1275-1279` 和
`tools/train_progressive.py:1308-1313`。这意味着解除绑定的正确位置是输入 provider
和预处理层，不是 CUDA rasterizer 或 Gaussian 优化器。

## 4. 第一性原理下的渐进式不变量

把阶段 `t` 的训练抽象为：

```text
G_t = F(G_(t-1), I_t, K_t, T_t, P_t, S_t)
```

其中：

- `G_(t-1)` 是上一阶段 Gaussian 状态；
- `I_t` 是截至阶段 `t` 可见的 RGB 观测；
- `K_t` 是相机内参；
- `T_t` 是相机位姿；
- `P_t` 是本阶段允许注入的新增几何；
- `S_t` 是语义观测和身份状态。

只要满足以下约束，输入来源从 SfM 换成官方 pose/LiDAR 不会使训练失去渐进性：

1. 阶段顺序确定且可重放；
2. 已出现的影像身份不会因为局部快照暂时省略而从模型中删除；
3. 阶段 `t` 不使用 `t+1` 之后的 RGB 或 LiDAR；
4. 新几何只由当前及历史扫描生成，不能预读完整 Terra 场景；
5. Gaussian、优化器和语义状态从 `t-1` 延续到 `t`；
6. pose provider 的来源和是否具有离线先验在产物 provenance 中如实记录；
7. 恢复训练必须重建完全相同的 provider 状态、体素占用状态和随机采样状态。

位姿是渲染测量模型的一部分，而点云主要是优化初值。错误 pose 会让不同帧对同一
Gaussian 施加彼此冲突的梯度，模型只能通过复制、膨胀、降低不透明度或模糊外观来
折中；更密的点云不能补偿系统性 pose 漂移。因此开发优先级应当是先验证 pose，
再替换初始化几何，而不是一次性同时改变全部输入。

## 5. 本地数据清单与已验证事实

### 5.1 场景输入

场景根目录：

```text
/data3/xzp/data/UAVScenes_Onthefly/interval5_AMtown01
```

关键内容：

| 内容 | 路径 | 本地事实 |
|---|---|---|
| RGB | `interval5_CAM` | 490 张，2448×2048 |
| LiDAR | `interval5_LIDAR` | 490 个 TXT，约 355 MiB |
| 官方相机—地图标定 | `sampleinfos_interpolated.json` | 504 条记录 |
| RTK | `rtk_positions_raw.csv` | 位置轨迹，不含可直接替代相机旋转的完整姿态 |
| 当前渐进 SfM | `GaussianSplatting/13...487` | 475 个连续数字阶段 |
| 语义标签 | `semantic_label/interval5_CAM_label_id` | 490/490 已适配 |

`sampleinfos_interpolated.json` 比当前 RGB/LiDAR 多 14 条早期元数据。490 个本地 RGB
时间戳全部能在 JSON 中精确找到，490 个 LiDAR 文件也与 RGB 一一对应。实现必须按
完整十进制时间戳 join，禁止使用数组位置 join。

`selection_manifest.json` 记录选择了 504 个相机帧和 504 个 LiDAR 帧，但当前磁盘上
只有 490 对。这是已存在的数据 provenance 差异，预检应报告但不能擅自补造文件。

### 5.2 LiDAR 文件格式

文件名示例：

```text
image1658137205.136885439_lidar1658137205.124739662.txt
```

名称同时编码相机时间戳和匹配的 LiDAR 时间戳。每行只有三个浮点数：

```text
x y z
```

没有 header、强度、ring、每点时间、颜色或语义字段。490 个扫描总计
12,271,870 点，抽样扫描约为 2.4--2.6 万点。由于没有每点时间，无法执行严格的
scan deskew；后续若观察到高速运动方向上的厚表面，需要把它视为数据限制，而不是
通过 Gaussian 膨胀隐藏。

相机时间减 LiDAR 时间的统计为：

- 最小值约 -41.50 ms；
- 中位数约 -11.96 ms；
- 平均值约 -12.12 ms；
- 最大值约 +15.85 ms。

### 5.3 官方 pose 和相机参数

每条 `sampleinfos_interpolated.json` 记录包含：

```text
OriginalImageName
T4x4
P3x3
K1 K2 K3 P1 P2
Width Height
```

本地投影验证表明，`T4x4` 应按 camera-to-world 使用。对于世界点 `p_W`，投影到
相机前需要先使用 `inverse(T4x4)`。随机抽取 Terra 点并使用这一方向投影时，在场景
早、中、晚三处都能得到位于图像视锥内且深度为正的点；反方向不能形成合理投影。

JSON 中 AMtown01 的代表性 `P3x3` 为：

```text
fx = fy = 1469.4898681640625
cx = 1174.0027077275518
cy = 1049.91204868583
```

记录中的畸变系数为零。这不能直接证明磁盘 RGB 已经使用该矩阵完成整流。官方
AMtown 原始相机标定还提供约 `fx=1453.72`、`fy=1453.28`、非中心主点以及非零
`k1,k2,p1,p2,k3`。开发时必须显式决定“原始 K/distortion → 目标 pinhole K”的
映射，不允许把 JSON 的零畸变字段和原始 RGB 静默组合。

### 5.4 相机—LiDAR 外参闭环

已验证的 AMtown LiDAR→Camera 形式为：

```text
p_C = R_CL * p_L + t_CL
```

其中 `R_CL` 的主要轴关系约为：LiDAR 前向轴映射到相机深度轴，另外两轴映射到
图像横纵方向。使用官方原始相机 K/distortion 投影三处扫描时：

- 早期扫描约 99.26% 点落入原始图像；
- 中期扫描约 99.41% 点落入原始图像；
- 末期扫描约 99.17% 点落入原始图像。

实际叠加检查显示点与道路、屋顶和树冠结构一致。这个结果证明本地 RGB、LiDAR、
官方外参在数值上可以组成训练输入，但它不是亚像素精度证明。正式开发仍需对不少于
20 个覆盖全轨迹的帧执行边缘对齐和深度连续性 QA。

### 5.5 Terra 模型

本机存在：

```text
/data3/xzp/data/semantic_gs/UAVScenes/terra_3dmap_pointcloud_mesh/AMtown/cloud_merged.ply
/data3/xzp/data/semantic_gs/UAVScenes/terra_3dmap_pointcloud_mesh/AMtown/Mesh.ply
```

`cloud_merged.ply` 是约 1.2 GiB 的二进制 RGB 点云，包含 83,359,173 个点。
`Mesh.ply` 包含约 11,048,765 个顶点和 22,050,702 个面。官方 pose 的逆变换能把
Terra 点投影到对应相机视锥，说明二者共享地图坐标语义。

完整 Terra 模型聚合了全场景和离线处理信息。它可以用于：

- 验证 LiDAR→World 变换；
- 生成离线深度或几何质量参考；
- 运行 `official_terra` 质量上界实验。

它不能用于严格在线主实验的初始阶段，否则模型在第一帧就获得未来区域几何。

### 5.6 当前 SfM 与官方轨迹差异

最终 `GaussianSplatting/487/images_info.txt` 记录 489 个输入身份，其中 487 个注册，
两张未注册：

```text
1658138322.586457295.jpg
1658138325.087337065.jpg
```

本地 RGB 中另有一张不在该累计列表：

```text
1658138327.584852595.jpg
```

因此当前 SfM 最终覆盖 487/490。

将 487 个 SfM 相机中心与官方轨迹做最优全局 Sim(3) 对齐后，观测到：

| 指标 | 数值 |
|---|---:|
| 位置误差均值 | 10.07 m |
| 位置误差中位数 | 9.84 m |
| 位置 RMSE | 11.20 m |
| 位置最大误差 | 23.03 m |
| 旋转误差均值 | 5.47° |
| 旋转误差中位数 | 5.37° |
| 旋转最大误差 | 9.72° |

这些数值表示一个全局相似变换无法解释两条轨迹之间的差异，存在明显的非刚性漂移或
局部姿态差异。它们证明“需要验证替代 pose”具有充分依据，但不能单独证明官方 pose
一定得到更高 PSNR；后者仍受时间同步、镜头模型、滚动快门和光度一致性影响。

当前最终阶段 COLMAP 相机为 `RADIAL`，代表性参数约为：

```text
f  = 1602.51195
cx = 1224
cy = 1024
k1 = 0.0123779
k2 = 0.0314490
```

它与官方相机标定差异明显，也是 pose-only 实验必须同时控制相机模型的原因。

## 6. 唯一坐标与投影契约

### 6.1 坐标记号

后续代码、配置和测试统一使用以下记号：

- `L`：单帧 LiDAR 局部坐标；
- `C`：相机坐标，`x` 向右、`y` 向下、`z` 向前；
- `W`：UAVScenes/Terra 世界坐标；
- `G`：3DGS 内部训练坐标；
- `T_AB`：把 B 坐标中的点变换到 A 坐标。

禁止使用没有方向后缀的变量名，如 `extrinsic`、`pose` 或 `transform`。应使用
`T_WC`、`R_CL`、`K_raw`、`K_rectified` 等名字。

### 6.2 LiDAR 到世界坐标

官方链路应写成：

```text
p_C = R_CL p_L + t_CL
p_W = R_WC p_C + t_WC
```

其中 `T_WC = sampleinfo["T4x4"]`。

组合矩阵为：

```text
T_WL = T_WC * T_CL
```

实现必须用已知合成点和真实投影测试锁定乘法顺序，不能依靠矩阵变量名猜测。

### 6.3 官方 pose 到 COLMAP/渲染器形式

如果内部需要 world-to-camera：

```text
R_CW = transpose(R_WC)
t_CW = -R_CW * t_WC
```

转换测试至少验证：

1. `T_CW * T_WC` 为单位阵；
2. 相机中心由 `-R_CW^T t_CW` 恢复为 `t_WC`；
3. 相机前方合成点具有正深度；
4. 真实 LiDAR 点能投进对应图像；
5. 保存—读取 round trip 不改变旋转和相机中心。

### 6.4 训练坐标规范化

官方世界坐标是米制，现有 SfM 是任意尺度。全轨迹对齐估计的尺度约为 5.9，直接切换
可能改变位置学习率、Gaussian 初始尺度、densification/pruning 行为和 TDOM 网格
解释。

第一轮推荐在初始 13 帧上建立一次固定 `W→G` Sim(3)：

```text
p_G = s * R_GW * p_W + t_GW
```

同一个变换必须应用于所有官方相机中心、LiDAR 点和离线参考点。相机方向只应用
`R_GW`，不把尺度乘入旋转矩阵：

```text
C_G  = s * R_GW * C_W + t_GW
R_GC = R_GW * R_WC
```

建议使用初始 13 帧而不是完整轨迹估计规范化，以免严格因果实验借用未来相机分布。
转换参数必须写入 run provenance，最终如需 Terra/RTK 米制产品再应用逆变换。

另一种方案是保留米制 `W` 作为 `G`，让 `cameras_extent` 调整位置学习率。该方案更
适合最终地理产品，但需要重新审计所有尺度敏感阈值，不应作为第一轮低风险 A/B。

### 6.5 图像整流契约

Gaussian rasterizer 使用中心主点 pinhole。当前
`src/semantic_atdom/utils/camera_calibration.py` 已能让 RGB、语义标签、有效 mask 和
SfM 二维点共享 remap，但官方适配需要补齐：

1. 五参数畸变 `k1,k2,p1,p2,k3`；
2. 显式 source K 和 target K，而不是固定复用 source focal；
3. LiDAR 投影到原始图像和整流图像的双向测试；
4. 缓存键包含原始 K、全部畸变、目标 K、尺寸和实现版本；
5. RGB 与语义标签保持线性/最近邻插值差异，但使用完全相同的映射；
6. 黑边和无效区域继续进入训练 mask，不能被当成真实黑色观测。

推荐新增从显式 UAVScenes 标定构造 `RectifiedPinhole` 的工厂，而不是伪装成不完整的
COLMAP `OPENCV` 参数。

## 7. 目标输入架构

### 7.1 不修改原始数据

现有 `GaussianSplatting`、RGB、LiDAR、JSON 和语义标签全部视为只读输入。任何中间
点云、重新三角化模型、投影缓存和规范化参数写入 run 的 `.work` 或带哈希的数据集
缓存，不回写 `/data3/.../GaussianSplatting`。

### 7.2 保留阶段结构，覆盖几何权威来源

推荐保留现有 `GaussianSplatting/<stage>` 作为阶段 provider，并新增两个相互独立的
选项：

```text
pose_source:
  legacy_sfm
  uavscenes_official

geometry_source:
  legacy_sfm
  official_retriangulated
  official_lidar
  official_terra
```

含义：

- `legacy_sfm + legacy_sfm`：当前可复现基线；
- `uavscenes_official + official_retriangulated`：pose-only 控制实验；
- `uavscenes_official + official_lidar`：推荐的渐进在线几何实验；
- `uavscenes_official + official_terra`：离线质量上界。

不建议用单个布尔参数 `use_official_data`，因为它无法表达 pose 和 geometry 的独立
消融，也容易在恢复训练时产生含混状态。

### 7.3 统一阶段对象

建议新增只读阶段对象，至少包含：

```python
ProgressiveStageInput(
    stage_id,
    active_image_ids,
    newly_observed_image_ids,
    camera_records,
    new_point_cloud,
    overlap,
    coordinate_contract,
    provenance,
)
```

训练器只消费该对象转换出的当前消息，不再直接判断 `bin` 或 `sparse/0`。legacy 和
official provider 对训练器暴露相同接口。

### 7.4 推荐运行时覆盖方式

第一版不需要复制 475 个阶段目录。运行时流程为：

1. 从原 `GaussianSplatting/<stage>` 读取阶段号、局部影像身份和 overlap；
2. 从预检报告的 `new_image_ids` 计算真实新增身份；
3. 按精确时间戳从官方 JSON 取得这些相机及历史快照中需要更新的相机；
4. 根据 `geometry_source` 生成本阶段增量点；
5. 构造现有 `CameraInfo` 和 `BasicPointCloud`；
6. 将 provider 配置、源文件哈希和坐标变换写入 provenance。

这种方式保留当前目录结构，又不会把旧 `bin` 中的错误 pose 继续当作几何权威。

如果后续需要可被其他程序消费的物化数据集，可以另行生成
`GaussianSplattingOfficial/<stage>/official/*`，但它不是第一版运行所必需。

## 8. 官方 LiDAR 增量点策略

### 8.1 为什么不能全部注入

将 490 帧 LiDAR 变换到统一世界坐标后进行全局体素去重，实测结果为：

| 体素尺寸 | 唯一点数 | 每帧平均新增 |
|---|---:|---:|
| 0.30 m | 7,618,103 | 15,547 |
| 0.50 m | 4,133,624 | 8,436 |
| 0.75 m | 2,077,629 | 4,240 |
| 1.00 m | 1,197,827 | 2,445 |
| 1.50 m | 520,778 | 1,063 |

当前生产配置的 Gaussian 预算约为 100 万，历史正式模型约 103 万。1 m 体素仅 LiDAR
初值就超过预算，还没有给图像驱动的 densification 留空间；0.3--0.75 m 更不可行。

简单 CPU 原型读取文本、完成坐标变换和全局体素去重约需 40--50 ms/帧。预处理算力
本身不是首要瓶颈，Gaussian 数量增长才是训练延迟和显存的首要风险。

### 8.2 推荐选择流水线

每个新扫描按以下顺序处理：

1. 按文件名精确解析 image/lidar 时间戳；
2. 丢弃 NaN、Inf、非正深度和显式 range 异常；
3. 使用 `T_CL` 和当前相机 `T_WC` 变换到世界坐标；
4. 投影到对应原始或统一整流图像，丢弃无效区域；
5. 从 RGB 采样初始颜色；
6. 在细粒度全局网格中执行几何 novelty 判断；
7. 对候选点做确定性优先级排序；
8. 截断到本阶段新增点预算；
9. 变换到训练坐标并构造 `BasicPointCloud`；
10. 更新可恢复的全局体素占用状态。

### 8.3 点优先级

不建议仅用大体素产生最终点，因为统一 1.5 m 网格会牺牲小屋顶、路缘和树冠边界。
建议使用“细网格判重 + 固定预算采样”：

- 0.3--0.5 m 网格用于判断点是否代表新空间；
- 平面内部低密度保留；
- 深度不连续、图像梯度和语义边界提高权重；
- 对动态类别、孤立点和只出现一次的高残差点降低权重；
- 稳定排序键至少包含阶段、体素 key、点坐标量化值和固定 seed；
- 相同输入和 checkpoint 恢复必须生成相同的新增点集合。

建议第一轮预算：

- 初始 13 帧合计约 20,000--50,000 个 LiDAR seed；
- 后续每个新影像约 500--1,000 个 seed；
- 全程 LiDAR seed 目标约 300,000--500,000；
- 剩余 Gaussian 容量交给光度 densification；
- 继续使用现有 100 万预算和 5% 滞回进行总量控制。

这些数值是开发起点，不是已经验收的最优参数。最终选择必须由固定 A/B 实验决定。

### 8.4 颜色与可见性

LiDAR TXT 不含颜色。由于约 99% 的点能投进配对 RGB，可以从同帧图像双线性采样
初始颜色。必须先完成统一整流再采样，避免颜色和训练像素处于不同投影模型。

以下点需要特别处理：

- 被前景遮挡但投影到背景颜色的点；
- 车辆、行人等动态物体；
- 风致植被在多帧融合后形成的厚表面；
- 扫描时间和相机时间不同导致的运动偏移；
- 图像边缘的整流黑边。

第一版可以使用深度 z-buffer 保留同一像素最近点，并对动态语义类别关闭或降低初始化
权重。后续可要求跨扫描或跨视图支持后再激活点，但该策略会引入观测延迟，需要在
online latency 指标中单独记录。

## 9. Pose 使用策略

### 9.1 第一阶段：固定官方 pose

第一轮实验应固定官方 pose，不同时引入可学习 pose residual。目的是真正测量“移除
SfM 漂移”带来的影响。如果一开始允许自由 pose 优化，模型可能重新漂回光度局部最优，
无法判断官方约束是否有效。

### 9.2 Pose-only 几何

为了隔离 pose 和 LiDAR 两个变量，建议保留现有二维特征轨迹，但在官方 K/T 下重新
三角化三维点：

1. 读取现有每阶段二维观测和 track identity；
2. 用官方相机模型将观测统一整流；
3. 在官方 pose 下重新三角化；
4. 过滤负深度、大重投影误差和低视差点；
5. 只使用截至当前阶段的观测；
6. 缓存结果并绑定所有输入哈希。

这条路径仍使用上游 SfM 的特征关联，但不使用其 pose 和三维坐标，适合作为
`official_retriangulated` 控制组。

### 9.3 后续：有界 pose residual

官方 pose 可能不是光度意义上的亚像素最优，原因包括时间偏差、滚动快门、相机—
LiDAR 外参误差和地图标定误差。如果固定官方 pose 的几何一致性明显改善、但高频
光度指标受限，可以在后续实验中优化小的 `SE(3)` residual，并施加强先验和上限。

这必须是单独实验，不得默认开启。需要记录每帧最终平移/旋转修正，并拒绝重新出现
米级、度级累计漂移。

## 10. Overlap、语义和身份后果

### 10.1 第一轮保留现有 overlap

当前 `match_matrix_value.txt` 是 SfM 匹配计数，不是严格的几何可见面积。第一轮
487 帧对照可以继续使用它，因为这样能保持语义分组、阶段身份和旧基线一致，只替换
pose/geometry 变量。

现有 overlap 不应被描述为官方几何 overlap，也不能用于证明 LiDAR 覆盖质量。

### 10.2 后续重新计算几何 overlap

在官方 pipeline 稳定后，可用以下信息计算新的 overlap：

- 相机视锥相交；
- LiDAR/Terra 深度可见性；
- 双向有效投影像素比例；
- 遮挡处理后的共享体素数量。

新矩阵必须保持：对角线为各影像有效支持量、非负有限、身份顺序显式、结果确定性和
版本化。切换 overlap 是独立实验，不与首次 pose/LiDAR 替换同时发生。

### 10.3 语义状态

更可靠 pose 预计会减少语义投影错位和不同物体之间的错误融合，但点云拓扑改变会使
在线 semantic state 的索引和 Gaussian 数量轨迹发生变化。必须维持：

- topology event 与 Gaussian 增删一一对应；
- `persistent_union` 身份不因快照临时省略而删除；
- 中间状态如实报告 `stale_online`；
- final boundary 仍执行覆盖全部有效观测的 exact rebuild；
- RGB TDOM 和 Semantic TDOM 使用同一坐标规范化与最终逆变换。

## 11. 对训练质量、速度和产品的预期后果

### 11.1 预期收益

- 减少不同阶段 pose 漂移导致的重影、重复 Gaussian 和异常尺度；
- 为弱纹理屋顶、道路和植被提供真实深度初始化；
- 获得可恢复的米制尺度，便于与 RTK/Terra 对齐；
- 不再依赖 SfM 是否成功注册全部图像；
- 官方 pose 固定后，历史相机不再因每次 bundle adjustment 被反复移动；
- 直接点云路径可以移除当前 Delaunay 和二维→三维插值热点；
- 更一致的几何预计有利于最终语义投影和 TDOM 边界。

### 11.2 可能的质量回归

- 官方绝对 pose 可能不如 SfM pose 适应具体图像的滚动快门和光度误差；
- 错误的 K/distortion 组合可以抵消全部 pose 收益；
- LiDAR 太稀或过度体素化会丢失小结构；
- 无约束注入植被和动态物体会形成厚表面；
- 初始颜色错误会增加早期优化负担；
- 点预算过小会导致新视角短期欠覆盖，过大则挤压 densification；
- 固定 pose 不再通过 bundle adjustment 吸收时间同步和外参残差。

### 11.3 可能的速度回归

LiDAR 文本解析和坐标变换本身预计只占几十毫秒/帧，但以下成本随 Gaussian 数增长：

- rasterization tile/point 相交；
- 梯度与 Adam 状态；
- densification/pruning；
- 语义 state 扩容和 topology 同步；
- RGB/Semantic TDOM 渲染；
- checkpoint 和进程间传输。

所以性能验收必须同时记录“每帧新增 seed”“训练前/后 Gaussian 数”和“单帧墙钟
时间”，不能只记录点云预处理耗时。

### 11.4 产品坐标变化

如果训练使用规范化 `G` 坐标，输出点云、相机和 TDOM 仍需保存 `T_WG`，否则结果
无法重新对齐 Terra/RTK。发布 manifest 应明确：

- internal training frame；
- official world frame；
- 正向和逆向 Sim(3)；
- 单位；
- 是否使用完整 Terra 先验；
- pose/geometry/overlap provider 版本。

## 12. 实验矩阵与因果隔离

所有第一轮对照保持相同的 487 个已注册身份、13--487 阶段边界、随机 seed、图像
分辨率、迭代数、Gaussian 预算、语义输入和输出设置。

| 实验 | Pose | Geometry | 目的 |
|---|---|---|---|
| B0 | legacy SfM | legacy SfM | 当前权威基线 |
| P1 | official fixed | official retriangulated | 隔离 pose/相机模型影响 |
| L1 | official fixed | progressive LiDAR | 验证推荐在线几何路径 |
| L2 | official fixed | progressive LiDAR，不同点预算 | 找到质量/速度折中 |
| T1 | official fixed | cropped/downsampled Terra | 离线质量上界 |
| R490 | official fixed | progressive LiDAR | 在方案稳定后恢复完整 490 帧 |

禁止直接从 B0 跳到 R490 后只比较最终指标，因为 pose、geometry 和影像数量同时变化，
无法归因。

### 12.1 数据拆分

仅在全部训练图像上计算训练重建指标不能证明泛化质量。至少固定一组 held-out 相机，
使用官方 pose 渲染，并在所有实验中保持相同身份。建议同时报告：

- 训练帧 L1/PSNR/SSIM/LPIPS；
- held-out 帧 L1/PSNR/SSIM/LPIPS；
- LiDAR 深度或点到渲染深度的一致性；
- 相机边缘附近和弱纹理区域的分区指标；
- 语义类别一致率、JS 散度和边界误差；
- 最终 Gaussian 数及其空间分布。

### 12.2 非劣验收建议

“不影响训练质量”不能只靠主观截图。第一轮建议把下列阈值作为暂定非劣门槛，最终
可在实现前由项目负责人确认：

- held-out 平均 PSNR 不低于 B0 超过 0.10 dB；
- held-out 平均 SSIM 不低于 B0 超过 0.002；
- held-out 平均 LPIPS 不高于 B0 超过 0.005；
- 不允许出现新的大面积双影、漂浮层或地图断裂；
- 最终语义必须为 `exact_final`，lag=0，errors/deferred 为空；
- Gaussian 数不得靠持续超预算后强制裁剪来维持；
- 相同 seed 重跑的身份集合、点选择和阶段计数必须一致。

如果官方 pose 明显改善几何指标但轻微降低光度指标，应先检查相机模型和同步，不应
立即放宽质量门槛或允许无界 pose 优化。

### 12.3 性能验收

每个阶段记录：

- 官方元数据 lookup；
- LiDAR 文本读取；
- 坐标变换和投影；
- novelty/预算选择；
- 新点跨进程传输；
- Gaussian 扩张；
- 增量训练；
- 语义更新；
- 地图/视频渲染；
- CPU RSS、GPU allocated/reserved/max；
- 阶段前后 Gaussian 数。

比较 P50/P95/P99 和最大阶段延迟。第一轮不得通过减少迭代、关闭语义或降低分辨率获得
表面速度收益。

## 13. 推荐开发阶段

### Phase 0：只读审计和显式配置

1. 新增 pose/geometry provider 枚举和 provenance schema；
2. 实现 490 帧时间戳精确 join 与 14 条缺失元数据报告；
3. 把官方 AMtown 标定作为版本化配置纳入仓库，不在运行时联网读取；
4. 完成坐标变换和投影单元测试；
5. 扩展整流器支持完整畸变和显式 target K；
6. 不改变 legacy 默认行为。

验收：现有测试全过，默认命令产物不变，官方 preflight 只读通过。

### Phase 1：Pose-only

1. 读取现有阶段/身份/overlap；
2. 用官方 pose 和相机模型构造 `CameraInfo`；
3. 用现有 track 在官方相机下重新三角化；
4. 使用初始 13 帧固定 Sim(3)；
5. 跑 stage 13 单阶段和 13--17 smoke；
6. 再跑能覆盖已知非严格累计行为的 13--22；
7. 与 B0 做同参数质量/性能比较。

验收：身份和阶段契约不变，无负深度/NaN，恢复训练确定，质量达到暂定非劣门槛。

### Phase 2：Progressive LiDAR

1. 新增 LiDAR provider；
2. 实现投影、颜色、z-buffer、novelty 和固定预算；
3. 把体素占用与 provider RNG 纳入 checkpoint；
4. 直接返回 `BasicPointCloud`，绕过 SfM 三角插值；
5. 跑不同点预算小范围消融；
6. 验证 Gaussian budget 和语义 topology。

验收：不预读未来扫描；恢复前后新增点完全一致；单帧预处理和训练延迟无不可接受回归；
质量达到非劣门槛。

### Phase 3：官方几何 overlap

1. 保持 L1 其他条件不变；
2. 计算基于深度可见性的 overlap；
3. 单独比较在线语义分组和最终 exact rebuild；
4. 版本化矩阵算法和阈值。

验收：身份集合、矩阵维度、归一化和语义结果均通过预检。

### Phase 4：恢复 490 帧

1. 在当前 acquisition order 中确定三个缺失 SfM 身份的正确插入位置；
2. 生成新的阶段范围和 overlap；
3. 更新标签/preflight 契约；
4. 与 487 帧 L1 分开报告，不能覆盖原基线。

### Phase 5：Terra 上界

1. 按可见视锥裁剪 Terra；
2. 分层体素降采样并限制初始 Gaussian；
3. 明确标记 `offline_map_prior=true`；
4. 只用于评估 pose/LiDAR 方案距离可达到上界还有多远。

## 14. 代码改造路由

建议的最小改造范围：

| 文件/模块 | 责任 |
|---|---|
| `src/semantic_atdom/arguments/__init__.py` | pose/geometry provider 和路径参数 |
| `src/semantic_atdom/scene/progressive_layout.py` | 保留 stage 解析，暴露 scene root 与共享传感器目录 |
| 新增 `src/semantic_atdom/scene/progressive_sources.py` | provider 接口、时间戳 join、官方 stage 对象 |
| `src/semantic_atdom/utils/camera_calibration.py` | 完整 UAVScenes source/target 标定与 k3 |
| `src/semantic_atdom/scene/dataset_readers.py` | 官方 CameraInfo、直接 BasicPointCloud 路径 |
| `tools/train_progressive.py` | 按 provider 选择输入，保持 trainer 消息不变 |
| `src/semantic_atdom/semantics/preflight.py` | 模式相关必需文件、因果性和哈希验证 |
| `src/semantic_atdom/semantics/orchestrator.py` | 后续支持官方几何 overlap；第一版保持 legacy |
| `src/semantic_atdom/training_checkpoint.py` | provider、体素占用、采样状态恢复 |
| `src/semantic_atdom/utils/run_output.py` | provenance 和坐标契约发布 |

不建议把 UAVScenes 特例直接散落在 `train_progressive.py` 主循环。数据集特定逻辑应在
provider 和 calibration adapter 内，训练器只消费统一对象。

## 15. 测试清单

### 15.1 单元测试

- 504 metadata 与 490 RGB/LiDAR 的精确 join；
- 14 条缺失记录稳定报告；
- image/lidar 文件名高精度时间戳解析；
- `T_WC ↔ T_CW` round trip；
- `T_WL = T_WC * T_CL` 合成点；
- 官方 distortion 包含 k3；
- RGB/语义/LiDAR 使用相同 rectification map；
- LiDAR 投影正深度和图像边界；
- 固定 Sim(3) 对相机和点的一致应用；
- novelty hash 和预算采样确定性；
- checkpoint 恢复后的体素占用和下一批新增点一致；
- stage `t` 不能打开 `t+1` 的 LiDAR；
- legacy provider 默认行为不变。

### 15.2 集成测试

- 真实 stage 13 官方 pose-only；
- 真实 stage 13 官方 LiDAR；
- stage 13--17 initialization/progressive/final refinement；
- stage 13--22 覆盖 AMtown 非严格累计身份；
- checkpoint 中断和新进程恢复；
- RGB 与语义 exact-final 发布；
- 训练进度视频开启时 lag=0 和画布一致；
- Gaussian 预算边界和 topology event 完整；
- 官方坐标逆变换后的点云/相机可重新投影到原始图像。

### 15.3 数据 QA

- 至少 20 帧 LiDAR/RGB overlay，覆盖轨迹早中晚和不同高度；
- 投影点落入图像比例、深度分布和边缘距离；
- LiDAR 融合后点到 Terra 的距离分布，仅作离线 QA；
- 动态类、植被、屋顶边缘和图像黑边的专项可视化；
- 三个 SfM 缺失帧的 RGB、标签、pose、LiDAR 完整性。

## 16. Provenance 和缓存契约

每个官方运行必须记录：

- pose source 与版本；
- geometry source 与版本；
- overlap source 与版本；
- metadata、LiDAR、标定、阶段身份输入的 SHA-256；
- 490/504 join 报告；
- source/target K 与全部 distortion；
- `T_CL`、`T_WG`/`T_GW`；
- 点过滤、体素、预算、排序和随机参数；
- 每阶段输入点数、候选点数、保留点数；
- 是否读取 Terra；
- 是否允许 pose residual；
- 代码提交和工作树状态。

缓存 key 至少包含上述几何相关输入和算法版本。只用 stage 编号或文件修改时间作为
缓存 key 不足以防止错误复用。

## 17. 回滚与兼容策略

1. `legacy_sfm` 始终是默认 provider，除非新的正式实验明确切换；
2. 不修改原始 `GaussianSplatting`；
3. official provider 失败时直接终止，不静默回退到 SfM pose 或点云；
4. run manifest 明确记录 provider，恢复训练要求完全一致；
5. 旧 checkpoint 只能由 legacy provider 恢复，除非实现并验证显式迁移；
6. P1/L1/T1 使用不同 run name，不覆盖历史 baseline；
7. 任何全量失败保留诊断目录，但不更新 `outputs/latest`；
8. 如果官方方案未通过非劣门槛，保留实现用于诊断，不将其设为默认路径。

## 18. No-go 条件

出现以下任一情况时，不得进入全量训练：

- 无法建立唯一、可测试的 `L→C→W→G` 变换链；
- RGB、语义和 LiDAR 的整流像素不一致；
- 大范围 LiDAR overlay 与道路/屋顶边缘系统性错位；
- official provider 在 stage `t` 读取未来 LiDAR/Terra；
- 点选择在同 seed 重跑或 checkpoint 恢复后不确定；
- 小范围运行持续超过 Gaussian 预算并依赖大规模强制裁剪；
- pose-only 已明显劣于基线且相机模型/同步问题尚未排除；
- semantic topology 不完整、final exact rebuild 失败或 lag 非零；
- 不能从产物恢复到官方世界坐标；
- 为通过速度验收而改变分辨率、迭代数或模型容量。

## 19. 推荐的当前决策

基于现有证据，建议后续开发采用以下默认决策：

1. 保留原 `GaussianSplatting/13...487` 作为阶段与身份 provider；
2. 不复制、不覆盖原始输入；
3. 先实现 `uavscenes_official + official_retriangulated`，隔离 pose 影响；
4. 再实现 `uavscenes_official + official_lidar`；
5. 初始阶段使用固定 13 帧 Sim(3) 规范化以降低超参数迁移风险；
6. LiDAR 使用细网格 novelty 和固定点预算，不把大体素当作唯一选择机制；
7. 第一轮保持现有 487 帧和 legacy overlap；
8. 三个缺失帧、官方 overlap、pose residual 和 Terra 分别作为后续独立实验；
9. 把严格在线结论限定为“模型和几何观测按阶段因果更新”，并明确官方 pose 是外部
   真值输入；
10. 按 `13`、`13--17`、`13--22`、完整范围逐级放行，不跳过小范围验收。

## 20. 最终判断

保留现有 `GaussianSplatting` 阶段结构、替换 pose 和新增几何，在算法定义和工程结构
上都可行。它不会破坏渐进式 Gaussian 训练，因为渐进性由阶段观测和状态延续决定，
而不是由 SfM 决定。当前代码的主要阻碍是输入层把相机、特征关联和三维点绑定在同一
COLMAP 文件契约中；通过 provider 抽象和直接 `BasicPointCloud` 路径可以解除这一
绑定，训练核心不需要重写。

最大的技术风险不是 LiDAR 文件读取，而是相机整流正确性、坐标尺度、点预算和实验
定义。最大的科学风险是把离线官方 pose 或完整 Terra 先验误称为端到端在线估计。
只要严格执行本文的变量隔离、因果边界、provenance 和验收门槛，这条路线值得进入
开发，并且比继续依赖已经观测到明显全局漂移的 SfM pose 更有希望提高最终几何和
语义质量。
