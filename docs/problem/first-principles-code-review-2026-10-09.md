# 当前代码的第一性原理审查与工程问题记录

状态：问题已记录，尚未修复；架构建议尚未实施
审查日期：2026-10-09
审查分支：`semantic/g1-evidence-v2`
代码基线：`0665f4ba911ddb09001aa63618c729309011d90d`

## 1. 范围与证据边界

本次审查重点覆盖渐进训练、几何采样、语义融合、断点恢复、进程通信和最终发布，
并以 `origin/main` 的 `fb4290ebb640af156c2f3583a9bf3ca976cfc5c2` 为历史比较基线。
本文不宣称穷尽仓库中的所有缺陷，也不将一次完整训练成功等同于工程边界已经验证。

正式 `semantic-atdom` 环境执行以下检查，结果为 **245 passed，3 warnings**，
包括 CUDA observation/TDOM 测试；运行时 Python 3.10.21、PyTorch 2.4.0，CUDA 可用。

```bash
/home/xuzipeng/anaconda3/envs/semantic-atdom/bin/python \
  -m pytest -q -p no:cacheprovider tests
```

额外使用现有测试 fixture、实际函数调用和小规模进程故障模拟检查未覆盖的行为。
这些检查证明相应条件下的错误机制，不替代全量训练、磁盘故障注入或质量消融。
审查期间未修改算法或工程实现。

当前本机数据交付与历史实验进展另见
[数据与实验状态核对](../project/current-data-experiment-status.md)。文档事实更新不关闭
本文问题；历史成功运行也不替代这里各项故障边界的修复验收。

本文使用的最低原则是：

1. 从图像采样得到的三维点应满足其声称的投影对应关系。
2. 数据身份与运行选项分离，原始输入只读。
3. 恢复操作能延续同一数据与训练状态，并拒绝真正不兼容的输入。
4. 辅助进程失败必须可检测，队列和退出等待必须有界。
5. 产品的证据资格、来源和精确性声明与实际计算一致。

权威约束来自 [全局认知](../project/global-context.md)、
[语义数学与产品合同](../design/semantic-tdom-v1.md)、
[单向语义依赖 ADR](../decisions/0001-one-way-semantic-dependency.md)、
[输出生命周期 ADR](../decisions/0008-run-output-contract.md)、
[训练视频 ADR](../decisions/0009-published-renders-and-map-training-video.md) 和
[平面补洞 ADR](../decisions/0010-bounded-planar-semantic-fallback.md)。

所有代码行号均指向上述审查基线；后续修复可能使行号移动，应同时通过函数名定位。
P1 表示应优先处理的正确性或可用性问题；P2 表示有特定触发条件的契约或工程缺口。
优先级不是对所有历史产物已经失效的声明。

## 2. 问题索引

| ID | 优先级 | 问题 | 证据 | 状态 |
|---|---|---|---|---|
| R01 | P1 | 语义训练恢复使用不稳定的 preflight hash | 同数据 fixture 对比、恢复代码核对 | 待修复 |
| R02 | P1 | 输出 worker 死亡后生产者退出可能挂起 | CPU 队列退出模拟、进程生命周期核对 | 待修复 |
| R03 | P2 | 独立渲染路径写入原始 SfM 目录 | 实际 reader 的 CPU mock 调用 | 待修复 |
| R04 | 架构改进 | 训练入口责任混杂与位置消息耦合 | 代码结构审查；属于工程判断 | 待设计 |
| R05 | P2 | 在线分歧错误计入开放组且误标 exact | CPU 状态复现、渲染声明核对 | 待修复 |
| R06 | P2 | 独立最终重建不能隔离单张无效标签 | 独立 hook 的 CPU 复现 | 待修复 |
| R07 | P1 | 二维重心权重直接插值三维点不保持投影 | 实际采样函数的数值反例 | 待修复与质量验证 |
| R08 | P2 | 最终输入绑定遗漏生成配置身份 | 实际 final binding 接受 metadata 漂移 | 待修复 |
| R09 | P2 | 尺度过滤与补洞对可信几何定义不同 | 偶数集合中位数反例、两条实现核对 | 待修复 |

## 3. 工程规范与运行生命周期

### R01：语义训练恢复使用不稳定的 preflight hash

**触发条件与影响。** 首次开启语义训练并保存 stage-boundary checkpoint，之后用
`start_checkpoint` 恢复同一数据。首次运行的 `require_complete_initial_snapshot`
为 True，恢复运行变为 False，而整个 preflight report 被计算为一个哈希。
checkpoint 存储旧哈希，恢复校验要求它与新报告哈希完全相同，导致恢复被拒绝。

**代码证据。**

- `tools/train_progressive.py:513-515`，`TrainingPreparation()` 根据是否有 checkpoint
  设置 `require_complete_initial_snapshot`。
- `src/semantic_atdom/semantics/preflight.py:345-346`，`run_dataset_preflight()`
  将该检查选项写入参与哈希的报告。
- `tools/train_progressive.py:1771-1774`，初始化恢复路径逐字比较报告哈希。

**复现。** 使用 `tests/unit/test_dataset_preflight.py` 中的 fixture，在同一临时数据上
分别执行 `require_complete_initial_snapshot=True/False` 的预检。两份报告唯一不同
字段是这个布尔值，`report_sha256` 却不同。这里验证的是数据身份检查的拒绝机制，
未执行完整 GPU 训练保存后再恢复的实验。

**修复方向。** 将稳定的输入内容指纹与本次检查选项、报告呈现分离；checkpoint
验证数据身份及必要的恢复兼容参数，不使用含运行模式的整个报告作为唯一数据身份。

**验收。** 相同数据的首次训练 checkpoint 在新进程中恢复成功，下一阶段能执行；
RGB、标签、相机或 overlap 内容真正变化时仍被拒绝。恢复与连续运行的随机状态、
相机集合和阶段边界应一致。

### R02：输出 worker 死亡后生产者退出可能挂起

**触发条件与影响。** `DataSaver` 保存视频帧时发生写盘异常并退出，训练器继续向
无界 `multiprocessing.Queue` 发送完整帧。主进程先等待 trainer 退出，随后才检查
saver 状态；trainer 又可能等待 queue feeder 把无人消费的大消息写完，形成无界等待。
这种情况下运行无法及时结束或记录失败。

**代码证据。**

- `tools/train_progressive.py:3192-3193`，`DataSaver.SaveTrainingProgress()` 的图像保存
  异常会从 worker 传播。
- `tools/train_progressive.py:3721`，输出使用无界 `mp.Queue()`。
- `tools/train_progressive.py:2183`，trainer 向输出队列发送帧。
- `tools/train_progressive.py:3801-3810`，主进程先无超时 join trainer，再检查 saver。

**复现边界。** CPU 模拟无人消费的 Queue，生产者发送 1 MB 消息后 `put()` 返回，
但 `join(timeout=0.5)` 后生产者仍存活，阻塞在退出时发送队列缓冲的路径。
这与代码中的依赖顺序共同建立故障机制；尚未对正式训练目录实际注入磁盘满故障。

**修复方向。** 共同监督所有 worker；设置有界队列、失败通知和可中断发送；
输出失败后停止继续发送。统一 finally 清理存活进程和队列，join 必须有超时。

**验收。** 注入输出异常、worker 提前退出和队列积压时，运行在规定时间内完成失败
收尾，留下失败原因、已有模型及显式启用的恢复点，无遗留进程和不断增长的帧缓冲。

### R03：独立渲染路径写入原始 SfM 目录

**触发条件与影响。** 使用 `tools/render.py` 加载没有 `points3D.ply` 的原始数据。
默认 reader 会把 `points3D.bin` 转换成 PLY 并写回原始 `bin/` 或 `sparse/0/`。
这违反原始输入只读约束，在只读挂载上还可能直接导致渲染失败。

**代码证据。**

- `tools/render.py:40` 调用 `Scene(dataset, ...)`。
- `src/semantic_atdom/scene/__init__.py:71/75` 传入的 cache 路径缺省为空。
- `src/semantic_atdom/scene/dataset_readers.py:1368-1377`，
  `readColmapSceneInfo()` 默认 `get_ply=True`，向源目录创建 PLY。
- `ModelParams.extract()` 仅提取已声明字段，不能保留训练中临时附加的
  `point_cloud_cache_path`。

**复现边界。** mock 相机和二进制解析，调用实际 `readColmapSceneInfo()`，使用
实际 PLY 写入函数；观察到临时 `raw_dataset/bin/points3D.ply` 被创建。
未运行完整独立渲染的 GPU 主流程。

**修复方向与验收。** 场景读取层默认不写输入；渲染直接使用内存点云，或显式传入
位于输出根的缓存。对只读输入执行训练和独立渲染都应成功，输入文件树与内容哈希不变。

### R04：训练入口责任混杂与位置消息耦合

**性质。** 这是架构判断，不单独声称文件长度构成 bug。

`tools/train_progressive.py` 在审查基线有 4,019 行，同时实现 CLI、数据采样、RGB
优化、语义调度、进程管理、视频绘制/编码、恢复和发布事务。阶段消息使用字符串命令
及位置列表，例如生产者的阶段消息与 `GaussianTrainer.run()` 中九字段消费逻辑需要
同步修改。不同生命周期的状态缺少明确所有者，R01/R02 暴露了相应接口检查缺口。

**建议边界。** 优先提取类型明确的 stage payload、统一 worker supervisor、视频服务
及最终发布服务，再逐步分离数据 provider 和训练器。避免仅拆文件却保留相同的隐式
共享状态，也避免无证据的大规模抽象。

**验收。** 单一消息 schema 能检查字段和版本；进程失败行为可独立测试；视频和最终
发布逻辑不要求加载整个优化循环。重构保持数值路径和现有发布合同。

## 4. 设计契约与语义状态

### R05：在线分歧错误计入开放组且误标 exact

**合同。** `docs/design/semantic-tdom-v1.md:245-266` 规定少于两个有效闭合组时
分歧为 NaN，开放组只能提供 provisional 统计；诊断层区分 `online_estimate` 和
`exact_final`。

**代码与复现。**

- `src/semantic_atdom/semantics/state.py:397-401` 用包含开放组的 `new_group_count`
  判断分歧是否可用。
- `src/semantic_atdom/semantics/tdom.py:452` 用总 `group_count >= 2` 判断像素支撑资格。
- 默认最大组长 4，同阶段 ingest 两个无 overlap 的观测，分布分别为 `[1,0]`、
  `[0,1]`，支持均为 1；得到闭合组数 1、开放组数 1，却输出 group support
  约 0.6667、disagreement 0.5。
- `tdom.py:1036-1040` 对所有可用分歧设置 `exact_rebuild`，包括在线统计。

**影响。** 诊断有效性和精确性声明错误；这不等于已经证明 hard-class gate 或全部
最终类别产品错误。

**修复方向与验收。** 显式保存闭合统计及其资格；分别测试零组、一闭合加一开放、
两闭合组的输出。由实际计算来源设置在线/最终状态，不能从 available 布尔值推导 exact。

### R06：独立最终重建不能隔离单张无效标签

**适用范围。** 独立 hook、非权威 partial 重建及部分恢复路径。默认正式训练的预检
通常提前阻止无效标签，不能把下面复现描述成每个默认运行都会发生的问题。

**合同。** 最终重建可使用缺口后的有效标签，对明确缺失或无效输入发布有声明的
`exact_partial`，并记录有效、缺失及失败身份。

**代码与复现。** `src/semantic_atdom/semantics/orchestrator.py:1324` 主要按文件存在
加入 final_ready，`rebuild.py:232` 的单观测读取错误未被隔离。CPU fixture 中
`000001/000003` 标签和 metadata 有效，`000002` metadata 为非法 JSON；调用
`synchronize(..., final=True)` 后退回 `online_fallback`，projector 只处理了
`000001` 两次，`000003` 没有参与最终重建。

**修复方向与验收。** 在明确允许 partial 的路径逐张验证、记录错误，再重建全部有效
观测；上述反例应得到包含 `000001/000003` 的 partial 重建及 `000002` 的失败声明。
权威 preflight 已绑定的输入发生漂移时仍应拒绝，不允许以忽略错误的方式继续声明
`exact_final`。

## 5. 几何与输入身份的交叉验证

### R07：二维重心权重直接插值三维点不保持投影

**代码事实。** `src/semantic_atdom/scene/dataset_readers.py:216-225` 的
`Get_3D_Sampled_Points_GPU()` 计算二维重心权重，随后直接线性组合三维顶点；CPU
`Get_3D_Sampled_Points()` 有相同计算。渐进新增点路径实际调用 GPU 版本。

**数值反例。** 使用相机坐标系和归一化投影 `u=X/Z, v=Y/Z`：

| 顶点 | 三维坐标 | 正确二维投影 |
|---|---|---|
| A | `(0,0,1)` | `(0,0)` |
| B | `(2,0,2)` | `(1,0)` |
| C | `(0,4,4)` | `(0,1)` |

二维目标为 `(1/3,1/3)`，实际函数给出三维点 `(2/3,4/3,7/3)`；它的投影为
`(2/7,4/7)`，不是目标。焦距 1,000 像素时，水平/垂直有符号误差约为
`(-47.619,238.095)` 像素。反例通过实际函数在 CPU tensor 上运行，顶点及输入二维
对应严格一致，不依赖 SfM 噪声。

**原则与影响边界。** 二维重心权重不能一般地直接作为透视相机下的三维重心权重。
等深度时这个反例不会出现。它证明采样对应关系不成立，尚未证明所有新增点都错误，
也未量化其对真实训练 PSNR、尺度异常或 Semantic TDOM 的贡献。

**修复方向。** 对同一三角表面，按相机空间深度做透视修正：

```text
w_i = (lambda_i / z_i) / sum_j(lambda_j / z_j)
P = sum_i(w_i * P_i)
```

或计算采样像素射线与三角平面的交点。还需处理负深度、退化平面和真实 SfM
对应误差。二维 Delaunay 连接跨深度边界的风险不能只靠修正这条公式消除。

**验收。** 对等深度、不同深度、相机旋转平移和退化输入添加投影性质测试；生成点
应在约定容差内重投影回目标像素。修改采样后执行同参数/随机状态的真实场景 A/B，
检查 RGB 指标、尺度谱、几何覆盖及语义产品，不预先声称质量提升。

### R08：最终输入绑定遗漏生成配置身份

**代码事实。** `src/semantic_atdom/semantics/preflight.py:327-335` 的 observation
记录绑定 image ID、label version/hash、RGB hash 和 source binding；
`input_binding.py:160-186` 的最终检查没有绑定完整生成配置身份。

**复现。** 使用 `tests/unit/test_semantic_input_binding.py` 的现有 fixture，保持
RGB、label、label version 和 report hash 不变，修改 metadata 的 checkpoint hash、
prompt、thresholds，并新增有效 score 文件及其自洽 hash；实际
`validate_final_preflight_binding()` 仍接受绑定。

**影响边界。** 生成来源在预检后变化却未被阻止，不符合生成配置变化须发布新版本的
要求。V2 明确不依赖可选 scores，因此不能把 scores 漂移说成当前融合数值已改变。
final rebuild 的两遍检查可检测两遍之间的变化，但不能替代与训练前身份的比较。

**修复方向与验收。** 在预检和最终检查中绑定规范化生成配置指纹或 metadata 内容
哈希，明确可变字段与可选 scores 的策略。checkpoint/prompt/thresholds 的漂移必须
被检测；若允许特定审计字段变化，应有明确版本规则，不能隐式放行。

### R09：尺度过滤与补洞对可信几何定义不同

**代码事实。** `src/semantic_atdom/semantics/tdom.py:263-272` 的
`_trusted_geometry_mask()` 使用 Torch median；
`planar_fallback.py:344-352` 独立使用 NumPy median。偶数数量时，前者使用较小的
中间值，后者使用两个中间值的平均数。

**反例。** 64 个高斯的最大轴尺度为 32 个 1 和 32 个 20，ratio=14：

| 路径 | 中位数 | 阈值 | 可信高斯数 |
|---|---:|---:|---:|
| TDOM 过滤 | 1 | 14 | 32 |
| 平面补洞 | 10.5 | 147 | 64 |

同一个模型中的 32 个高斯被 TDOM 排除，却仍有资格参与补洞平面的拟合；这与
ADR 0010 要求用附近可信高斯定义平面不一致。反例证明两种定义不同，不宣称现有
Phantom3 全量结果恰好触发这一分布；补洞默认关闭，其他投票条件仍然生效。

**修复方向与验收。** 补洞复用渲染时同一可信几何 mask，避免再次推导领域概念。
测试奇数、偶数、空集合、关闭过滤及异常输入，并验证同一配置下两条路径接受完全
相同的 Gaussian IDs。不能通过放宽尺度过滤来绕过这个问题。

## 6. 已知设计债务与不应误判的行为

以下问题已在现有文档记录，属于需要实验决定的设计近似，不计作新增已复现缺陷：

- `GaussianModel.expand_from_pcd()` 仅用当前新增批次的邻域初始化尺度，未包含
  已有地图支撑；研究与修复实验见
  [Phantom3 尺度异常指导](phantom3-water-scale-outlier-training-guidance.md)。
- 异常尺度过滤让 RGB 和 Semantic 使用不同 Gaussian 子集，仍缺少通用的权威
  像素有效域。当前过滤与有界平面补洞不能被解释为几何根治。
- 在线历史证据陈旧、因果顺序的鲁棒权重及拓扑状态复制均有明确近似边界；没有
  新的违反合同证据时，不将它们本身列为 bug。
- ADR 0009 明确要求请求的视频在边界不同步时失败；本次没有将这一规定单独判为
  错误。R02 针对的是输出进程死亡无法有界收尾，而不是要求发布不一致的视频。

历史性能结论及后续修复记录见
[渐进训练性能瓶颈](progressive-training-performance-bottlenecks.md)；官方 pose/LiDAR
方案见 [输入适配指导](official-pose-lidar-progressive-training-guidance.md)。
新输入 provider 可能绕过部分采样路径，但在旧 provider 仍受支持时，不能将 R07
未经修复地标为已解决。

## 7. 后续修复与状态更新要求

工程侧先验证恢复身份、进程监督和只读数据边界；语义侧分别修正诊断资格/来源与
partial API 行为；几何侧先补投影性质测试，再做真实训练质量对照。
这些工作可以形成不同变更，避免将工程修复、算法变化和大规模重构混在一次提交中。

关闭问题时，在对应 ID 下追加修复提交、回归测试、实验结果和剩余边界，并更新索引
状态。不得仅凭测试总数增加、代码重构完成或一次训练成功就关闭相关问题。
目前九项记录均未关闭。
