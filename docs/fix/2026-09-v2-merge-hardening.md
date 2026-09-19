# V2 合并前第一性原理审查与加固

日期：2026-09-19

来源分支：`feature/uavscenes-interval5-amtown01`

目标分支：`semantic/g1-evidence-v2`

## 决策边界

- 当前分支定义的 output schema v2 是唯一权威布局：`model/`、
  `products/maps/`、`metrics/`、`logs/`、`provenance/`、可选
  `renders/cameras/` 与 `videos/training_progress.mp4`。
- 不恢复旧分支的 `qa/`、`visualizations/`、数字 stage 公开目录或空产品占位。
- 本轮只修复正确性、数据安全、输出契约和可恢复性问题；训练入口的大规模拆分与
  共享缓存优化延期，避免在合并前扩大回归面。

## 第一性原理不变量与修复

### 输入和配准

- `persistent_union` 允许中间阶段暂时省略 identity，但最终 overlap 必须重新包含
  所有已观测 identity；否则没有完整 overlap 图就不能声称完成 exact rebuild。
- 相机是否需要替换不能只比较 FOV。现在同时比较完整 `RectifiedPinhole`，因此焦距
  相同但畸变或主点变化时也会重新载入 RGB、有效区域和稀疏点映射。
- 稀疏点在生成 mask 和访问下采样 mask 前统一过滤 NaN、Inf 与图像外坐标；少于
  三点或共线导致 Delaunay 失败时产生明确 sentinel，不再越界或异常崩溃。Part2
  会在任何采样或 `tri.simplices` 访问前跳过这类相机的新高斯初始化，
  但不会将相机从最终全图评估中删除。
- RGB、COLMAP preflight 与语义观测共用同一个 image identity 规则：只剥离
  已知图像后缀，不再把小数时间戳误当成文件扩展名截断。
- 一个阶段没有任何有效稀疏支持相机时立即失败，不再通过回退 iteration 形成无限
  循环。最终全相机渲染对 sentinel 相机采用全图 mask，并按最终 camera inventory
  精确验证 JPG 与 CSV 集合。
- 语义 TDOM 的 CUDA feature 通道上限等价于最多 15 类。请求地图或训练视频时，
  preflight 在启动训练进程前拒绝 16/26 类配置；仅 fusion-only 路径不受此限制。

### 终态和发布

- 开启语义时，成功提交前必须验证 `status=exact_final`、最终 geometry stage、
  `lag_batches=0`、errors/deferred 为空且历史证据不 stale；请求 TDOM 时还必须验证
  `products/maps/latest.json` 的哈希及 pair manifest 为 `complete/exact_final`。
- 最终 Gaussian 与语义状态由“移动唯一源文件”改为“同文件系统优先 hardlink、
  必要时复制到临时文件、校验大小和 SHA-256、原子安装”。任何收尾失败都保留
  final stage、语义恢复槽和显式 recovery checkpoint，避免完整训练结果不可恢复。
- semantic archive 额外验证 canonical recovery-context hash，防止 state、geometry
  与 context 错配后仍进入成功模型。
- manifest 的终态不可重写；`succeeded` 必须记录 `final_stage` 和
  `optimizer_iteration`。`outputs/latest` 只接受 succeeded run，但它只是非权威便利
  索引：更新失败保留旧指针并记录警告，不反向篡改已成功运行。
- `.work` 只在 succeeded manifest 提交后 best-effort 清理。未请求地图时不创建
  `products/maps` 空目录，也不在 artifacts 中声明 maps。

### 恢复和迁移

- rolling checkpoint 以原子替换的 PT payload 为 stage 权威；JSON sidecar 仅供人读。
  即使进程在 PT 与 sidecar 两次替换之间退出，新 PT 仍然可恢复。
- legacy → schema v2 迁移先在 run 的 sibling staging 构建完整结果，再以目录 rename
  交换；失败会恢复原 legacy root。相机渲染只发布到 `renders/cameras/`，不再安装或
  生成废弃的 `qa/`；没有相机渲染时，manifest 完全省略 `renders`，而不是
  发布含义不清的 `null`。

## 验收

- `semantic-atdom` 环境完整测试：227 passed。
- 真实 UAVScenes 小范围：stage 13–14，最小训练迭代，启用语义、成对 TDOM、
  `--render_all_cameras` 与 `--render_training_progress`。
- 结果：manifest schema v2 / `succeeded`，`final_stage=14`、
  `optimizer_iteration=4`；semantic `exact_final`、lag 0、errors 空；地图 pair
  `complete`；14/14 相机 JPG 和 CSV 完整；视频为 3 帧 H.264、3672×1080、2 fps；
  成功后无 `.work`。

本次真实验收使用临时输出根，避免改变数据集正式 `outputs/latest` 或增加诊断运行。
