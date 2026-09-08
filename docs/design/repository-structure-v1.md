# Semantic-A-TDOM 仓库结构重构设计

## 状态

- 日期：2026-09-08
- 状态：已经讨论确认，待编写实施计划
- 适用仓库：`code/Semantic-A-TDOM/Semantic-A-TDOM/`

## 目标

将当前混在仓库根目录的项目源码、命令入口、第三方依赖、文档、图片和占位内容分开，形成清晰的 Python `src` 布局，为后续 Semantic TDOM 开发提供稳定的仓库边界。

本次是行为保持的目录迁移，不是训练、渲染或语义算法重构。

## 仓库边界

- 只修改 `code/Semantic-A-TDOM/Semantic-A-TDOM/`。
- `code/Semantic-A-TDOM/ref/A-TDOM/` 是原版对照基线，必须保持完全不变。
- `A-TDOM_Online` 只说明过去的课题工作，不用于本次目录设计或代码迁移。
- 原始 A-TDOM 状态由 Git 历史和新仓库中的 `baseline-atdom-e50a5f7` tag 保留，新仓库内不建立第二份 `legacy/atdom` 源码副本。

## 非目标

本次不做以下工作：

- 不实现 `SemanticGaussianField` 或任何语义算法；
- 不设计 densify、clone、split 或 prune 时的语义状态规则；
- 不拆分或重写现有大型训练脚本；
- 不修复 `tools/legacy/` 中的硬编码路径或历史逻辑；
- 不改变训练参数含义、数据集格式、checkpoint 格式或渲染结果；
- 不引入 YAML 配置系统或统一子命令 CLI；
- 不执行完整训练或语义评测。

## 已选方案

采用单一 `src/semantic_atdom` Python 包，但在第一阶段保留现有内部包名和文件边界。这样可以清理根目录并建立项目命名空间，同时避免在目录迁移中夹带代码职责拆分。

未选方案：

1. **只整理根目录的最小方案**：风险低，但 `scene/`、`utils/` 等仍作为顶层包散落在根目录，无法解决根本问题。
2. **A-TDOM 核心加语义插件**：隔离强，但当前需要额外设计插件协议和事件系统，超出目录重构范围。
3. **立即按 `data/geometry/training/common` 拆分**：长期方向正确，但会要求拆分现有文件和重写内部接口，不再是行为保持的机械迁移。

## 目标目录

第一阶段完成后，仓库应形成以下结构。没有内容的规划目录不使用占位文件强行提交。

```text
Semantic-A-TDOM/
├── pyproject.toml
├── environment.yml
├── README.md
│
├── src/
│   └── semantic_atdom/
│       ├── __init__.py
│       ├── arguments/
│       ├── scene/
│       ├── gaussian_renderer/
│       └── utils/
│
├── tools/
│   ├── train_progressive.py
│   ├── train_windowed.py
│   ├── train_gui.py
│   ├── render.py
│   ├── convert_colmap.py
│   ├── evaluate_metrics.py
│   └── legacy/
│       ├── prepare_dataset.py
│       ├── downsample_images.py
│       ├── create_demo_video.py
│       ├── export_colmap_stages.py
│       └── full_eval.py
│
├── third_party/
│   ├── archives/
│   │   ├── diff-gaussian-rasterization.zip
│   │   ├── fused-ssim.zip
│   │   └── simple-knn.zip
│   └── src/
│       ├── diff-gaussian-rasterization/
│       ├── fused-ssim/
│       ├── simple-knn/
│       └── lpipsPyTorch/
│
├── assets/
│   └── docs/
│       ├── overview.png
│       └── logos/
│
├── docs/
│   ├── design/
│   ├── decisions/
│   ├── research/
│   └── guides/
│
├── plans/
│   ├── active/
│   └── completed/
│
└── tests/
    ├── unit/
    ├── integration/
    └── fixtures/
```

`plans/` 和 `tests/` 只在首个实际文件需要落地时创建对应子目录，不使用 `.gitkeep` 或名为 `1` 的文件占位。

## 源码迁移映射

| 当前路径 | 目标路径 | 处理 |
|---|---|---|
| `arguments/` | `src/semantic_atdom/arguments/` | 整体移动，不拆分 |
| `scene/` | `src/semantic_atdom/scene/` | 整体移动，不拆分 |
| `gaussian_renderer/` | `src/semantic_atdom/gaussian_renderer/` | 整体移动，不拆分 |
| `utils/` | `src/semantic_atdom/utils/` | 整体移动，不拆分 |
| `lpipsPyTorch/` | `third_party/src/lpipsPyTorch/` | 作为第三方源码保留 |

项目内部 import 只做命名空间转换，例如：

```python
from scene import Scene
from utils.loss_utils import l1_loss
```

转换为：

```python
from semantic_atdom.scene import Scene
from semantic_atdom.utils.loss_utils import l1_loss
```

本次不将 `scene` 改名为 `geometry`，不将 `utils` 拆为 `common`，也不创建没有实现的语义源码目录。

## 命令入口迁移映射

| 当前文件 | 目标文件 | 定位 |
|---|---|---|
| `ContinuosProgressiveTrain4.py` | `tools/train_progressive.py` | 当前主要渐进训练入口 |
| `WindowsTrain.py` | `tools/train_windowed.py` | 窗口化训练入口 |
| `Window-On-the-Fly-Train.py` | `tools/train_gui.py` | GUI/在线训练入口 |
| `render.py` | `tools/render.py` | 渲染入口 |
| `convert.py` | `tools/convert_colmap.py` | COLMAP 转换入口 |
| `metrics.py` | `tools/evaluate_metrics.py` | 现有 RGB 指标入口 |
| `DatasetPrepare.py` | `tools/legacy/prepare_dataset.py` | 硬编码数据准备脚本 |
| `DownSample.py` | `tools/legacy/downsample_images.py` | 硬编码图像处理脚本 |
| `GetDemo.py` | `tools/legacy/create_demo_video.py` | 硬编码视频生成脚本 |
| `Test.py` | `tools/legacy/export_colmap_stages.py` | 历史 COLMAP 阶段导出脚本 |
| `full_eval.py` | `tools/legacy/full_eval.py` | 依赖不存在的 `train.py` 的历史入口 |

本次不保留原脚本名的顶层兼容包装。旧命令与新命令的映射记录在迁移后 README 中。

三个训练入口在第一阶段允许作为临时例外继续包含现有大型实现。将它们变为真正的薄入口，必须在后续独立的训练模块重构中完成。

## Python 打包与环境

- 增加最小 `pyproject.toml`，只负责项目元数据、构建后端和本地 Python 包映射。
- 项目源码使用 `pip install -e .` 进行可编辑安装。
- `environment.yml` 继续是 CUDA、PyTorch 和运行依赖的环境来源；本次只更新因目录迁移而失效的本地路径。
- 不在命令入口中修改 `sys.path`，也不依赖从仓库根目录启动才能 import 项目包。
- `pyproject.toml` 显式将 `src/semantic_atdom/` 映射为 `semantic_atdom` 包，并将 `third_party/src/lpipsPyTorch/` 映射为保持原名的 `lpipsPyTorch` 包。因此一次 `pip install -e .` 同时提供项目包和该本地第三方包，不修改其内部行为。

## 第三方依赖

以下三个 ZIP 是必要的离线依赖资产，不得删除：

- `diff-gaussian-rasterization.zip`
- `fused-ssim.zip`
- `simple-knn.zip`

处理规则：

1. 原始 ZIP 移入 `third_party/archives/`。
2. 各 ZIP 解压到 `third_party/src/` 下的同名目录。
3. 解压源码中不保留 `build/`、已编译 `.so`、Ninja 缓存、`__pycache__/` 或嵌套 `.git/`。
4. 新生成的构建产物由 `.gitignore` 忽略。
5. 未来由 Semantic-A-TDOM 自行维护的 CUDA/C++ 语义扩展才放入 `extensions/`；本次不创建空扩展。

## 数据与运行产物

真实影像、`semantic_labels`、checkpoint、训练输出和渲染结果不进入 Git 仓库。它们继续位于 `/data3/...` 等外部路径，并通过现有命令参数传入。

本次删除只含 1 字节占位文件的 `On-The-Fly/` 和 `demo_data/`，不删除或移入任何真实数据。以后的小型测试资产只在实际测试需要时加入 `tests/fixtures/`。

## 文档、计划与图片

- `docs/design/`：已确认的系统和模块设计。
- `docs/decisions/`：难以逆转且存在真实取舍的关键决策记录。
- `docs/research/`：论文调研、方案比较和推导记录。
- `docs/guides/`：安装、数据准备、训练和渲染说明。
- `plans/active/` 与 `plans/completed/`：只放实施计划，不放长期架构知识。

现有文档的目标映射：

| 当前路径 | 目标路径 |
|---|---|
| `docs/semantic-tdom-v1-decisions.md` | `docs/design/semantic-tdom-v1.md` |
| `tasks/adr/*.md` | `docs/decisions/*.md` |
| `tasks/CONTEXT.md` | `docs/design/semantic-tdom-glossary.md` |

`tasks/` 在内容迁移后删除。

所有现有真实图片移入 `assets/docs/`，其中根目录 `0.png` 改名为 `assets/docs/overview.png`，README 同步更新引用。Logo 归入 `assets/docs/logos/`。不删除真实图片。

## 占位文件

当前名为 `1` 的文件均为 1 字节占位符，不是程序输入或资产。迁移时删除所有这类文件，不使用新占位符替代。

## 验证标准

### 原版保护

- 记录并比较 `ref/A-TDOM/` 迁移前后的 HEAD、tracked diff 和工作区状态。
- 新仓库不从 `ref/A-TDOM/` 再次复制文件。

### 迁移完整性

- 每个原始 Python 文件必须在迁移表中有唯一目标。
- 除 import、本地依赖路径和因命名变更必须更新的命令引用外，不改动业务逻辑。
- 根目录不再存在项目 Python 入口或顶层内部包。

### 静态检查

```bash
python -m compileall src tools
git diff --check
```

搜索并消除项目内部的旧顶层 import：

```text
from scene ...
from utils ...
from arguments ...
from gaussian_renderer ...
```

### 安装与导入

- `pip install -e .` 成功。
- 当前环境具备必要依赖时，导入 `semantic_atdom.scene`、`semantic_atdom.gaussian_renderer`、`semantic_atdom.arguments` 和 `semantic_atdom.utils`。
- 对支持 `--help` 的主要命令入口运行帮助命令。

### 轻量运行

如果当前 CUDA 环境和已有小数据可直接使用，执行一次最小数据加载或单帧渲染冒烟测试。不启动完整训练。因环境或外部数据阻塞的检查必须记录为“未验证”，不得声称通过。

### 第三方检查

- 三个 ZIP 均存在于 `third_party/archives/`。
- 三个解压目录均包含可用的 `setup.py`。
- `third_party/src/` 不含压缩包附带的旧构建目录、已编译共享库、缓存或嵌套 Git 元数据。

## 实施顺序约束

实施计划应使迁移可分段检查：

1. 记录两个仓库基线状态；
2. 整理第三方依赖和忽略规则；
3. 迁移项目包并修正 import；
4. 迁移命令入口；
5. 迁移文档和图片，清理占位文件；
6. 更新安装与使用说明；
7. 逐项执行验收检查并记录未覆盖项。

任何在迁移中发现的训练、渲染或语义缺陷都应单独记录，不在本次目录重构中顺手修复。
