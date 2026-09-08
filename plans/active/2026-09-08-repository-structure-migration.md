# Repository Structure Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the inherited A-TDOM files into a clean `src/semantic_atdom` repository layout without changing training, rendering, dataset, or checkpoint behavior.

**Architecture:** Move the existing first-party packages unchanged beneath one `semantic_atdom` namespace, classify existing executable scripts under `tools/`, and separate offline third-party sources, documentation, and assets. Keep the three large training scripts intact during this migration; only imports and path references may change.

**Tech Stack:** Python 3.7, setuptools, Conda, PyTorch 1.12.1, CUDA 11.6, built-in `unittest`, Git.

**Spec:** `docs/design/repository-structure-v1.md`

## Global Constraints

- Modify only `/home/xuzipeng/code/Semantic-A-TDOM/Semantic-A-TDOM/`.
- Treat `/home/xuzipeng/code/Semantic-A-TDOM/ref/A-TDOM/` as read-only; its HEAD must remain `e50a5f7986664ca97b135c066b4cd462eb0bb4bf` and its tracked worktree must remain unchanged.
- Do not read from or copy code from `ref/A-TDOM_Online/`.
- Do not implement semantic behavior or create a placeholder semantic source package.
- Do not split or rewrite the three training implementations.
- Do not fix historical logic inside `tools/legacy/`.
- Do not change CLI argument meanings, dataset formats, checkpoint formats, or rendering behavior.
- Do not retain top-level compatibility wrappers for old script names.
- Preserve all three dependency ZIP archives and all real image assets.
- Do not commit build outputs, compiled libraries, caches, real datasets, checkpoints, or run outputs.
- Use `apply_patch` for textual edits and `git mv` for tracked relocations.
- Run each task from `/home/xuzipeng/code/Semantic-A-TDOM/Semantic-A-TDOM/` unless the command says otherwise.

---

### Task 1: Establish the baseline guard and clean third-party layout

**Files:**
- Create: `.gitignore`
- Create: `tests/test_repository_layout.py`
- Modify: `environment.yml`
- Modify: `README.md`
- Move: `submodules/diff-gaussian-rasterization.zip` → `third_party/archives/diff-gaussian-rasterization.zip`
- Move: `submodules/fused-ssim.zip` → `third_party/archives/fused-ssim.zip`
- Move: `submodules/simple-knn.zip` → `third_party/archives/simple-knn.zip`
- Extract: `third_party/archives/*.zip` → `third_party/src/<dependency>/`
- Move: `lpipsPyTorch/` → `third_party/src/lpipsPyTorch/`

**Interfaces:**
- Consumes: existing dependency ZIP files and the original `environment.yml` local pip entries.
- Produces: clean, offline-installable source directories at `third_party/src/{diff-gaussian-rasterization,fused-ssim,simple-knn}` and preserved archives at `third_party/archives/`.

Vendored sources under `third_party/src/` retain their upstream formatting, including any existing whitespace. Task 1 therefore excludes that tree from Git whitespace checks instead of rewriting third-party code; the structural test and explicit artifact searches separately enforce third-party cleanliness.

- [ ] **Step 1: Verify and tag the inherited baseline**

Run:

```bash
git -C ../ref/A-TDOM rev-parse HEAD
git -C ../ref/A-TDOM diff --quiet
git -C ../ref/A-TDOM diff --cached --quiet
git -C ../ref/A-TDOM status --short
git tag baseline-atdom-e50a5f7 e50a5f7
git show-ref --verify refs/tags/baseline-atdom-e50a5f7
```

Expected:

- HEAD prints `e50a5f7986664ca97b135c066b4cd462eb0bb4bf`.
- Both tracked-diff commands return exit code 0.
- Status shows only the three pre-existing untracked `__pycache__/` directories under `arguments/`, `scene/`, and `utils/`.
- The tag resolves to the inherited A-TDOM commit `e50a5f7` in the Semantic-A-TDOM repository.

- [ ] **Step 2: Add a failing structural test for third-party dependencies**

Create `tests/test_repository_layout.py` with:

```python
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = (
    "diff-gaussian-rasterization",
    "fused-ssim",
    "simple-knn",
)


class RepositoryLayoutTests(unittest.TestCase):
    def test_dependency_archives_are_preserved(self):
        archive_dir = ROOT / "third_party" / "archives"
        for name in DEPENDENCIES:
            self.assertTrue((archive_dir / f"{name}.zip").is_file())

    def test_dependency_sources_are_clean(self):
        source_dir = ROOT / "third_party" / "src"
        forbidden_directory_names = {".git", "build", "__pycache__"}
        forbidden_file_names = {".ninja_deps", ".ninja_log"}

        for name in DEPENDENCIES:
            dependency = source_dir / name
            self.assertTrue((dependency / "setup.py").is_file())
            for path in dependency.rglob("*"):
                if path.is_dir():
                    self.assertNotIn(path.name, forbidden_directory_names)
                    self.assertFalse(path.name.endswith(".egg-info"))
                elif path.is_file():
                    self.assertNotIn(path.name, forbidden_file_names)
                    self.assertNotIn(path.suffix, {".o", ".pyc", ".so"})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run the structural test and verify it fails before migration**

Run:

```bash
python -m unittest discover -s tests -p 'test_repository_layout.py' -v
```

Expected: FAIL because `third_party/archives/` and the clean extracted dependency directories do not exist yet.

- [ ] **Step 4: Add repository-wide ignore rules**

Create `.gitignore` with:

```gitignore
__pycache__/
*.py[cod]
*.so
*.o
.pytest_cache/
.mypy_cache/
.ruff_cache/
build/
dist/
*.egg-info/

outputs/
runs/
checkpoints/
data/
semantic_labels/
```

- [ ] **Step 5: Move the offline archives and vendored LPIPS source**

Run these explicit relocations:

```bash
mkdir -p third_party/archives third_party/src
git mv submodules/diff-gaussian-rasterization.zip third_party/archives/diff-gaussian-rasterization.zip
git mv submodules/fused-ssim.zip third_party/archives/fused-ssim.zip
git mv submodules/simple-knn.zip third_party/archives/simple-knn.zip
git mv lpipsPyTorch third_party/src/lpipsPyTorch
```

Do not move or delete `submodules/1` in this task; all one-byte placeholders are removed together in Task 4.

- [ ] **Step 6: Extract each archive and inspect removal targets**

Run:

```bash
unzip -q third_party/archives/diff-gaussian-rasterization.zip -d third_party/src
unzip -q third_party/archives/fused-ssim.zip -d third_party/src
unzip -q third_party/archives/simple-knn.zip -d third_party/src

find third_party/src/diff-gaussian-rasterization third_party/src/fused-ssim third_party/src/simple-knn \
  -type d \( -name .git -o -name build -o -name __pycache__ -o -name '*.egg-info' \) -print
find third_party/src/diff-gaussian-rasterization third_party/src/fused-ssim third_party/src/simple-knn \
  -type f \( -name '*.so' -o -name '*.o' -o -name '*.pyc' -o -name .ninja_deps -o -name .ninja_log \) -print
```

Expected: the commands print the stale build, package metadata, nested Git, compiled-library, and cache paths that came from the archives. Confirm every printed path is inside one of the three explicit dependency roots before continuing.

- [ ] **Step 7: Remove only the inspected generated artifacts**

Run against the three explicit dependency roots:

```bash
find third_party/src/diff-gaussian-rasterization third_party/src/fused-ssim third_party/src/simple-knn \
  -type d \( -name .git -o -name build -o -name __pycache__ -o -name '*.egg-info' \) -prune -exec rm -rf -- {} +
find third_party/src/diff-gaussian-rasterization third_party/src/fused-ssim third_party/src/simple-knn \
  -type f \( -name '*.so' -o -name '*.o' -o -name '*.pyc' -o -name .ninja_deps -o -name .ninja_log \) -delete
```

The three ZIP files are not removal targets and must remain untouched in `third_party/archives/`.

- [ ] **Step 8: Update local dependency paths**

In `environment.yml`, replace exactly:

```yaml
    - submodules/diff-gaussian-rasterization
    - submodules/simple-knn
    - submodules/fused-ssim
```

with:

```yaml
    - third_party/src/diff-gaussian-rasterization
    - third_party/src/simple-knn
    - third_party/src/fused-ssim
```

In the Windows installation block of `README.md`, make the same three path replacements. Replace the cloning note about manually decompressing `submodules/` with this repository-specific statement:

```markdown
The required CUDA dependencies are preserved as offline archives in
`third_party/archives/` and as clean build sources in `third_party/src/`.
```

- [ ] **Step 9: Run the third-party structural test**

Run:

```bash
python -m unittest discover -s tests -p 'test_repository_layout.py' -v
git diff --check -- . ':(exclude)third_party/src/**'
```

Expected: both dependency tests PASS and the scoped whitespace check produces no output.

- [ ] **Step 10: Commit the third-party layout**

Run:

```bash
git add .gitignore tests/test_repository_layout.py environment.yml README.md
git add -A -- third_party submodules lpipsPyTorch
git diff --cached --check -- . ':(exclude)third_party/src/**'
git commit -m "build: organize offline dependencies"
```

Expected: the commit preserves all three ZIPs, adds clean source trees, moves `lpipsPyTorch`, and does not include compiled or cache files.

---

### Task 2: Move first-party packages under `src/semantic_atdom`

**Files:**
- Create: `pyproject.toml`
- Create: `src/semantic_atdom/__init__.py`
- Modify: `tests/test_repository_layout.py`
- Move: `arguments/` → `src/semantic_atdom/arguments/`
- Move: `scene/` → `src/semantic_atdom/scene/`
- Move: `gaussian_renderer/` → `src/semantic_atdom/gaussian_renderer/`
- Move: `utils/` → `src/semantic_atdom/utils/`
- Modify imports in: all Python files under the four moved packages
- Modify imports in: `ContinuosProgressiveTrain4.py`, `WindowsTrain.py`, `Window-On-the-Fly-Train.py`, `render.py`, `metrics.py`

**Interfaces:**
- Consumes: the original `arguments`, `scene`, `gaussian_renderer`, and `utils` packages without changing their public classes or functions.
- Produces: importable modules named `semantic_atdom.arguments`, `semantic_atdom.scene`, `semantic_atdom.gaussian_renderer`, and `semantic_atdom.utils`; preserves the third-party import name `lpipsPyTorch`.

- [ ] **Step 1: Extend the structural test with the required package layout**

Add these methods to `RepositoryLayoutTests` in `tests/test_repository_layout.py`:

```python
    def test_first_party_packages_live_under_semantic_atdom(self):
        package_root = ROOT / "src" / "semantic_atdom"
        self.assertTrue((package_root / "__init__.py").is_file())
        for name in ("arguments", "scene", "gaussian_renderer", "utils"):
            self.assertTrue((package_root / name / "__init__.py").is_file())

    def test_old_top_level_package_directories_are_absent(self):
        for name in ("arguments", "scene", "gaussian_renderer", "utils", "lpipsPyTorch"):
            self.assertFalse((ROOT / name).exists())
```

- [ ] **Step 2: Run the new tests and verify they fail before migration**

Run:

```bash
python -m unittest discover -s tests -p 'test_repository_layout.py' -v
```

Expected: the two new tests FAIL because the first-party packages are still at repository root and `src/semantic_atdom` does not exist.

- [ ] **Step 3: Create the minimal project package metadata**

Create `pyproject.toml` with:

```toml
[build-system]
requires = ["setuptools>=61,<68"]
build-backend = "setuptools.build_meta"

[project]
name = "semantic-atdom"
version = "0.1.0"
description = "Semantic extension of A-TDOM for progressive orthophoto generation"
readme = "README.md"
requires-python = ">=3.7"

[tool.setuptools]
packages = [
    "semantic_atdom",
    "semantic_atdom.arguments",
    "semantic_atdom.scene",
    "semantic_atdom.gaussian_renderer",
    "semantic_atdom.utils",
    "lpipsPyTorch",
    "lpipsPyTorch.modules",
]
package-dir = {semantic_atdom = "src/semantic_atdom", lpipsPyTorch = "third_party/src/lpipsPyTorch"}
```

Create `src/semantic_atdom/__init__.py` with:

```python
"""Semantic-A-TDOM project package."""
```

- [ ] **Step 4: Move the four first-party packages without splitting files**

Run:

```bash
mkdir -p src/semantic_atdom
git mv arguments src/semantic_atdom/arguments
git mv scene src/semantic_atdom/scene
git mv gaussian_renderer src/semantic_atdom/gaussian_renderer
git mv utils src/semantic_atdom/utils
```

Do not rename internal files or classes.

- [ ] **Step 5: Apply the exact namespace-only import conversion**

Use `apply_patch` to apply these replacements everywhere they occur in the moved packages and the five listed root scripts:

```text
from arguments                         → from semantic_atdom.arguments
from scene                             → from semantic_atdom.scene
from gaussian_renderer                 → from semantic_atdom.gaussian_renderer
from utils                             → from semantic_atdom.utils
```

The complete files that contain these imports are:

```text
ContinuosProgressiveTrain4.py
Window-On-the-Fly-Train.py
WindowsTrain.py
metrics.py
render.py
src/semantic_atdom/gaussian_renderer/__init__.py
src/semantic_atdom/gaussian_renderer/network_gui.py
src/semantic_atdom/scene/__init__.py
src/semantic_atdom/scene/cameras.py
src/semantic_atdom/scene/dataset_readers.py
src/semantic_atdom/scene/gaussian_model.py
src/semantic_atdom/utils/camera_utils.py
```

For example, the moved renderer imports must read:

```python
from semantic_atdom.scene.gaussian_model import GaussianModel
from semantic_atdom.utils.sh_utils import eval_sh
```

Keep this third-party import unchanged:

```python
from lpipsPyTorch import lpips
```

- [ ] **Step 6: Verify no old first-party imports remain**

Run:

```bash
rg -n '^(from|import) (arguments|scene|gaussian_renderer|utils)(\.|\s|$)' src *.py
```

Expected: no matches. An `rg` exit code of 1 is the expected no-match result.

- [ ] **Step 7: Install the editable package and run package checks**

Run:

```bash
python -m pip install --no-deps -e .
python -m compileall -q src
python -m unittest discover -s tests -p 'test_repository_layout.py' -v
python -c "from importlib.util import find_spec; assert find_spec('semantic_atdom.scene'); assert find_spec('semantic_atdom.gaussian_renderer'); assert find_spec('lpipsPyTorch')"
git diff --check
```

Expected: installation succeeds; compilation is silent; all current tests PASS; all three package specs are found; whitespace check is silent.

- [ ] **Step 8: Commit the project package migration**

Run:

```bash
git add pyproject.toml src tests/test_repository_layout.py ContinuosProgressiveTrain4.py Window-On-the-Fly-Train.py WindowsTrain.py render.py metrics.py
git diff --cached --check
git commit -m "refactor: namespace project source package"
```

Expected: Git reports package-directory renames plus import-only changes; no function body is split or rewritten.

---

### Task 3: Classify and relocate all executable scripts

**Files:**
- Modify: `tests/test_repository_layout.py`
- Modify: `README.md`
- Move: `ContinuosProgressiveTrain4.py` → `tools/train_progressive.py`
- Move: `WindowsTrain.py` → `tools/train_windowed.py`
- Move: `Window-On-the-Fly-Train.py` → `tools/train_gui.py`
- Move: `render.py` → `tools/render.py`
- Move: `convert.py` → `tools/convert_colmap.py`
- Move: `metrics.py` → `tools/evaluate_metrics.py`
- Move: `DatasetPrepare.py` → `tools/legacy/prepare_dataset.py`
- Move: `DownSample.py` → `tools/legacy/downsample_images.py`
- Move: `GetDemo.py` → `tools/legacy/create_demo_video.py`
- Move: `Test.py` → `tools/legacy/export_colmap_stages.py`
- Move: `full_eval.py` → `tools/legacy/full_eval.py`

**Interfaces:**
- Consumes: the package imports created in Task 2.
- Produces: the supported entry paths `tools/train_progressive.py`, `tools/train_windowed.py`, `tools/train_gui.py`, `tools/render.py`, `tools/convert_colmap.py`, and `tools/evaluate_metrics.py`; historical helpers remain available only under `tools/legacy/`.

- [ ] **Step 1: Extend the structural test with exact tool expectations**

Add these methods to `RepositoryLayoutTests`:

```python
    def test_supported_tools_have_explicit_names(self):
        supported = {
            "train_progressive.py",
            "train_windowed.py",
            "train_gui.py",
            "render.py",
            "convert_colmap.py",
            "evaluate_metrics.py",
        }
        self.assertEqual(supported, {path.name for path in (ROOT / "tools").glob("*.py")})

    def test_historical_helpers_are_isolated(self):
        historical = {
            "prepare_dataset.py",
            "downsample_images.py",
            "create_demo_video.py",
            "export_colmap_stages.py",
            "full_eval.py",
        }
        self.assertEqual(historical, {path.name for path in (ROOT / "tools" / "legacy").glob("*.py")})

    def test_repository_root_has_no_python_entry_scripts(self):
        self.assertEqual([], list(ROOT.glob("*.py")))
```

- [ ] **Step 2: Run the new tool tests and verify they fail before migration**

Run:

```bash
python -m unittest discover -s tests -p 'test_repository_layout.py' -v
```

Expected: the three new tests FAIL because the executable scripts are still at repository root.

- [ ] **Step 3: Move each executable to its approved location**

Run:

```bash
mkdir -p tools/legacy
git mv ContinuosProgressiveTrain4.py tools/train_progressive.py
git mv WindowsTrain.py tools/train_windowed.py
git mv Window-On-the-Fly-Train.py tools/train_gui.py
git mv render.py tools/render.py
git mv convert.py tools/convert_colmap.py
git mv metrics.py tools/evaluate_metrics.py
git mv DatasetPrepare.py tools/legacy/prepare_dataset.py
git mv DownSample.py tools/legacy/downsample_images.py
git mv GetDemo.py tools/legacy/create_demo_video.py
git mv Test.py tools/legacy/export_colmap_stages.py
git mv full_eval.py tools/legacy/full_eval.py
```

- [ ] **Step 4: Update only path references made stale by the moves**

In `tools/legacy/full_eval.py`, replace:

```python
os.system("python render.py --iteration 7000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)
os.system("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)
os.system("python metrics.py -m " + scenes_string)
```

with:

```python
os.system("python tools/render.py --iteration 7000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)
os.system("python tools/render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)
os.system("python tools/evaluate_metrics.py -m " + scenes_string)
```

Leave its existing `python train.py` calls unchanged and document the file as historical; creating or guessing a replacement training command would be a behavior fix outside this migration.

- [ ] **Step 5: Update README command names and add the migration map**

Replace every README occurrence as follows:

```text
python Window-On-the-Fly-Train.py   → python tools/train_gui.py
Window-On-the-Fly-Train.py          → tools/train_gui.py
python ContinuosProgressiveTrain4.py → python tools/train_progressive.py
ContinuosProgressiveTrain4.py       → tools/train_progressive.py
```

Add this concise section after the repository setup instructions:

```markdown
## Command layout

Project entry points live under `tools/`. The directory migration renamed the
original scripts as follows:

| Original name | Current path |
|---|---|
| `ContinuosProgressiveTrain4.py` | `tools/train_progressive.py` |
| `WindowsTrain.py` | `tools/train_windowed.py` |
| `Window-On-the-Fly-Train.py` | `tools/train_gui.py` |
| `render.py` | `tools/render.py` |
| `convert.py` | `tools/convert_colmap.py` |
| `metrics.py` | `tools/evaluate_metrics.py` |

The scripts under `tools/legacy/` are retained for traceability and are not
supported command interfaces.
```

- [ ] **Step 6: Verify the tool layout and syntax**

Run:

```bash
python -m compileall -q src tools
python -m unittest discover -s tests -p 'test_repository_layout.py' -v
python tools/convert_colmap.py --help
rg -n 'ContinuosProgressiveTrain4\.py|Window-On-the-Fly-Train\.py|WindowsTrain\.py' README.md tools
git diff --check
```

Expected:

- Compilation is silent.
- All current tests PASS.
- The COLMAP tool prints its argument help and exits with code 0.
- The old-name search finds only the intentional migration table in `README.md`; it must not find executable files or live command examples.
- Whitespace check is silent.

- [ ] **Step 7: Commit the executable layout**

Run:

```bash
git add tools README.md tests/test_repository_layout.py
git diff --cached --check
git commit -m "refactor: organize project entry scripts"
```

Expected: all eleven root scripts are represented by one renamed file each, and repository root contains no Python entry script.

---

### Task 4: Organize documentation and assets, remove placeholders, and run final verification

**Files:**
- Modify: `tests/test_repository_layout.py`
- Modify: `README.md`
- Move: `docs/semantic-tdom-v1-decisions.md` → `docs/design/semantic-tdom-v1.md`
- Move: `tasks/CONTEXT.md` → `docs/design/semantic-tdom-glossary.md`
- Move: `tasks/adr/*.md` → `docs/decisions/*.md`
- Move: `0.png` → `assets/docs/overview.png`
- Move: documentation figures under `assets/` → `assets/docs/`
- Move: logo files under `assets/` → `assets/docs/logos/`
- Delete: every one-byte file named `1`
- Delete when empty: `tasks/`, `On-The-Fly/`, `demo_data/`, `submodules/`

**Interfaces:**
- Consumes: the approved semantic design documents already present in the worktree and the existing README image assets.
- Produces: canonical design and decision documentation under `docs/`, documentation-only images under `assets/docs/`, and a placeholder-free repository.

- [ ] **Step 1: Extend the structural test with documentation, asset, and placeholder rules**

Add these methods to `RepositoryLayoutTests`:

```python
    def test_semantic_documents_have_canonical_locations(self):
        self.assertTrue((ROOT / "docs" / "design" / "semantic-tdom-v1.md").is_file())
        self.assertTrue((ROOT / "docs" / "design" / "semantic-tdom-glossary.md").is_file())
        decisions = ROOT / "docs" / "decisions"
        self.assertEqual(4, len(list(decisions.glob("*.md"))))
        self.assertFalse((ROOT / "tasks").exists())

    def test_documentation_assets_are_grouped(self):
        self.assertTrue((ROOT / "assets" / "docs" / "overview.png").is_file())
        self.assertTrue((ROOT / "assets" / "docs" / "logos").is_dir())
        self.assertFalse((ROOT / "0.png").exists())

    def test_one_byte_placeholder_files_are_absent(self):
        placeholders = [
            path.relative_to(ROOT)
            for path in ROOT.rglob("1")
            if path.is_file() and ".git" not in path.parts
        ]
        self.assertEqual([], placeholders)
```

- [ ] **Step 2: Run the new tests and verify they fail before migration**

Run:

```bash
python -m unittest discover -s tests -p 'test_repository_layout.py' -v
```

Expected: the three new tests FAIL because the documents, images, and placeholder files still use their old locations.

- [ ] **Step 3: Move the semantic design material into its canonical documentation directories**

Run:

```bash
mkdir -p docs/design docs/decisions
mv docs/semantic-tdom-v1-decisions.md docs/design/semantic-tdom-v1.md
mv tasks/CONTEXT.md docs/design/semantic-tdom-glossary.md
mv tasks/adr/0001-one-way-semantic-dependency.md docs/decisions/0001-one-way-semantic-dependency.md
mv tasks/adr/0002-dirichlet-semantic-field.md docs/decisions/0002-dirichlet-semantic-field.md
mv tasks/adr/0003-stage-synchronous-semantics.md docs/decisions/0003-stage-synchronous-semantics.md
mv tasks/adr/0004-clean-atdom-lineage.md docs/decisions/0004-clean-atdom-lineage.md
```

Use `apply_patch` to change the first heading of `docs/design/semantic-tdom-glossary.md` from:

```markdown
# Semantic TDOM Context
```

to:

```markdown
# Semantic TDOM Glossary
```

Do not alter the confirmed semantic definitions or decisions.

- [ ] **Step 4: Move every real documentation image without deleting content**

Run:

```bash
mkdir -p assets/docs/logos
git mv 0.png assets/docs/overview.png
git mv assets/better.png assets/docs/better.png
git mv assets/select.png assets/docs/select.png
git mv assets/teaser.png assets/docs/teaser.png
git mv assets/times_compare.png assets/docs/times_compare.png
git mv assets/worse.png assets/docs/worse.png
git mv assets/logo_graphdeco.png assets/docs/logos/logo_graphdeco.png
git mv assets/logo_inria.png assets/docs/logos/logo_inria.png
git mv assets/logo_mpi.png assets/docs/logos/logo_mpi.png
git mv assets/logo_mpi.svg assets/docs/logos/logo_mpi.svg
git mv assets/logo_uca.png assets/docs/logos/logo_uca.png
```

Update the README image reference exactly:

```text
![image](0.png) → ![image](assets/docs/overview.png)
```

- [ ] **Step 5: Remove the known one-byte placeholders from their post-migration paths**

Run only these explicit removals:

```bash
git rm On-The-Fly/1
git rm On-The-Fly/phantom3/images/1
git rm assets/1
git rm demo_data/1
git rm submodules/1
git rm src/semantic_atdom/arguments/1
git rm src/semantic_atdom/gaussian_renderer/1
git rm src/semantic_atdom/scene/1
git rm src/semantic_atdom/utils/1
git rm third_party/src/lpipsPyTorch/1
git rm third_party/src/lpipsPyTorch/modules/1
```

After these files are removed, Git will not retain their now-empty parent directories. Do not add `.gitkeep` replacements.

- [ ] **Step 6: Run the complete required verification suite**

Run:

```bash
python -m pip install --no-deps -e .
python -m compileall -q src tools
python -m unittest discover -s tests -p 'test_repository_layout.py' -v
python -c "from importlib.util import find_spec; assert find_spec('semantic_atdom.scene'); assert find_spec('semantic_atdom.gaussian_renderer'); assert find_spec('semantic_atdom.arguments'); assert find_spec('semantic_atdom.utils'); assert find_spec('lpipsPyTorch')"
python tools/convert_colmap.py --help
git diff --check
find . -maxdepth 1 -type f -name '*.py' -print
find . -type f -name 1 -not -path './.git/*' -print
```

Expected:

- Editable installation succeeds.
- Compilation is silent.
- Every `unittest` test PASSes.
- All five package specs resolve.
- `convert_colmap.py --help` exits with code 0.
- Both `find` commands produce no output.
- `git diff --check` produces no output.

- [ ] **Step 7: Verify forbidden legacy imports and generated third-party artifacts are absent**

Run:

```bash
rg -n '^(from|import) (arguments|scene|gaussian_renderer|utils)(\.|\s|$)' src tools
find third_party/src/diff-gaussian-rasterization third_party/src/fused-ssim third_party/src/simple-knn \
  -type d \( -name .git -o -name build -o -name __pycache__ -o -name '*.egg-info' \) -print
find third_party/src/diff-gaussian-rasterization third_party/src/fused-ssim third_party/src/simple-knn \
  -type f \( -name '*.so' -o -name '*.o' -o -name '*.pyc' -o -name .ninja_deps -o -name .ninja_log \) -print
```

Expected: all three searches produce no matches. `rg` exit code 1 is the expected no-match result.

- [ ] **Step 8: Attempt environment-dependent entry-point smoke tests**

Run each command separately so one missing optional dependency does not hide the remaining results:

```bash
python tools/render.py --help
python tools/evaluate_metrics.py --help
python tools/train_progressive.py --help
python tools/train_windowed.py --help
python tools/train_gui.py --help
```

Expected when the existing CUDA/PyTorch/PyQt dependencies are installed: each command prints help and exits with code 0. If a command cannot import an existing external dependency, record that exact command and exception in the final handoff as **not verified**; do not modify training or rendering logic to solve environment setup during this migration.

- [ ] **Step 9: Recheck the read-only baseline repository**

Run:

```bash
git -C ../ref/A-TDOM rev-parse HEAD
git -C ../ref/A-TDOM diff --quiet
git -C ../ref/A-TDOM diff --cached --quiet
git -C ../ref/A-TDOM status --short
```

Expected:

- HEAD remains `e50a5f7986664ca97b135c066b4cd462eb0bb4bf`.
- Both tracked-diff commands return exit code 0.
- The same three pre-existing untracked `__pycache__/` entries remain; no additional path appears.

- [ ] **Step 10: Review the staged migration and commit it**

Run:

```bash
git add README.md assets docs tests/test_repository_layout.py
git status --short
git diff --cached --find-renames=80% --summary
git diff --cached --check
git commit -m "docs: organize project documentation and assets"
git status --short
```

Expected:

- Before commit, all intended documentation and image moves are visible and no real dataset path is staged.
- After commit, the Semantic-A-TDOM worktree is clean apart from ignored build or cache files.
- The read-only `ref/A-TDOM/` repository remains unchanged.

---

## Completion Criteria

The migration is complete only when all of the following are true:

- The four first-party packages import through `semantic_atdom.*` and no longer exist at repository root.
- All eleven executable scripts have exactly one approved destination under `tools/`.
- The three dependency ZIPs and their clean source directories coexist under `third_party/`.
- `pip install -e .`, compilation, structural unit tests, and the dependency-independent CLI help check pass.
- Existing semantic documents and all real image assets are preserved in their approved directories.
- No one-byte `1` placeholder file remains.
- No semantic algorithm, training flow, checkpoint format, or data contract has changed.
- `ref/A-TDOM/` remains at its original commit with no tracked changes.
- Any environment-dependent smoke test that could not run is explicitly reported as not verified.
