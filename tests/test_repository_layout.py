import hashlib
from pathlib import Path
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = (
    "diff-gaussian-rasterization",
    "fused-ssim",
    "simple-knn",
)
ARCHIVE_SHA256 = {
    "diff-gaussian-rasterization": (
        "4d45bf7975811afdc2f5758ca93187811fd0c4a751ddae02a773c147e0b32579"
    ),
    "fused-ssim": (
        "c0d0cb6df5113f7671df6523631abc6b7d952bda69b099f65bc4b318ab27f858"
    ),
    "simple-knn": (
        "33e48afb8c243ae49433f6f49869a115513f48fc643af1b80ea6021be8a89842"
    ),
}


class RepositoryLayoutTests(unittest.TestCase):
    def test_dependency_archives_are_preserved(self):
        archive_dir = ROOT / "third_party" / "archives"
        for name in DEPENDENCIES:
            archive = archive_dir / f"{name}.zip"
            self.assertTrue(archive.is_file())

            with zipfile.ZipFile(archive) as dependency_zip:
                self.assertIsNone(dependency_zip.testzip())

            digest = hashlib.sha256()
            with archive.open("rb") as archive_file:
                for chunk in iter(lambda: archive_file.read(1024 * 1024), b""):
                    digest.update(chunk)
            self.assertEqual(digest.hexdigest(), ARCHIVE_SHA256[name])

    def test_dependency_sources_are_clean(self):
        source_dir = ROOT / "third_party" / "src"
        forbidden_directory_names = {".git", "build", "__pycache__"}
        forbidden_file_names = {
            ".git",
            ".gitmodules",
            ".ninja_deps",
            ".ninja_log",
        }

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

    def test_first_party_packages_live_under_semantic_atdom(self):
        package_root = ROOT / "src" / "semantic_atdom"
        self.assertTrue((package_root / "__init__.py").is_file())
        for name in ("arguments", "scene", "gaussian_renderer", "utils"):
            self.assertTrue((package_root / name / "__init__.py").is_file())

    def test_old_top_level_package_directories_are_absent(self):
        for name in ("arguments", "scene", "gaussian_renderer", "utils", "lpipsPyTorch"):
            self.assertFalse((ROOT / name).exists())

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


if __name__ == "__main__":
    unittest.main()
