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
