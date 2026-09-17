from argparse import Namespace
from datetime import datetime
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from semantic_atdom.utils.run_output import (
    RUN_OUTPUT_SCHEMA_VERSION,
    RunLayout,
    allocate_run_root,
    archive_semantic_state,
    finalize_stage_model,
    normalized_run_name,
)


class RunOutputTests(unittest.TestCase):
    def test_run_name_is_safe_and_descriptive(self):
        self.assertEqual(normalized_run_name("AMtown01 v2"), "AMtown01-v2")
        with self.assertRaises(ValueError):
            normalized_run_name("../")
        with self.assertRaises(ValueError):
            normalized_run_name("latest")

    def test_allocate_run_root_is_human_readable(self):
        with TemporaryDirectory() as directory:
            root = allocate_run_root(
                Path(directory),
                "amtown01-v2",
                repository_root=Path(__file__).parents[2],
                timestamp=datetime(2026, 9, 16, 14, 22, 2),
            )
            self.assertEqual(root.parent.name, "runs")
            self.assertRegex(
                root.name, r"^amtown01-v2_20260916-142202_[0-9a-f]{7}$"
            )

    def test_manifest_lifecycle_and_latest_only_after_success(self):
        with TemporaryDirectory() as directory:
            output_root = Path(directory) / "outputs"
            layout = RunLayout.from_root(output_root / "runs" / "run-a")
            layout.begin(
                source_dataset=Path(directory) / "dataset",
                repository_root=Path(__file__).parents[2],
                command=["train"],
                options={"render_training_progress": False},
            )
            self.assertEqual(
                layout.read_manifest()["schema_version"],
                RUN_OUTPUT_SCHEMA_VERSION,
            )
            self.assertEqual(layout.renders, layout.root / "renders")
            self.assertEqual(layout.videos, layout.root / "videos")
            self.assertEqual(layout.read_manifest()["status"], "running")
            self.assertFalse((output_root / "latest").exists())

            layout.set_status("failed", failure={"message": "boom"})
            self.assertEqual(layout.read_manifest()["status"], "failed")
            self.assertFalse((output_root / "latest").exists())

            layout.set_status("succeeded", artifacts={"model": "model"})
            latest = layout.publish_latest()
            self.assertEqual(latest.resolve(), layout.root)
            self.assertEqual(layout.read_manifest()["status"], "succeeded")

    def test_non_empty_output_is_never_silently_reused(self):
        with TemporaryDirectory() as directory:
            layout = RunLayout.from_root(Path(directory) / "run")
            layout.root.mkdir()
            (layout.root / "unrelated.txt").write_text("keep")

            with self.assertRaisesRegex(FileExistsError, "non-empty"):
                layout.begin(
                    source_dataset=Path(directory) / "dataset",
                    repository_root=Path(__file__).parents[2],
                    command=["train"],
                    options={},
                )

            self.assertEqual((layout.root / "unrelated.txt").read_text(), "keep")

    def test_config_uses_relative_internal_paths(self):
        with TemporaryDirectory() as directory:
            layout = RunLayout.from_root(Path(directory) / "run")
            layout.begin(
                source_dataset=Path(directory) / "dataset",
                repository_root=Path(__file__).parents[2],
                command=["train"],
                options={},
            )
            path = layout.write_config(
                vars(
                    Namespace(
                        Model_Path_Dir=str(layout.root),
                        logs_dir=str(layout.logs),
                        Source_Path_Dir="/external/dataset",
                    )
                )
            )
            payload = json.loads(path.read_text())["arguments"]
            self.assertEqual(payload["Model_Path_Dir"], ".")
            self.assertEqual(payload["logs_dir"], "logs")
            self.assertEqual(payload["Source_Path_Dir"], "/external/dataset")

    def test_final_stage_is_published_without_numeric_directories(self):
        with TemporaryDirectory() as directory:
            layout = RunLayout.from_root(Path(directory) / "run")
            layout.root.mkdir()
            stage = layout.stage(487)
            cloud = stage / "point_cloud" / "iteration_487" / "point_cloud.ply"
            cloud.parent.mkdir(parents=True)
            cloud.write_bytes(b"ply")
            (stage / "chkpnt487.pth").write_bytes(b"state")
            (stage / "cameras.json").write_text("[]")
            records = finalize_stage_model(layout, 487)

            self.assertEqual((layout.model / "gaussians.ply").read_bytes(), b"ply")
            self.assertEqual(
                (layout.model / "training_state.pt").read_bytes(), b"state"
            )
            self.assertFalse(stage.exists())
            self.assertEqual(records["gaussians"]["size_bytes"], 3)

    def test_only_active_verified_semantic_slot_is_archived(self):
        with TemporaryDirectory() as directory:
            layout = RunLayout.from_root(Path(directory) / "run")
            source = layout.semantic_work
            source.mkdir(parents=True)
            active_state = source / "semantic_state.1.pt"
            active_geometry = source / "geometry_checkpoint.1.pt"
            active_context = source / "recovery_context.1.json"
            active_state.write_bytes(b"active")
            active_geometry.write_bytes(b"geometry")
            active_context.write_text("{}")
            (source / "semantic_state.0.pt").write_bytes(b"stale")
            state_hash = hashlib.sha256(b"active").hexdigest()
            geometry_hash = hashlib.sha256(b"geometry").hexdigest()
            (source / "latest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 5,
                        "slot": 1,
                        "semantic_state_sha256": state_hash,
                        "geometry_checkpoint_sha256": geometry_hash,
                    }
                )
            )
            (source / "ledger.json").write_text("{}")

            records = archive_semantic_state(layout)

            self.assertEqual((layout.semantic_model / "state.pt").read_bytes(), b"active")
            self.assertTrue((layout.semantic_model / "manifest.json").is_file())
            self.assertTrue((source / "semantic_state.0.pt").is_file())
            self.assertEqual(records["state"]["sha256"], state_hash)

    def test_semantic_archive_rejects_hash_mismatch(self):
        with TemporaryDirectory() as directory:
            layout = RunLayout.from_root(Path(directory) / "run")
            source = layout.semantic_work
            source.mkdir(parents=True)
            (source / "semantic_state.0.pt").write_bytes(b"state")
            (source / "geometry_checkpoint.0.pt").write_bytes(b"geometry")
            (source / "recovery_context.0.json").write_text("{}")
            (source / "latest.json").write_text(
                json.dumps(
                    {
                        "slot": 0,
                        "semantic_state_sha256": "0" * 64,
                        "geometry_checkpoint_sha256": "1" * 64,
                    }
                )
            )
            with self.assertRaisesRegex(RuntimeError, "state hash"):
                archive_semantic_state(layout)
            self.assertFalse(layout.semantic_model.exists())


if __name__ == "__main__":
    unittest.main()
