from argparse import Namespace
from datetime import datetime
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from semantic_atdom.utils.run_output import (
    RUN_OUTPUT_SCHEMA_VERSION,
    RunLayout,
    allocate_run_root,
    archive_semantic_state,
    copy_file_verified,
    finalize_stage_model,
    normalized_run_name,
    sha256_file,
    validate_camera_render_outputs,
    validate_final_semantic_outputs,
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

            with self.assertRaisesRegex(RuntimeError, "succeeded run"):
                layout.publish_latest()
            layout.set_status(
                "succeeded",
                artifacts={"model": "model"},
                final_stage=487,
                optimizer_iteration=100800,
            )
            latest = layout.publish_latest()
            self.assertEqual(latest.resolve(), layout.root)
            manifest = layout.read_manifest()
            self.assertEqual(manifest["status"], "succeeded")
            self.assertEqual(manifest["final_stage"], 487)
            self.assertEqual(manifest["optimizer_iteration"], 100800)

    def test_terminal_status_cannot_be_rewritten_or_published(self):
        with TemporaryDirectory() as directory:
            output_root = Path(directory) / "outputs"
            layout = RunLayout.from_root(output_root / "runs" / "run-a")
            layout.begin(
                source_dataset=Path(directory) / "dataset",
                repository_root=Path(__file__).parents[2],
                command=["train"],
                options={},
            )
            layout.set_status("failed", failure={"message": "boom"})

            with self.assertRaisesRegex(RuntimeError, "terminal"):
                layout.set_status(
                    "succeeded",
                    artifacts={"model": "model"},
                    final_stage=1,
                    optimizer_iteration=1,
                )
            with self.assertRaisesRegex(RuntimeError, "succeeded run"):
                layout.publish_latest()
            self.assertFalse((output_root / "latest").exists())

    def test_latest_failure_preserves_succeeded_run_and_previous_pointer(self):
        with TemporaryDirectory() as directory:
            output_root = Path(directory) / "outputs"
            previous = output_root / "runs" / "previous"
            previous.mkdir(parents=True)
            latest = output_root / "latest"
            latest.symlink_to("runs/previous", target_is_directory=True)
            layout = RunLayout.from_root(output_root / "runs" / "run-a")
            layout.begin(
                source_dataset=Path(directory) / "dataset",
                repository_root=Path(__file__).parents[2],
                command=["train"],
                options={},
            )
            layout.set_status(
                "succeeded",
                artifacts={},
                final_stage=20,
                optimizer_iteration=100,
            )
            real_replace = __import__("os").replace

            def fail_latest(source, destination):
                if Path(destination) == latest:
                    raise OSError("injected latest failure")
                return real_replace(source, destination)

            with mock.patch(
                "semantic_atdom.utils.run_output.os.replace",
                side_effect=fail_latest,
            ):
                with self.assertRaisesRegex(OSError, "latest failure"):
                    layout.publish_latest()

            self.assertEqual(layout.read_manifest()["status"], "succeeded")
            self.assertEqual(latest.resolve(), previous)
            self.assertFalse((output_root / ".latest.tmp").exists())

    def test_success_requires_completion_metadata(self):
        with TemporaryDirectory() as directory:
            layout = RunLayout.from_root(Path(directory) / "run")
            layout.begin(
                source_dataset=Path(directory) / "dataset",
                repository_root=Path(__file__).parents[2],
                command=["train"],
                options={},
            )

            with self.assertRaisesRegex(ValueError, "final_stage"):
                layout.set_status("succeeded", artifacts={})
            self.assertEqual(layout.read_manifest()["status"], "running")

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

    def test_prepare_does_not_create_optional_product_placeholders(self):
        with TemporaryDirectory() as directory:
            layout = RunLayout.from_root(Path(directory) / "run")
            layout.begin(
                source_dataset=Path(directory) / "dataset",
                repository_root=Path(__file__).parents[2],
                command=["train"],
                options={},
            )

            self.assertFalse(layout.products.exists())
            self.assertFalse(layout.renders.exists())
            self.assertFalse(layout.videos.exists())

    def test_camera_render_validation_requires_exact_inventory(self):
        with TemporaryDirectory() as directory:
            layout = RunLayout.from_root(Path(directory) / "run")
            layout.model.mkdir(parents=True)
            layout.metrics.mkdir(parents=True)
            renders = layout.renders / "cameras"
            renders.mkdir(parents=True)
            (layout.model / "cameras.json").write_text(
                json.dumps(
                    [
                        {"img_name": "camera-a"},
                        {"img_name": "camera-b"},
                    ]
                )
            )
            (renders / "camera-a.jpg").write_bytes(b"jpeg")
            (layout.metrics / "per_camera.csv").write_text(
                "camera_id,iteration,psnr,visible_gaussians,render\n"
                "camera-a,10,20.0,1,renders/cameras/camera-a.jpg\n"
            )

            with self.assertRaisesRegex(RuntimeError, r"missing=\['camera-b'\]"):
                validate_camera_render_outputs(layout)

            (renders / "camera-b.jpg").write_bytes(b"jpeg")
            with (layout.metrics / "per_camera.csv").open("a") as stream:
                stream.write(
                    "camera-b,10,20.0,1,renders/cameras/camera-b.jpg\n"
                )
            self.assertEqual(validate_camera_render_outputs(layout), 2)

    def test_final_stage_is_copied_without_destroying_recovery_source(self):
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
            self.assertFalse((layout.model / "training_state.pt").exists())
            self.assertNotIn("training_state", records)
            self.assertTrue(stage.exists())
            self.assertEqual(records["gaussians"]["size_bytes"], 3)

    def test_final_training_state_is_retained_only_when_requested(self):
        with TemporaryDirectory() as directory:
            layout = RunLayout.from_root(Path(directory) / "run")
            layout.root.mkdir()
            stage = layout.stage(487)
            cloud = stage / "point_cloud" / "iteration_487" / "point_cloud.ply"
            cloud.parent.mkdir(parents=True)
            cloud.write_bytes(b"ply")
            (stage / "chkpnt487.pth").write_bytes(b"state")

            records = finalize_stage_model(
                layout, 487, include_training_state=True
            )

            self.assertEqual(
                (layout.model / "training_state.pt").read_bytes(), b"state"
            )
            self.assertIn("training_state", records)

    def test_final_model_copy_failure_preserves_complete_stage_source(self):
        with TemporaryDirectory() as directory:
            layout = RunLayout.from_root(Path(directory) / "run")
            layout.root.mkdir()
            stage = layout.stage(20)
            cloud = stage / "point_cloud" / "iteration_20" / "point_cloud.ply"
            cloud.parent.mkdir(parents=True)
            cloud.write_bytes(b"ply")
            cameras = stage / "cameras.json"
            cameras.write_text("[]")
            def fail_cameras(source, destination):
                if Path(source) == cameras:
                    raise OSError("injected copy failure")
                return copy_file_verified(source, destination)

            with mock.patch(
                "semantic_atdom.utils.run_output.copy_file_verified",
                side_effect=fail_cameras,
            ):
                with self.assertRaisesRegex(OSError, "injected copy failure"):
                    finalize_stage_model(layout, 20)

            self.assertEqual(cloud.read_bytes(), b"ply")
            self.assertEqual(cameras.read_text(), "[]")
            self.assertTrue(stage.is_dir())

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
            context_hash = hashlib.sha256(b"{}").hexdigest()
            (source / "latest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 5,
                        "slot": 1,
                        "semantic_state_sha256": state_hash,
                        "geometry_checkpoint_sha256": geometry_hash,
                        "recovery_context_sha256": context_hash,
                    }
                )
            )
            (source / "ledger.json").write_text("{}")

            records = archive_semantic_state(layout)

            self.assertEqual((layout.semantic_model / "state.pt").read_bytes(), b"active")
            self.assertTrue((layout.semantic_model / "manifest.json").is_file())
            self.assertTrue(active_state.is_file())
            self.assertTrue(active_geometry.is_file())
            self.assertTrue(active_context.is_file())
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

    def test_semantic_archive_rejects_recovery_context_hash_mismatch(self):
        with TemporaryDirectory() as directory:
            layout = RunLayout.from_root(Path(directory) / "run")
            source = layout.semantic_work
            source.mkdir(parents=True)
            state = source / "semantic_state.0.pt"
            geometry = source / "geometry_checkpoint.0.pt"
            context = source / "recovery_context.0.json"
            state.write_bytes(b"state")
            geometry.write_bytes(b"geometry")
            context.write_text('{"stage": 20}')
            (source / "latest.json").write_text(
                json.dumps(
                    {
                        "slot": 0,
                        "semantic_state_sha256": sha256_file(state),
                        "geometry_checkpoint_sha256": sha256_file(geometry),
                        "recovery_context_sha256": "0" * 64,
                    }
                )
            )

            with self.assertRaisesRegex(RuntimeError, "recovery context hash"):
                archive_semantic_state(layout)
            self.assertTrue(state.is_file())
            self.assertTrue(geometry.is_file())
            self.assertTrue(context.is_file())
            self.assertFalse(layout.semantic_model.exists())

    def test_final_semantic_validation_rejects_stale_or_failed_state(self):
        with TemporaryDirectory() as directory:
            layout = RunLayout.from_root(Path(directory) / "run")
            layout.semantic_work.mkdir(parents=True)
            (layout.semantic_work / "sync_status.json").write_text(
                json.dumps(
                    {
                        "status": "stale_online",
                        "geometry_stage": 20,
                        "lag_batches": 1,
                        "errors": ["rebuild failed"],
                        "deferred_image_ids": ["camera-20"],
                        "historical_evidence_stale": True,
                        "publication_status": "failed",
                    }
                )
            )

            with self.assertRaisesRegex(RuntimeError, "not authoritative"):
                validate_final_semantic_outputs(
                    layout, final_stage=20, require_map=True
                )

    def test_final_semantic_validation_accepts_exact_hashed_map_pair(self):
        with TemporaryDirectory() as directory:
            layout = RunLayout.from_root(Path(directory) / "run")
            layout.semantic_work.mkdir(parents=True)
            status = {
                "status": "exact_final",
                "geometry_stage": 20,
                "lag_batches": 0,
                "errors": [],
                "deferred_image_ids": [],
                "historical_evidence_stale": False,
                "publication_status": "complete",
            }
            (layout.semantic_work / "sync_status.json").write_text(
                json.dumps(status)
            )
            pair = layout.map_releases / "stage_000020" / "pair_manifest.json"
            pair.parent.mkdir(parents=True)
            pair.write_text(
                json.dumps(
                    {
                        "publication_status": "complete",
                        "semantic_status": "exact_final",
                        "geometry_stage": 20,
                    }
                )
            )
            (layout.maps / "latest.json").write_text(
                json.dumps(
                    {
                        "pair_manifest": (
                            "releases/stage_000020/pair_manifest.json"
                        ),
                        "pair_manifest_sha256": sha256_file(pair),
                    }
                )
            )

            self.assertEqual(
                validate_final_semantic_outputs(
                    layout, final_stage=20, require_map=True
                ),
                status,
            )

    def test_online_finalization_accepts_stale_history_but_requires_complete_inputs(self):
        with TemporaryDirectory() as directory:
            layout = RunLayout.from_root(Path(directory) / "run")
            layout.semantic_work.mkdir(parents=True)
            status = {"status": "approx_final", "geometry_stage": 20,
                "lag_batches": 0, "errors": [], "deferred_image_ids": [],
                "historical_evidence_stale": True}
            path = layout.semantic_work / "sync_status.json"
            path.write_text(json.dumps(status))
            self.assertEqual(validate_final_semantic_outputs(
                layout, final_stage=20, require_map=False, require_exact=False), status)
            with self.assertRaisesRegex(RuntimeError, "not authoritative"):
                validate_final_semantic_outputs(layout, final_stage=20, require_map=False)
            for field, value in [("lag_batches", 1), ("errors", ["broken input"]),
                                 ("deferred_image_ids", ["000020"])]:
                with self.subTest(field=field):
                    path.write_text(json.dumps(dict(status, **{field: value})))
                    with self.assertRaises(RuntimeError):
                        validate_final_semantic_outputs(layout, final_stage=20,
                            require_map=False, require_exact=False)


if __name__ == "__main__":
    unittest.main()
