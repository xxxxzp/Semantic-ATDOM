import hashlib
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from tools.migrate_run_output import migrate


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


class MigrateRunOutputTests(unittest.TestCase):
    @staticmethod
    def _create_legacy_run(root):
        stage = root / "1"
        cloud = stage / "point_cloud" / "iteration_1" / "point_cloud.ply"
        cloud.parent.mkdir(parents=True)
        cloud.write_bytes(b"ply")
        (stage / "chkpnt1.pth").write_bytes(b"training-state")
        (stage / "cfg_args").write_text(
            "Namespace(Source_Path_Dir='/dataset')", encoding="utf-8"
        )

        release = root / "Map_Products" / "releases" / "stage_000001"
        (release / "semantic").mkdir(parents=True)
        pair = release / "pair_manifest.json"
        pair.write_text(
            json.dumps(
                {
                    "publication_status": "complete",
                    "semantic_status": "exact_final",
                    "geometry_stage": 1,
                }
            ),
            encoding="utf-8",
        )
        (release / "semantic" / "class_spec.json").write_text(
            "{}", encoding="utf-8"
        )
        (root / "Map_Products" / "latest.json").write_text(
            json.dumps(
                {
                    "product_key": "stage_000001",
                    "pair_manifest": "releases/stage_000001/pair_manifest.json",
                    "pair_manifest_sha256": _sha256(pair.read_bytes()),
                }
            ),
            encoding="utf-8",
        )

        semantic = root / "semantic"
        semantic.mkdir()
        state = b"semantic-state"
        geometry = b"semantic-geometry"
        (semantic / "semantic_state.0.pt").write_bytes(state)
        (semantic / "geometry_checkpoint.0.pt").write_bytes(geometry)
        (semantic / "recovery_context.0.json").write_text("{}", encoding="utf-8")
        (semantic / "sync_status.json").write_text(
            json.dumps(
                {
                    "status": "exact_final",
                    "geometry_stage": 1,
                    "lag_batches": 0,
                    "errors": [],
                    "deferred_image_ids": [],
                    "historical_evidence_stale": False,
                    "publication_status": "complete",
                }
            ),
            encoding="utf-8",
        )
        (semantic / "latest.json").write_text(
            json.dumps(
                {
                    "schema_version": 5,
                    "slot": 0,
                    "semantic_state_sha256": _sha256(state),
                    "geometry_checkpoint_sha256": _sha256(geometry),
                }
            ),
            encoding="utf-8",
        )

        intermediate = root / "Intermediate_Results"
        intermediate.mkdir()
        (intermediate / "eval.txt").write_text(
            "[ITER 100] Evaluating train: L1 0.1 Gaussians 10 "
            "PSNR 20.0 SSIM 0.5 LPIPS 0.4\n",
            encoding="utf-8",
        )
        camera = root / "OutputImages" / "camera-a"
        camera.mkdir(parents=True)
        (camera / "PredictionImages100.jpg").write_bytes(b"jpeg")
        (camera / "PSNR.txt").write_text(
            "100: 20.0, Visible_Gaussians_Num: 10\n", encoding="utf-8"
        )

    def test_camera_renders_are_installed_before_legacy_cleanup(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "legacy-run"
            self._create_legacy_run(root)

            migrate(root, final_stage=1, apply=True)

            render = root / "renders" / "cameras" / "camera-a.jpg"
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(render.read_bytes(), b"jpeg")
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["final_stage"], 1)
            self.assertEqual(manifest["optimizer_iteration"], 100)
            self.assertEqual(
                manifest["artifacts"]["renders"]["cameras"]["path"],
                "renders/cameras",
            )
            self.assertFalse((root / "OutputImages").exists())
            self.assertFalse(root.with_name(root.name + ".migration-output-v2").exists())
            self.assertFalse(root.with_name(root.name + ".migration-backup").exists())

    def test_failed_directory_exchange_restores_untouched_legacy_run(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "legacy-run"
            self._create_legacy_run(root)
            staging = root.with_name(root.name + ".migration-output-v2")
            real_replace = os.replace

            def fail_install(source, destination):
                if Path(source) == staging and Path(destination) == root:
                    raise OSError("injected install failure")
                return real_replace(source, destination)

            with mock.patch(
                "tools.migrate_run_output.os.replace", side_effect=fail_install
            ):
                with self.assertRaisesRegex(OSError, "injected install failure"):
                    migrate(root, final_stage=1, apply=True)

            self.assertTrue((root / "OutputImages" / "camera-a").is_dir())
            self.assertTrue((root / "Map_Products" / "latest.json").is_file())
            self.assertFalse((root / "manifest.json").exists())
            self.assertFalse(staging.exists())
            self.assertFalse(root.with_name(root.name + ".migration-backup").exists())

    def test_manifest_omits_renders_when_legacy_run_has_none(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "legacy-run"
            self._create_legacy_run(root)
            shutil.rmtree(root / "OutputImages")

            migrate(root, final_stage=1, apply=True)

            manifest = json.loads((root / "manifest.json").read_text())
            self.assertNotIn("renders", manifest["artifacts"])


if __name__ == "__main__":
    unittest.main()
