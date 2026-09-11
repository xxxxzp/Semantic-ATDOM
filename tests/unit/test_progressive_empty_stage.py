import io
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from semantic_atdom.scene import dataset_readers
from tools.train_progressive import DataPreProccesser


class _RecordingQueue:
    def __init__(self):
        self.messages = []

    def put(self, message):
        self.messages.append(message)


class ProgressiveEmptyStageTests(unittest.TestCase):
    def test_sparse_reader_accepts_checkpoint_with_no_new_images(self):
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.object(
                dataset_readers,
                "read_points3D_ID_binary",
                return_value=(
                    np.empty((0,), dtype=np.int64),
                    np.empty((0, 3), dtype=np.float32),
                    np.empty((0, 3), dtype=np.uint8),
                    np.empty((0,), dtype=np.float32),
                ),
            ):
                scene = dataset_readers.SingleImage_readColmapSceneInfo_Part2(
                    root,
                    train_cam_infos=[],
                    test_cam_infos=[],
                    nerf_normalization={"radius": 1.0},
                    point3D_ids_list={},
                    xys_list={},
                    tri_list={},
                    NewCams=[],
                    OriginImageHeight=1049,
                    OriginImageWidth=1870,
                    get_ply=False,
                    Diary=io.StringIO(),
                    Silence=True,
                )

        self.assertIsNone(scene.point_cloud)
        self.assertIsNone(scene.ply_path)

    def test_cpu_sparse_reader_combines_multiple_new_image_batches(self):
        vertex_dtype = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]

        def fake_store_ply(_path, xyz, rgb, get_ply=True):
            self.assertFalse(get_ply)
            vertices = np.zeros(xyz.shape[0], dtype=vertex_dtype)
            vertices["x"], vertices["y"], vertices["z"] = xyz.T
            vertices["red"], vertices["green"], vertices["blue"] = rgb.T
            return {"vertex": vertices}

        sampled_batches = [
            (
                np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
                np.array([[10, 20, 30]], dtype=np.uint8),
            ),
            (
                np.array([[4.0, 5.0, 6.0]], dtype=np.float32),
                np.array([[40, 50, 60]], dtype=np.uint8),
            ),
        ]
        new_cameras = [
            SimpleNamespace(image_name=name, LogMask=np.ones((2, 2), dtype=bool))
            for name in ("a", "b")
        ]
        with tempfile.TemporaryDirectory() as root, mock.patch.multiple(
            dataset_readers,
            read_points3D_ID_binary=mock.DEFAULT,
            SamplePointsFromTri=mock.DEFAULT,
            PointsFilter=mock.DEFAULT,
            Get_3D_From_2D_3=mock.DEFAULT,
            Get_3D_Sampled_Points=mock.DEFAULT,
            storePly=mock.DEFAULT,
        ) as patched:
            patched["read_points3D_ID_binary"].return_value = (
                np.empty((0,), dtype=np.int64),
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
                np.empty((0,), dtype=np.float32),
            )
            patched["SamplePointsFromTri"].return_value = (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )
            patched["PointsFilter"].side_effect = (
                lambda points, *_args, **kwargs: (
                    (points, kwargs["Tri_IDs"])
                    if "Tri_IDs" in kwargs
                    else (points, kwargs.get("point3D_ids"))
                )
            )
            empty_points = np.empty((0, 3), dtype=np.float32)
            empty_colors = np.empty((0, 3), dtype=np.uint8)
            patched["Get_3D_From_2D_3"].return_value = (
                empty_points,
                empty_colors,
                empty_points,
                empty_colors,
            )
            patched["Get_3D_Sampled_Points"].side_effect = sampled_batches
            patched["storePly"].side_effect = fake_store_ply

            scene = dataset_readers.SingleImage_readColmapSceneInfo_Part2(
                root,
                train_cam_infos=[],
                test_cam_infos=[],
                nerf_normalization={"radius": 1.0},
                point3D_ids_list={
                    "a": np.empty((0,), dtype=np.int64),
                    "b": np.empty((0,), dtype=np.int64),
                },
                xys_list={
                    "a": np.empty((0, 2), dtype=np.float32),
                    "b": np.empty((0, 2), dtype=np.float32),
                },
                tri_list={"a": object(), "b": object()},
                NewCams=new_cameras,
                OriginImageHeight=1049,
                OriginImageWidth=1870,
                device="cpu",
                get_ply=False,
                Diary=io.StringIO(),
                Silence=True,
            )

        np.testing.assert_array_equal(
            scene.point_cloud.points,
            np.array(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                dtype=np.float32,
            ),
        )
        np.testing.assert_allclose(
            scene.point_cloud.colors,
            np.array(
                [[10, 20, 30], [40, 50, 60]],
                dtype=np.float32,
            )
            / 255.0,
            rtol=1e-7,
            atol=0.0,
        )

    def test_new_image_count_comes_from_loaded_cameras(self):
        processor = object.__new__(DataPreProccesser)
        processor.NewImagesNum = 1
        processor.Silence = True
        processor.Diary = io.StringIO()
        processor.resolution_scales = [1.0]
        processor.GaussianTrainer_queue = _RecordingQueue()

        _, new_cameras, updated_positions = processor.AddNewImages2(
            [SimpleNamespace(image=None, image_name="000402")],
            TrainCameras={},
        )

        self.assertEqual(processor.NewImagesNum, 0)
        self.assertEqual(new_cameras, [])
        self.assertEqual(updated_positions, {})
        self.assertEqual(
            processor.GaussianTrainer_queue.messages,
            [["RenderNewDone"]],
        )


if __name__ == "__main__":
    unittest.main()
