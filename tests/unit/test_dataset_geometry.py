from types import SimpleNamespace
from io import StringIO
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import numpy as np

from semantic_atdom.scene.dataset_readers import (
    Get_Tri_Mask,
    PointsFilter,
    SingleImage_readColmapSceneInfo_Part2,
)


class DatasetGeometryTests(unittest.TestCase):
    def test_sparse_support_rejects_too_few_or_collinear_points(self):
        for points in (
            np.asarray([[1.0, 1.0], [2.0, 2.0]]),
            np.asarray([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        ):
            extrinsic = SimpleNamespace(
                point3D_ids=np.arange(points.shape[0]),
                xys=points,
            )

            mask, point_ids, retained, triangulation = Get_Tri_Mask(
                extrinsic,
                width=8,
                height=8,
                Get3D_ID=True,
                device="cpu",
                Silence=True,
            )

            self.assertEqual(tuple(mask.shape), (2,))
            self.assertEqual(len(point_ids), len(retained))
            self.assertIsNone(triangulation)

    def test_part2_skips_gaussian_initialization_for_degenerate_camera(self):
        points = np.asarray([[1.0, 1.0], [2.0, 2.0]])
        extrinsic = SimpleNamespace(
            point3D_ids=np.arange(points.shape[0]),
            xys=points,
        )
        _, _, retained, triangulation = Get_Tri_Mask(
            extrinsic,
            width=8,
            height=8,
            Get3D_ID=True,
            device="cpu",
            Silence=True,
        )

        self.assertIsNone(triangulation)
        with TemporaryDirectory() as directory, mock.patch(
            "semantic_atdom.scene.dataset_readers.read_points3D_ID_binary",
            return_value=(
                np.asarray([0, 1]),
                np.zeros((2, 3)),
                np.zeros((2, 3)),
                None,
            ),
        ):
            scene = SingleImage_readColmapSceneInfo_Part2(
                directory,
                train_cam_infos=[],
                test_cam_infos=[],
                nerf_normalization={},
                point3D_ids_list={"camera": np.asarray([0, 1])},
                xys_list={"camera": retained},
                tri_list={"camera": triangulation},
                NewCams=[],
                OriginImageHeight=8,
                OriginImageWidth=8,
                device="cpu",
                get_ply=False,
                Diary=StringIO(),
                Silence=True,
            )

        self.assertIsNone(scene.point_cloud)

    def test_sparse_support_filters_nonfinite_and_out_of_bounds_points(self):
        points = np.asarray(
            [
                [1.0, 1.0],
                [6.0, 1.0],
                [1.0, 6.0],
                [-1.0, 1.0],
                [8.0, 1.0],
                [np.nan, 2.0],
                [2.0, np.inf],
            ]
        )
        extrinsic = SimpleNamespace(
            point3D_ids=np.arange(points.shape[0]),
            xys=points,
        )

        mask, point_ids, retained, triangulation = Get_Tri_Mask(
            extrinsic,
            width=8,
            height=8,
            Get3D_ID=True,
            device="cpu",
            Silence=True,
        )

        self.assertEqual(tuple(mask.shape), (1, 8, 8))
        np.testing.assert_array_equal(point_ids, np.asarray([0, 1, 2]))
        np.testing.assert_array_equal(retained, points[:3])
        self.assertIsNotNone(triangulation)

    def test_point_filter_never_indexes_outside_mask(self):
        points = np.asarray(
            [
                [0.0, 0.0],
                [7.9, 7.9],
                [-0.1, 1.0],
                [8.0, 1.0],
                [np.nan, 2.0],
                [2.0, np.inf],
            ]
        )
        point_ids = np.arange(points.shape[0])

        selected, selected_ids = PointsFilter(
            points,
            np.ones((8, 8), dtype=bool),
            8,
            8,
            8,
            8,
            point3D_ids=point_ids,
        )

        np.testing.assert_array_equal(selected, points[:2])
        np.testing.assert_array_equal(selected_ids, point_ids[:2])


if __name__ == "__main__":
    unittest.main()
