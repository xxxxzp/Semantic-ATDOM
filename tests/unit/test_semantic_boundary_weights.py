"""Boundary reliability must retain the original Euclidean-distance contract."""
import unittest

import numpy as np
from scipy.ndimage import distance_transform_edt

from semantic_atdom.semantics.orchestrator import label_boundary_weights


def reference_weights(labels, bandwidth, floor):
    # An edge exists when any in-image four-neighbor has another ID.
    padded = np.pad(labels, 1, mode="edge")
    neighbors = (padded[:-2, 1:-1], padded[2:, 1:-1],
                 padded[1:-1, :-2], padded[1:-1, 2:])
    edge = np.any([neighbor != labels for neighbor in neighbors], axis=0)
    if bandwidth <= 0 or not edge.any():
        result = np.ones(labels.shape, dtype=np.float32)
    else:
        distances = distance_transform_edt(~edge)
        result = (floor + (1 - floor) * np.clip(distances / bandwidth, 0, 1)).astype(np.float32)
    result[labels == 255] = 0
    return result


class BoundaryWeightTests(unittest.TestCase):
    def test_vertical_boundary_has_the_worked_euclidean_ramp(self):
        labels = np.tile(np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.uint8), (5, 1))
        expected = np.tile(np.array([1., .625, .25, .25, .625, 1., 1.], dtype=np.float32), (5, 1))
        np.testing.assert_array_equal(label_boundary_weights(labels, bandwidth=2.), expected)

    def test_exact_weights_match_scipy_for_sparse_diagonal_and_unknown_edges(self):
        labels = np.zeros((151, 173), dtype=np.uint16)
        labels[23:30, 19:28] = 1
        labels[66:91, 102:117] = 1024
        labels[110:123, 67:92] = 255
        rows, columns = np.indices(labels.shape)
        labels[(rows > 10) & (rows < 30) & (columns - rows == 100)] = 2
        labels.flags.writeable = False
        before = labels.copy()
        for bandwidth in (0., .01, 1., 1.000001, 2., np.sqrt(5) - 1e-12,
                          np.sqrt(5) + 1e-12, 5.002481389578164, 31.99, 128., 128.01, 256., np.inf):
            for floor in (0., .1, .25, 1.):
                with self.subTest(bandwidth=bandwidth, floor=floor):
                    actual = label_boundary_weights(labels, bandwidth=bandwidth, floor=floor)
                    np.testing.assert_array_equal(actual, reference_weights(labels, bandwidth, floor))
                    self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_array_equal(labels, before)

    def test_single_axes_and_noncontiguous_ids_preserve_the_same_weights(self):
        for labels in (np.array([[0, 0, 255, 2, 2]], dtype=np.int32),
                       np.array([[0], [0], [255], [2]], dtype=np.int64),
                       np.array([[0, 0, 2], [2, 255, 2]], dtype=np.uint8)[:, ::-1],
                       np.zeros((5, 7), dtype=np.uint8),
                       np.full((5, 7), 255, dtype=np.uint8)):
            with self.subTest(shape=labels.shape):
                np.testing.assert_array_equal(label_boundary_weights(labels, bandwidth=5.002481389578164),
                    reference_weights(labels, 5.002481389578164, .25))


if __name__ == "__main__":
    unittest.main()
