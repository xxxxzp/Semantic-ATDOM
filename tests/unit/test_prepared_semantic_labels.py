"""Reuse label preparation without reusing geometry-dependent projections."""
from types import SimpleNamespace
from unittest import mock
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
import torch

from semantic_atdom.semantics.orchestrator import CudaSemanticProjector, OnlineSemanticConfig


def inputs():
    labels = np.array([[0, 0, 1], [0, 1, 1]], dtype=np.uint8)
    camera = SimpleNamespace(image_width=3, image_height=2, calibration=None,
        Tri_Mask=torch.ones(2, 3, dtype=torch.bool), FoVx=1., FoVy=1.,
        world_view_transform=torch.eye(4), full_proj_transform=torch.eye(4), camera_center=torch.zeros(3))
    model = SimpleNamespace(get_xyz=torch.zeros(1, 3), get_opacity=torch.ones(1, 1),
        get_scaling=torch.ones(1, 3), get_rotation=torch.ones(1, 4))
    return labels, camera, model


def test_repeated_preparation_is_reused_but_geometry_is_rasterized_again():
    from semantic_atdom.semantics.orchestrator import label_boundary_weights
    labels, camera, model = inputs()
    projector = CudaSemanticProjector(SimpleNamespace(num_classes=2, ignore_id=255), OnlineSemanticConfig())
    result = SimpleNamespace(distribution=torch.zeros(1, 2), support=torch.zeros(1), valid_mass=torch.zeros(1))
    with mock.patch('semantic_atdom.semantics.orchestrator.label_boundary_weights', wraps=label_boundary_weights) as boundary, mock.patch(
        'semantic_atdom.semantics.orchestrator.rasterize_semantic_observation', return_value=result) as raster:
        projector.project(model, camera, SimpleNamespace(labels=labels))
        first = raster.call_args.kwargs
        model.get_xyz += 1
        projector.project(model, camera, SimpleNamespace(labels=labels.copy()))
        assert boundary.call_count == 1
        assert raster.call_count == 2
        torch.testing.assert_close(first['labels'], raster.call_args.kwargs['labels'])
        torch.testing.assert_close(first['boundary_weights'], raster.call_args.kwargs['boundary_weights'])


def test_content_mask_calibration_grid_and_parameters_invalidate_preparation():
    from semantic_atdom.semantics.prepared_labels import PreparedLabelCache
    from semantic_atdom.semantics.orchestrator import label_boundary_weights
    from semantic_atdom.semantics.label_grid import rectified_source_labels
    from semantic_atdom.utils.pixel_grid import resize_nearest
    from semantic_atdom.utils.camera_calibration import rectified_pinhole_from_colmap
    labels, camera, _ = inputs()
    cache = PreparedLabelCache()
    boundary = mock.Mock(wraps=label_boundary_weights)
    def check(bandwidth=2., floor=.25):
        ids, weights = cache.prepare(labels, camera, 255, bandwidth, floor, boundary)
        source = rectified_source_labels(labels, camera, 255)
        resolution = (camera.image_width, camera.image_height)
        np.testing.assert_array_equal(ids, resize_nearest(source, resolution))
        np.testing.assert_array_equal(weights, resize_nearest(label_boundary_weights(source, bandwidth=bandwidth, floor=floor), resolution))
        assert not ids.flags.writeable and not weights.flags.writeable
    check()
    labels[0, 0] = 1
    check()
    camera.Tri_Mask[1, 1] = False
    check()
    camera.calibration = rectified_pinhole_from_colmap('PINHOLE', (3., 3., 1., 1.), width=3, height=2)
    check()
    camera.image_width, camera.image_height, camera.Tri_Mask = 2, 1, None
    check()
    check(bandwidth=3.)
    check(floor=.5)
    assert boundary.call_count == 7


def test_disk_reuse_checks_content_and_recovers_corruption_without_stale_arrays():
    from semantic_atdom.semantics.prepared_labels import PreparedLabelCache
    from semantic_atdom.semantics.orchestrator import label_boundary_weights
    labels, camera, _ = inputs()
    with TemporaryDirectory() as root:
        boundary = mock.Mock(wraps=label_boundary_weights)
        def prepare(cache):
            return cache.prepare(labels, camera, 255, 2., .25, boundary)
        expected = prepare(PreparedLabelCache(root, max_bytes=0))
        reader = PreparedLabelCache(root, max_bytes=0)
        actual = prepare(reader)
        assert boundary.call_count == 1 and reader.stats['disk_hits'] == 1
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
        path = next(Path(root).glob('*.npz'))
        with np.load(path, allow_pickle=False) as archive:
            payload = {name: archive[name].copy() for name in archive.files}
        payload['weights'][:] = .123
        np.savez_compressed(path, **payload)
        reader = PreparedLabelCache(root, max_bytes=0)
        actual = prepare(reader)
        assert boundary.call_count == 2 and reader.stats['corrupt_entries'] == 1
        np.testing.assert_array_equal(actual[1], expected[1])
        path.write_bytes(b'broken archive')
        actual = prepare(PreparedLabelCache(root, max_bytes=0))
        assert boundary.call_count == 3
        np.testing.assert_array_equal(actual[1], expected[1])


def test_cpu_and_disk_limits_evict_derived_entries():
    from semantic_atdom.semantics.prepared_labels import PreparedLabelCache
    from semantic_atdom.semantics.orchestrator import label_boundary_weights
    labels, camera, _ = inputs()
    with TemporaryDirectory() as root:
        cache = PreparedLabelCache(root, max_bytes=30, max_disk_bytes=1)
        for floor in (.1, .2, .3):
            cache.prepare(labels, camera, 255, 2., floor, label_boundary_weights)
            assert cache.nbytes <= 30
            assert sum(p.stat().st_size for p in Path(root).glob('*.npz')) <= 1
