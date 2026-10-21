"""Pose-only updates must not rebuild masks that the trainer never consumes."""
from types import SimpleNamespace
from unittest import mock
import numpy as np
import torch

from semantic_atdom.scene import dataset_readers as readers


def fixture():
    extr = SimpleNamespace(camera_id=1, qvec=np.array([1., 0., 0., 0.]),
                           tvec=np.array([1., 2., 3.]), name='existing.jpg')
    intr = SimpleNamespace(id=1, model='RADIAL', width=8, height=6,
                           params=np.array([5., 4., 3., .2, .03]))
    fy, fx, calibration = readers._camera_projection(intr)
    retained = {'existing':dict(FoVx=fx, FoVy=fy, calibration=calibration)}
    return extr, intr, retained


def read(extr, intr, retained):
    return readers.SingleImage_readColmapCameras({1:extr}, {1:intr}, '/unused',
        CurrentImagesNames=['existing'], Do_Get_Tri_Mask=True, Silence=True,
        retained_camera_mappings=retained)[0][0]


def test_matching_retained_projection_skips_all_unused_mask_preparation():
    extr, intr, retained = fixture()
    with mock.patch.object(readers, 'Get_Tri_Mask', side_effect=AssertionError('unused triangulation')), mock.patch.object(
        readers, '_rectify_training_mask', side_effect=AssertionError('unused rectification')):
        info = read(extr, intr, retained)
    np.testing.assert_array_equal(info.T, extr.tvec)
    assert info.calibration == retained['existing']['calibration']
    assert info.image is None and info.Tri_Mask is None


def test_changed_projection_still_prepares_replacement_mask():
    extr, intr, retained = fixture()
    intr.params[0] = 6.
    with mock.patch.object(readers, 'Get_Tri_Mask', return_value=torch.ones(1,6,8,dtype=torch.bool)) as get_mask:
        info = read(extr, intr, retained)
    get_mask.assert_called_once()
    assert info.Tri_Mask is not None
    assert info.calibration != retained['existing']['calibration']


def test_legacy_reader_without_retained_metadata_keeps_mask_behavior():
    extr, intr, _ = fixture()
    with mock.patch.object(readers, 'Get_Tri_Mask', return_value=torch.ones(1,6,8,dtype=torch.bool)) as get_mask:
        info = read(extr, intr, None)
    get_mask.assert_called_once()
    assert info.Tri_Mask is not None
