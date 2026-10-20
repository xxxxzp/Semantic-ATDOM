"""Read-only source labels mapped onto the camera's rectified training grid."""

import numpy as np

from semantic_atdom.semantics.schema import SemanticSchemaError
from semantic_atdom.utils.camera_calibration import rectify_array
from semantic_atdom.utils.pixel_grid import resize_nearest


def camera_mask(camera):
    mask = getattr(camera, "Tri_Mask", None)
    if mask is None or tuple(mask.shape) in ((1,), (2,)):
        return None
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask, dtype=bool)
    expected = (int(camera.image_height), int(camera.image_width))
    if mask.shape == (1,) + expected:
        mask = mask[0]
    if mask.shape != expected:
        raise SemanticSchemaError("camera mask does not match the training grid")
    return mask


def rectified_source_labels(labels, camera, ignore_id=255):
    """Validate the original grid, rectify IDs, and gate source reliability.

    The camera retains the ORIGINAL calibration; its image dimensions describe
    the TRAINING grid. A coarse camera mask is lifted to the source grid only to
    compute boundary reliability in source pixels. IDs are never interpolated.
    """
    labels = np.asarray(labels)
    calibration = getattr(camera, "calibration", None)
    expected = (int(camera.image_height), int(camera.image_width))
    source_shape = (
        expected if calibration is None else (calibration.height, calibration.width)
    )
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise SemanticSchemaError("semantic labels must be a two-dimensional integer array")
    if labels.shape != source_shape:
        raise SemanticSchemaError(
            "label has shape {}, source calibration expects {}".format(labels.shape, source_shape)
        )
    if calibration is None:
        result = labels.copy()
    else:
        result = rectify_array(
            labels, calibration, interpolation="nearest", border_value=ignore_id
        )
    mask = camera_mask(camera)
    if mask is not None:
        if mask.shape != result.shape:
            mask = resize_nearest(mask, (result.shape[1], result.shape[0]))
        result[~mask] = ignore_id
    return result


def labels_on_camera_grid(labels, camera, ignore_id=255):
    source = rectified_source_labels(labels, camera, ignore_id)
    return resize_nearest(source, (int(camera.image_width), int(camera.image_height)))
