"""Render one frozen training boundary on the ordered semantic worker."""
from dataclasses import dataclass
import time
import numpy as np
import torch

from semantic_atdom.semantics.tdom import rasterize_rgb_tdom_striped, rasterize_semantic_tdom
from semantic_atdom.utils.image_utils import tdom_to_uint8


def semantic_frame(gaussians, state, camera, class_spec, config):
    result = rasterize_semantic_tdom(gaussians, state, camera, class_spec, config=config)
    if result.colors is not None:
        return result.colors.permute(1, 2, 0).detach().cpu().numpy()
    palette = config.render_palette
    if palette is None:
        palette = tuple(item.color for item in class_spec.classes)
    palette = np.asarray(palette, dtype=np.uint8)
    ids = result.class_ids.detach().cpu().numpy().astype(np.int64)
    colors = palette[np.clip(ids, 0, len(palette) - 1)]
    colors[result.unknown_mask.detach().cpu().numpy()] = 0
    return colors


@dataclass(frozen=True)
class ProgressCapture:
    camera: object
    background: torch.Tensor
    config: object
    metadata: dict
    output_queue: object
    submitted_at: float

    def __call__(self, core, gaussians, status):
        if (status.geometry_stage != self.metadata['stage']
            or status.semantic_through_image is None or status.lag_batches != 0):
            raise RuntimeError('training progress requires the matching committed semantic boundary')
        started = time.perf_counter()
        with torch.no_grad():
            rendered = rasterize_rgb_tdom_striped(gaussians, self.camera, self.background, self.config)
            rgb = tdom_to_uint8(rendered)
            del rendered
            rgb_done = time.perf_counter()
            semantic = semantic_frame(gaussians, core.orchestrator.state, self.camera,
                                      core.dataset.class_spec, self.config)
        semantic_done = time.perf_counter()
        metadata = dict(self.metadata, semantic_through_image=status.semantic_through_image,
            semantic_status=status.status, semantic_lag_batches=int(status.lag_batches),
            semantic_historical_evidence_stale=bool(status.historical_evidence_stale),
            semantic_errors=list(status.errors))
        metadata['capture_timing'] = dict(
            worker_delay_seconds=started-self.submitted_at,
            rgb_tdom_seconds=rgb_done-started,
            semantic_tdom_seconds=semantic_done-rgb_done,
            result_ready_seconds=semantic_done-self.submitted_at)
        self.output_queue.put(['SaveTrainingProgress', rgb, semantic, metadata])
        return metadata['capture_timing']
