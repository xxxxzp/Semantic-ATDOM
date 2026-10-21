"""Requested video stages stay ordered and frozen while training proceeds."""
import threading
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

from semantic_atdom.semantics.orchestrator import AsyncProgressiveSemanticHook, SemanticSyncResult
from semantic_atdom.semantics.publication import MapPublicationPolicy
from semantic_atdom.semantics.state import SemanticGaussianState
from semantic_atdom.semantics.topology import TopologyDelta


def model():
    class Model:
        _xyz = torch.zeros(1, 3)
        _opacity = torch.zeros(1, 1)
        _scaling = torch.zeros(1, 3)
        _rotation = torch.tensor([[1., 0., 0., 0.]])
        _features_dc = torch.zeros(1, 1, 3)
        _features_rest = torch.zeros(1, 0, 3)
        get_xyz = property(lambda self: self._xyz)
    return Model()


def camera(stage):
    return SimpleNamespace(image_name=f'{stage:06d}', image_height=2, image_width=3,
        FoVx=1., FoVy=1., world_view_transform=torch.eye(4), full_proj_transform=torch.eye(4),
        camera_center=torch.zeros(3), Tri_Mask=None)


def core(started, release):
    class Core:
        orchestrator = SimpleNamespace(state=SemanticGaussianState.initialize(1, 2))
        publication_policy = MapPublicationPolicy()
        dataset = SimpleNamespace(class_spec=SimpleNamespace(num_classes=2))
        def synchronize(self, stage, cameras, snapshot, root, final, boundaries, publication):
            assert snapshot.get_xyz.shape[0] == self.orchestrator.state.num_gaussians
            assert boundaries[-1] == f'{stage:06d}'
            if stage == 1:
                started.set()
                assert release.wait(5)
            return SemanticSyncResult(stage, cameras[-1].image_name, (), (), 0, True, ())
    return Core()


def test_capture_submission_returns_before_semantics_and_preserves_stage_before_topology():
    started, release = threading.Event(), threading.Event()
    hook, geometry, observed = AsyncProgressiveSemanticHook(core(started, release)), model(), []
    def capture(core, frozen, status):
        observed.append((status.geometry_stage, frozen.get_xyz.clone(), core.orchestrator.state.num_gaussians))
    try:
        hook.synchronize(1, [camera(1)], geometry, Path('unused'), on_synchronized=capture)
        assert started.wait(2)
        # The caller can already perform the next geometry update.
        geometry._xyz += 10
        hook.apply_topology(TopologyDelta.append_prior(1, 1))
        for name in ('_xyz', '_opacity', '_scaling', '_rotation', '_features_dc', '_features_rest'):
            setattr(geometry, name, torch.cat((getattr(geometry, name), getattr(geometry, name))))
        hook.synchronize(2, [camera(1), camera(2)], geometry, Path('unused'), on_synchronized=capture)
        release.set()
        hook.wait_for_idle()
        assert [x[0] for x in observed] == [1, 2]
        assert [x[2] for x in observed] == [1, 2]
        torch.testing.assert_close(observed[0][1], torch.zeros(1, 3))
        torch.testing.assert_close(observed[1][1], torch.full((2, 3), 10.))
        assert hook.capture_stats['completed'] == 2
    finally:
        release.set()
        hook.close()


def test_capture_capacity_applies_explicit_backpressure_without_dropping_frames():
    started, release, submitted = threading.Event(), threading.Event(), threading.Event()
    hook, geometry, observed = AsyncProgressiveSemanticHook(core(started, release)), model(), []
    capture = lambda core, frozen, status: observed.append(status.geometry_stage)
    failures = []
    def third():
        try:
            hook.synchronize(3, [camera(3)], geometry, Path('unused'), on_synchronized=capture)
            submitted.set()
        except BaseException as error:
            failures.append(error)
    try:
        hook.synchronize(1, [camera(1)], geometry, Path('unused'), on_synchronized=capture)
        assert started.wait(2)
        hook.synchronize(2, [camera(2)], geometry, Path('unused'), on_synchronized=capture)
        thread = threading.Thread(target=third)
        thread.start()
        assert not submitted.wait(.1)
        release.set()
        thread.join(3)
        assert submitted.is_set() and not failures
        hook.wait_for_idle()
        assert observed == [1, 2, 3]
        assert hook.capture_stats['max_pending'] == 2
        assert hook.capture_stats['backpressure_seconds'] >= .09
    finally:
        release.set()
        hook.close()


def test_capture_failure_is_visible_at_drain_and_releases_capacity():
    started, release = threading.Event(), threading.Event()
    release.set()
    hook = AsyncProgressiveSemanticHook(core(started, release))
    def fail(*args):
        raise ValueError('broken video capture')
    try:
        hook.synchronize(1, [camera(1)], model(), Path('unused'), on_synchronized=fail)
        with pytest.raises(RuntimeError, match='broken video capture'):
            hook.wait_for_idle()
        assert hook.capture_stats['pending'] == 0
    finally:
        hook.close()
