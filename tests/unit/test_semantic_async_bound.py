import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from semantic_atdom.semantics.orchestrator import AsyncProgressiveSemanticHook, MapPublicationRequest, SemanticSyncResult
from semantic_atdom.semantics.publication import MapPublicationPolicy
from semantic_atdom.semantics.state import SemanticGaussianState
from semantic_atdom.semantics.tdom import SemanticTdomConfig
from semantic_atdom.semantics.topology import TopologyDelta


class AsyncSnapshotBoundTests(unittest.TestCase):
    def test_pending_map_does_not_retain_all_intermediate_projection_snapshots(self):
        started, release = threading.Event(), threading.Event()
        observed = []
        class Core:
            orchestrator = SimpleNamespace(state=SemanticGaussianState.initialize(1, 2))
            dataset = SimpleNamespace(class_spec=SimpleNamespace(num_classes=2))
            publication_policy = MapPublicationPolicy()
            def checkpoint_required(self, *args): return False
            def synchronize(self, stage, cameras, snapshot, *args):
                if self.orchestrator.state.num_gaussians != snapshot.get_xyz.shape[0]:
                    raise RuntimeError("state and snapshot topology diverged")
                observed.append((stage, tuple(c.image_name for c in cameras)))
                if stage == 1:
                    started.set()
                    if not release.wait(10): raise RuntimeError('test release timed out')
                return SemanticSyncResult(stage, cameras[-1].image_name, (), (), 0, True, ())
        def camera(name):
            return SimpleNamespace(image_name=name, image_height=1, image_width=1,
                FoVx=1., FoVy=1., world_view_transform=torch.eye(4),
                full_proj_transform=torch.eye(4), camera_center=torch.zeros(3), Tri_Mask=None)
        gaussians = SimpleNamespace(_xyz=torch.zeros(1, 3), _opacity=torch.zeros(1, 1),
            _scaling=torch.zeros(1, 3), _rotation=torch.tensor([[1., 0., 0., 0.]]),
            _features_dc=torch.zeros(1, 1, 3), _features_rest=torch.zeros(1, 0, 3))
        gaussians.get_xyz = gaussians._xyz
        hook = AsyncProgressiveSemanticHook(Core())
        try:
            hook.synchronize(1, [camera('000001')], gaussians, Path('unused'))
            self.assertTrue(started.wait(2))
            hook.synchronize(2, [camera('000001'), camera('000002')], gaussians, Path('unused'),
                publication_request=MapPublicationRequest(product_key='stage2', output_root=Path('unused'),
                    camera=camera('tdom'), alignment={'geometry_stage': 2, 'grid': {'grid_id':'g'}, 'window':{'grid_id':'g'}},
                    rgb_renderer=lambda *args: None, rgb_pipeline=SimpleNamespace(compute_cov3D_python=False, GetNormal=False),
                    rgb_background=torch.zeros(3), semantic_config=SemanticTdomConfig()))
            for stage in range(3, 203):
                if stage == 101:
                    hook.apply_topology(TopologyDelta.append_prior(1, 1))
                    for name in ("_xyz", "_opacity", "_scaling", "_rotation", "_features_dc", "_features_rest"):
                        value = getattr(gaussians, name)
                        setattr(gaussians, name, torch.cat((value, value), dim=0))
                    gaussians.get_xyz = gaussians._xyz
                hook.synchronize(stage, [camera(f'{i:06d}') for i in range(1, stage+1)], gaussians, Path('unused'))
            release.set()
            hook.wait_for_idle()
            self.assertEqual([x[0] for x in observed], [1, 2, 202])
            self.assertEqual(len(observed[-1][1]), 202)
        finally:
            release.set()
            hook.close()
