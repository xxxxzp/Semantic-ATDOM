import numpy as np
import pytest
from types import SimpleNamespace
import torch

from semantic_atdom.semantics.planar_fallback import (
    PlanarFallbackConfig,
    accept_planar_votes,
    apply_planar_multiview_fallback,
    find_bounded_class_candidates,
)
from semantic_atdom.semantics.schema import SemanticClassSpec
from semantic_atdom.semantics.tdom import SemanticTdomResult


def test_candidates_exclude_boundary_connected_unknown_and_other_classes():
    ignore_id = 255
    water_id = 6
    class_ids = np.full((16, 16), 3, dtype=np.uint8)
    class_ids[2:14, 2:14] = water_id
    class_ids[6:10, 6:10] = ignore_id
    class_ids[0:3, 0:2] = ignore_id
    class_ids[12:14, 12:14] = 3
    class_ids[12:13, 12:13] = ignore_id

    config = PlanarFallbackConfig(
        class_name="water",
        min_component_pixels=4,
        ring_radius_pixels=1,
        min_boundary_consensus=0.95,
    )
    candidates = find_bounded_class_candidates(
        class_ids,
        ignore_id=ignore_id,
        target_class_id=water_id,
        config=config,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.area == 16
    assert candidate.bbox == (6, 6, 4, 4)
    assert candidate.boundary_consensus == 1.0


def test_vote_gate_requires_views_agreement_and_independent_groups():
    config = PlanarFallbackConfig(
        class_name="water",
        min_views=5,
        min_agreement=0.95,
        min_groups=3,
    )
    target_votes = np.array([10, 10, 4, 10, 10], dtype=np.uint16)
    valid_votes = np.array([10, 11, 4, 10, 10], dtype=np.uint16)
    target_groups = np.array([4, 4, 4, 2, 3], dtype=np.uint16)
    observed_groups = np.array([4, 4, 4, 4, 2], dtype=np.uint16)

    accepted, agreement = accept_planar_votes(
        target_votes,
        valid_votes,
        target_groups,
        observed_groups,
        config,
    )

    np.testing.assert_array_equal(
        accepted,
        np.array([True, False, False, False, False]),
    )
    np.testing.assert_allclose(
        agreement,
        np.array([1.0, 10.0 / 11.0, 1.0, 1.0, 1.0]),
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"min_views": 0}, "min_views"),
        ({"min_agreement": 1.01}, "min_agreement"),
        ({"max_plane_mad_world": 0.0}, "max_plane_mad_world"),
    ],
)
def test_config_rejects_unsafe_thresholds(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PlanarFallbackConfig(class_name="water", **kwargs)


@pytest.mark.parametrize("scale_values, ratio, expected_plane_count", [
    ([.1] * 3, 14., 3),
    ([1.] * 32 + [20.] * 32, 14., 32),
    ([1.] * 33 + [20.] * 32, 14., 33),
    ([1.] * 32 + [20.] * 32, 0., 64),
])
def test_end_to_end_fallback_updates_class_diagnostics_and_provenance(scale_values, ratio, expected_plane_count):
    class_spec = SemanticClassSpec.from_dict(
        {
            "schema_version": 1,
            "ignore_id": 255,
            "classes": [
                {"id": 0, "name": "ground", "color": [1, 2, 3]},
                {"id": 1, "name": "water", "color": [4, 5, 6]},
            ],
        }
    )
    class_ids = torch.ones((8, 8), dtype=torch.uint8)
    class_ids[3:5, 3:5] = 255
    unknown = class_ids == 255
    result = SemanticTdomResult(
        class_ids=class_ids,
        colors=torch.zeros((3, 8, 8), dtype=torch.uint8),
        unknown_mask=unknown,
        unknown_reason=unknown.to(torch.uint8),
        probabilities=torch.zeros((2, 8, 8), dtype=torch.float16),
        coverage=None,
        absolute_support=torch.zeros((8, 8)),
        support_ratio=None,
        top1_probability=torch.full((8, 8), float("nan")),
        entropy=torch.full((8, 8), float("nan")),
        normalized_entropy=torch.full((8, 8), float("nan")),
        margin=torch.full((8, 8), float("nan")),
        mixture_disagreement=None,
        group_disagreement=None,
        prior_regularized_variance_proxy=None,
        radii=None,
        metrics={},
    )
    camera = SimpleNamespace(
        image_name="000001",
        image_width=8,
        image_height=8,
        full_proj_transform=torch.eye(4),
        calibration=None,
        Tri_Mask=None,
    )
    dataset = SimpleNamespace(
        class_spec=class_spec,
        load_observation=lambda image_id: SimpleNamespace(
            labels=np.ones((8, 8), dtype=np.uint8)
        ),
    )
    gaussians = SimpleNamespace(
        get_xyz=torch.zeros((len(scale_values), 3)),
        get_scaling=torch.tensor(scale_values)[:, None].repeat(1, 3),
    )
    state = SimpleNamespace(
        alpha_robust=torch.tensor([[0.5, 4.5]]).repeat(len(scale_values), 1),
        prior_total=1.0,
        num_classes=2,
    )
    config = PlanarFallbackConfig(
        class_name="water",
        min_component_pixels=4,
        ring_radius_pixels=1,
        min_boundary_consensus=1.0,
        plane_context_pixels=4,
        min_plane_gaussians=1,
        max_plane_mad_world=0.1,
        min_views=1,
        min_agreement=1.0,
        min_groups=1,
    )

    outcome = apply_planar_multiview_fallback(
        gaussians=gaussians,
        semantic_state=state,
        tdom_camera=camera,
        alignment={"grid": {"plane_normal_world": [0.0, 0.0, 1.0]}},
        semantic_result=result,
        class_spec=class_spec,
        dataset=dataset,
        source_cameras=(camera,),
        observation_groups=(("000001",),),
        geometry_scale_ratio=ratio,
        config=config,
    )

    assert outcome.metrics["components"][0]["plane_gaussian_count"] == expected_plane_count
    assert outcome.metrics["filled_pixel_count"] == 4
    assert torch.all(outcome.result.class_ids[3:5, 3:5] == 1)
    assert torch.all(outcome.result.class_provenance[3:5, 3:5] == 2)
    assert torch.all(outcome.result.probabilities[1, 3:5, 3:5] == 1)
    assert torch.all(outcome.result.top1_probability[3:5, 3:5] == 1)
    assert torch.all(outcome.result.entropy[3:5, 3:5] == 0)
    assert outcome.result.colors is None
