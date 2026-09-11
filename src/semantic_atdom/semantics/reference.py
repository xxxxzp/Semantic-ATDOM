"""Small, deterministic semantic fusion reference implementation.

These functions intentionally detach geometry-derived inputs. They define the
numerical contract for future CUDA forward-only kernels, not a training loss.
"""

from typing import NamedTuple, Optional

import torch


class ObservationStats(NamedTuple):
    evidence: torch.Tensor
    all_mass: torch.Tensor
    valid_mass: torch.Tensor
    distribution: torch.Tensor
    support: torch.Tensor


class GroupStats(NamedTuple):
    distribution: torch.Tensor
    support: torch.Tensor


class EvidenceBelief(NamedTuple):
    evidence: torch.Tensor
    distribution: torch.Tensor
    evidence_ratio: torch.Tensor
    concentration: torch.Tensor


class RobustPosterior(NamedTuple):
    raw_alpha: torch.Tensor
    robust_alpha: torch.Tensor
    js_divergence: torch.Tensor
    weights: torch.Tensor


class SemanticRender(NamedTuple):
    class_ids: torch.Tensor
    probabilities: torch.Tensor
    coverage: torch.Tensor
    pixel_evidence_support: torch.Tensor
    support_ratio: torch.Tensor
    top1_probability: torch.Tensor
    entropy: torch.Tensor
    normalized_entropy: torch.Tensor
    margin: torch.Tensor
    mixture_disagreement: torch.Tensor
    group_disagreement: torch.Tensor
    prior_regularized_variance_proxy: torch.Tensor

    @property
    def absolute_support(self) -> torch.Tensor:
        """Deprecated alias for the V1 reference field name."""
        return self.pixel_evidence_support


def _check_finite_nonnegative(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError("{} must contain only finite values".format(name))
    if bool((value < 0).any()):
        raise ValueError("{} must be non-negative".format(name))


def evidence_conditioned_belief(
    dirichlet_alpha: torch.Tensor,
    prior_total: float = 1.0,
    eps: float = 1e-8,
) -> EvidenceBelief:
    """Separate observed class evidence from a symmetric Dirichlet prior."""
    dirichlet_alpha = dirichlet_alpha.detach()
    if dirichlet_alpha.ndim != 2 or dirichlet_alpha.shape[1] < 1:
        raise ValueError("dirichlet_alpha must have shape [gaussians, classes]")
    if prior_total <= 0 or eps <= 0:
        raise ValueError("prior_total and eps must be positive")
    _check_finite_nonnegative("dirichlet_alpha", dirichlet_alpha)

    prior_per_class = prior_total / dirichlet_alpha.shape[1]
    evidence = dirichlet_alpha - prior_per_class
    if bool((evidence < -eps).any()):
        raise ValueError("dirichlet_alpha must not be below its prior")
    evidence = evidence.clamp_min(0.0)
    evidence_total = evidence.sum(dim=1)
    concentration = dirichlet_alpha.sum(dim=1)
    distribution = evidence / (evidence_total.unsqueeze(1) + eps)
    evidence_ratio = evidence_total / (concentration + eps)
    return EvidenceBelief(
        evidence=evidence,
        distribution=distribution,
        evidence_ratio=evidence_ratio,
        concentration=concentration,
    )


def footprint_observation(
    contributions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    boundary_weights: Optional[torch.Tensor] = None,
    ignore_id: int = 255,
    eps: float = 1e-8,
) -> ObservationStats:
    """Accumulate one camera's full-footprint alpha-T contributions.

    Args:
        contributions: Pixel-by-Gaussian matrix with alpha-T contributions.
        labels: One integer label per pixel.
        num_classes: Number of real classes, excluding the ignore id.
        boundary_weights: Optional reliability in [0, 1] for each pixel.
    """
    contributions = contributions.detach()
    labels = labels.detach().to(device=contributions.device)
    if contributions.ndim != 2:
        raise ValueError("contributions must have shape [pixels, gaussians]")
    if labels.ndim != 1 or labels.shape[0] != contributions.shape[0]:
        raise ValueError("labels must have shape [pixels]")
    integer_dtypes = (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    )
    if labels.dtype not in integer_dtypes:
        raise ValueError("labels must use an integer dtype")
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    _check_finite_nonnegative("contributions", contributions)

    if boundary_weights is None:
        boundary_weights = torch.ones_like(
            labels, dtype=contributions.dtype, device=contributions.device
        )
    else:
        boundary_weights = boundary_weights.detach().to(
            device=contributions.device, dtype=contributions.dtype
        )
    if boundary_weights.ndim != 1 or boundary_weights.shape != labels.shape:
        raise ValueError("boundary_weights must have shape [pixels]")
    _check_finite_nonnegative("boundary_weights", boundary_weights)
    if bool((boundary_weights > 1).any()):
        raise ValueError("boundary_weights must not exceed one")

    valid = labels != ignore_id
    invalid_class = valid & ((labels < 0) | (labels >= num_classes))
    if bool(invalid_class.any()):
        invalid_ids = torch.unique(labels[invalid_class]).cpu().tolist()
        raise ValueError("labels contain invalid class ids: {}".format(invalid_ids))

    safe_labels = labels.clone().long()
    safe_labels[~valid] = 0
    one_hot = torch.nn.functional.one_hot(
        safe_labels, num_classes=num_classes
    ).to(dtype=contributions.dtype)
    pixel_weight = boundary_weights * valid.to(dtype=contributions.dtype)
    evidence = contributions.transpose(0, 1).matmul(
        one_hot * pixel_weight.unsqueeze(1)
    )
    all_mass = contributions.sum(dim=0)
    valid_mass = (contributions * pixel_weight.unsqueeze(1)).sum(dim=0)
    evidence_mass = evidence.sum(dim=1)
    distribution = evidence / (evidence_mass.unsqueeze(1) + eps)
    support = (
        (1.0 - torch.exp(-all_mass)) * valid_mass / (all_mass + eps)
    ).clamp(min=0.0, max=1.0)
    return ObservationStats(
        evidence=evidence,
        all_mass=all_mass,
        valid_mass=valid_mass,
        distribution=distribution,
        support=support,
    )


def aggregate_observation_group(
    distributions: torch.Tensor,
    supports: torch.Tensor,
    eps: float = 1e-8,
) -> GroupStats:
    """Collapse correlated camera observations into at most one sample."""
    distributions = distributions.detach()
    supports = supports.detach().to(
        device=distributions.device, dtype=distributions.dtype
    )
    if distributions.ndim != 3:
        raise ValueError("distributions must have shape [views, gaussians, classes]")
    if supports.shape != distributions.shape[:2]:
        raise ValueError("supports must have shape [views, gaussians]")
    _check_finite_nonnegative("distributions", distributions)
    _check_finite_nonnegative("supports", supports)
    if bool((supports > 1).any()):
        raise ValueError("supports must not exceed one")

    weighted = (distributions * supports.unsqueeze(2)).sum(dim=0)
    support_sum = supports.sum(dim=0)
    group_distribution = weighted / (support_sum.unsqueeze(1) + eps)
    # Old PyTorch releases used by A-TDOM may try to JIT a reduction kernel
    # for torch.prod that does not recognize Ada GPUs. Groups contain at most
    # four rows online, so this equivalent loop is both bounded and portable.
    remaining = torch.ones_like(supports[0])
    for view_support in supports:
        remaining = remaining * (1.0 - view_support)
    group_support = 1.0 - remaining
    return GroupStats(group_distribution, group_support)


def _js_divergence(
    first: torch.Tensor, second: torch.Tensor, eps: float
) -> torch.Tensor:
    first = first / (first.sum(dim=-1, keepdim=True) + eps)
    second = second / (second.sum(dim=-1, keepdim=True) + eps)
    midpoint = 0.5 * (first + second)
    first_term = first * (
        torch.log(first.clamp_min(eps)) - torch.log(midpoint.clamp_min(eps))
    )
    second_term = second * (
        torch.log(second.clamp_min(eps)) - torch.log(midpoint.clamp_min(eps))
    )
    return 0.5 * (first_term.sum(dim=-1) + second_term.sum(dim=-1))


def exact_robust_posterior(
    group_distributions: torch.Tensor,
    group_supports: torch.Tensor,
    prior_total: float = 1.0,
    reliability_floor: float = 0.1,
    temperature: float = 0.1,
    min_consensus_support: float = 1.0,
    eps: float = 1e-8,
) -> RobustPosterior:
    """Build raw and exact leave-one-group-out robust Dirichlet posteriors."""
    group_distributions = group_distributions.detach()
    group_supports = group_supports.detach().to(
        device=group_distributions.device, dtype=group_distributions.dtype
    )
    if group_distributions.ndim != 3:
        raise ValueError(
            "group_distributions must have shape [groups, gaussians, classes]"
        )
    if group_supports.shape != group_distributions.shape[:2]:
        raise ValueError("group_supports must have shape [groups, gaussians]")
    if prior_total <= 0:
        raise ValueError("prior_total must be positive")
    if reliability_floor < 0 or reliability_floor > 1:
        raise ValueError("reliability_floor must be in [0, 1]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if min_consensus_support < 0:
        raise ValueError("min_consensus_support must be non-negative")
    _check_finite_nonnegative("group_distributions", group_distributions)
    _check_finite_nonnegative("group_supports", group_supports)
    if bool((group_supports > 1).any()):
        raise ValueError("group_supports must not exceed one")

    group_count, gaussian_count, class_count = group_distributions.shape
    prior = torch.full(
        (gaussian_count, class_count),
        prior_total / class_count,
        dtype=group_distributions.dtype,
        device=group_distributions.device,
    )
    evidence = group_supports.unsqueeze(2) * group_distributions
    raw = prior + evidence.sum(dim=0)
    js = torch.zeros_like(group_supports)
    reliability = torch.ones_like(group_supports)
    leave_out_support = torch.zeros_like(group_supports)

    effective_group_count = (group_supports > eps).sum(dim=0)
    if bool((effective_group_count >= 3).any()):
        total_evidence = evidence.sum(dim=0)
        for group_index in range(group_count):
            leave_one_out = total_evidence - evidence[group_index]
            leave_out_support[group_index] = leave_one_out.sum(dim=1)
            consensus = leave_one_out / (
                leave_one_out.sum(dim=1, keepdim=True) + eps
            )
            js[group_index] = _js_divergence(
                group_distributions[group_index], consensus, eps
            )
        candidate = reliability_floor + (1.0 - reliability_floor) * torch.exp(
            -js / temperature
        )
        other_group_count = effective_group_count.unsqueeze(0) - (
            group_supports > eps
        ).to(dtype=effective_group_count.dtype)
        eligible = (
            (other_group_count >= 2)
            & (leave_out_support >= min_consensus_support)
            & (group_supports > eps)
        )
        reliability = torch.where(eligible, candidate, reliability)

    robust = prior + (
        reliability.unsqueeze(2) * group_supports.unsqueeze(2) * group_distributions
    ).sum(dim=0)
    return RobustPosterior(raw, robust, js, reliability)


def render_semantic_tdom(
    contributions: torch.Tensor,
    dirichlet_alpha: torch.Tensor,
    group_disagreement: Optional[torch.Tensor] = None,
    prior_total: float = 1.0,
    min_coverage: float = 0.0,
    min_absolute_support: float = 0.0,
    min_support_ratio: float = 0.0,
    min_margin: float = 0.0,
    ignore_id: int = 255,
    eps: float = 1e-8,
) -> SemanticRender:
    """Render evidence-conditioned semantics and apply the three V2 gates.

    ``min_margin`` is accepted only for compatibility with the superseded V1
    reference API.  Margin remains a diagnostic and never changes class ids.
    """
    contributions = contributions.detach()
    dirichlet_alpha = dirichlet_alpha.detach().to(
        device=contributions.device, dtype=contributions.dtype
    )
    if contributions.ndim != 2:
        raise ValueError("contributions must have shape [pixels, gaussians]")
    if (
        dirichlet_alpha.ndim != 2
        or dirichlet_alpha.shape[0] != contributions.shape[1]
    ):
        raise ValueError("dirichlet_alpha must have shape [gaussians, classes]")
    _check_finite_nonnegative("contributions", contributions)
    _check_finite_nonnegative("dirichlet_alpha", dirichlet_alpha)
    if group_disagreement is not None:
        group_disagreement = group_disagreement.detach().to(
            device=contributions.device, dtype=contributions.dtype
        )
        if group_disagreement.shape != (contributions.shape[1],):
            raise ValueError(
                "group_disagreement must have shape [gaussians]"
            )
        _check_finite_nonnegative(
            "group_disagreement", group_disagreement
        )

    belief = evidence_conditioned_belief(
        dirichlet_alpha, prior_total=prior_total, eps=eps
    )
    semantic_weights = contributions * belief.evidence_ratio.unsqueeze(0)
    coverage = contributions.sum(dim=1)
    absolute_support = semantic_weights.sum(dim=1)
    support_ratio = absolute_support / (coverage + eps)
    safe_probabilities = semantic_weights.matmul(belief.distribution) / (
        absolute_support.unsqueeze(1) + eps
    )

    entropy = -(
        safe_probabilities * torch.log(safe_probabilities.clamp_min(eps))
    ).sum(dim=1)
    if safe_probabilities.shape[1] > 1:
        top_two = torch.topk(safe_probabilities, k=2, dim=1).values
        top1_probability = top_two[:, 0]
        margin = top_two[:, 0] - top_two[:, 1]
        entropy_scale = torch.log(
            torch.tensor(
                float(safe_probabilities.shape[1]),
                dtype=contributions.dtype,
                device=contributions.device,
            )
        )
    else:
        top1_probability = safe_probabilities[:, 0]
        margin = safe_probabilities[:, 0]
        entropy_scale = torch.ones(
            (), dtype=contributions.dtype, device=contributions.device
        )
    normalized_entropy = entropy / entropy_scale
    distribution_second_moment = belief.distribution.square().sum(dim=1)
    mixture_disagreement = (
        semantic_weights.matmul(distribution_second_moment)
        / (absolute_support + eps)
        - safe_probabilities.square().sum(dim=1)
    ).clamp_min(0.0)
    normalized_weights = semantic_weights / (
        absolute_support.unsqueeze(1) + eps
    )
    alpha = dirichlet_alpha
    concentration = belief.concentration
    per_class_variance = alpha * (
        concentration.unsqueeze(1) - alpha
    ) / (
        concentration.square().unsqueeze(1)
        * (concentration + 1.0).unsqueeze(1)
        + eps
    )
    gaussian_trace_variance = per_class_variance.sum(dim=1)
    prior_regularized_variance_proxy = normalized_weights.square().matmul(
        gaussian_trace_variance
    )
    if group_disagreement is None:
        pixel_group_disagreement = torch.full_like(
            absolute_support, float("nan")
        )
    else:
        pixel_group_disagreement = semantic_weights.matmul(
            group_disagreement
        ) / (absolute_support + eps)
    has_evidence = absolute_support > eps
    undefined = torch.full_like(absolute_support, float("nan"))
    entropy = torch.where(has_evidence, entropy, undefined)
    normalized_entropy = torch.where(
        has_evidence, normalized_entropy, undefined
    )
    top1_probability = torch.where(
        has_evidence, top1_probability, undefined
    )
    margin = torch.where(has_evidence, margin, undefined)
    mixture_disagreement = torch.where(
        has_evidence, mixture_disagreement, undefined
    )
    prior_regularized_variance_proxy = torch.where(
        has_evidence, prior_regularized_variance_proxy, undefined
    )
    if group_disagreement is not None:
        pixel_group_disagreement = torch.where(
            has_evidence, pixel_group_disagreement, undefined
        )
    probabilities = torch.where(
        has_evidence.unsqueeze(1),
        safe_probabilities,
        torch.full_like(safe_probabilities, float("nan")),
    )
    class_ids = torch.argmax(safe_probabilities, dim=1)
    valid = (
        (coverage >= min_coverage)
        & (absolute_support >= min_absolute_support)
        & (support_ratio >= min_support_ratio)
        & has_evidence
    )
    class_ids = torch.where(
        valid, class_ids, torch.full_like(class_ids, ignore_id)
    )
    return SemanticRender(
        class_ids=class_ids,
        probabilities=probabilities,
        coverage=coverage,
        pixel_evidence_support=absolute_support,
        support_ratio=support_ratio,
        top1_probability=top1_probability,
        entropy=entropy,
        normalized_entropy=normalized_entropy,
        margin=margin,
        mixture_disagreement=mixture_disagreement,
        group_disagreement=pixel_group_disagreement,
        prior_regularized_variance_proxy=(
            prior_regularized_variance_proxy
        ),
    )
