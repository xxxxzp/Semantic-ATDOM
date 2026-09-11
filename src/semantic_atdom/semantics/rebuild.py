"""Exact two-pass semantic rebuild on frozen final Gaussian geometry."""

from dataclasses import asdict, dataclass
import time
from typing import Dict, Mapping, NamedTuple, Optional, Sequence, Tuple

import torch

from semantic_atdom.semantics.reference import (
    aggregate_observation_group,
    evidence_conditioned_belief,
)
from semantic_atdom.semantics.schema import (
    LedgerEntry,
    SemanticDataset,
    SemanticLedger,
    SemanticSchemaError,
)
from semantic_atdom.semantics.state import SemanticGaussianState


@dataclass(frozen=True)
class FinalRebuildMetrics:
    image_count: int
    group_count: int
    label_read_seconds: float
    projection_seconds: float
    fusion_seconds: float
    total_seconds: float
    peak_cuda_bytes: int
    online_mean_js: Optional[float] = None
    online_max_js: Optional[float] = None
    online_class_agreement: Optional[float] = None

    def to_dict(self):
        return asdict(self)


class FinalRebuildResult(NamedTuple):
    state: SemanticGaussianState
    ledger: SemanticLedger
    label_provenance: Mapping[str, object]
    metrics: FinalRebuildMetrics
    image_groups: Tuple[Tuple[str, ...], ...]


class FinalSemanticRebuilder:
    """Reproject every final label twice to obtain exact G0 and LOO G1."""

    def __init__(self, dataset: SemanticDataset, projector, config):
        self.dataset = dataset
        self.projector = projector
        self.config = config

    def rebuild(
        self,
        image_ids: Sequence[str],
        camera_by_id: Mapping[str, object],
        gaussians,
        overlap,
        online_state: Optional[SemanticGaussianState] = None,
    ) -> FinalRebuildResult:
        self._synchronize(gaussians.get_xyz.device)
        started = time.monotonic()
        if gaussians.get_xyz.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(gaussians.get_xyz.device)
        groups = self._groups(image_ids, overlap)
        device = gaussians.get_xyz.device
        gaussian_count = gaussians.get_xyz.shape[0]
        class_count = self.dataset.class_spec.num_classes
        state = SemanticGaussianState.initialize(
            gaussian_count,
            class_count,
            prior_total=self.config.prior_total,
            device=device,
        )
        raw = state.prior()
        robust = state.prior()
        group_count = torch.zeros(
            gaussian_count, dtype=torch.long, device=device
        )
        semantic_support = torch.zeros(
            gaussian_count, dtype=state.dtype, device=device
        )
        group_weight = torch.zeros(
            gaussian_count, dtype=state.dtype, device=device
        )
        group_first_moment = torch.zeros_like(raw)
        group_second_moment = torch.zeros_like(raw)
        ledger = SemanticLedger()
        provenance = {"schema_version": 1, "versions": {}}
        label_seconds = 0.0
        projection_seconds = 0.0
        fusion_seconds = 0.0

        for group in groups:
            distributions, supports, raw_support, timings = self._project_group(
                group,
                camera_by_id,
                gaussians,
                ledger,
                provenance,
            )
            label_seconds += timings[0]
            projection_seconds += timings[1]
            fusion_start = time.monotonic()
            stats = aggregate_observation_group(
                distributions, supports, eps=self.config.eps
            )
            evidence = stats.support.unsqueeze(1) * stats.distribution
            raw.add_(evidence)
            group_count.add_((stats.support > self.config.eps).long())
            semantic_support.add_(raw_support)
            group_weight.add_(stats.support)
            group_first_moment.add_(
                stats.support.unsqueeze(1) * stats.distribution
            )
            group_second_moment.add_(
                stats.support.unsqueeze(1) * stats.distribution.square()
            )
            self._synchronize(device)
            fusion_seconds += time.monotonic() - fusion_start

        if len(provenance["versions"]) > 1:
            raise SemanticSchemaError(
                "final semantic rebuild cannot mix label versions"
            )

        for group in groups:
            distributions, supports, _, timings = self._project_group(
                group,
                camera_by_id,
                gaussians,
                None,
                None,
                expected_ledger=ledger,
                expected_provenance=provenance,
            )
            label_seconds += timings[0]
            projection_seconds += timings[1]
            fusion_start = time.monotonic()
            stats = aggregate_observation_group(
                distributions, supports, eps=self.config.eps
            )
            evidence = stats.support.unsqueeze(1) * stats.distribution
            leave_one_out = (raw - state.prior()) - evidence
            leave_out_support = leave_one_out.sum(dim=1)
            consensus = leave_one_out / (
                leave_out_support.unsqueeze(1) + self.config.eps
            )
            js = self._js_divergence(stats.distribution, consensus)
            candidate = self.config.reliability_floor + (
                1.0 - self.config.reliability_floor
            ) * torch.exp(-js / self.config.temperature)
            other_group_count = group_count - (
                stats.support > self.config.eps
            ).long()
            eligible = (
                (other_group_count >= 2)
                & (leave_out_support >= self.config.min_consensus_support)
                & (stats.support > self.config.eps)
            )
            reliability = torch.where(
                eligible, candidate, torch.ones_like(candidate)
            )
            robust.add_(reliability.unsqueeze(1) * evidence)
            self._synchronize(device)
            fusion_seconds += time.monotonic() - fusion_start

        group_mean = group_first_moment / (
            group_weight.unsqueeze(1) + self.config.eps
        )
        disagreement = (
            group_second_moment
            / (group_weight.unsqueeze(1) + self.config.eps)
            - group_mean.square()
        ).clamp_min(0.0)
        state.alpha_raw = raw
        state.alpha_robust = robust
        state.group_variance = disagreement
        state.group_count = group_count
        state.semantic_support = semantic_support
        state.group_disagreement_available = bool(
            ((group_count == 0) | (group_count >= 2)).all()
        )
        state.mark_historical_evidence_current()
        state.mark_dirty()
        state._validate()
        comparison = self._compare_online(online_state, state)
        peak_cuda_bytes = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        )
        self._synchronize(device)
        metrics = FinalRebuildMetrics(
            image_count=sum(len(group) for group in groups),
            group_count=len(groups),
            label_read_seconds=label_seconds,
            projection_seconds=projection_seconds,
            fusion_seconds=fusion_seconds,
            total_seconds=time.monotonic() - started,
            peak_cuda_bytes=peak_cuda_bytes,
            **comparison,
        )
        return FinalRebuildResult(
            state, ledger, provenance, metrics, groups
        )

    def _project_group(
        self,
        image_ids,
        camera_by_id,
        gaussians,
        ledger,
        provenance,
        expected_ledger=None,
        expected_provenance=None,
    ):
        distributions = []
        supports = []
        raw_support = torch.zeros(
            gaussians.get_xyz.shape[0],
            dtype=torch.float32,
            device=gaussians.get_xyz.device,
        )
        label_seconds = 0.0
        projection_seconds = 0.0
        for image_id in image_ids:
            before = time.monotonic()
            observation = self.dataset.load_observation(image_id)
            label_seconds += time.monotonic() - before
            if image_id not in camera_by_id:
                raise ValueError(
                    "final semantic image has no camera: {}".format(image_id)
                )
            self._synchronize(gaussians.get_xyz.device)
            before = time.monotonic()
            projection = self.projector.project(
                gaussians, camera_by_id[image_id], observation
            )
            self._synchronize(gaussians.get_xyz.device)
            projection_seconds += time.monotonic() - before
            distributions.append(projection.distribution)
            supports.append(projection.support)
            raw_support.add_(projection.semantic_support)
            if ledger is not None:
                ledger.record(
                    LedgerEntry(
                        image_id=image_id,
                        label_version=observation.key.label_version,
                        label_sha256=observation.metadata["label_sha256"],
                        score_sha256=observation.metadata.get("score_sha256"),
                    )
                )
                candidate = {
                    "model": observation.metadata["model"],
                    "checkpoint_sha256": observation.metadata["checkpoint_sha256"],
                    "prompt_config": observation.metadata["prompt_config"],
                    "thresholds": dict(observation.metadata["thresholds"]),
                }
                previous = provenance["versions"].setdefault(
                    observation.key.label_version, candidate
                )
                if previous != candidate:
                    raise ValueError("label provenance changed during final rebuild")
            if expected_ledger is not None:
                expected = expected_ledger.get(
                    image_id, observation.key.label_version
                )
                if expected is None:
                    raise SemanticSchemaError(
                        "label identity changed between final rebuild passes: {}".format(
                            image_id
                        )
                    )
                actual = LedgerEntry(
                    image_id=image_id,
                    label_version=observation.key.label_version,
                    label_sha256=observation.metadata["label_sha256"],
                    score_sha256=observation.metadata.get("score_sha256"),
                )
                if actual != expected:
                    raise SemanticSchemaError(
                        "label content changed between final rebuild passes: {}".format(
                            image_id
                        )
                    )
                candidate = {
                    "model": observation.metadata["model"],
                    "checkpoint_sha256": observation.metadata[
                        "checkpoint_sha256"
                    ],
                    "prompt_config": observation.metadata["prompt_config"],
                    "thresholds": dict(observation.metadata["thresholds"]),
                }
                if expected_provenance["versions"].get(
                    observation.key.label_version
                ) != candidate:
                    raise SemanticSchemaError(
                        "label provenance changed between final rebuild passes"
                    )
        return (
            torch.stack(distributions),
            torch.stack(supports),
            raw_support,
            (label_seconds, projection_seconds),
        )

    @staticmethod
    def _synchronize(device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def _groups(self, image_ids, overlap):
        groups = []
        current = []
        for image_id in image_ids:
            if current and (
                len(current) >= self.config.max_group_size
                or overlap.normalized(current[0], image_id)
                < self.config.overlap_threshold
            ):
                groups.append(tuple(current))
                current = []
            current.append(image_id)
        if current:
            groups.append(tuple(current))
        return tuple(groups)

    def _js_divergence(self, first, second):
        eps = self.config.eps
        first = first / (first.sum(dim=1, keepdim=True) + eps)
        second = second / (second.sum(dim=1, keepdim=True) + eps)
        midpoint = 0.5 * (first + second)
        return 0.5 * (
            (
                first
                * (
                    torch.log(first.clamp_min(eps))
                    - torch.log(midpoint.clamp_min(eps))
                )
            ).sum(dim=1)
            + (
                second
                * (
                    torch.log(second.clamp_min(eps))
                    - torch.log(midpoint.clamp_min(eps))
                )
            ).sum(dim=1)
        )

    def _compare_online(self, online_state, final_state) -> Dict[str, float]:
        if (
            online_state is None
            or online_state.num_gaussians != final_state.num_gaussians
        ):
            return {}
        eps = self.config.eps
        online = evidence_conditioned_belief(
            online_state.alpha_robust,
            prior_total=online_state.prior_total,
            eps=eps,
        ).distribution
        exact = evidence_conditioned_belief(
            final_state.alpha_robust,
            prior_total=final_state.prior_total,
            eps=eps,
        ).distribution
        js = self._js_divergence(online, exact)
        observed = final_state.group_count > 0
        if not bool(observed.any()):
            return {}
        return {
            "online_mean_js": float(js[observed].mean().item()),
            "online_max_js": float(js[observed].max().item()),
            "online_class_agreement": float(
                (
                    online[observed].argmax(dim=1)
                    == exact[observed].argmax(dim=1)
                ).float().mean().item()
            ),
        }
