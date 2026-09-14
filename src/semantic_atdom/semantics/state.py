"""Non-parameter semantic state aligned one-to-one with Gaussian geometry."""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import torch

from semantic_atdom.semantics.topology import TopologyDelta
from semantic_atdom.semantics.provenance import SEMANTIC_METHOD_VERSION
from semantic_atdom.semantics.schema import (
    LedgerEntry,
    SemanticClassSpec,
    SemanticLedger,
    SemanticSchemaError,
)


class SemanticStateError(ValueError):
    """Raised when semantic state and geometry cannot be aligned safely."""


class SemanticGaussianState:
    """Mutable sidecar tensors that are deliberately outside the RGB optimizer."""

    SCHEMA_VERSION = 3
    RECOVERY_CONTEXT_KEYS = (
        "ledger",
        "label_provenance",
        "open_group",
        "current_geometry_stage",
        "class_spec",
        "semantic_config",
    )

    def __init__(
        self,
        alpha_raw: torch.Tensor,
        alpha_robust: torch.Tensor,
        group_variance: torch.Tensor,
        group_count: torch.Tensor,
        semantic_support: torch.Tensor,
        dirty: torch.Tensor,
        gaussian_ids: torch.Tensor,
        prior_total: float,
        topology_version: int = 0,
        next_gaussian_id: Optional[int] = None,
        recovery_context: Optional[Mapping[str, object]] = None,
        open_raw_evidence: Optional[torch.Tensor] = None,
        open_robust_evidence: Optional[torch.Tensor] = None,
        open_group_count: Optional[torch.Tensor] = None,
        open_distributions: Optional[torch.Tensor] = None,
        open_supports: Optional[torch.Tensor] = None,
        open_semantic_supports: Optional[torch.Tensor] = None,
        historical_evidence_stale: bool = False,
        group_disagreement_available: bool = False,
        active_projection_dirty: Optional[torch.Tensor] = None,
    ):
        self.alpha_raw = alpha_raw.detach()
        self.alpha_robust = alpha_robust.detach()
        self.group_variance = group_variance.detach()
        self.group_count = group_count.detach()
        self.semantic_support = semantic_support.detach()
        self.dirty = dirty.detach()
        if active_projection_dirty is None:
            active_projection_dirty = torch.zeros_like(self.dirty)
        self.active_projection_dirty = active_projection_dirty.detach().to(
            device=self.alpha_raw.device, dtype=torch.bool
        )
        self.gaussian_ids = gaussian_ids.detach()
        self.prior_total = float(prior_total)
        self.topology_version = int(topology_version)
        if next_gaussian_id is None:
            next_gaussian_id = (
                int(self.gaussian_ids.max().item()) + 1
                if self.gaussian_ids.numel()
                else 0
            )
        self.next_gaussian_id = int(next_gaussian_id)
        self.recovery_context = dict(recovery_context or {})
        self.open_raw_evidence = self._optional_like_alpha(
            open_raw_evidence
        )
        self.open_robust_evidence = self._optional_like_alpha(
            open_robust_evidence
        )
        self.open_group_count = self._optional_row_tensor(
            open_group_count, torch.long
        )
        if open_distributions is None:
            open_distributions = torch.empty(
                (0, self.alpha_raw.shape[0], self.alpha_raw.shape[1]),
                dtype=self.alpha_raw.dtype,
                device=self.alpha_raw.device,
            )
        if open_supports is None:
            open_supports = torch.empty(
                (0, self.alpha_raw.shape[0]),
                dtype=self.alpha_raw.dtype,
                device=self.alpha_raw.device,
            )
        if open_semantic_supports is None:
            open_semantic_supports = torch.empty(
                (0, self.alpha_raw.shape[0]),
                dtype=self.alpha_raw.dtype,
                device=self.alpha_raw.device,
            )
        self.open_distributions = open_distributions.detach().to(
            device=self.alpha_raw.device, dtype=self.alpha_raw.dtype
        )
        self.open_supports = open_supports.detach().to(
            device=self.alpha_raw.device, dtype=self.alpha_raw.dtype
        )
        self.open_semantic_supports = open_semantic_supports.detach().to(
            device=self.alpha_raw.device, dtype=self.alpha_raw.dtype
        )
        self.is_valid = True
        self.invalid_reason = None
        self.historical_evidence_stale = bool(historical_evidence_stale)
        self.group_disagreement_available = bool(group_disagreement_available)
        self._validate()

    @classmethod
    def initialize(
        cls,
        num_gaussians: int,
        num_classes: int,
        prior_total: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "SemanticGaussianState":
        if num_gaussians < 0:
            raise SemanticStateError("num_gaussians must be non-negative")
        if num_classes < 1:
            raise SemanticStateError("num_classes must be positive")
        if prior_total <= 0:
            raise SemanticStateError("prior_total must be positive")
        prior = torch.full(
            (num_gaussians, num_classes),
            prior_total / num_classes,
            device=device,
            dtype=dtype,
        )
        return cls(
            alpha_raw=prior.clone(),
            alpha_robust=prior.clone(),
            group_variance=torch.zeros_like(prior),
            group_count=torch.zeros(
                num_gaussians, dtype=torch.long, device=device
            ),
            semantic_support=torch.zeros(
                num_gaussians, dtype=dtype, device=device
            ),
            dirty=torch.ones(num_gaussians, dtype=torch.bool, device=device),
            gaussian_ids=torch.arange(
                num_gaussians, dtype=torch.long, device=device
            ),
            prior_total=prior_total,
            next_gaussian_id=num_gaussians,
        )

    @property
    def num_gaussians(self) -> int:
        return self.alpha_raw.shape[0]

    @property
    def num_classes(self) -> int:
        return self.alpha_raw.shape[1]

    @property
    def device(self) -> torch.device:
        return self.alpha_raw.device

    @property
    def dtype(self) -> torch.dtype:
        return self.alpha_raw.dtype

    @property
    def gaussian_ids_sha256(self) -> str:
        values = self.gaussian_ids.detach().cpu().contiguous().numpy()
        return hashlib.sha256(values.tobytes()).hexdigest()

    def invalidate(self, reason: object) -> None:
        """Make a failed sidecar unusable until it is explicitly rebuilt."""
        self.is_valid = False
        self.invalid_reason = str(reason)

    def _require_valid(self) -> None:
        if not self.is_valid:
            raise SemanticStateError(
                "semantic state requires rebuild: {}".format(
                    self.invalid_reason or "unknown topology observer failure"
                )
            )

    def prior(self, row_count: Optional[int] = None) -> torch.Tensor:
        if row_count is None:
            row_count = self.num_gaussians
        return torch.full(
            (row_count, self.num_classes),
            self.prior_total / self.num_classes,
            device=self.device,
            dtype=self.dtype,
        )

    @torch.no_grad()
    def add_evidence(
        self,
        raw_evidence: torch.Tensor,
        robust_evidence: Optional[torch.Tensor] = None,
        support: Optional[torch.Tensor] = None,
        group_count_increment: Optional[torch.Tensor] = None,
    ) -> None:
        self._require_valid()
        raw = self._evidence_tensor(raw_evidence, "raw_evidence")
        robust = (
            raw
            if robust_evidence is None
            else self._evidence_tensor(robust_evidence, "robust_evidence")
        )
        validated_support = None
        if support is not None:
            validated_support = support.detach().to(
                device=self.device, dtype=self.dtype
            )
            if validated_support.shape != (self.num_gaussians,):
                raise SemanticStateError(
                    "support must have shape [num_gaussians]"
                )
            self._check_nonnegative_finite("support", validated_support)
        validated_increment = None
        if group_count_increment is not None:
            integer_types = (
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            )
            if group_count_increment.dtype not in integer_types:
                raise SemanticStateError(
                    "group_count_increment must use an integer dtype"
                )
            validated_increment = group_count_increment.detach().to(
                device=self.device, dtype=torch.long
            )
            if validated_increment.shape != (self.num_gaussians,):
                raise SemanticStateError(
                    "group_count_increment must have shape [num_gaussians]"
                )
            if bool((validated_increment < 0).any()):
                raise SemanticStateError(
                    "group_count_increment must be non-negative"
                )

        changed = (raw > 0).any(dim=1) | (robust > 0).any(dim=1)
        if validated_support is not None:
            changed |= validated_support > 0
        if validated_increment is not None:
            changed |= validated_increment > 0
        self.alpha_raw.add_(raw)
        self.alpha_robust.add_(robust)
        self.group_disagreement_available = False
        if validated_support is not None:
            self.semantic_support.add_(validated_support)
        if validated_increment is not None:
            self.group_count.add_(validated_increment)
        self.dirty[changed] = True

    @torch.no_grad()
    def replace_open_group(
        self,
        raw_evidence: torch.Tensor,
        robust_evidence: torch.Tensor,
        group_count_increment: torch.Tensor,
        distributions: torch.Tensor,
        supports: torch.Tensor,
        semantic_supports: torch.Tensor,
        semantic_support_increment: Optional[torch.Tensor] = None,
        commit_existing: bool = False,
    ) -> None:
        """Replace, rather than re-add, the current provisional group."""
        self._require_valid()
        raw = self._evidence_tensor(raw_evidence, "raw_evidence")
        robust = self._evidence_tensor(
            robust_evidence, "robust_evidence"
        )
        increment = self._group_count_tensor(group_count_increment)
        distributions, supports = self._open_observation_tensors(
            distributions, supports
        )
        semantic_supports = semantic_supports.detach().to(
            device=self.device, dtype=self.dtype
        )
        if semantic_supports.shape != supports.shape:
            raise SemanticStateError(
                "open semantic supports must match open supports"
            )
        self._check_nonnegative_finite(
            "open_semantic_supports", semantic_supports
        )
        support_increment = self._support_tensor(
            semantic_support_increment, "semantic_support_increment"
        )

        if commit_existing:
            closed_raw = self.alpha_raw
            closed_robust = self.alpha_robust
            closed_count = self.group_count
        else:
            closed_raw = self.alpha_raw - self.open_raw_evidence
            closed_robust = self.alpha_robust - self.open_robust_evidence
            closed_count = self.group_count - self.open_group_count
        tolerance = torch.finfo(self.dtype).eps * 8
        if bool((closed_raw < -tolerance).any()) or bool(
            (closed_robust < -tolerance).any()
        ):
            raise SemanticStateError(
                "open-group evidence exceeds the stored posterior"
            )
        if bool((closed_count < 0).any()):
            raise SemanticStateError(
                "open-group count exceeds the stored group count"
            )

        changed = (
            (self.open_raw_evidence != raw).any(dim=1)
            | (self.open_robust_evidence != robust).any(dim=1)
            | (self.open_group_count != increment)
        )
        if support_increment is not None:
            changed |= support_increment > 0
        new_alpha_raw = closed_raw + raw
        new_alpha_robust = closed_robust + robust
        new_group_count = closed_count + increment
        new_semantic_support = self.semantic_support.clone()
        if support_increment is not None:
            new_semantic_support.add_(support_increment)
        elif not commit_existing:
            new_semantic_support.sub_(self.open_semantic_supports.sum(dim=0))
            new_semantic_support.add_(semantic_supports.sum(dim=0))
            if bool((new_semantic_support < -tolerance).any()):
                raise SemanticStateError(
                    "open semantic support exceeds total semantic support"
                )
            new_semantic_support.clamp_min_(0)
        new_open_raw = raw.clone()
        new_open_robust = robust.clone()
        new_open_count = increment.clone()
        new_distributions = distributions.clone()
        new_supports = supports.clone()
        new_semantic_supports = semantic_supports.clone()

        self.alpha_raw = new_alpha_raw
        self.alpha_robust = new_alpha_robust
        self.group_disagreement_available = False
        self.group_count = new_group_count
        self.semantic_support = new_semantic_support
        self.open_raw_evidence = new_open_raw
        self.open_robust_evidence = new_open_robust
        self.open_group_count = new_open_count
        self.open_distributions = new_distributions
        self.open_supports = new_supports
        self.open_semantic_supports = new_semantic_supports
        self.dirty[changed] = True
        self._validate()

    @torch.no_grad()
    def close_open_group(self) -> None:
        """Commit the provisional contribution and forget its replacement data."""
        self._require_valid()
        self.open_raw_evidence.zero_()
        self.open_robust_evidence.zero_()
        self.open_group_count.zero_()
        self.open_distributions = torch.empty(
            (0, self.num_gaussians, self.num_classes),
            dtype=self.dtype,
            device=self.device,
        )
        self.open_supports = torch.empty(
            (0, self.num_gaussians),
            dtype=self.dtype,
            device=self.device,
        )
        self.open_semantic_supports = torch.empty(
            (0, self.num_gaussians),
            dtype=self.dtype,
            device=self.device,
        )

    @torch.no_grad()
    def apply_topology(self, delta: TopologyDelta) -> None:
        self._require_valid()
        delta.validate()
        if delta.old_count != self.num_gaussians:
            raise SemanticStateError(
                "topology expected {} Gaussians, state has {}".format(
                    delta.old_count, self.num_gaussians
                )
            )
        if delta.is_noop:
            return
        if bool((self.group_count - self.open_group_count > 0).any()):
            self.mark_historical_evidence_stale()
        sources = delta.source_indices.to(device=self.device)
        weights = delta.evidence_weights.to(device=self.device, dtype=self.dtype)
        preserve_ids = delta.preserve_source_ids.to(device=self.device)
        changed = delta.changed_rows.to(device=self.device)
        valid = sources >= 0

        new_raw = self.prior(delta.new_count)
        new_robust = self.prior(delta.new_count)
        new_variance = torch.zeros_like(new_raw)
        new_group_count = torch.zeros(
            delta.new_count, dtype=torch.long, device=self.device
        )
        new_support = torch.zeros(
            delta.new_count, dtype=self.dtype, device=self.device
        )
        new_open_raw = torch.zeros_like(new_raw)
        new_open_robust = torch.zeros_like(new_raw)
        new_open_group_count = torch.zeros(
            delta.new_count, dtype=torch.long, device=self.device
        )
        open_view_count = self.open_distributions.shape[0]
        new_open_distributions = torch.zeros(
            (open_view_count, delta.new_count, self.num_classes),
            dtype=self.dtype,
            device=self.device,
        )
        new_open_supports = torch.zeros(
            (open_view_count, delta.new_count),
            dtype=self.dtype,
            device=self.device,
        )
        new_open_semantic_supports = torch.zeros(
            (open_view_count, delta.new_count),
            dtype=self.dtype,
            device=self.device,
        )
        new_dirty = torch.ones(
            delta.new_count, dtype=torch.bool, device=self.device
        )
        new_active_projection_dirty = changed & (open_view_count > 0)
        new_ids = torch.full(
            (delta.new_count,), -1, dtype=torch.long, device=self.device
        )

        # Belief is local primitive state rather than globally conserved mass:
        # every geometric descendant copies it. The legacy weights below scale
        # only support-like quantities used for refresh/accounting.
        inherited = valid
        if bool(inherited.any()):
            old_indices = sources[inherited]
            row_weights = weights[inherited]
            old_prior = self.prior()
            new_raw[inherited] += (
                self.alpha_raw[old_indices] - old_prior[old_indices]
            )
            new_robust[inherited] += (
                self.alpha_robust[old_indices] - old_prior[old_indices]
            )
            new_variance[inherited] = self.group_variance[old_indices]
            new_group_count[inherited] = self.group_count[old_indices]
            new_support[inherited] = (
                self.semantic_support[old_indices] * row_weights
            )
            new_open_raw[inherited] = (
                self.open_raw_evidence[old_indices]
            )
            new_open_robust[inherited] = (
                self.open_robust_evidence[old_indices]
            )
            new_open_group_count[inherited] = self.open_group_count[
                old_indices
            ]
            if open_view_count:
                new_open_distributions[:, inherited] = (
                    self.open_distributions[:, old_indices]
                )
                new_open_supports[:, inherited] = (
                    self.open_supports[:, old_indices]
                    * row_weights.unsqueeze(0)
                )
                new_open_semantic_supports[:, inherited] = (
                    self.open_semantic_supports[:, old_indices]
                    * row_weights.unsqueeze(0)
                )
            new_dirty[inherited] = self.dirty[old_indices] | changed[inherited]
            new_active_projection_dirty[inherited] = (
                self.active_projection_dirty[old_indices]
                | (changed[inherited] & (open_view_count > 0))
            )

        retained = valid & preserve_ids
        if bool(retained.any()):
            new_ids[retained] = self.gaussian_ids[sources[retained]]
        needs_id = ~retained
        new_id_count = int(needs_id.sum().item())
        if new_id_count:
            new_ids[needs_id] = torch.arange(
                self.next_gaussian_id,
                self.next_gaussian_id + new_id_count,
                dtype=torch.long,
                device=self.device,
            )
            self.next_gaussian_id += new_id_count

        self.alpha_raw = new_raw
        self.alpha_robust = new_robust
        self.group_variance = new_variance
        self.group_count = new_group_count
        self.semantic_support = new_support
        self.open_raw_evidence = new_open_raw
        self.open_robust_evidence = new_open_robust
        self.open_group_count = new_open_group_count
        self.open_distributions = new_open_distributions
        self.open_supports = new_open_supports
        self.open_semantic_supports = new_open_semantic_supports
        self.dirty = new_dirty
        self.active_projection_dirty = new_active_projection_dirty
        self.gaussian_ids = new_ids
        if delta.kind != "identity":
            self.topology_version += 1
        self._validate()

    @torch.no_grad()
    def mark_dirty(self, mask: Optional[torch.Tensor] = None) -> None:
        self._require_valid()
        if mask is None:
            self.dirty.fill_(True)
            return
        mask = mask.detach().to(device=self.device, dtype=torch.bool)
        if mask.shape != (self.num_gaussians,):
            raise SemanticStateError("dirty mask must have shape [num_gaussians]")
        self.dirty[mask] = True

    @torch.no_grad()
    def mark_clean(self, mask: Optional[torch.Tensor] = None) -> None:
        self._require_valid()
        if mask is None:
            self.dirty.zero_()
            return
        mask = mask.detach().to(device=self.device, dtype=torch.bool)
        if mask.shape != (self.num_gaussians,):
            raise SemanticStateError("clean mask must have shape [num_gaussians]")
        self.dirty[mask] = False

    @torch.no_grad()
    def mark_active_projection_current(
        self, mask: Optional[torch.Tensor] = None
    ) -> None:
        """Mark refreshed active-window projections as current."""
        self._require_valid()
        if mask is None:
            self.active_projection_dirty.zero_()
            return
        mask = mask.detach().to(device=self.device, dtype=torch.bool)
        if mask.shape != (self.num_gaussians,):
            raise SemanticStateError(
                "active projection mask must have shape [num_gaussians]"
            )
        self.active_projection_dirty[mask] = False

    @torch.no_grad()
    def mark_active_projection_dirty(
        self, mask: Optional[torch.Tensor] = None
    ) -> None:
        """Require active-window reprojection after geometry changes."""
        self._require_valid()
        if mask is None:
            self.active_projection_dirty.fill_(True)
            return
        mask = mask.detach().to(device=self.device, dtype=torch.bool)
        if mask.shape != (self.num_gaussians,):
            raise SemanticStateError(
                "active projection mask must have shape [num_gaussians]"
            )
        self.active_projection_dirty[mask] = True

    @torch.no_grad()
    def mark_historical_evidence_stale(self) -> None:
        """Record that closed evidence predates the current geometry."""
        self._require_valid()
        self.historical_evidence_stale = True
        self.group_disagreement_available = False

    @torch.no_grad()
    def mark_historical_evidence_current(self) -> None:
        """Record completion of an exact rebuild on frozen geometry."""
        self._require_valid()
        self.historical_evidence_stale = False

    def state_dict(
        self,
        geometry_hash: str,
        class_spec_hash: str,
        recovery_context: Mapping[str, object],
    ) -> Dict[str, object]:
        self._require_valid()
        if not self._is_sha256(geometry_hash):
            raise SemanticStateError(
                "geometry_hash must be a lowercase SHA-256 digest"
            )
        if not self._is_sha256(class_spec_hash):
            raise SemanticStateError(
                "class_spec_hash must be a lowercase SHA-256 digest"
            )
        context = self._validated_recovery_context(
            recovery_context,
            class_spec_hash=class_spec_hash,
            prior_total=self.prior_total,
            num_classes=self.num_classes,
        )
        self._validate_open_group_context(context.get("open_group"))
        context_sha256 = self.recovery_context_sha256(context)
        return {
            "schema_version": self.SCHEMA_VERSION,
            "semantic_method_version": SEMANTIC_METHOD_VERSION,
            "geometry_hash": geometry_hash,
            "class_spec_hash": class_spec_hash,
            "num_classes": self.num_classes,
            "prior_total": self.prior_total,
            "topology_version": self.topology_version,
            "next_gaussian_id": self.next_gaussian_id,
            "gaussian_ids_sha256": self.gaussian_ids_sha256,
            "recovery_context": context,
            "recovery_context_sha256": context_sha256,
            "alpha_raw": self.alpha_raw,
            "alpha_robust": self.alpha_robust,
            "group_variance": self.group_variance,
            "group_count": self.group_count,
            "semantic_support": self.semantic_support,
            "open_raw_evidence": self.open_raw_evidence,
            "open_robust_evidence": self.open_robust_evidence,
            "open_group_count": self.open_group_count,
            "open_distributions": self.open_distributions,
            "open_supports": self.open_supports,
            "open_semantic_supports": self.open_semantic_supports,
            "dirty": self.dirty,
            "active_projection_dirty": self.active_projection_dirty,
            "historical_evidence_stale": self.historical_evidence_stale,
            "group_disagreement_available": self.group_disagreement_available,
            "gaussian_ids": self.gaussian_ids,
        }

    def save(
        self,
        path: Path,
        geometry_hash: str,
        class_spec_hash: str,
        recovery_context: Mapping[str, object],
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        torch.save(
            self.state_dict(
                geometry_hash, class_spec_hash, recovery_context
            ),
            str(temporary),
        )
        os.replace(str(temporary), str(path))

    @classmethod
    def load(
        cls,
        path: Path,
        expected_geometry_hash: str,
        expected_class_spec_hash: str,
        expected_gaussian_count: int,
        expected_recovery_context_sha256: str,
        map_location: Optional[str] = "cpu",
    ) -> "SemanticGaussianState":
        try:
            value = torch.load(str(path), map_location=map_location)
        except (OSError, ValueError, RuntimeError) as error:
            raise SemanticStateError(
                "unable to load semantic state {}: {}".format(path, error)
            )
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != cls.SCHEMA_VERSION
            or value.get("semantic_method_version") != SEMANTIC_METHOD_VERSION
        ):
            raise SemanticStateError("unsupported semantic state schema")
        if value.get("geometry_hash") != expected_geometry_hash:
            raise SemanticStateError("semantic state geometry hash does not match")
        if value.get("class_spec_hash") != expected_class_spec_hash:
            raise SemanticStateError(
                "semantic state class spec hash does not match"
            )
        if not cls._is_sha256(expected_recovery_context_sha256):
            raise SemanticStateError(
                "expected recovery context hash must be a SHA-256 digest"
            )
        required = (
            "alpha_raw",
            "alpha_robust",
            "group_variance",
            "group_count",
            "semantic_support",
            "dirty",
            "active_projection_dirty",
            "historical_evidence_stale",
            "group_disagreement_available",
            "gaussian_ids",
            "prior_total",
            "topology_version",
            "next_gaussian_id",
            "recovery_context",
            "recovery_context_sha256",
            "gaussian_ids_sha256",
            "num_classes",
        )
        required += (
            "open_raw_evidence",
            "open_robust_evidence",
            "open_group_count",
            "open_distributions",
            "open_supports",
            "open_semantic_supports",
        )
        missing = [name for name in required if name not in value]
        if missing:
            raise SemanticStateError(
                "semantic state is missing fields: {}".format(missing)
            )
        try:
            context = cls._validated_recovery_context(
                value["recovery_context"],
                class_spec_hash=expected_class_spec_hash,
                prior_total=value["prior_total"],
                num_classes=value["num_classes"],
            )
            context_sha256 = cls.recovery_context_sha256(context)
            if value["recovery_context_sha256"] != context_sha256:
                raise SemanticStateError(
                    "semantic recovery context payload is corrupted"
                )
            if context_sha256 != expected_recovery_context_sha256:
                raise SemanticStateError(
                    "semantic recovery context does not match current inputs"
                )
            state = cls(
                alpha_raw=value["alpha_raw"],
                alpha_robust=value["alpha_robust"],
                group_variance=value["group_variance"],
                group_count=value["group_count"],
                semantic_support=value["semantic_support"],
                dirty=value["dirty"],
                gaussian_ids=value["gaussian_ids"],
                prior_total=value["prior_total"],
                topology_version=value["topology_version"],
                next_gaussian_id=value["next_gaussian_id"],
                recovery_context=context,
                open_raw_evidence=value.get("open_raw_evidence"),
                open_robust_evidence=value.get("open_robust_evidence"),
                open_group_count=value.get("open_group_count"),
                open_distributions=value.get("open_distributions"),
                open_supports=value.get("open_supports"),
                open_semantic_supports=value.get("open_semantic_supports"),
                active_projection_dirty=value["active_projection_dirty"],
                historical_evidence_stale=value["historical_evidence_stale"],
                group_disagreement_available=value[
                    "group_disagreement_available"
                ],
            )
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, SemanticStateError):
                raise
            raise SemanticStateError(
                "invalid semantic state payload: {}".format(error)
            )
        if state.num_gaussians != expected_gaussian_count:
            raise SemanticStateError(
                "semantic state Gaussian count does not match geometry"
            )
        if state.gaussian_ids_sha256 != value["gaussian_ids_sha256"]:
            raise SemanticStateError(
                "semantic state Gaussian ID payload is corrupted"
            )
        if state.num_classes != value["num_classes"]:
            raise SemanticStateError(
                "semantic state class count is inconsistent"
            )
        state._validate_open_group_context(context.get("open_group"))
        return state

    @classmethod
    def _validated_recovery_context(
        cls,
        value: object,
        class_spec_hash: str,
        prior_total: float,
        num_classes: int,
    ) -> Dict[str, object]:
        if not isinstance(value, Mapping):
            raise SemanticStateError(
                "recovery_context must be a mapping"
            )
        missing = [
            name for name in cls.RECOVERY_CONTEXT_KEYS if name not in value
        ]
        if missing:
            raise SemanticStateError(
                "recovery_context is missing fields: {}".format(missing)
            )
        ledger_value = value["ledger"]
        if (
            not isinstance(ledger_value, Mapping)
            or ledger_value.get("schema_version") != 1
            or not isinstance(ledger_value.get("observations"), list)
        ):
            raise SemanticStateError("invalid semantic ledger recovery payload")
        try:
            entries = tuple(
                LedgerEntry.from_dict(item)
                for item in ledger_value["observations"]
            )
            ledger = SemanticLedger(entries)
        except SemanticSchemaError as error:
            raise SemanticStateError(
                "invalid semantic ledger recovery payload: {}".format(error)
            )
        if len(entries) != len(ledger.entries):
            raise SemanticStateError(
                "semantic ledger recovery payload contains duplicate keys"
            )

        provenance = value["label_provenance"]
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("schema_version") != 1
            or not isinstance(provenance.get("versions"), Mapping)
        ):
            raise SemanticStateError("invalid label provenance recovery payload")
        versions = provenance["versions"]
        for label_version, item in versions.items():
            if not isinstance(label_version, str) or not label_version:
                raise SemanticStateError("label provenance version must be a string")
            if not isinstance(item, Mapping):
                raise SemanticStateError("label provenance entry must be a mapping")
            if not isinstance(item.get("model"), str) or not item["model"]:
                raise SemanticStateError("label provenance requires model")
            if not cls._is_sha256(item.get("checkpoint_sha256")):
                raise SemanticStateError(
                    "label provenance requires checkpoint_sha256"
                )
            if "prompt_config" not in item or not isinstance(
                item.get("thresholds"), Mapping
            ):
                raise SemanticStateError(
                    "label provenance requires prompt_config and thresholds"
                )
        unknown_versions = {
            entry.label_version for entry in entries
        } - set(versions)
        if unknown_versions:
            raise SemanticStateError(
                "ledger references unknown label versions: {}".format(
                    sorted(unknown_versions)
                )
            )
        if entries and not versions:
            raise SemanticStateError(
                "label provenance is required when ledger entries exist"
            )

        open_group = cls._validated_open_group(value["open_group"])

        current_geometry_stage = value["current_geometry_stage"]
        if (
            isinstance(current_geometry_stage, bool)
            or not isinstance(current_geometry_stage, int)
            or current_geometry_stage < 0
        ):
            raise SemanticStateError(
                "recovery current_geometry_stage must be a non-negative integer"
            )
        if (
            open_group is not None
            and open_group["projection_geometry_stage"]
            > current_geometry_stage
        ):
            raise SemanticStateError(
                "open-group projection stage exceeds recovery geometry stage"
            )

        try:
            class_spec = SemanticClassSpec.from_dict(value["class_spec"])
        except (SemanticSchemaError, AttributeError) as error:
            raise SemanticStateError(
                "invalid class spec recovery payload: {}".format(error)
            )
        if class_spec.fingerprint != class_spec_hash:
            raise SemanticStateError(
                "recovery class spec does not match class_spec_hash"
            )
        if class_spec.num_classes != num_classes:
            raise SemanticStateError(
                "recovery class spec does not match semantic state width"
            )

        config = value["semantic_config"]
        if not isinstance(config, Mapping) or config.get("schema_version") != 1:
            raise SemanticStateError("invalid semantic config recovery payload")
        config_prior = config.get("prior_total")
        if isinstance(config_prior, bool) or not isinstance(
            config_prior, (int, float)
        ):
            raise SemanticStateError("semantic config requires prior_total")
        if float(config_prior) != float(prior_total):
            raise SemanticStateError(
                "semantic config prior_total does not match state"
            )

        normalized = {
            "ledger": {
                "schema_version": 1,
                "observations": [entry.to_dict() for entry in ledger.entries],
            },
            "label_provenance": dict(provenance),
            "open_group": open_group,
            "current_geometry_stage": current_geometry_stage,
            "class_spec": class_spec.to_dict(),
            "semantic_config": dict(config),
        }
        try:
            return json.loads(
                json.dumps(
                    normalized,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError) as error:
            raise SemanticStateError(
                "recovery_context must be JSON-compatible: {}".format(error)
            )

    @classmethod
    def recovery_context_sha256(cls, value: Mapping[str, object]) -> str:
        try:
            canonical = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise SemanticStateError(
                "recovery_context must be JSON-compatible: {}".format(error)
            )
        return hashlib.sha256(canonical).hexdigest()

    @classmethod
    def validated_recovery_context_sha256(
        cls,
        value: Mapping[str, object],
        class_spec_hash: str,
        prior_total: float,
        num_classes: int,
    ) -> str:
        """Fingerprint externally loaded recovery inputs after validation."""
        normalized = cls._validated_recovery_context(
            value,
            class_spec_hash=class_spec_hash,
            prior_total=prior_total,
            num_classes=num_classes,
        )
        return cls.recovery_context_sha256(normalized)

    @staticmethod
    def _is_sha256(value: object) -> bool:
        if not isinstance(value, str) or len(value) != 64:
            return False
        return all(character in "0123456789abcdef" for character in value)

    def _evidence_tensor(
        self, value: torch.Tensor, name: str
    ) -> torch.Tensor:
        value = value.detach().to(device=self.device, dtype=self.dtype)
        if value.shape != self.alpha_raw.shape:
            raise SemanticStateError(
                "{} must have shape {}".format(name, tuple(self.alpha_raw.shape))
            )
        self._check_nonnegative_finite(name, value)
        return value

    def _optional_like_alpha(
        self, value: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if value is None:
            return torch.zeros_like(self.alpha_raw)
        return value.detach().to(device=self.device, dtype=self.dtype)

    def _optional_row_tensor(
        self, value: Optional[torch.Tensor], dtype: torch.dtype
    ) -> torch.Tensor:
        if value is None:
            return torch.zeros(
                self.num_gaussians, dtype=dtype, device=self.device
            )
        return value.detach().to(device=self.device, dtype=dtype)

    def _group_count_tensor(self, value: torch.Tensor) -> torch.Tensor:
        integer_types = (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        )
        if value.dtype not in integer_types:
            raise SemanticStateError(
                "group_count_increment must use an integer dtype"
            )
        result = value.detach().to(device=self.device, dtype=torch.long)
        if result.shape != (self.num_gaussians,):
            raise SemanticStateError(
                "group_count_increment must have shape [num_gaussians]"
            )
        if bool((result < 0).any()):
            raise SemanticStateError(
                "group_count_increment must be non-negative"
            )
        return result

    def _support_tensor(
        self, value: Optional[torch.Tensor], name: str
    ) -> Optional[torch.Tensor]:
        if value is None:
            return None
        result = value.detach().to(device=self.device, dtype=self.dtype)
        if result.shape != (self.num_gaussians,):
            raise SemanticStateError(
                "{} must have shape [num_gaussians]".format(name)
            )
        self._check_nonnegative_finite(name, result)
        return result

    def _open_observation_tensors(
        self, distributions: torch.Tensor, supports: torch.Tensor
    ):
        distributions = distributions.detach().to(
            device=self.device, dtype=self.dtype
        )
        supports = supports.detach().to(
            device=self.device, dtype=self.dtype
        )
        if distributions.ndim != 3 or distributions.shape[1:] != (
            self.num_gaussians,
            self.num_classes,
        ):
            raise SemanticStateError(
                "open distributions must have shape [views, gaussians, classes]"
            )
        if supports.shape != distributions.shape[:2]:
            raise SemanticStateError(
                "open supports must have shape [views, gaussians]"
            )
        self._check_nonnegative_finite("open_distributions", distributions)
        self._check_nonnegative_finite("open_supports", supports)
        if bool((supports > 1).any()):
            raise SemanticStateError("open supports must not exceed one")
        return distributions, supports

    @staticmethod
    def _validated_open_group(value: object) -> Optional[Dict[str, object]]:
        if value is None:
            return None
        if not isinstance(value, Mapping) or value.get("schema_version") != 1:
            raise SemanticStateError("invalid open_group recovery payload")
        anchor = value.get("anchor_image_id")
        members = value.get("member_image_ids")
        label_version = value.get("label_version")
        topology_version = value.get("projection_topology_version", 0)
        geometry_stage = value.get("projection_geometry_stage", 0)
        representatives = value.get("representative_image_ids")
        if not isinstance(anchor, str) or not anchor:
            raise SemanticStateError("open_group requires anchor_image_id")
        if (
            not isinstance(members, Sequence)
            or isinstance(members, (str, bytes))
            or not members
            or len(members) > 4
            or any(not isinstance(item, str) or not item for item in members)
            or len(set(members)) != len(members)
            or members[0] != anchor
        ):
            raise SemanticStateError(
                "open_group requires 1..4 unique ordered member_image_ids"
            )
        if (
            isinstance(geometry_stage, bool)
            or not isinstance(geometry_stage, int)
            or geometry_stage < 0
        ):
            raise SemanticStateError(
                "open_group projection_geometry_stage must be non-negative"
            )
        if not isinstance(label_version, str) or not label_version:
            raise SemanticStateError("open_group requires label_version")
        if (
            isinstance(topology_version, bool)
            or not isinstance(topology_version, int)
            or topology_version < 0
        ):
            raise SemanticStateError(
                "open_group projection_topology_version must be non-negative"
            )
        if representatives is None:
            representatives = [members[0]]
            if members[-1] != members[0]:
                representatives.append(members[-1])
        if (
            not isinstance(representatives, Sequence)
            or isinstance(representatives, (str, bytes))
            or not representatives
            or len(representatives) > 2
            or any(item not in members for item in representatives)
            or len(set(representatives)) != len(representatives)
        ):
            raise SemanticStateError(
                "open_group requires one or two representative image ids"
            )
        return {
            "schema_version": 1,
            "anchor_image_id": anchor,
            "member_image_ids": list(members),
            "representative_image_ids": list(representatives),
            "label_version": label_version,
            "projection_topology_version": topology_version,
            "projection_geometry_stage": geometry_stage,
        }

    def _validate_open_group_context(self, value: object) -> None:
        open_group = self._validated_open_group(value)
        view_count = self.open_distributions.shape[0]
        if open_group is None:
            if view_count != 0 or bool((self.open_group_count > 0).any()):
                raise SemanticStateError(
                    "open_group metadata is missing for provisional tensors"
                )
            return
        if len(open_group["representative_image_ids"]) != view_count:
            raise SemanticStateError(
                "open_group representative count does not match provisional tensors"
            )

    @staticmethod
    def _check_nonnegative_finite(name: str, value: torch.Tensor) -> None:
        if not bool(torch.isfinite(value).all()):
            raise SemanticStateError("{} must be finite".format(name))
        if bool((value < 0).any()):
            raise SemanticStateError("{} must be non-negative".format(name))

    def _validate(self) -> None:
        if self.alpha_raw.ndim != 2 or self.alpha_raw.shape[1] < 1:
            raise SemanticStateError("alpha_raw must have shape [N, K]")
        if not self.alpha_raw.is_floating_point():
            raise SemanticStateError("alpha_raw must use a floating dtype")
        if self.alpha_robust.shape != self.alpha_raw.shape:
            raise SemanticStateError("alpha_robust must match alpha_raw")
        if self.group_variance.shape != self.alpha_raw.shape:
            raise SemanticStateError("group_variance must match alpha_raw")
        if self.open_raw_evidence.shape != self.alpha_raw.shape:
            raise SemanticStateError("open_raw_evidence must match alpha_raw")
        if self.open_robust_evidence.shape != self.alpha_raw.shape:
            raise SemanticStateError("open_robust_evidence must match alpha_raw")
        if self.alpha_robust.dtype != self.alpha_raw.dtype:
            raise SemanticStateError("alpha_robust must match alpha_raw dtype")
        if self.group_variance.dtype != self.alpha_raw.dtype:
            raise SemanticStateError("group_variance must match alpha_raw dtype")
        row_shape = (self.alpha_raw.shape[0],)
        for name, value in (
            ("group_count", self.group_count),
            ("semantic_support", self.semantic_support),
            ("dirty", self.dirty),
            ("active_projection_dirty", self.active_projection_dirty),
            ("gaussian_ids", self.gaussian_ids),
            ("open_group_count", self.open_group_count),
        ):
            if value.shape != row_shape:
                raise SemanticStateError(
                    "{} must have shape [N]".format(name)
                )
            if value.device != self.alpha_raw.device:
                raise SemanticStateError(
                    "{} must share the alpha device".format(name)
                )
        for name, value in (
            ("alpha_robust", self.alpha_robust),
            ("group_variance", self.group_variance),
            ("semantic_support", self.semantic_support),
            ("open_raw_evidence", self.open_raw_evidence),
            ("open_robust_evidence", self.open_robust_evidence),
        ):
            if value.dtype != self.alpha_raw.dtype:
                raise SemanticStateError(
                    "{} must share the alpha dtype".format(name)
                )
        if self.group_count.dtype != torch.long:
            raise SemanticStateError("group_count must use torch.long")
        if self.open_group_count.dtype != torch.long:
            raise SemanticStateError("open_group_count must use torch.long")
        if self.semantic_support.dtype != self.alpha_raw.dtype:
            raise SemanticStateError(
                "semantic_support must match alpha_raw dtype"
            )
        if self.dirty.dtype != torch.bool:
            raise SemanticStateError("dirty must use torch.bool")
        if self.active_projection_dirty.dtype != torch.bool:
            raise SemanticStateError(
                "active_projection_dirty must use torch.bool"
            )
        if self.gaussian_ids.dtype != torch.long:
            raise SemanticStateError("gaussian_ids must use torch.long")
        if self.gaussian_ids.numel() != torch.unique(self.gaussian_ids).numel():
            raise SemanticStateError("gaussian_ids must be unique")
        if bool((self.gaussian_ids < 0).any()):
            raise SemanticStateError("gaussian_ids must be non-negative")
        if self.topology_version < 0:
            raise SemanticStateError("topology_version must be non-negative")
        minimum_next_id = (
            int(self.gaussian_ids.max().item()) + 1
            if self.gaussian_ids.numel()
            else 0
        )
        if self.next_gaussian_id < minimum_next_id:
            raise SemanticStateError(
                "next_gaussian_id must exceed all allocated Gaussian IDs"
            )
        if bool((self.group_count < 0).any()):
            raise SemanticStateError("group_count must be non-negative")
        if bool((self.open_group_count < 0).any()) or bool(
            (self.open_group_count > self.group_count).any()
        ):
            raise SemanticStateError(
                "open_group_count must be within the total group_count"
            )
        if self.prior_total <= 0:
            raise SemanticStateError("prior_total must be positive")
        self._check_nonnegative_finite("alpha_raw", self.alpha_raw)
        self._check_nonnegative_finite("alpha_robust", self.alpha_robust)
        self._check_nonnegative_finite("group_variance", self.group_variance)
        self._check_nonnegative_finite(
            "open_raw_evidence", self.open_raw_evidence
        )
        self._check_nonnegative_finite(
            "open_robust_evidence", self.open_robust_evidence
        )
        self._check_nonnegative_finite(
            "semantic_support", self.semantic_support
        )
        self._open_observation_tensors(
            self.open_distributions, self.open_supports
        )
        if self.open_semantic_supports.shape != self.open_supports.shape:
            raise SemanticStateError(
                "open_semantic_supports must match open_supports"
            )
        self._check_nonnegative_finite(
            "open_semantic_supports", self.open_semantic_supports
        )
        tolerance = torch.finfo(self.dtype).eps * 8
        prior = self.prior()
        if bool((self.alpha_raw - prior - self.open_raw_evidence < -tolerance).any()):
            raise SemanticStateError(
                "open raw evidence exceeds total data evidence"
            )
        if bool(
            (
                self.alpha_robust
                - prior
                - self.open_robust_evidence
                < -tolerance
            ).any()
        ):
            raise SemanticStateError(
                "open robust evidence exceeds total data evidence"
            )
