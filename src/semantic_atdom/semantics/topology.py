"""Explicit Gaussian topology changes consumed by semantic sidecar state."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class TopologyDelta:
    """Map every output Gaussian to zero or one source Gaussian.

    A source index of -1 creates a new prior-only semantic row. The legacy
    ``evidence_weights`` name now describes each descendant's rendered/support
    fraction; V1.1 semantic state may copy the source belief to preserve the
    rendered field. Stable source IDs are retained only where
    ``preserve_source_ids`` is true.
    """

    old_count: int
    source_indices: torch.Tensor
    evidence_weights: torch.Tensor
    preserve_source_ids: torch.Tensor
    kind: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_indices", self.source_indices.detach().clone()
        )
        object.__setattr__(
            self, "evidence_weights", self.evidence_weights.detach().clone()
        )
        object.__setattr__(
            self,
            "preserve_source_ids",
            self.preserve_source_ids.detach().clone(),
        )
        self.validate()

    def validate(self) -> None:
        """Recheck invariants in case a public tensor was mutated in place."""
        if self.old_count < 0:
            raise ValueError("old_count must be non-negative")
        if self.kind not in {"identity", "append", "prune", "clone", "split"}:
            raise ValueError("unsupported topology delta kind")
        if self.source_indices.ndim != 1:
            raise ValueError("source_indices must be one-dimensional")
        if self.source_indices.dtype != torch.long:
            raise ValueError("source_indices must use torch.long")
        if self.evidence_weights.shape != self.source_indices.shape:
            raise ValueError("evidence_weights must match source_indices")
        if not self.evidence_weights.is_floating_point():
            raise ValueError("evidence_weights must use a floating dtype")
        if self.preserve_source_ids.shape != self.source_indices.shape:
            raise ValueError("preserve_source_ids must match source_indices")
        if self.preserve_source_ids.dtype != torch.bool:
            raise ValueError("preserve_source_ids must use torch.bool")
        if (
            self.evidence_weights.device != self.source_indices.device
            or self.preserve_source_ids.device != self.source_indices.device
        ):
            raise ValueError("all topology tensors must use the same device")
        if not bool(torch.isfinite(self.evidence_weights).all()):
            raise ValueError("evidence_weights must be finite")
        if bool((self.evidence_weights < 0).any()):
            raise ValueError("evidence_weights must be non-negative")
        if bool((self.source_indices < -1).any()) or bool(
            (self.source_indices >= self.old_count).any()
        ):
            raise ValueError("source index is outside the old Gaussian state")
        new_rows = self.source_indices == -1
        if bool((self.evidence_weights[new_rows] != 0).any()):
            raise ValueError("new prior rows must have zero evidence weight")
        if bool(self.preserve_source_ids[new_rows].any()):
            raise ValueError("new prior rows cannot preserve a source id")

        preserved = self.source_indices[self.preserve_source_ids]
        if preserved.numel() != torch.unique(preserved).numel():
            raise ValueError("a source Gaussian id may be preserved at most once")
        valid_sources = self.source_indices >= 0
        totals = torch.zeros(
            self.old_count,
            dtype=self.evidence_weights.dtype,
            device=self.evidence_weights.device,
        )
        if bool(valid_sources.any()):
            totals.scatter_add_(
                0,
                self.source_indices[valid_sources],
                self.evidence_weights[valid_sources],
            )
            if bool((totals > 1.0 + 1e-6).any()):
                raise ValueError(
                    "a topology delta may not duplicate source contribution"
                )

        identity = torch.arange(
            self.old_count,
            dtype=torch.long,
            device=self.source_indices.device,
        )
        if self.kind == "identity" and not self.is_noop:
            raise ValueError("identity topology delta must be a no-op")
        if self.kind == "append":
            if self.new_count < self.old_count:
                raise ValueError("append topology cannot remove rows")
            retained = slice(0, self.old_count)
            appended = slice(self.old_count, self.new_count)
            if (
                not torch.equal(self.source_indices[retained], identity)
                or not bool(self.preserve_source_ids[retained].all())
                or not bool(
                    (torch.abs(self.evidence_weights[retained] - 1.0) <= 1e-7).all()
                )
                or bool((self.source_indices[appended] != -1).any())
            ):
                raise ValueError("append topology has an invalid row layout")
        if self.kind == "prune":
            if (
                bool((self.source_indices < 0).any())
                or not bool(self.preserve_source_ids.all())
                or not bool(
                    (torch.abs(self.evidence_weights - 1.0) <= 1e-7).all()
                )
                or (
                    self.new_count > 1
                    and not bool((torch.diff(self.source_indices) > 0).all())
                )
            ):
                raise ValueError("prune topology has an invalid row layout")
        if self.kind in {"clone", "split"} and not bool(
            torch.allclose(
                totals,
                torch.ones_like(totals),
                rtol=0.0,
                atol=1e-6,
            )
        ):
            raise ValueError(
                "clone/split topology must conserve source contribution fractions"
            )
        if self.kind == "clone" and (
            self.new_count < self.old_count
            or not torch.equal(self.source_indices[: self.old_count], identity)
            or not bool(self.preserve_source_ids[: self.old_count].all())
            or bool(self.preserve_source_ids[self.old_count :].any())
            or bool((self.source_indices < 0).any())
        ):
            raise ValueError("clone topology has an invalid row layout")
        if self.kind == "split":
            if bool((self.source_indices < 0).any()):
                raise ValueError("split topology cannot create prior-only rows")
            preserved_sources = torch.zeros(
                self.old_count,
                dtype=torch.bool,
                device=self.source_indices.device,
            )
            child_sources = torch.zeros_like(preserved_sources)
            if bool(self.preserve_source_ids.any()):
                preserved_sources[
                    self.source_indices[self.preserve_source_ids]
                ] = True
                if not bool(
                    (
                        torch.abs(
                            self.evidence_weights[self.preserve_source_ids] - 1.0
                        )
                        <= 1e-7
                    ).all()
                ):
                    raise ValueError(
                        "split retained rows must preserve full source evidence"
                    )
            children = ~self.preserve_source_ids
            if bool(children.any()):
                child_sources[self.source_indices[children]] = True
            if bool((preserved_sources & child_sources).any()):
                raise ValueError(
                    "split sources must be either retained rows or children"
                )

    @property
    def new_count(self) -> int:
        return self.source_indices.numel()

    @property
    def changed_rows(self) -> torch.Tensor:
        return (
            (self.source_indices < 0)
            | (~self.preserve_source_ids)
            | (torch.abs(self.evidence_weights - 1.0) > 1e-7)
        )

    @property
    def is_noop(self) -> bool:
        if self.new_count != self.old_count:
            return False
        identity = torch.arange(
            self.old_count,
            dtype=torch.long,
            device=self.source_indices.device,
        )
        return (
            torch.equal(self.source_indices, identity)
            and bool(self.preserve_source_ids.all())
            and bool((torch.abs(self.evidence_weights - 1.0) <= 1e-7).all())
        )

    @classmethod
    def identity(
        cls, count: int, device: Optional[torch.device] = None
    ) -> "TopologyDelta":
        if count < 0:
            raise ValueError("count must be non-negative")
        sources = torch.arange(count, dtype=torch.long, device=device)
        return cls(
            old_count=count,
            source_indices=sources,
            evidence_weights=torch.ones(count, device=device),
            preserve_source_ids=torch.ones(count, dtype=torch.bool, device=device),
            kind="identity",
        )

    @classmethod
    def prune(cls, keep_mask: torch.Tensor) -> "TopologyDelta":
        if keep_mask.dtype != torch.bool:
            raise ValueError("keep_mask must use torch.bool")
        keep_mask = keep_mask.detach()
        if keep_mask.ndim != 1:
            raise ValueError("keep_mask must be one-dimensional")
        sources = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
        return cls(
            old_count=keep_mask.numel(),
            source_indices=sources,
            evidence_weights=torch.ones(
                sources.numel(), device=keep_mask.device
            ),
            preserve_source_ids=torch.ones(
                sources.numel(), dtype=torch.bool, device=keep_mask.device
            ),
            kind="prune",
        )

    @classmethod
    def append_prior(
        cls,
        old_count: int,
        append_count: int,
        device: Optional[torch.device] = None,
    ) -> "TopologyDelta":
        if old_count < 0:
            raise ValueError("old_count must be non-negative")
        if append_count < 0:
            raise ValueError("append_count must be non-negative")
        sources = torch.cat(
            (
                torch.arange(old_count, dtype=torch.long, device=device),
                torch.full(
                    (append_count,), -1, dtype=torch.long, device=device
                ),
            )
        )
        return cls(
            old_count=old_count,
            source_indices=sources,
            evidence_weights=torch.cat(
                (
                    torch.ones(old_count, device=device),
                    torch.zeros(append_count, device=device),
                )
            ),
            preserve_source_ids=torch.cat(
                (
                    torch.ones(old_count, dtype=torch.bool, device=device),
                    torch.zeros(append_count, dtype=torch.bool, device=device),
                )
            ),
            kind="append",
        )

    @classmethod
    def clone(
        cls,
        old_count: int,
        parent_indices: torch.Tensor,
        child_fractions: Optional[torch.Tensor] = None,
    ) -> "TopologyDelta":
        cls._validate_integer_indices(parent_indices)
        parents = parent_indices.detach().to(dtype=torch.long)
        cls._validate_parents(old_count, parents)
        if child_fractions is None:
            fractions = torch.full(
                (parents.numel(),), 0.5, device=parents.device
            )
        else:
            fractions = child_fractions.detach().to(
                device=parents.device, dtype=torch.float32
            )
        if fractions.shape != parents.shape:
            raise ValueError("child_fractions must match parent_indices")
        if bool((fractions < 0).any()) or bool((fractions > 1).any()):
            raise ValueError("child_fractions must be in [0, 1]")

        original_weights = torch.ones(old_count, device=parents.device)
        original_weights[parents] -= fractions
        return cls(
            old_count=old_count,
            source_indices=torch.cat(
                (torch.arange(old_count, device=parents.device), parents)
            ),
            evidence_weights=torch.cat((original_weights, fractions)),
            preserve_source_ids=torch.cat(
                (
                    torch.ones(old_count, dtype=torch.bool, device=parents.device),
                    torch.zeros(
                        parents.numel(), dtype=torch.bool, device=parents.device
                    ),
                )
            ),
            kind="clone",
        )

    @classmethod
    def split(
        cls,
        old_count: int,
        parent_indices: torch.Tensor,
        children_per_parent: int = 2,
        child_weights: Optional[torch.Tensor] = None,
    ) -> "TopologyDelta":
        cls._validate_integer_indices(parent_indices)
        parents = parent_indices.detach().to(dtype=torch.long)
        cls._validate_parents(old_count, parents)
        if children_per_parent < 1:
            raise ValueError("children_per_parent must be positive")
        keep_mask = torch.ones(old_count, dtype=torch.bool, device=parents.device)
        keep_mask[parents] = False
        retained = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
        child_sources = parents.repeat(children_per_parent)

        if child_weights is None:
            weights = torch.full(
                (child_sources.numel(),),
                1.0 / children_per_parent,
                device=parents.device,
            )
        else:
            weights = child_weights.detach().to(
                device=parents.device, dtype=torch.float32
            ).reshape(-1)
        if weights.shape != child_sources.shape:
            raise ValueError(
                "child_weights must contain one value per generated child"
            )
        if bool((weights < 0).any()):
            raise ValueError("child_weights must be non-negative")
        if parents.numel():
            totals = weights.reshape(
                children_per_parent, parents.numel()
            ).sum(dim=0)
            if not bool(torch.allclose(totals, torch.ones_like(totals))):
                raise ValueError(
                    "split child weights must sum to one for every parent"
                )

        return cls(
            old_count=old_count,
            source_indices=torch.cat((retained, child_sources)),
            evidence_weights=torch.cat(
                (torch.ones(retained.numel(), device=parents.device), weights)
            ),
            preserve_source_ids=torch.cat(
                (
                    torch.ones(
                        retained.numel(), dtype=torch.bool, device=parents.device
                    ),
                    torch.zeros(
                        child_sources.numel(),
                        dtype=torch.bool,
                        device=parents.device,
                    ),
                )
            ),
            kind="split",
        )

    @staticmethod
    def _validate_integer_indices(indices: torch.Tensor) -> None:
        integer_types = (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        )
        if indices.dtype not in integer_types:
            raise ValueError("parent_indices must use an integer dtype")

    @staticmethod
    def _validate_parents(old_count: int, parents: torch.Tensor) -> None:
        if parents.ndim != 1:
            raise ValueError("parent_indices must be one-dimensional")
        if bool((parents < 0).any()) or bool((parents >= old_count).any()):
            raise ValueError("parent index is outside the old Gaussian state")
        if parents.numel() != torch.unique(parents).numel():
            raise ValueError("parent_indices must be unique")
