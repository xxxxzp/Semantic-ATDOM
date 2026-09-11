"""Semantic data contracts and reference algorithms for Semantic A-TDOM."""

from semantic_atdom.semantics.schema import (
    LedgerEntry,
    ObservationKey,
    SemanticClass,
    SemanticClassSpec,
    SemanticDataset,
    SemanticLedger,
    SemanticObservation,
    SemanticSchemaError,
    sha256_file,
)
from semantic_atdom.semantics.reference import (
    EvidenceBelief,
    GroupStats,
    ObservationStats,
    RobustPosterior,
    SemanticRender,
    aggregate_observation_group,
    evidence_conditioned_belief,
    exact_robust_posterior,
    footprint_observation,
    render_semantic_tdom,
)
from semantic_atdom.semantics.state import (
    SemanticGaussianState,
    SemanticStateError,
)
from semantic_atdom.semantics.topology import TopologyDelta
from semantic_atdom.semantics.provenance import (
    RGB_MODEL_ATTRIBUTE_NAMES,
    SEMANTIC_METHOD_VERSION,
    SEMANTIC_PROJECTION_ATTRIBUTE_NAMES,
    rgb_model_sha256,
    semantic_projection_sha256,
)
from semantic_atdom.semantics.orchestrator import (
    AcquisitionOverlap,
    AsyncProgressiveSemanticHook,
    CudaSemanticProjector,
    OnlineSemanticConfig,
    OnlineSemanticOrchestrator,
    ProgressiveSemanticHook,
    ProjectedObservation,
    SemanticSyncResult,
    label_boundary_weights,
)
from semantic_atdom.semantics.cuda_observation import (
    CudaObservationStats,
    rasterize_semantic_observation,
)
from semantic_atdom.semantics.tdom import (
    SemanticTdomConfig,
    SemanticTdomResult,
    finalize_semantic_tdom,
    rasterize_semantic_tdom,
    write_semantic_tdom,
)
from semantic_atdom.semantics.rebuild import (
    FinalRebuildMetrics,
    FinalRebuildResult,
    FinalSemanticRebuilder,
)

__all__ = [
    "LedgerEntry",
    "ObservationKey",
    "SemanticClass",
    "SemanticClassSpec",
    "SemanticDataset",
    "SemanticLedger",
    "SemanticObservation",
    "SemanticSchemaError",
    "SemanticGaussianState",
    "SemanticStateError",
    "TopologyDelta",
    "RGB_MODEL_ATTRIBUTE_NAMES",
    "SEMANTIC_METHOD_VERSION",
    "SEMANTIC_PROJECTION_ATTRIBUTE_NAMES",
    "rgb_model_sha256",
    "semantic_projection_sha256",
    "AcquisitionOverlap",
    "AsyncProgressiveSemanticHook",
    "CudaSemanticProjector",
    "OnlineSemanticConfig",
    "OnlineSemanticOrchestrator",
    "ProgressiveSemanticHook",
    "ProjectedObservation",
    "SemanticSyncResult",
    "label_boundary_weights",
    "CudaObservationStats",
    "EvidenceBelief",
    "GroupStats",
    "ObservationStats",
    "RobustPosterior",
    "SemanticRender",
    "aggregate_observation_group",
    "evidence_conditioned_belief",
    "exact_robust_posterior",
    "footprint_observation",
    "render_semantic_tdom",
    "rasterize_semantic_observation",
    "sha256_file",
    "SemanticTdomConfig",
    "SemanticTdomResult",
    "finalize_semantic_tdom",
    "rasterize_semantic_tdom",
    "write_semantic_tdom",
    "FinalRebuildMetrics",
    "FinalRebuildResult",
    "FinalSemanticRebuilder",
]
