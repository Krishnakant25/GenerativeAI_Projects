"""Pydantic data models for the statistical layer (Layer 1).

The central artifact is `CausalCandidate`: one directional, candidate causal
relationship (asset_a "Granger-causes" asset_b at some lag) together with the
full statistical evidence behind it. Per the project's hard rule, a candidate
can never carry a causal direction without the corrected p-value that
justifies it — so `corrected_p_value` is required, not optional.

Naming is deliberately hedged ("candidate", "statistical_confidence") to keep
the honesty framing in the type system itself: nothing here asserts proven
causation.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, computed_field, model_validator

from config import (
    DEFAULT_ALPHA,
    DEFAULT_CORRECTION,
    DEFAULT_END_DATE,
    DEFAULT_MAX_LAG,
    DEFAULT_START_DATE,
)


class Direction(str, Enum):
    """Direction of the candidate relationship. Layer 1 only ever emits
    A_CAUSES_B for an ordered (asset_a, asset_b) pair; the reverse direction is
    represented as a separate candidate with the assets swapped."""

    A_CAUSES_B = "a_causes_b"


class RegimePeriod(BaseModel):
    """A time-bounded window during which the relationship was (in)active.

    Regime results MUST be time-bound — a relationship that held in 2019-2021
    is not a permanent claim. This model exists so that validity is always
    attached to an explicit date range, never asserted globally.
    """

    start: date
    end: date
    active: bool = Field(
        ..., description="Whether the relationship was active in this window."
    )
    mean_correlation: float | None = Field(
        default=None,
        description="Mean lead-lag correlation within the window, if computed.",
    )

    @model_validator(mode="after")
    def _check_order(self) -> "RegimePeriod":
        if self.end < self.start:
            raise ValueError("RegimePeriod.end must be >= start")
        return self


FlipStatus = Literal["pending", "confirmed", "reverted"]


class RegimeFlipEvent(BaseModel):
    """A detected change in a pair's *current* regime status between two runs.

    Phase 3 (scheduled monitoring): when a re-run finds that a previously-
    significant relationship's coupling switched on or off (its most recent
    regime window flipped ``active``), that is a candidate "regime-change early
    warning" event. Per the project hard rule it carries the statistic that
    justifies it — there is no flip without its corrected p-value.

    Trust is earned, not assumed. A flip starts ``pending`` and is only
    ``confirmed`` once its new status has persisted across
    ``MONITOR_CONFIRMATION_RUNS`` consecutive monitor runs, guarding against
    yfinance data-revision artifacts on the most recent bars. A flip that snaps
    back before then is marked ``reverted`` and never presented as a finding.

    This is the SAME discipline as the Layer-1 stationarity stress-test: a lone
    extreme p-value is not trusted until it survives ADF, and a lone regime flip
    is not trusted until it survives the confirmation window. ``pending`` flips
    are *provisional* (they may simply not have been observed enough times yet —
    read ``consecutive_confirmations`` / ``MONITOR_CONFIRMATION_RUNS`` to see how
    far along); only ``confirmed`` flips are the real signal. The three states
    are mutually exclusive and exhaustive, so "genuinely confirmed" is never
    conflated with "hasn't reverted *yet*". A reverted flip that later re-flips
    in the same direction is recorded as a fresh event, not a resurrection of the
    old one (the dedup gate in ``record_and_update_flips`` only suppresses still-
    open pending/confirmed duplicates).
    """

    asset_a: str = Field(..., description="Driver ticker of the pair.")
    asset_b: str = Field(..., description="Affected ticker of the pair.")

    prior_run_id: str = Field(..., description="Run the pair flipped FROM.")
    new_run_id: str = Field(
        ..., description="Run the flip was first detected in (carries the stat)."
    )

    old_active: bool = Field(..., description="Coupling status in the prior run.")
    new_active: bool = Field(..., description="Coupling status in the new run.")
    old_mean_correlation: float | None = Field(
        default=None, description="Mean corr of the prior run's current window."
    )
    new_mean_correlation: float | None = Field(
        default=None, description="Mean corr of the new run's current window."
    )

    # Hard rule: a flip never travels without the statistic behind it. These come
    # from the NEW run's candidate (the current evidence for the relationship).
    corrected_p_value: float = Field(
        ..., ge=0.0, le=1.0, description="Corrected Granger p of the new-run candidate."
    )
    lag: int = Field(..., ge=1, description="Lag (days) of the new-run candidate.")
    correlation_strength: float | None = Field(
        default=None, ge=-1.0, le=1.0, description="Peak lead-lag corr (new run)."
    )

    status: FlipStatus = Field(
        default="pending",
        description="'pending' until it survives the confirmation window, then "
        "'confirmed'; 'reverted' if the status snapped back first.",
    )
    consecutive_confirmations: int = Field(
        default=1,
        ge=1,
        description="Number of consecutive monitor runs the new status has held "
        "(including the detection run).",
    )
    detected_at: str = Field(..., description="ISO-8601 UTC time of first detection.")
    last_seen_run_id: str | None = Field(
        default=None, description="Most recent run in which the new status still held."
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def confirmed(self) -> bool:
        """True only once the flip has survived the confirmation window. The
        single boolean the UI/consumers should gate a 'real signal' on."""
        return self.status == "confirmed"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def direction(self) -> str:
        """Human-readable flip direction."""
        if self.new_active and not self.old_active:
            return "decoupled -> coupled (activated)"
        if self.old_active and not self.new_active:
            return "coupled -> decoupled (deactivated)"
        return "unchanged"  # should not occur for a stored flip


class CausalCandidate(BaseModel):
    """One directional candidate causal relationship with its full evidence.

    A candidate hypothesis for human review — NOT a proven causal claim. See
    the README's "Critical Framing" section.
    """

    candidate_id: str = Field(
        ..., description="Stable ID, typically f'{run_id}:{asset_a}->{asset_b}'."
    )
    run_id: str = Field(..., description="ID of the analysis run that produced it.")

    asset_a: str = Field(..., description="Driver ticker (its past precedes B).")
    asset_b: str = Field(..., description="Affected ticker (it follows A).")
    direction: Direction = Field(default=Direction.A_CAUSES_B)

    lag: int = Field(
        ..., ge=1, description="Lag in trading days at which the effect is strongest."
    )

    granger_p_value: float = Field(
        ..., ge=0.0, le=1.0, description="Raw Granger-causality p-value."
    )
    corrected_p_value: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "p-value after multiple-comparisons correction (FDR/Bonferroni). "
            "Required — no causal arrow is reported without it."
        ),
    )
    correlation_strength: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Peak time-lagged cross-correlation at `lag`.",
    )

    regime_periods: list[RegimePeriod] = Field(
        default_factory=list,
        description="Time-bound windows of (in)activity from HMM regime detection.",
    )

    statistical_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Derived 0-1 confidence from the statistical evidence (e.g. "
            "1 - corrected_p_value, optionally scaled by |correlation|). "
            "Distinct from any LLM confidence added in Phase 2."
        ),
    )
    is_significant: bool = Field(
        ...,
        description="True if corrected_p_value passes the run's alpha threshold.",
    )

    model_config = {"use_enum_values": True}


class AnalysisRun(BaseModel):
    """Metadata describing one end-to-end run of the engine."""

    run_id: str
    created_at: str = Field(..., description="ISO-8601 UTC timestamp.")
    start_date: str
    end_date: str
    asset_universe: list[str]
    max_lag: int = Field(..., ge=1)
    correction_method: str = Field(..., description="'fdr_bh' | 'bonferroni'")
    alpha: float = Field(..., gt=0.0, lt=1.0)
    notes: str | None = None


# --- API request / response schemas (Layer 1 only) -------------------------


class AnalyzeRequest(BaseModel):
    """Body for ``POST /analyze``. Every field defaults to the centralized
    ``config`` value, so an empty ``{}`` body runs the standard pipeline over
    the full asset universe."""

    start_date: str = Field(default=DEFAULT_START_DATE)
    end_date: str | None = Field(default=DEFAULT_END_DATE)
    max_lag: int = Field(default=DEFAULT_MAX_LAG, ge=1, le=30)
    alpha: float = Field(default=DEFAULT_ALPHA, gt=0.0, lt=1.0)
    correction_method: str = Field(default=DEFAULT_CORRECTION)
    notes: str | None = None

    @model_validator(mode="after")
    def _check_method(self) -> "AnalyzeRequest":
        if self.correction_method not in ("fdr_bh", "bonferroni"):
            raise ValueError("correction_method must be 'fdr_bh' or 'bonferroni'")
        return self


class GraphEdge(BaseModel):
    """One serialized edge of the discovered causal graph.

    Honesty rule made structural: every edge carries the corrected p-value (and
    lag / correlation) that justifies it — there is no field for a bare arrow.
    """

    source: str = Field(..., description="Driver ticker (edge tail).")
    target: str = Field(..., description="Affected ticker (edge head).")
    edge_type: str = Field(
        ..., description="'directed' | 'undirected' | 'bidirected' (from the CPDAG)."
    )
    orientation_source: str = Field(
        ..., description="How the arrow was oriented: 'pc' | 'granger' | 'none'."
    )
    corrected_p_value: float = Field(
        ..., ge=0.0, le=1.0, description="Corrected Granger p-value for this edge."
    )
    lag: int | None = Field(default=None, description="Lag in trading days.")
    correlation_strength: float | None = Field(
        default=None, ge=-1.0, le=1.0, description="Peak lead-lag correlation."
    )


class CausalGraph(BaseModel):
    """Node-link serialization of the discovered NetworkX ``DiGraph``."""

    run_id: str
    nodes: list[str] = Field(..., description="All assets in the run's universe.")
    edges: list[GraphEdge] = Field(default_factory=list)


class PairRegimes(BaseModel):
    """Time-bound regime windows for a single directional pair."""

    asset_a: str
    asset_b: str
    regime_periods: list[RegimePeriod] = Field(default_factory=list)


# --- Phase 3 (Option 5): walk-forward / out-of-sample edge-stability ---------


class ReplicationResult(BaseModel):
    """One directional pair's out-of-sample stability check.

    The full Layer-1 pipeline is run independently on a *discovery* window and a
    later, untouched *holdout* window. For every pair this records the discovery
    evidence, the holdout evidence (the same pair re-tested on new data), and
    whether the relationship **replicated** — i.e. was significant after
    correction in BOTH windows in the SAME direction.

    Same-direction is structural: Layer 1 emits one candidate per *ordered*
    (asset_a, asset_b) pair, so matching on the ordered pair already requires the
    arrow to point the same way. A pair that is significant A->B in discovery but
    only B->A in holdout is NOT counted as a replication — it appears as two
    separate ``ReplicationResult`` rows, neither of which replicates.

    This is the statistical analog of the Layer-2 spurious-rationalization
    control: it asks, honestly, whether "156 candidates, ~105 significant" is a
    durable finding or partly in-sample overfitting across many simultaneous
    tests. A relationship that holds out-of-sample is the strongest evidence the
    project can produce; one that does not is reported with the same candour.
    """

    asset_a: str = Field(..., description="Driver ticker (its past precedes B).")
    asset_b: str = Field(..., description="Affected ticker (it follows A).")
    direction: Direction = Field(default=Direction.A_CAUSES_B)

    # --- Discovery-period evidence (always present) ---
    lag_discovery: int = Field(..., ge=1, description="Best lag in the discovery window.")
    discovery_granger_p: float = Field(..., ge=0.0, le=1.0)
    discovery_corrected_p: float = Field(
        ..., ge=0.0, le=1.0, description="FDR/Bonferroni-corrected p in discovery."
    )
    discovery_correlation: float | None = Field(default=None, ge=-1.0, le=1.0)
    discovery_significant: bool = Field(
        ..., description="Passed the corrected alpha threshold in discovery."
    )
    in_graph_discovery: bool = Field(
        ..., description="PC kept this pair as a graph edge in the discovery run."
    )

    # --- Holdout-period evidence (null only if the pair was untested in holdout,
    #     e.g. a ticker dropped out of that window; with the fixed 13-asset
    #     universe both windows test all ordered pairs, so these are populated). ---
    lag_holdout: int | None = Field(default=None, ge=1)
    holdout_corrected_p: float | None = Field(default=None, ge=0.0, le=1.0)
    holdout_correlation: float | None = Field(default=None, ge=-1.0, le=1.0)
    holdout_significant: bool = Field(
        default=False, description="Passed the corrected alpha threshold in holdout."
    )
    in_graph_holdout: bool = Field(
        default=False, description="PC kept this pair as a graph edge in the holdout run."
    )

    replicated: bool = Field(
        ...,
        description="True iff significant in BOTH discovery and holdout (same "
        "ordered pair => same direction). The single bool the study turns on.",
    )


class ReplicationSummary(BaseModel):
    """Aggregate out-of-sample replication statistics over a discovery edge set.

    Rates are computed over the **discovery-significant** subset only — the
    candidates that the discovery run actually surfaced as findings. The PC-graph
    vs Granger-only breakdown tests a real, falsifiable hypothesis from the Phase-1
    design: PC-kept edges (which survived conditional-independence pruning of
    confounded / indirect links) should replicate at a higher rate than edges that
    were Granger-significant but PC dropped.
    """

    alpha: float = Field(..., gt=0.0, lt=1.0)

    # Overall (over discovery-significant pairs)
    n_discovery_significant: int = Field(..., ge=0)
    n_replicated: int = Field(..., ge=0)
    replication_rate: float = Field(..., ge=0.0, le=1.0)
    non_replication_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="1 - replication_rate. Empirical out-of-sample non-replication "
        "indicator, to be compared against nominal alpha.",
    )

    # PC-graph edges (in_graph_discovery=True AND discovery_significant)
    n_pc_discovery: int = Field(..., ge=0)
    n_pc_replicated: int = Field(..., ge=0)
    pc_replication_rate: float | None = Field(
        default=None, description="None if there were no significant PC edges."
    )

    # Granger-only edges (discovery_significant AND NOT in_graph_discovery)
    n_granger_only_discovery: int = Field(..., ge=0)
    n_granger_only_replicated: int = Field(..., ge=0)
    granger_only_replication_rate: float | None = Field(
        default=None, description="None if there were no Granger-only significant edges."
    )
