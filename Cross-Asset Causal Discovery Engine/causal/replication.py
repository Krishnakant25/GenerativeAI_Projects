"""Out-of-sample edge-stability (walk-forward replication) logic.

The full Layer-1 pipeline is run independently on a *discovery* window and a
later, untouched *holdout* window (see ``scripts/run_replication_study.py``).
This module holds the pure, side-effect-free comparison of the two runs, so it
can be unit-tested on synthetic data without touching yfinance or the DB:

    compute_replication(...)   -> list[ReplicationResult]   (one per ordered pair)
    summarize_replication(...) -> ReplicationSummary        (aggregate rates)

Replication rule (deliberately strict): a discovery-significant pair *replicates*
iff the SAME ordered pair (asset_a -> asset_b) is also significant-after-correction
in the holdout run. Because Layer 1 emits one candidate per *ordered* pair, an
ordered-pair match already enforces same-direction — a flipped direction is a
different row and never counts as replication.

Nothing here trusts a number it cannot show: each ``ReplicationResult`` carries
both periods' corrected p-values, so a "replicated" verdict can always be audited
against the evidence behind it.
"""

from __future__ import annotations

from causal.models import (
    CausalCandidate,
    ReplicationResult,
    ReplicationSummary,
)


def _index_by_pair(
    candidates: list[CausalCandidate],
) -> dict[tuple[str, str], CausalCandidate]:
    """Map (asset_a, asset_b) -> candidate. Layer 1 emits one candidate per
    ordered pair, so this is 1:1."""
    return {(c.asset_a, c.asset_b): c for c in candidates}


def _graph_pairs(
    candidates: list[CausalCandidate],
    graph_meta: dict[str, tuple[str, str]],
) -> set[tuple[str, str]]:
    """The set of ordered pairs PC kept as graph edges (in_graph=True)."""
    by_id = {c.candidate_id: c for c in candidates}
    pairs: set[tuple[str, str]] = set()
    for cand_id in graph_meta:
        c = by_id.get(cand_id)
        if c is not None:
            pairs.add((c.asset_a, c.asset_b))
    return pairs


def compute_replication(
    discovery_candidates: list[CausalCandidate],
    holdout_candidates: list[CausalCandidate],
    discovery_graph_meta: dict[str, tuple[str, str]],
    holdout_graph_meta: dict[str, tuple[str, str]],
) -> list[ReplicationResult]:
    """Build one ``ReplicationResult`` per discovery candidate (every ordered
    pair tested in discovery), pairing it with the same pair's holdout evidence.

    Emitting a row for *every* discovery candidate — not just the significant
    ones — keeps the persisted artifact a complete picture; the summary then
    restricts the rates to the discovery-significant subset.
    """
    holdout_by_pair = _index_by_pair(holdout_candidates)
    discovery_graph = _graph_pairs(discovery_candidates, discovery_graph_meta)
    holdout_graph = _graph_pairs(holdout_candidates, holdout_graph_meta)

    results: list[ReplicationResult] = []
    for d in discovery_candidates:
        pair = (d.asset_a, d.asset_b)
        h = holdout_by_pair.get(pair)

        holdout_significant = bool(h.is_significant) if h is not None else False
        # Replication requires significance in BOTH windows for the same ordered
        # pair (=> same direction).
        replicated = bool(d.is_significant and holdout_significant)

        results.append(
            ReplicationResult(
                asset_a=d.asset_a,
                asset_b=d.asset_b,
                direction=d.direction,
                lag_discovery=d.lag,
                discovery_granger_p=d.granger_p_value,
                discovery_corrected_p=d.corrected_p_value,
                discovery_correlation=d.correlation_strength,
                discovery_significant=bool(d.is_significant),
                in_graph_discovery=pair in discovery_graph,
                lag_holdout=h.lag if h is not None else None,
                holdout_corrected_p=h.corrected_p_value if h is not None else None,
                holdout_correlation=h.correlation_strength if h is not None else None,
                holdout_significant=holdout_significant,
                in_graph_holdout=pair in holdout_graph,
                replicated=replicated,
            )
        )
    return results


def summarize_replication(
    results: list[ReplicationResult],
    *,
    alpha: float,
) -> ReplicationSummary:
    """Aggregate replication rates over the **discovery-significant** subset.

    Splits that subset into PC-graph edges (``in_graph_discovery``) vs
    Granger-only edges to test whether PC-kept edges replicate more reliably.
    Per-category rates are ``None`` when that category is empty (so an empty
    category is never silently reported as 0% replication).
    """
    sig = [r for r in results if r.discovery_significant]
    n_sig = len(sig)
    n_repl = sum(1 for r in sig if r.replicated)

    pc = [r for r in sig if r.in_graph_discovery]
    granger_only = [r for r in sig if not r.in_graph_discovery]
    n_pc_repl = sum(1 for r in pc if r.replicated)
    n_go_repl = sum(1 for r in granger_only if r.replicated)

    replication_rate = (n_repl / n_sig) if n_sig else 0.0

    return ReplicationSummary(
        alpha=alpha,
        n_discovery_significant=n_sig,
        n_replicated=n_repl,
        replication_rate=replication_rate,
        non_replication_rate=1.0 - replication_rate if n_sig else 0.0,
        n_pc_discovery=len(pc),
        n_pc_replicated=n_pc_repl,
        pc_replication_rate=(n_pc_repl / len(pc)) if pc else None,
        n_granger_only_discovery=len(granger_only),
        n_granger_only_replicated=n_go_repl,
        granger_only_replication_rate=(n_go_repl / len(granger_only))
        if granger_only
        else None,
    )
