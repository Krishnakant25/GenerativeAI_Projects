"""Tests for out-of-sample edge-stability (walk-forward replication).

Two layers of test, mirroring the rest of the suite's style:

1. A *statistical* test on constructed series where ground truth is known: a
   relationship engineered to hold across the WHOLE series must replicate; one
   engineered to exist only in the first half must NOT. This drives real
   Granger + FDR correction through ``compute_replication`` and confirms the
   replication verdict tracks reality (the analog of test_granger's manufactured
   textbook example).

2. A *pure-logic* test that hand-constructs candidate lists to pin down the
   bookkeeping exactly: same-direction requirement, missing-holdout handling,
   and the PC-graph vs Granger-only summary buckets — without depending on any
   statistical behaviour.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from causal.correction import correct_p_values
from causal.granger import run_pairwise_granger
from causal.lead_lag import lagged_correlation
from causal.models import CausalCandidate, Direction
from causal.replication import compute_replication, summarize_replication


# ---------------------------------------------------------------------------
# 1. Statistical test on constructed series
# ---------------------------------------------------------------------------


def _candidates_from_panel(
    panel: pl.DataFrame, run_id: str, *, alpha: float = 0.05, max_lag: int = 5
) -> list[CausalCandidate]:
    """Mirror pipeline steps 3-5 (Granger -> FDR -> build candidates) for a panel,
    without PC/HMM. Returns one CausalCandidate per ordered pair."""
    granger = run_pairwise_granger(panel, max_lag=max_lag)
    correction = correct_p_values(
        [g.p_value for g in granger], method="fdr_bh", alpha=alpha
    )
    series = {t: panel[t].to_numpy() for t in panel.columns if t != "date"}
    cands: list[CausalCandidate] = []
    for g, corrected_p, reject in zip(
        granger, correction.corrected_p_values, correction.reject
    ):
        corr = lagged_correlation(series[g.asset_a], series[g.asset_b], g.best_lag)
        cands.append(
            CausalCandidate(
                candidate_id=f"{run_id}:{g.asset_a}->{g.asset_b}",
                run_id=run_id,
                asset_a=g.asset_a,
                asset_b=g.asset_b,
                direction=Direction.A_CAUSES_B,
                lag=g.best_lag,
                granger_p_value=g.p_value,
                corrected_p_value=corrected_p,
                correlation_strength=corr,
                statistical_confidence=max(0.0, 1.0 - corrected_p),
                is_significant=bool(reject),
            )
        )
    return cands


def _build_split_panels(
    n: int = 1200, lag: int = 2, seed: int = 7
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Two windows (first/second half).

      X -> Y  holds across the WHOLE series   -> must replicate
      W -> Z  holds only in the FIRST half     -> must NOT replicate

    All driver series are white noise; only the engineered dependencies link
    them, so reverse and cross pairs stay insignificant.
    """
    rng = np.random.default_rng(seed)
    half = n // 2

    x = rng.standard_normal(n)
    w = rng.standard_normal(n)
    y = np.zeros(n)
    z = np.zeros(n)
    for t in range(lag, n):
        # Y always depends on X's past.
        y[t] = 0.8 * x[t - lag] + 0.3 * rng.standard_normal()
        if t < half:
            # Z depends on W's past only in the first half...
            z[t] = 0.8 * w[t - lag] + 0.3 * rng.standard_normal()
        else:
            # ...and is pure noise in the second half (relationship gone).
            z[t] = rng.standard_normal()

    def _panel(lo: int, hi: int) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "date": pl.Series("date", range(hi - lo)),
                "X": x[lo:hi],
                "Y": y[lo:hi],
                "W": w[lo:hi],
                "Z": z[lo:hi],
            }
        )

    return _panel(0, half), _panel(half, n)


def test_replication_check_catches_durable_and_transient_edges():
    discovery_panel, holdout_panel = _build_split_panels()
    disc = _candidates_from_panel(discovery_panel, "run_disc")
    hold = _candidates_from_panel(holdout_panel, "run_hold")

    # Mark X->Y as a PC edge in discovery so the PC bucket is exercised too.
    disc_graph_meta = {"run_disc:X->Y": ("directed", "pc")}
    results = compute_replication(disc, hold, disc_graph_meta, {})
    by_pair = {(r.asset_a, r.asset_b): r for r in results}

    xy = by_pair[("X", "Y")]
    wz = by_pair[("W", "Z")]

    # X->Y holds across the whole series: significant in both, replicates.
    assert xy.discovery_significant is True
    assert xy.holdout_significant is True
    assert xy.replicated is True

    # W->Z holds only in the first half: significant in discovery, gone in
    # holdout, so it must NOT replicate.
    assert wz.discovery_significant is True
    assert wz.holdout_significant is False
    assert wz.replicated is False

    # Summary: at least the two engineered edges are discovery-significant, and
    # exactly the durable one among them replicated.
    summary = summarize_replication(results, alpha=0.05)
    assert summary.n_discovery_significant >= 2
    assert summary.n_replicated >= 1
    # The PC bucket holds X->Y (replicated) -> 100% for that bucket.
    assert summary.n_pc_discovery == 1
    assert summary.pc_replication_rate == 1.0
    # W->Z is Granger-only and did not replicate.
    assert summary.n_granger_only_discovery >= 1
    assert wz.in_graph_discovery is False


# ---------------------------------------------------------------------------
# 2. Pure-logic bookkeeping
# ---------------------------------------------------------------------------


def _cand(
    run_id: str,
    a: str,
    b: str,
    *,
    corrected_p: float,
    significant: bool,
) -> CausalCandidate:
    return CausalCandidate(
        candidate_id=f"{run_id}:{a}->{b}",
        run_id=run_id,
        asset_a=a,
        asset_b=b,
        direction=Direction.A_CAUSES_B,
        lag=2,
        granger_p_value=corrected_p,
        corrected_p_value=corrected_p,
        correlation_strength=0.4,
        statistical_confidence=max(0.0, 1.0 - corrected_p),
        is_significant=significant,
    )


def test_same_direction_required_and_missing_holdout_handled():
    # Discovery: A->B significant, B->A significant, C->D significant.
    disc = [
        _cand("d", "A", "B", corrected_p=0.001, significant=True),
        _cand("d", "B", "A", corrected_p=0.002, significant=True),
        _cand("d", "C", "D", corrected_p=0.003, significant=True),
    ]
    # Holdout: only B->A significant (A->B flipped to non-significant); C->D is
    # entirely absent from the holdout candidate set (pair untested there).
    hold = [
        _cand("h", "A", "B", corrected_p=0.40, significant=False),
        _cand("h", "B", "A", corrected_p=0.001, significant=True),
    ]
    results = compute_replication(disc, hold, {}, {})
    by_pair = {(r.asset_a, r.asset_b): r for r in results}

    # A->B significant in discovery but not holdout -> not replicated. The fact
    # that the *reverse* B->A is significant in holdout must NOT rescue it.
    assert by_pair[("A", "B")].replicated is False
    assert by_pair[("A", "B")].holdout_significant is False

    # B->A significant in both, same direction -> replicated.
    assert by_pair[("B", "A")].replicated is True

    # C->D absent from holdout -> holdout stats null, not replicated, no crash.
    cd = by_pair[("C", "D")]
    assert cd.holdout_corrected_p is None
    assert cd.holdout_significant is False
    assert cd.replicated is False


def test_summary_buckets_pc_vs_granger_only():
    disc = [
        _cand("d", "A", "B", corrected_p=0.001, significant=True),  # PC, replicates
        _cand("d", "C", "D", corrected_p=0.002, significant=True),  # Granger-only, no
        _cand("d", "E", "F", corrected_p=0.60, significant=False),  # not significant
    ]
    hold = [
        _cand("h", "A", "B", corrected_p=0.001, significant=True),
        _cand("h", "C", "D", corrected_p=0.50, significant=False),
    ]
    disc_graph_meta = {"d:A->B": ("directed", "pc")}  # only A->B is a PC edge
    results = compute_replication(disc, hold, disc_graph_meta, {})
    summary = summarize_replication(results, alpha=0.05)

    # Only the two significant discovery edges count; the non-significant one is
    # excluded from every rate.
    assert summary.n_discovery_significant == 2
    assert summary.n_replicated == 1
    assert summary.replication_rate == 0.5
    assert summary.non_replication_rate == 0.5

    # PC bucket = {A->B}, replicated -> 100%. Granger-only = {C->D}, did not -> 0%.
    assert summary.n_pc_discovery == 1 and summary.pc_replication_rate == 1.0
    assert (
        summary.n_granger_only_discovery == 1
        and summary.granger_only_replication_rate == 0.0
    )


def test_empty_category_rate_is_none_not_zero():
    # All significant discovery edges are PC edges -> Granger-only bucket empty.
    disc = [_cand("d", "A", "B", corrected_p=0.001, significant=True)]
    hold = [_cand("h", "A", "B", corrected_p=0.001, significant=True)]
    results = compute_replication(disc, hold, {"d:A->B": ("directed", "pc")}, {})
    summary = summarize_replication(results, alpha=0.05)
    assert summary.n_granger_only_discovery == 0
    # Empty category must be None, never a misleading 0.0.
    assert summary.granger_only_replication_rate is None
    assert summary.pc_replication_rate == 1.0
