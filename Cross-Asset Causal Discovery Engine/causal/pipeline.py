"""End-to-end orchestration of the Layer-1 statistical pipeline.

This is the single place that wires the stages together:

    fetch  ->  preprocess  ->  Granger  ->  multiple-comparisons correction
           ->  lead-lag correlation  ->  PC graph discovery  ->  HMM regimes

It is deliberately separate from the HTTP layer (``api/main.py``) so the
pipeline can be unit-tested and reused without FastAPI, and so the API stays a
thin persistence/serialization shell.

Honesty rules enforced here, not just hoped for:
  * Every candidate carries its corrected p-value (correction runs across *all*
    pairs before anything is labelled significant).
  * ``statistical_confidence`` is a transparent function of the corrected
    p-value (and correlation), documented at ``_statistical_confidence``.
  * Regime detection is attached only to significant candidates, and the
    windows are always time-bound (``RegimePeriod``), never global claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

import polars as pl

from causal.correction import correct_p_values
from causal.granger import run_pairwise_granger
from causal.graph_discovery import discover_causal_graph
from causal.lead_lag import lagged_correlation
from causal.models import AnalysisRun, CausalCandidate, Direction
from causal.regime_detection import detect_regimes
from config import (
    DEFAULT_ALPHA,
    DEFAULT_CORRECTION,
    DEFAULT_END_DATE,
    DEFAULT_INDEP_TEST,
    DEFAULT_MAX_LAG,
    DEFAULT_START_DATE,
    MIN_OBSERVATIONS,
)
from data.fetcher import fetch_panel
from data.preprocessor import preprocess


class InsufficientDataError(ValueError):
    """Raised when the aligned panel has too few observations for the engine to
    produce a trustworthy result (so the API can answer 422, not 500)."""


@dataclass(frozen=True)
class AnalysisResult:
    """Everything one run produced, ready to persist or serialize."""

    run: AnalysisRun
    candidates: list[CausalCandidate]
    # candidate_id -> (edge_type, orientation_source) for PC graph members.
    graph_meta: dict[str, tuple[str, str]]
    missing_tickers: list[str]


def _statistical_confidence(corrected_p: float, correlation: float | None) -> float:
    """Transparent 0-1 confidence derived from the statistical evidence.

    Base is ``1 - corrected_p`` (so a barely-significant edge scores low). When
    a lead-lag correlation is available we scale by ``0.5 + 0.5*|corr|`` so a
    significant edge with a stronger effect size ranks above an equally
    significant but weaker one. This is a ranking heuristic, NOT a probability
    of causation — that framing is documented in the README.
    """
    base = max(0.0, min(1.0, 1.0 - corrected_p))
    if correlation is None:
        return base
    return max(0.0, min(1.0, base * (0.5 + 0.5 * abs(correlation))))


def run_analysis(
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = DEFAULT_END_DATE,
    max_lag: int = DEFAULT_MAX_LAG,
    alpha: float = DEFAULT_ALPHA,
    correction_method: str = DEFAULT_CORRECTION,
    indep_test: str = DEFAULT_INDEP_TEST,
    tickers: list[str] | None = None,
    notes: str | None = None,
) -> AnalysisResult:
    """Run the full Layer-1 pipeline and return a persistable result.

    Raises:
        data.fetcher.DataUnavailableError: the data provider returned nothing.
        InsufficientDataError: the aligned panel is too short to analyse.
    """
    run_id = f"run_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"

    # 1. Fetch + 2. preprocess -------------------------------------------------
    fetched = fetch_panel(tickers, start=start_date, end=end_date)
    _, returns = preprocess(fetched.prices)

    if returns.height < MIN_OBSERVATIONS:
        raise InsufficientDataError(
            f"Only {returns.height} aligned return rows after preprocessing "
            f"(need >= {MIN_OBSERVATIONS}). Widen the date range or check that "
            f"the tickers share enough overlapping history."
        )

    analysed = [c for c in returns.columns if c != "date"]
    series = {t: returns[t].to_numpy() for t in analysed}

    # 3. Pairwise Granger ------------------------------------------------------
    granger = run_pairwise_granger(returns, max_lag=max_lag)

    # 4. Multiple-comparisons correction across ALL pairs ----------------------
    correction = correct_p_values(
        [g.p_value for g in granger], method=correction_method, alpha=alpha
    )

    # 5. Build candidates with corrected p + lead-lag effect size --------------
    candidates: list[CausalCandidate] = []
    by_pair: dict[tuple[str, str], CausalCandidate] = {}
    for g, corrected_p, reject in zip(
        granger, correction.corrected_p_values, correction.reject
    ):
        corr = lagged_correlation(series[g.asset_a], series[g.asset_b], g.best_lag)
        cand = CausalCandidate(
            candidate_id=f"{run_id}:{g.asset_a}->{g.asset_b}",
            run_id=run_id,
            asset_a=g.asset_a,
            asset_b=g.asset_b,
            direction=Direction.A_CAUSES_B,
            lag=g.best_lag,
            granger_p_value=g.p_value,
            corrected_p_value=corrected_p,
            correlation_strength=corr,
            statistical_confidence=_statistical_confidence(corrected_p, corr),
            is_significant=bool(reject),
        )
        candidates.append(cand)
        by_pair[(g.asset_a, g.asset_b)] = cand

    # 6. PC graph discovery; map kept edges back to candidate ids --------------
    graph = discover_causal_graph(
        returns, alpha=alpha, indep_test=indep_test, granger_results=granger
    )
    graph_meta: dict[str, tuple[str, str]] = {}
    for u, v, data in graph.edges(data=True):
        cand = by_pair.get((u, v))
        if cand is None:
            continue  # PC should only emit pairs we Granger-tested, but be safe
        graph_meta[cand.candidate_id] = (
            data["edge_type"],
            data["orientation_source"],
        )

    # 7. HMM regime detection for significant candidates -----------------------
    # A regime window only makes sense for a relationship that exists, so we
    # spend the HMM fits on the significant edges rather than all 100+ pairs.
    for cand in candidates:
        if cand.is_significant:
            cand.regime_periods = detect_regimes(returns, cand.asset_a, cand.asset_b)

    # 8. Run metadata — store the ACTUAL covered window, not just what was asked
    covered_start = str(returns["date"].min())
    covered_end = str(returns["date"].max())
    run_notes = notes
    if fetched.report.missing:
        miss = f"missing tickers (no data): {fetched.report.missing}"
        run_notes = f"{notes}; {miss}" if notes else miss

    run = AnalysisRun(
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        start_date=covered_start,
        end_date=covered_end,
        asset_universe=analysed,
        max_lag=max_lag,
        correction_method=correction_method,
        alpha=alpha,
        notes=run_notes,
    )

    return AnalysisResult(
        run=run,
        candidates=candidates,
        graph_meta=graph_meta,
        missing_tickers=fetched.report.missing,
    )


if __name__ == "__main__":
    result = run_analysis()
    sig = [c for c in result.candidates if c.is_significant]
    print(f"Run {result.run.run_id}")
    print(f"  window:      {result.run.start_date} .. {result.run.end_date}")
    print(f"  assets:      {len(result.run.asset_universe)}")
    print(f"  candidates:  {len(result.candidates)}  ({len(sig)} significant)")
    print(f"  graph edges: {len(result.graph_meta)}")
    if result.missing_tickers:
        print(f"  MISSING:     {result.missing_tickers}")
