"""Pairwise Granger-causality testing across the asset universe.

Granger causality asks a narrow, honest question: *do the past values of A
help predict B beyond what B's own past already explains?* That is predictive
precedence, NOT proof of causation (a common driver, or a faster third asset,
produces the same signal). Every name and docstring here keeps that framing.

Inputs must be (weakly) stationary — we operate on the log-return panel from
``data.preprocessor``, not raw prices.

Implementation notes:
  - We use ``statsmodels.tsa.stattools.grangercausalitytests``. Its convention
    is that, given a 2-column array ``[[y, x], ...]``, it tests whether column
    **x (the second column)** Granger-causes column **y (the first)**. So to
    test "asset_a -> asset_b" we pass columns ordered ``[b, a]``.
  - We scan lags ``1..max_lag`` and keep the lag with the smallest p-value,
    reporting both that lag and its raw p-value. This is itself a form of
    multiple testing across lags; the headline correction across *pairs*
    happens later in ``causal.correction``. (Picking the best lag inflates
    significance a little; documented here rather than hidden.)
  - The raw ``ssr_ftest`` p-value is used as the test statistic.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import numpy as np
import polars as pl
from statsmodels.tsa.stattools import grangercausalitytests

from config import DEFAULT_MAX_LAG


@dataclass(frozen=True)
class GrangerResult:
    """Raw (uncorrected) result of one directional Granger test."""

    asset_a: str        # candidate driver
    asset_b: str        # candidate affected
    best_lag: int
    p_value: float      # raw ssr_ftest p-value at best_lag
    f_stat: float


def _pair_test(y: np.ndarray, x: np.ndarray, max_lag: int) -> tuple[int, float, float]:
    """Test whether ``x`` Granger-causes ``y``. Returns (best_lag, p, F).

    Picks the lag in 1..max_lag with the smallest ssr_ftest p-value.
    """
    # statsmodels wants column 0 = the series being predicted (y),
    # column 1 = the candidate causal series (x).
    data = np.column_stack([y, x])
    # grangercausalitytests prints a full results table to stdout for every
    # lag (the `verbose` kwarg is deprecated/ignored in current statsmodels).
    # That noise would pollute API logs once this runs per-request, so we
    # silence stdout and read the p-values from the returned object instead.
    with contextlib.redirect_stdout(io.StringIO()):
        results = grangercausalitytests(data, maxlag=max_lag)

    best_lag, best_p, best_f = 1, 1.0, 0.0
    for lag, (stats, _) in results.items():
        f_stat, p_value = stats["ssr_ftest"][0], stats["ssr_ftest"][1]
        if p_value < best_p:
            best_lag, best_p, best_f = lag, float(p_value), float(f_stat)
    return best_lag, best_p, best_f


def run_pairwise_granger(
    returns: pl.DataFrame,
    max_lag: int = DEFAULT_MAX_LAG,
) -> list[GrangerResult]:
    """Run Granger tests for every ordered pair of tickers (both directions).

    ``returns`` is the wide log-return panel (``date`` + one column per
    ticker). Returns one `GrangerResult` per ordered pair (a, b), a != b —
    i.e. n*(n-1) results for n tickers. p-value correction is applied
    separately by the caller via ``causal.correction``.
    """
    tickers = [c for c in returns.columns if c != "date"]
    series = {t: returns[t].to_numpy() for t in tickers}

    out: list[GrangerResult] = []
    for a in tickers:
        for b in tickers:
            if a == b:
                continue
            best_lag, p_value, f_stat = _pair_test(
                y=series[b], x=series[a], max_lag=max_lag
            )
            out.append(
                GrangerResult(
                    asset_a=a, asset_b=b,
                    best_lag=best_lag, p_value=p_value, f_stat=f_stat,
                )
            )
    return out


if __name__ == "__main__":
    from data.fetcher import fetch_prices
    from data.preprocessor import preprocess

    _, rets = preprocess(fetch_prices())
    results = run_pairwise_granger(rets, max_lag=5)
    results.sort(key=lambda r: r.p_value)
    print("Top 10 directional Granger results (raw p, uncorrected):")
    for r in results[:10]:
        print(f"  {r.asset_a:8s} -> {r.asset_b:8s} lag={r.best_lag} p={r.p_value:.4g}")
