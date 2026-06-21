"""Time-lagged cross-correlation — lead-lag relationship detection.

Granger answers "does A's past help predict B?"; cross-correlation answers
"at what lag, and how strongly and in which sign, do A and B co-move?". The two
are complementary: Granger gives significance, the lead-lag correlation gives
an interpretable effect size and direction that populates
``CausalCandidate.correlation_strength``.

For a positive lag k, we correlate A_t with B_{t+k} — i.e. A *leads* B by k
days. We scan k in 1..max_lag and report the lag with the largest absolute
correlation, keeping its signed value (sign matters: oil *up* preceding
airlines *down* is a different economic story than oil up preceding airlines
up).

Pure numpy over the stationary log-return panel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from config import DEFAULT_MAX_LAG


@dataclass(frozen=True)
class LeadLagResult:
    """Peak lead-lag cross-correlation for an ordered pair (a leads b)."""

    asset_a: str        # leader
    asset_b: str        # follower
    best_lag: int
    correlation: float  # signed Pearson correlation at best_lag


def _lagged_corr(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    """Pearson correlation of a_t with b_{t+lag} (a leads b by `lag`)."""
    if lag <= 0:
        raise ValueError("lag must be >= 1")
    a_lead = a[:-lag]
    b_follow = b[lag:]
    if a_lead.size < 3:
        return 0.0
    # np.corrcoef returns the 2x2 matrix; off-diagonal is the correlation.
    c = np.corrcoef(a_lead, b_follow)[0, 1]
    return 0.0 if np.isnan(c) else float(c)


def lagged_correlation(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    """Signed Pearson correlation of ``a`` leading ``b`` by exactly ``lag`` days.

    Public entry point used to attach a ``correlation_strength`` to a candidate
    *at the lag Granger picked*, so the reported effect size and the reported
    lag describe the same relationship.
    """
    return _lagged_corr(a, b, lag)


def peak_lead_lag(
    a: np.ndarray, b: np.ndarray, max_lag: int = DEFAULT_MAX_LAG
) -> LeadLagResult:
    """Find the lag in 1..max_lag maximising |corr(a_t, b_{t+lag})|."""
    best_lag, best_corr = 1, 0.0
    for lag in range(1, max_lag + 1):
        c = _lagged_corr(a, b, lag)
        if abs(c) > abs(best_corr):
            best_lag, best_corr = lag, c
    return LeadLagResult(asset_a="", asset_b="", best_lag=best_lag, correlation=best_corr)


def run_pairwise_lead_lag(
    returns: pl.DataFrame, max_lag: int = DEFAULT_MAX_LAG
) -> dict[tuple[str, str], LeadLagResult]:
    """Compute peak lead-lag correlation for every ordered pair (a leads b)."""
    tickers = [c for c in returns.columns if c != "date"]
    series = {t: returns[t].to_numpy() for t in tickers}

    out: dict[tuple[str, str], LeadLagResult] = {}
    for a in tickers:
        for b in tickers:
            if a == b:
                continue
            res = peak_lead_lag(series[a], series[b], max_lag=max_lag)
            out[(a, b)] = LeadLagResult(
                asset_a=a, asset_b=b,
                best_lag=res.best_lag, correlation=res.correlation,
            )
    return out
