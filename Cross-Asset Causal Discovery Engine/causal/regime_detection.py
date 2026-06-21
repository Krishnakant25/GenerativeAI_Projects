"""HMM-based regime detection for a pairwise relationship (hmmlearn).

A causal candidate is rarely valid for all time — a relationship that held in
2019 may break in 2020. This module asks *when* a pair's co-movement was
active, so every finding carries a time-bound validity window rather than a
permanent claim (a project hard rule).

Approach: compute a rolling correlation between the two assets' log returns,
then fit a 2-state Gaussian HMM over that series. One latent state captures the
"coupled" regime (high |correlation|), the other the "decoupled" regime. We
collapse the Viterbi path into contiguous ``RegimePeriod`` windows.

Two facts about hmmlearn 0.3.3 drive the code — both verified, not assumed:
  * ``GaussianHMM.fit`` needs ``X`` shape ``(n_samples, n_features)``; a
    univariate series must be reshaped to ``(-1, 1)``.
  * State labels are arbitrary (label-switching), so we never assume
    "state 1 = active" — we map states to regimes by their fitted mean
    |correlation|. ``random_state`` is fixed and ``n_iter`` raised above the
    library default of 10 so results are reproducible and EM actually
    converges.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
from hmmlearn.hmm import GaussianHMM

from causal.models import RegimePeriod
from config import (
    DEFAULT_HMM_N_ITER,
    DEFAULT_HMM_RANDOM_STATE,
    DEFAULT_REGIME_WINDOW,
)


def _rolling_correlation(
    returns: pl.DataFrame, asset_a: str, asset_b: str, window: int
) -> tuple[list[date], np.ndarray]:
    """Rolling Pearson correlation of two return series, dropping warm-up nulls."""
    corr = (
        returns.select(
            "date",
            pl.rolling_corr(pl.col(asset_a), pl.col(asset_b), window_size=window)
            .alias("corr"),
        )
        .drop_nulls("corr")
    )
    return corr["date"].to_list(), corr["corr"].to_numpy()


def detect_regimes(
    returns: pl.DataFrame,
    asset_a: str,
    asset_b: str,
    window: int = DEFAULT_REGIME_WINDOW,
    n_iter: int = DEFAULT_HMM_N_ITER,
    random_state: int = DEFAULT_HMM_RANDOM_STATE,
) -> list[RegimePeriod]:
    """Detect coupled/decoupled regimes for the ``(asset_a, asset_b)`` pair.

    Returns a chronological list of ``RegimePeriod`` windows. The ``active``
    flag marks the higher-|correlation| (coupled) regime. Returns an empty list
    if there are too few observations to fit a 2-state model.
    """
    dates, corr = _rolling_correlation(returns, asset_a, asset_b, window)
    if len(corr) < 2 * window:  # not enough signal for a stable 2-state fit
        return []

    X = corr.reshape(-1, 1)
    model = GaussianHMM(
        n_components=2,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(X)
    states = model.predict(X)

    # Label-switching: the "active" state is the one with higher mean |corr|.
    means = model.means_.ravel()
    active_state = int(np.argmax(np.abs(means)))

    return _collapse_runs(dates, corr, states, active_state)


def _collapse_runs(
    dates: list[date],
    corr: np.ndarray,
    states: np.ndarray,
    active_state: int,
) -> list[RegimePeriod]:
    """Merge contiguous same-state observations into ``RegimePeriod`` windows."""
    periods: list[RegimePeriod] = []
    run_start = 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[run_start]:
            seg = corr[run_start:i]
            periods.append(
                RegimePeriod(
                    start=dates[run_start],
                    end=dates[i - 1],
                    active=bool(states[run_start] == active_state),
                    mean_correlation=float(np.mean(seg)),
                )
            )
            run_start = i
    return periods


if __name__ == "__main__":
    from data.fetcher import fetch_prices
    from data.preprocessor import preprocess

    _, rets = preprocess(fetch_prices())
    regimes = detect_regimes(rets, "CL=F", "JETS")  # oil vs airlines
    print(f"Oil -> Airlines: {len(regimes)} regime windows")
    for r in regimes:
        flag = "ACTIVE  " if r.active else "inactive"
        print(f"  {r.start} .. {r.end}  [{flag}] mean_corr={r.mean_correlation:+.2f}")
