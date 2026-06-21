"""Tests for pairwise Granger causality.

Rather than depend on live market data (slow, non-deterministic), we validate
the engine against a *constructed* relationship where the ground truth is
known: build series X as noise and Y as a lagged copy of X plus noise. By
construction X's past predicts Y, but Y's past does NOT predict X. A correct
Granger implementation must find X -> Y far more significant than Y -> X.

This is the unit-test analogue of the doc's "validate against a known textbook
example (oil -> airlines)": we manufacture the textbook example so the test is
deterministic.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from causal.granger import run_pairwise_granger


def _make_lagged_panel(n: int = 600, lag: int = 2, seed: int = 7) -> pl.DataFrame:
    """X is white noise; Y_t depends on X_{t-lag}. So X Granger-causes Y."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(lag, n):
        y[t] = 0.8 * x[t - lag] + 0.3 * rng.standard_normal()
    return pl.DataFrame(
        {
            "date": pl.Series(
                "date", range(n)
            ),  # ordinal index is fine — granger ignores `date`
            "X": x,
            "Y": y,
        }
    )


def test_detects_known_direction():
    panel = _make_lagged_panel(lag=2)
    results = run_pairwise_granger(panel, max_lag=5)
    by_pair = {(r.asset_a, r.asset_b): r for r in results}

    x_to_y = by_pair[("X", "Y")]
    y_to_x = by_pair[("Y", "X")]

    # X -> Y is the true direction: it should be strongly significant...
    assert x_to_y.p_value < 0.01
    # ...and far more significant than the reverse (spurious) direction.
    assert x_to_y.p_value < y_to_x.p_value
    # The detected lag should be at or near the injected lag of 2.
    assert x_to_y.best_lag in (1, 2, 3)


def test_independent_series_not_significant():
    rng = np.random.default_rng(11)
    panel = pl.DataFrame(
        {
            "date": pl.Series("date", range(600)),
            "A": rng.standard_normal(600),
            "B": rng.standard_normal(600),
        }
    )
    results = run_pairwise_granger(panel, max_lag=5)
    # Two independent noise series: neither direction should be strongly
    # significant. (Allow a loose bound — best-lag scanning inflates a little.)
    for r in results:
        assert r.p_value > 0.01


def test_emits_both_directions_per_pair():
    panel = _make_lagged_panel()
    results = run_pairwise_granger(panel, max_lag=3)
    pairs = {(r.asset_a, r.asset_b) for r in results}
    assert ("X", "Y") in pairs and ("Y", "X") in pairs
    assert len(results) == 2  # 2 tickers -> 2 ordered pairs
