"""Tests for the price-panel preprocessor.

The key regression here is the non-finite guard: log returns are undefined for
non-positive prices (the canonical real-world case being front-month WTI crude,
``CL=F``, whose settlement went negative on 2020-04-20). Polars treats NaN/inf
as valid floats, so ``drop_nulls`` would let them through and poison the
statistics downstream — the preprocessor must drop those rows itself.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl

from data.preprocessor import align_panel, compute_log_returns, preprocess


def _dates(n: int) -> pl.Series:
    base = date(2020, 1, 1)
    return pl.Series("date", [base + timedelta(days=i) for i in range(n)])


def test_negative_price_yields_only_finite_returns():
    # OIL prints a negative settlement on day index 2, like CL=F in April 2020.
    prices = pl.DataFrame(
        {
            "date": _dates(6),
            "A": [10.0, 11.0, 12.0, 11.5, 12.0, 12.5],
            "OIL": [50.0, 51.0, -5.0, 20.0, 21.0, 22.0],
        }
    )
    _, returns = preprocess(prices)

    # Every surviving return is finite (no NaN/inf leaked through).
    for ticker in ("A", "OIL"):
        assert all(math.isfinite(v) for v in returns[ticker].to_numpy())

    # The two sessions bracketing the sign flip (into and out of the negative
    # price) are dropped: 5 raw returns -> 3 finite ones.
    assert returns.height == 3


def test_forward_fill_then_drop_keeps_full_cross_sections():
    # B is missing one session (a holiday the other market observed); a short
    # gap is carried forward, and the result has no nulls.
    prices = pl.DataFrame(
        {
            "date": _dates(5),
            "A": [10.0, 10.5, 11.0, 11.2, 11.5],
            "B": [20.0, None, 21.0, 21.5, 22.0],
        }
    )
    aligned = align_panel(prices, max_ffill_gap=3)
    assert aligned.null_count().to_numpy().sum() == 0
    assert aligned.height == 5  # the gap was bridged, no row dropped


def test_long_gap_is_not_bridged():
    # A gap longer than max_ffill_gap must NOT be silently carried across.
    prices = pl.DataFrame(
        {
            "date": _dates(6),
            "A": [10.0, 10.5, 11.0, 11.2, 11.5, 11.8],
            "B": [20.0, None, None, None, None, 22.0],
        }
    )
    aligned = align_panel(prices, max_ffill_gap=2)
    # The four-session gap exceeds the cap, so those rows are dropped.
    assert aligned.height < 6
    assert aligned.null_count().to_numpy().sum() == 0
