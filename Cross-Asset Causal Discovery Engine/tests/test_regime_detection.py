"""Tests for HMM-based regime detection.

Constructed data again, so the ground truth is known: two assets that are
strongly coupled in the first half of the sample and independent in the second.
A correct detector must (a) recover at least one active and one inactive
window, (b) resolve label-switching so the "active" flag tracks the
higher-|correlation| regime regardless of the HMM's internal state numbering,
and (c) place the coupled regime in the first half.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from causal.regime_detection import detect_regimes


def _two_regime_panel(n: int = 600, brk: int = 300, seed: int = 1) -> pl.DataFrame:
    """A and B: tightly coupled for t < brk, independent for t >= brk."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = np.empty(n)
    b[:brk] = a[:brk] + 0.1 * rng.standard_normal(brk)        # coupled
    b[brk:] = rng.standard_normal(n - brk)                    # decoupled
    base = date(2020, 1, 1)
    dates = [base + timedelta(days=i) for i in range(n)]
    return pl.DataFrame({"date": pl.Series("date", dates), "A": a, "B": b})


def test_recovers_active_and_inactive_regimes():
    panel = _two_regime_panel()
    regimes = detect_regimes(panel, "A", "B", window=60)

    assert len(regimes) >= 2, "should split the sample into multiple windows"

    active = [r for r in regimes if r.active]
    inactive = [r for r in regimes if not r.active]
    assert active and inactive, "expected both an active and an inactive regime"

    # Every window is time-bound and well-ordered (never a global claim).
    for r in regimes:
        assert r.start <= r.end

    # Label-switching resolved: the "active" regime really is the high-|corr|
    # one, whatever the HMM numbered its states.
    active_corr = np.mean([abs(r.mean_correlation) for r in active])
    inactive_corr = np.mean([abs(r.mean_correlation) for r in inactive])
    assert active_corr > inactive_corr


def test_active_regime_lands_in_the_coupled_first_half():
    panel = _two_regime_panel(n=600, brk=300)
    regimes = detect_regimes(panel, "A", "B", window=60)
    active = [r for r in regimes if r.active]
    assert active
    # The dominant active window should start before the regime break (day 300).
    longest_active = max(active, key=lambda r: (r.end - r.start).days)
    assert longest_active.start < date(2020, 1, 1) + timedelta(days=300)


def test_too_short_returns_no_regimes():
    """Fewer than 2*window correlation points => no stable 2-state fit."""
    panel = _two_regime_panel(n=80, brk=40)
    assert detect_regimes(panel, "A", "B", window=60) == []
