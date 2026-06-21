"""Diagnostic: was the HMM label-switching safeguard actually EXERCISED across
the two real monitor windows (not just unit-tested)?

The safeguard (causal/regime_detection.py): the "active" (coupled) regime is the
fitted state with the higher mean |correlation| — ``argmax(|means|)`` — never a
raw HMM state index, which is arbitrary per fit. This script fits the HMM for
each significant pair on BOTH windows and reports, per pair, the raw active-state
index each fit chose. If that index differs between the two fits (0 vs 1) for any
pair, the raw HMM numbering genuinely flipped between independent fits and the
safeguard corrected it — i.e. it was exercised on real data, not just in tests.

Usage:
    python -m scripts.diagnose_label_switching <baseline_run_id> <end1> <end2>
"""
from __future__ import annotations

import sys
import warnings
from datetime import date

import numpy as np
import polars as pl
from hmmlearn.hmm import GaussianHMM

import config
from causal.regime_detection import _rolling_correlation
from config import (
    DEFAULT_HMM_N_ITER,
    DEFAULT_HMM_RANDOM_STATE,
    DEFAULT_REGIME_WINDOW,
)
from data.fetcher import fetch_prices
from data.preprocessor import preprocess
from db import storage

warnings.filterwarnings("ignore")


def active_state_index(returns, a, b) -> tuple[int, float, float] | None:
    """Refit the exact regime HMM and return (active_state_idx, mean|corr| of the
    two states). None if the window is too short to fit."""
    dates, corr = _rolling_correlation(returns, a, b, DEFAULT_REGIME_WINDOW)
    if len(corr) < 2 * DEFAULT_REGIME_WINDOW:
        return None
    model = GaussianHMM(
        n_components=2, covariance_type="full",
        n_iter=DEFAULT_HMM_N_ITER, random_state=DEFAULT_HMM_RANDOM_STATE,
    )
    model.fit(corr.reshape(-1, 1))
    means = np.abs(model.means_.ravel())
    return int(np.argmax(means)), float(means[0]), float(means[1])


def main() -> None:
    run_id, end1, end2 = sys.argv[1], sys.argv[2], sys.argv[3]
    db = "results/monitor_validation.db"

    sig = storage.load_candidates(run_id, significant_only=True, db_path=db)
    pairs = [(c.asset_a, c.asset_b) for c in sig]
    print(f"Baseline run {run_id}: {len(pairs)} significant pairs to check.\n")

    prices = fetch_prices(start=config.DEFAULT_START_DATE)
    _, rets_full = preprocess(prices)
    d1, d2 = date.fromisoformat(end1), date.fromisoformat(end2)
    w1 = rets_full.filter(pl.col("date") <= d1)
    w2 = rets_full.filter(pl.col("date") <= d2)
    print(f"Window 1 (<= {end1}): {len(w1)} rows | "
          f"Window 2 (<= {end2}): {len(w2)} rows\n")

    flipped = 0
    checked = 0
    for a, b in pairs:
        r1 = active_state_index(w1, a, b)
        r2 = active_state_index(w2, a, b)
        if r1 is None or r2 is None:
            continue
        checked += 1
        if r1[0] != r2[0]:
            flipped += 1
            print(f"  LABEL-SWITCH on {config.asset_name(a)} -> "
                  f"{config.asset_name(b)}: window1 active_state={r1[0]} "
                  f"(|means|={r1[1]:.3f},{r1[2]:.3f})  vs  window2 "
                  f"active_state={r2[0]} (|means|={r2[1]:.3f},{r2[2]:.3f})")

    print(f"\nChecked {checked} pairs across both windows.")
    print(f"Raw HMM state index flipped between fits on {flipped} pair(s).")
    if flipped:
        print("=> Label-switching safeguard WAS exercised on real data: the raw "
              "state numbering differed between independent fits, and the "
              "argmax-|mean| assignment kept 'active' consistent regardless.")
    else:
        print("=> The raw state index happened to be stable across both fits "
              "(fixed random_state + overlapping data), so the safeguard was not "
              "visibly triggered this time. It still guards every fit by "
              "construction; it simply was not needed to correct a flip here.")


if __name__ == "__main__":
    main()
