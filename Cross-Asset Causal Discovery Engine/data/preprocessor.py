"""Cleaning and transformation of the raw price panel (Polars throughout).

Two problems to solve before any statistics run:

1. **Alignment.** The 13 assets trade on different calendars (US equities,
   Indian equities, FX, futures all keep different holidays), so the raw panel
   is ragged. We forward-fill each series over short gaps (carry the last known
   price across a holiday the *other* market observed) and then drop any rows
   that are still incomplete — typically the early period before a given
   asset's history begins.

2. **Stationarity.** Granger causality and most of the downstream statistics
   assume (weakly) stationary inputs. Raw prices are non-stationary (they
   trend and drift). We work in **log returns**, ``ln(P_t / P_{t-1})``, which
   are approximately stationary and comparable across assets of very different
   price levels.

The preprocessor never invents data: forward-fill only carries a real prior
observation forward; it never back-fills or interpolates a value that did not
exist.
"""

from __future__ import annotations

import polars as pl

try:
    from config import DEFAULT_MAX_FFILL_GAP
except ImportError:  # pragma: no cover - path shim for direct execution
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import DEFAULT_MAX_FFILL_GAP


def _ticker_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "date"]


def align_panel(
    prices: pl.DataFrame, max_ffill_gap: int = DEFAULT_MAX_FFILL_GAP
) -> pl.DataFrame:
    """Forward-fill short gaps per ticker, then drop rows still containing
    nulls so every remaining row is a fully-observed cross-section.

    ``max_ffill_gap`` caps how many consecutive missing observations we carry
    forward — a long gap (e.g. a delisting or a data outage) should *not* be
    silently bridged, so anything longer is left null and the row is dropped.
    """
    tickers = _ticker_columns(prices)

    filled = prices.sort("date").with_columns(
        [pl.col(t).forward_fill(limit=max_ffill_gap) for t in tickers]
    )
    # Keep only fully-observed cross-sections.
    return filled.drop_nulls(subset=tickers)


def compute_log_returns(prices: pl.DataFrame) -> pl.DataFrame:
    """Convert an aligned price panel to daily log returns.

    Returns a Polars DataFrame with the same ``date`` column (minus the first
    row, which has no prior price) and one log-return column per ticker.

    Non-finite guard: ``ln(P_t / P_{t-1})`` is only defined for strictly
    positive prices. Some real series violate that — most famously front-month
    WTI crude (``CL=F``), whose settlement went *negative* on 2020-04-20 — and a
    sign flip yields ``NaN`` while a zero yields ``-inf``. Critically, Polars
    treats ``NaN``/``inf`` as *valid floats*, so ``drop_nulls`` does NOT remove
    them and they would poison the downstream statistics ("x contains NaN or inf
    values"). We therefore drop any row whose return is not finite for every
    ticker, keeping only fully-finite cross-sections. This silently discards at
    most the one or two sessions bracketing such an event, not real signal.
    """
    tickers = _ticker_columns(prices)
    prices = prices.sort("date")

    returns = prices.with_columns(
        [
            (pl.col(t) / pl.col(t).shift(1)).log().alias(t)
            for t in tickers
        ]
    )
    # First row is null by construction (no t-1 price). Then keep only rows
    # where every ticker's return is finite (excludes NaN/±inf from a
    # non-positive price, which drop_nulls would otherwise let through).
    finite = pl.all_horizontal([pl.col(t).is_finite() for t in tickers])
    return returns.drop_nulls(subset=tickers).filter(finite)


def preprocess(
    prices: pl.DataFrame, max_ffill_gap: int = DEFAULT_MAX_FFILL_GAP
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Full Phase-1 preprocessing.

    Returns ``(aligned_prices, log_returns)``:
      - ``aligned_prices``  — gap-filled, fully-observed price panel (for
        plotting / regime visualisation),
      - ``log_returns``     — stationary series for the statistical engine.
    """
    aligned = align_panel(prices, max_ffill_gap=max_ffill_gap)
    returns = compute_log_returns(aligned)
    return aligned, returns


if __name__ == "__main__":
    from data.fetcher import fetch_prices

    raw = fetch_prices()
    aligned, returns = preprocess(raw)
    print(f"Raw rows:     {raw.height}")
    print(f"Aligned rows: {aligned.height}  (fully-observed cross-sections)")
    print(f"Return rows:  {returns.height}")
    print(returns.head())
