"""Market-data acquisition for the fixed asset universe.

yfinance is the only place pandas appears in this project, and only because
yfinance returns pandas natively. We convert to Polars at the boundary
immediately (see ``_close_frame_to_polars``); everything downstream is Polars,
per the project's hard rule (no pandas in the processing pipeline).

The fetcher pulls *Close* prices (auto-adjusted for splits/dividends) and
returns a wide Polars DataFrame: one ``date`` column plus one column per
ticker that actually returned data.

Robustness contract (the universe spans commodities, FX, equities and rates,
which keep *different* trading calendars and histories):

  * **A ticker that returns no data at all** is dropped from the panel and
    recorded in ``FetchReport.missing`` rather than left as an all-null column
    that would later wipe out every row in the inner-join. The run continues
    on the tickers that *did* return data.
  * **Partial date coverage** (a ticker whose history starts late, or a
    sparse series like ``^TNX`` that prints far fewer bars) is left as-is here;
    the per-ticker non-null counts are reported in ``FetchReport.coverage`` and
    the actual calendar reconciliation (forward-fill short gaps, then drop
    incomplete cross-sections) is the preprocessor's job, documented there.
  * **Total failure** (nothing came back for any ticker) raises
    ``DataUnavailableError`` so the API can answer with a meaningful HTTP error
    instead of a 500.

Alignment strategy (chosen, and stated explicitly): we do NOT reindex every
asset onto a single exchange's calendar (that would invent or delete bars).
Instead the union of all observed dates is kept here; the preprocessor then
forward-fills each series across short holiday gaps and drops any date that is
still not fully observed. The result is an "intersection-with-carry" panel —
every surviving row is a real, simultaneously-observed cross-section.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
import yfinance as yf

try:
    from config import DEFAULT_END_DATE, DEFAULT_START_DATE, TICKERS
except ImportError:  # pragma: no cover - path shim for direct execution
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import DEFAULT_END_DATE, DEFAULT_START_DATE, TICKERS


class DataUnavailableError(RuntimeError):
    """Raised when the data provider returns nothing usable for any ticker."""


@dataclass(frozen=True)
class FetchReport:
    """Per-run provenance for a data pull — what was asked for vs. delivered."""

    requested: list[str]
    returned: list[str]            # tickers with >= 1 real observation
    missing: list[str]             # requested but no data at all
    rows: int                      # raw rows in the union-of-dates panel
    coverage: dict[str, int]       # ticker -> count of non-null observations


@dataclass(frozen=True)
class FetchResult:
    """A fetched price panel together with its provenance report."""

    prices: pl.DataFrame           # wide panel: `date` + one col per returned ticker
    report: FetchReport


def _close_frame_to_polars(close_df, tickers: list[str]) -> pl.DataFrame:
    """Convert a pandas Close-price frame (DatetimeIndex) to a wide Polars
    DataFrame with a ``date`` column. Single- and multi-ticker yfinance shapes
    are normalised to the same wide layout."""
    # Single ticker: yfinance returns a Series-like / single column frame.
    if len(tickers) == 1 and close_df.ndim == 1:
        close_df = close_df.to_frame(name=tickers[0])

    data = {"date": list(close_df.index.to_pydatetime())}
    for ticker in tickers:
        if ticker in close_df.columns:
            data[ticker] = close_df[ticker].to_list()
        else:
            # Ticker requested but absent from the response — surface as nulls
            # here; fetch_panel decides whether that means "drop the column".
            data[ticker] = [None] * len(close_df)

    return pl.DataFrame(data).with_columns(pl.col("date").cast(pl.Date)).sort("date")


def fetch_panel(
    tickers: list[str] | None = None,
    start: str = DEFAULT_START_DATE,
    end: str | None = DEFAULT_END_DATE,
) -> FetchResult:
    """Download adjusted Close prices for ``tickers`` over [start, end) and
    return them with a coverage report.

    Tickers that return no data at all are dropped from the panel and listed in
    ``report.missing``. Raises ``DataUnavailableError`` if *nothing* usable
    came back, so callers never silently analyse an empty panel.
    """
    requested = list(tickers or TICKERS)

    raw = yf.download(
        requested,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        group_by="column",
    )
    if raw is None or raw.empty:
        raise DataUnavailableError(
            f"yfinance returned no data for {requested} over {start}..{end}"
        )

    # With group_by="column", multi-ticker frames carry a ("Close", ticker)
    # column MultiIndex; single-ticker frames have flat columns. Both expose
    # a "Close" key.
    close = raw["Close"]
    panel = _close_frame_to_polars(close, requested)

    # Per-ticker real-observation counts; a ticker with zero is "missing".
    coverage = {t: panel.height - panel[t].null_count() for t in requested}
    returned = [t for t in requested if coverage[t] > 0]
    missing = [t for t in requested if coverage[t] == 0]

    if not returned:
        raise DataUnavailableError(
            f"Every requested ticker returned empty over {start}..{end}: {requested}"
        )

    # Keep only columns that actually carry data so a dead ticker can't null
    # out every cross-section downstream.
    panel = panel.select(["date", *returned])

    report = FetchReport(
        requested=requested,
        returned=returned,
        missing=missing,
        rows=panel.height,
        coverage=coverage,
    )
    return FetchResult(prices=panel, report=report)


def fetch_prices(
    tickers: list[str] | None = None,
    start: str = DEFAULT_START_DATE,
    end: str | None = DEFAULT_END_DATE,
) -> pl.DataFrame:
    """Backwards-compatible convenience: just the wide price panel.

    Thin wrapper over :func:`fetch_panel` for scripts and ``__main__`` blocks
    that don't need the coverage report.
    """
    return fetch_panel(tickers, start=start, end=end).prices


if __name__ == "__main__":
    result = fetch_panel()
    df = result.prices
    print(f"Fetched {df.height} rows x {df.width - 1} tickers")
    print(df.head())
    print("Returned:", result.report.returned)
    if result.report.missing:
        print("MISSING (no data):", result.report.missing)
    print("Coverage (non-null obs per ticker):", result.report.coverage)
