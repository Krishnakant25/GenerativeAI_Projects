"""Verify that every ticker in the fixed asset universe is currently valid
and returns data on yfinance.

Run this BEFORE building anything on top of the data layer — yfinance symbols
and availability drift over time, and a silently-dead ticker poisons the whole
analysis. Exits non-zero if any ticker fails.

Usage:
    python -m scripts.verify_tickers
"""

from __future__ import annotations

import sys

import yfinance as yf

# Allow running both as a module and as a script.
try:
    from config import ASSET_UNIVERSE, TICKERS, asset_name
except ImportError:  # pragma: no cover - path shim for direct execution
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import ASSET_UNIVERSE, TICKERS, asset_name


def verify(period: str = "5d") -> dict[str, bool]:
    """Download a short recent window for each ticker; report which return data."""
    results: dict[str, bool] = {}
    for ticker in TICKERS:
        name = asset_name(ticker)
        try:
            df = yf.download(
                ticker,
                period=period,
                progress=False,
                auto_adjust=True,
            )
            ok = df is not None and not df.empty
        except Exception as exc:  # noqa: BLE001 - report any failure per ticker
            ok = False
            print(f"  [ERROR] {ticker:10s} ({name}): {exc}")
        results[ticker] = ok
        status = "OK  " if ok else "FAIL"
        rows = 0 if not ok else len(df)
        print(f"  [{status}] {ticker:10s} {name:32s} rows={rows}")
    return results


def main() -> int:
    print(f"Verifying {len(TICKERS)} tickers against yfinance...\n")
    results = verify()
    failed = [t for t, ok in results.items() if not ok]
    print()
    if failed:
        print(f"{len(failed)} ticker(s) FAILED: {failed}")
        print("Update config.ASSET_UNIVERSE with current symbols before building.")
        return 1
    print(f"All {len(TICKERS)} tickers valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
