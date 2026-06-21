"""Fixed asset universe and shared configuration for the engine.

The **13-asset** universe is fixed by the build spec. (An early spec header
said "12", but the spec's own enumeration — and ``ASSET_UNIVERSE`` below —
list 13: 3 commodities + 3 currencies + 3 equity indices + 1 rate + 3 sector
ETFs. We use 13; this note reconciles the apparent off-by-one.) Tickers are
yfinance symbols and MUST be verified against yfinance before being relied
upon (symbols and availability drift over time) — see
``scripts/verify_tickers.py``.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path


class AssetClass(str, Enum):
    COMMODITY = "commodity"
    CURRENCY = "currency"
    EQUITY_INDEX = "equity_index"
    RATE = "rate"
    SECTOR_ETF = "sector_etf"


# ticker -> (human-readable name, asset class)
ASSET_UNIVERSE: dict[str, tuple[str, AssetClass]] = {
    # Commodities
    "CL=F": ("Crude Oil", AssetClass.COMMODITY),
    "GC=F": ("Gold", AssetClass.COMMODITY),
    "NG=F": ("Natural Gas", AssetClass.COMMODITY),
    # Currencies
    "EURUSD=X": ("EUR/USD", AssetClass.CURRENCY),
    "JPY=X": ("USD/JPY", AssetClass.CURRENCY),
    "INR=X": ("USD/INR", AssetClass.CURRENCY),
    # Equity Indices
    "^GSPC": ("S&P 500", AssetClass.EQUITY_INDEX),
    "^IXIC": ("Nasdaq", AssetClass.EQUITY_INDEX),
    "^NSEI": ("Nifty 50", AssetClass.EQUITY_INDEX),
    # Bonds / Rates
    "^TNX": ("10Y Treasury Yield", AssetClass.RATE),
    # Sector ETFs
    "XLE": ("Energy Sector ETF", AssetClass.SECTOR_ETF),
    "XLF": ("Financials Sector ETF", AssetClass.SECTOR_ETF),
    "JETS": ("US Global Jets (Airlines) ETF", AssetClass.SECTOR_ETF),
}

TICKERS: list[str] = list(ASSET_UNIVERSE.keys())


def asset_name(ticker: str) -> str:
    """Human-readable name for a ticker, or the ticker itself if unknown."""
    entry = ASSET_UNIVERSE.get(ticker)
    return entry[0] if entry else ticker


# ---------------------------------------------------------------------------
# Single source of truth for every tunable in the pipeline. Modules default
# their keyword arguments to these constants instead of hardcoding their own,
# so a run is configured in exactly one place (the build doc asks for this).
# ---------------------------------------------------------------------------

# --- Analysis window & significance ---
DEFAULT_START_DATE = "2018-01-01"   # spans the 2020 COVID regime break
DEFAULT_END_DATE: str | None = None  # None => through the latest available bar
DEFAULT_MAX_LAG = 5                  # trading days scanned by Granger / lead-lag
DEFAULT_ALPHA = 0.05                 # significance threshold (also PC alpha)
DEFAULT_CORRECTION = "fdr_bh"        # 'fdr_bh' | 'bonferroni'

# --- Preprocessing / alignment ---
# Cross-asset alignment strategy: forward-fill each series across at most
# DEFAULT_MAX_FFILL_GAP missing sessions (a holiday another market observed),
# then drop any row still incomplete. See data/preprocessor.py.
DEFAULT_MAX_FFILL_GAP = 3
# Minimum aligned log-return rows required to attempt Granger / PC at all.
# Guards against pathologically short date ranges; a multi-year run has ~1000+.
MIN_OBSERVATIONS = 50

# --- PC causal-graph discovery (causal-learn) ---
DEFAULT_INDEP_TEST = "fisherz"

# --- HMM regime detection (hmmlearn) ---
DEFAULT_REGIME_WINDOW = 60           # rolling-correlation window (trading days)
DEFAULT_HMM_N_COMPONENTS = 2         # coupled vs decoupled regimes
DEFAULT_HMM_N_ITER = 100             # EM iterations (lib default of 10 is too low)
DEFAULT_HMM_RANDOM_STATE = 0         # fixed for reproducibility

# --- Phase 3: scheduled regime-flip monitoring ---
# A flip (a pair's coupling switching on/off between two runs) is NOT trusted on
# first sight: yfinance can revise the most recent bars after the fact, so a
# single-run flip may be a data-revision artifact. A flip is only marked
# "confirmed" once its new regime status has persisted across this many
# consecutive monitor runs without reverting. Same discipline as not trusting a
# lone extreme p-value without the stationarity stress-test.
MONITOR_CONFIRMATION_RUNS = 3
# Alert-fatigue gate: with 100+ pairs, only pairs whose underlying candidate is
# significant after correction in the *new* run are eligible to surface a flip.
# Reuses the analysis alpha — a pair that was never significant is not a signal.
MONITOR_MIN_SIGNIFICANCE_ALPHA = DEFAULT_ALPHA
# Tag written to a monitor run's notes so monitor runs are distinguishable from
# ad-hoc /analyze runs when listing/diffing.
MONITOR_RUN_NOTE = "scheduled-monitor"

# --- Persistence ---
# SQLite lives under db/. Overridable at runtime via the CAUSAL_ENGINE_DB env
# var (used by the API and the test suite to point at a throwaway file).
DB_PATH = Path(__file__).resolve().parent / "db" / "causal_engine.db"

# --- Dashboard (Streamlit thin HTTP client over the API) ---
# The Streamlit dashboard NEVER imports the pipeline; it only talks to the
# FastAPI service over HTTP so the API stays the single source of truth. Every
# hardcoded dashboard parameter lives here, not scattered in dashboard/app.py.
API_BASE_URL = "http://127.0.0.1:8000"   # where `uvicorn api.main:app` is served
API_GET_TIMEOUT_SECONDS = 15             # metadata / findings reads are fast
API_ANALYZE_TIMEOUT_SECONDS = 1800       # a full multi-year run is synchronous & slow
# Default value for the dashboard's corrected-significance filter: edges with a
# corrected p-value at or below this are shown. Mirrors the analysis alpha.
DASHBOARD_DEFAULT_SIG_THRESHOLD = DEFAULT_ALPHA
# Fixed canvas size for the streamlit-agraph causal graph (pixels).
DASHBOARD_GRAPH_HEIGHT = 600
DASHBOARD_GRAPH_WIDTH = 900
