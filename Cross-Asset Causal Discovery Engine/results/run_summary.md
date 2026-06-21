# Recorded validation run — summary

**run_id:** `run_20260619_133653_683874bb`
**Recorded window (covered):** 2018-01-04 → 2026-06-11
**Requested window:** 2018-01-03 → 2026-06-12
**Correction:** fdr_bh · α = 0.05 · max_lag = 5
**HMM random_state (fixed):** 0

## Counts
- Assets analysed: **13**
- Candidate relationships (ordered pairs): **156**
- Significant after correction: **106**
- Edges kept in PC graph: **20** (16 of which are Granger-significant)
- Regime windows recorded: **2143** across 106 pairs
- Tickers that failed to return data: none — all 13 tickers returned data

## Top 5 graph edges (most significant)
| Relationship | Corrected p | Lag (d) | Correlation | type/orientation |
|---|---|---|---|---|
| US Global Jets (Airlines) ETF → Nifty 50 | 2.46e-27 | 4 | -0.047 | directed/pc |
| Energy Sector ETF → Nifty 50 | 4.78e-22 | 5 | +0.118 | directed/pc |
| 10Y Treasury Yield → Energy Sector ETF | 1.10e-10 | 5 | +0.132 | directed/pc |
| US Global Jets (Airlines) ETF → S&P 500 | 4.29e-08 | 2 | +0.167 | directed/pc |
| Financials Sector ETF → Energy Sector ETF | 1.80e-07 | 2 | +0.151 | directed/pc |

## Stationarity stress-test (ADF on every Granger-input series)
**All 13 return series are stationary** at the 5% ADF level (every ADF p < 0.05; the least-stationary is ^TNX at p=1.27e-12). The extreme Granger p-values are therefore NOT an artifact of non-stationary inputs — the strong results survive this check. Remaining caveats (best-lag selection, residual autocorrelation/heteroskedasticity, and multiple comparisons) are documented in the README.

Full per-series ADF results in `stationarity.csv`.

## Artifacts written to `results/`
- `candidates.csv` — all 156 candidates with corrected p, lag, correlation, edge_type, orientation_source, in_graph.
- `graph_edges.csv` — the 20 discovered graph edges with statistics.
- `regimes.csv` — 2143 time-bound regime windows.
- `stationarity.csv` — ADF result per return series.
- `graph.png` — headless render of the 16 significant graph edges (each labelled with its corrected p-value).
- `test_output.txt` — full pytest output (written separately by the test run).
- `screenshots/` — headless renders of the four dashboard views (graph, regime timeline, edge detail, disclaimer banner), generated from this recorded run because the sandbox has no browser. To capture live browser screenshots, run `streamlit run dashboard/app.py` and load `run_20260619_133653_683874bb`.

_Detail panel rendered for top edge: US Global Jets (Airlines) ETF → Nifty 50 (corrected p = 2.46e-27)._
