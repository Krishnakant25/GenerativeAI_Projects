# Out-of-sample edge-stability (walk-forward replication) — summary

**Method:** single train/test split (not rolling walk-forward — see script
docstring for the justification). The full Layer-1 pipeline (Granger → FDR → PC →
HMM) was run independently on each window; a discovery-significant pair is counted
as **replicated** iff the same ordered pair (same direction) is also
significant-after-correction in the holdout window.

| Window | Requested | Covered | Aligned rows | run_id |
|---|---|---|---|---|
| Discovery | 2018-01-04 → 2024-01-01 | 2018-01-05 → 2023-12-29 | 1301 | `run_20260620_195126_dee5ce5b` |
| Holdout | 2024-01-01 → 2026-06-12 | 2024-01-03 → 2026-06-11 | 520 | `run_20260620_195138_4f22b53c` |

**Correction:** fdr_bh · α = 0.05 · max_lag = 5

## Headline numbers

- Discovery-significant pairs: **103**
- Replicated in holdout (same direction, significant in both): **13**
- **Overall replication rate: 12.6%**
- **Empirical non-replication rate: 87.4%** (vs nominal α = 0.05)

### PC-graph edges vs Granger-only edges

| Category | Discovery-significant | Replicated | Replication rate |
|---|---|---|---|
| PC-graph (in_graph) | 14 | 2 | 14.3% |
| Granger-only | 89 | 11 | 12.4% |

**Directionally higher, but within noise — neither confirmed nor refuted.** PC-graph edges replicated at 14.3% (2/14) vs 12.4% (11/89) for Granger-only edges. The gap is 1.9 percentage points on a PC base of only 14 significant edges — one edge flipping would move it. The sign is consistent with the Phase-1 hypothesis (PC pruning → more stable edges), but this split is too thin to confirm or refute it with any confidence.

### Non-replication vs nominal alpha — honest reading

The empirical non-replication rate (87.4%) is
**not** a pure false-discovery rate and is not claimed to be one. A
discovery-significant edge can fail to replicate for three distinct reasons, only
the first of which is "the discovery finding was spurious":
1. **Genuine in-sample false positive** — overfitting across 156 simultaneous
   tests (this is what a non-replication rate far above α would flag).
2. **Real regime change** — a relationship that genuinely held in 2018–2023 and
   genuinely stopped holding in the structurally different 2024–2026 regime. The
   project's own regime-detection layer exists precisely because these
   relationships are *not* permanent.
3. **Power loss** — the holdout has ~520 rows vs ~1301 in
   discovery; a real but moderate edge can miss the corrected threshold on fewer
   observations.

The number is reported plainly so a reviewer can weigh these; the study does not
pre-attribute non-replication to any single cause.

**Mechanism (reverse coverage).** The holdout surfaced only **16**
significant edges in total (vs 103 in discovery, on
~520 rows vs ~1301). Of those 16 holdout edges,
**13 (81.2%)** were also in the
discovery set. So the low *forward* replication rate is driven largely by the
holdout detecting far fewer edges at all — and the durable edges it does detect
are overwhelmingly a subset of the discovery findings. This is mechanism, not
mitigation: the headline forward rate stands at 12.6%.

### Replication by discovery-signal strength

Discovery-significant edges ranked by corrected p (strongest first), in bands:

| Discovery rank band | Replicated | Rate |
|---|---|---|
| 1–20 (strongest) | 10/20 | 50.0% |
| 21–40 | 0/20 | 0.0% |
| 41–60 | 0/20 | 0.0% |
| 61+ (weakest) | 3/43 | 7.0% |

Durability tracks discovery strength: the very strongest edges carry the bulk of
the replication, while the broad middle of the significant set is fragile
out-of-sample. The headline 12.6% is not "the method finds nothing durable" — it
is "only the strongest signals are durable, and the long tail of marginally-
significant edges is where the in-sample fragility lives."

## Top 10 discovery edges — did they hold out-of-sample?

| Relationship | Discovery corrected p | Holdout corrected p | Type | Replicated? |
|---|---|---|---|---|
| S&P 500 → Nifty 50 | 4.14e-37 | 1.47e-04 | Granger-only | ✅ replicated |
| 10Y Treasury Yield → USD/JPY | 1.20e-35 | 3.24e-25 | Granger-only | ✅ replicated |
| Nasdaq → Nifty 50 | 3.62e-31 | 7.15e-04 | Granger-only | ✅ replicated |
| Gold → USD/JPY | 2.04e-30 | 4.31e-04 | Granger-only | ✅ replicated |
| Financials Sector ETF → Nifty 50 | 9.72e-30 | 6.22e-04 | Granger-only | ✅ replicated |
| S&P 500 → USD/INR | 2.19e-26 | 4.32e-02 | Granger-only | ✅ replicated |
| Gold → EUR/USD | 1.29e-24 | 7.86e-07 | Granger-only | ✅ replicated |
| Nasdaq → USD/INR | 1.09e-23 | 8.08e-02 | Granger-only | ❌ did not |
| US Global Jets (Airlines) ETF → Nifty 50 | 3.53e-23 | 2.54e-04 | PC | ✅ replicated |
| Financials Sector ETF → USD/INR | 1.02e-22 | 6.14e-02 | Granger-only | ❌ did not |

## Artifacts

- `replication.csv` — every ordered pair with discovery + holdout stats, in_graph
  flags for both windows, and the replicated verdict.
- `replication_summary.md` — this file.
