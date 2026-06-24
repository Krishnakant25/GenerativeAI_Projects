# Cross-Asset Causal Discovery Engine — Final Summary

*The elevator-pitch version. Full detail lives in [`../README.md`](../README.md);
every number below is reproduced from the recorded run
`run_20260619_133653_683874bb` and the artifacts in this `results/` directory.*

---

## 1. What was built

A hybrid **statistical + local-LLM** engine that surfaces *candidate* causal
relationships across a fixed 13-instrument cross-asset universe (commodities,
currencies, equity indices, rates, sector ETFs). **Layer 1** runs the full
statistical pipeline — pairwise Granger causality across all 156 ordered pairs →
FDR (Benjamini-Hochberg) correction → lead-lag cross-correlation → PC directed-graph
discovery → HMM regime detection — in Polars, persisted to SQLite, served by FastAPI
and explored in a thin-client Streamlit dashboard. **Layer 2** sends each surviving
candidate to a **local** Ollama model (`llama3.1:8b-instruct-q4_0`) that explains the
economic mechanism and rates plausibility into a Pydantic-validated `HypothesisCard`
— a heuristic triage filter, never validation, with every card still carrying its
corrected p-value. **Phase 3** adds scheduled regime-flip monitoring (three named
safeguards) and an out-of-sample walk-forward replication study. The whole system
runs entirely offline — no cloud APIs, no per-token cost, no data leaving the machine.

## 2. The four self-skepticism findings

The project's thesis is that **statistical significance, predictive durability, and
economic plausibility are three different things** — and it tests that thesis against
itself four independent ways:

1. **Stationarity — confirmed.** All **13/13** return series are stationary at the 5%
   ADF level (worst is the 10Y yield `^TNX`, ADF p = 1.27e-12). The extreme Granger
   p-values are **not** an artifact of non-stationary inputs. *(`stationarity.csv`)*
2. **Spurious-rationalization control — passed, with caveat.** Fed a fabricated
   candidate (USD/INR "Granger-causes" US Natural Gas, invented p-value), the model
   correctly returned `LIKELY_SPURIOUS` (conf 0.50) rather than rubber-stamping it.
   Caveat: this local model is **markedly conservative** (98/106 flagged spurious) and
   almost certainly under-credits some real channels (e.g. S&P 500 → Nifty 50). *(`hypothesis_cards.csv`)*
3. **Mechanism hallucination — found, diagnosed, fixed.** The first run pasted a
   textbook *"oil → input costs → airline margins"* channel onto **XLF → Crude Oil** —
   a different pair entirely; the defect was systemic (**8 of 10** plausible channels
   changed on re-validation). Fixed at two layers (prompt binds the channel to both
   real asset names + a structural `MECHANISM_MISMATCH` validator backstop); final
   mismatch count **0**. *(`cards_summary.md`)*
4. **Out-of-sample replication — 12.6%, the headline result.** Of 103
   discovery-significant edges (2018–2023), only **13 replicated** in the 2024–mid-2026
   holdout: **12.6%** overall (87.4% non-replication vs nominal α = 0.05), with
   durability concentrated in the strongest macro edges (top-20 band replicates 50%).
   The four durable macro-core edges (S&P→Nifty, 10Y→USD/JPY, Gold→USD/JPY, Nasdaq→Nifty)
   were **all flagged `LIKELY_SPURIOUS` by Layer 2 and all rejected as direct edges by
   PC (`in_graph=false`)** — a coherent two-layer *disagreement* that is exactly what
   "real but confounded" looks like. *(`replication_summary.md`, `replication.csv`)*

## 3. Final state — tests and live system

- **Test suite: 62 passed** (`pytest -v`, full output in [`test_output.txt`](test_output.txt)),
  including the Ollama-gated live spurious-control test.
- **Live system verified end-to-end (cold start):** Ollama up and the model responds;
  API `/health` → `{"status":"ok","database":true}` and `/llm/health` reports the model
  available; the Streamlit dashboard serves (HTTP 200). Loading
  `run_20260619_133653_683874bb` renders every tab from real data — causal graph
  (20 edges / 13 nodes), regime timelines (106 pairs), hypothesis-card feed
  (**106 cards, 98 spurious / 3 known / 5 novel**), and business use-case panel. The
  **Events tab is correctly empty** for this analysis run (its flips live in the
  dedicated `monitor_validation.db`, which serves **38 events — 10 confirmed, 28 pending,
  0 reverted** — exactly as the README discloses).

## 4. What was deliberately NOT built (and why)

Phase 3 offered a menu of extensions; Options **1** (scheduled regime-flip monitoring)
and **5** (out-of-sample replication) were built. Two others were **deliberately
deferred** after the planning pass:

- **Option 3 — a further Layer-2 stress test** (e.g. does `llm_confidence` calibrate
  against Option-5 replication or PC's `in_graph` decision?). *Not built:* confidence
  values cluster tightly (mostly 0.6–0.7) and attach almost entirely to one flag
  (98/106 spurious), while replication is rare (13/103) — the contingency table would
  be too sparse to support any real conclusion, with high risk of a low-signal null
  that merely restates the durable-edge cross-reference already done. A real open
  question, but the thesis is already demonstrated four times over.
- **Option 4 — docker-compose packaging** (API + dashboard + Ollama, one-command
  reproducibility). *Not built:* zero analytical content — pure developer-experience
  polish that discovers and tests nothing; no containerization precedent in the repo;
  Ollama's ~4.7 GB model plus Windows GPU passthrough is a known time sink; and it adds
  a permanent maintenance surface to a project whose reproducibility story (pinned
  `requirements.txt` + `requirements.lock.txt` + closed date windows) is already strong.

The recommendation taken instead was a **documentation polish pass (completed) then
stop** — the project already carries four independent self-skepticism findings, and the
binding constraint had become *making the existing findings cohere*, not finding a fifth.
