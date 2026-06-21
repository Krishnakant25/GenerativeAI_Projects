## Project: Cross-Asset Causal Discovery Engine
### Project Initiation Document — PHASE 1 (NO LLM / NO OLLAMA DEPENDENCY)
### PREMIER RESUME PROJECT

---

## How This Differs From the Original Doc

This is the same project, same architecture, same end-state. It's been split
into two phases so the entire statistical engine, API, database, and
dashboard (minus one panel) can be built and fully tested right now, with
zero dependency on Ollama or any local model being downloaded.

- **Phase 1 (this document):** Everything that does not call an LLM.
  Steps 1–7 of the original build order, plus Layer 3's static and
  interactive dashboard views, plus API endpoints that don't touch the
  validator. This is buildable and demoable today.
- **Phase 2 (deferred, separate prompt):** Layer 2 (LLM plausibility
  validation, hypothesis card generation) and the dashboard's hypothesis
  card feed + business use-case panel, which depend on hypothesis cards
  existing. Pick this up once llama3.1:8b-instruct-q4_0 is pulled.

Nothing architectural has been removed — Layer 2 is deferred, not deleted.
The Pydantic schema for `HypothesisCard` should still be stubbed in Phase 1
(see Step 3) so Phase 2 has a stable contract to write against, but no LLM
calls happen until Phase 2.

---

## What This Project Is
A hybrid statistical + LLM system that surfaces candidate causal
relationships between cross-asset financial instruments (commodities,
currencies, equity indices, rates). Statistical methods detect candidate
causal structure; a local LLM (Phase 2) explains the economic mechanism
behind each finding, checks plausibility, and flags confidence — producing
causal hypothesis cards, not bare numbers.

This is the flagship project for an AI Engineer resume — build it to
production quality, not demo quality. Code clarity, documentation, and
correctness all matter more here than in other portfolio projects.

Built entirely offline. No cloud APIs, zero per-token cost, zero data
leaving the machine (Phase 2 will use Ollama; Phase 1 has no LLM calls at all).

---

## CRITICAL FRAMING — must be reflected in README, UI copy, and code comments
This system does NOT claim to discover "true causality." Granger causality is
predictive precedence, not proof of causation. The LLM's plausibility check
(Phase 2) will be a heuristic filter, not validation. All outputs must be
framed as "candidate causal hypotheses for human review," never as
definitive causal claims.

This honesty is a deliberate design choice and a key talking point — it
demonstrates statistical maturity that most candidates lack. Document it
clearly and prominently, not buried in a footnote. This applies to Phase 1
output too: even without the LLM layer, the README and dashboard copy around
the statistical findings must carry this framing from day one.

---

## Business Case (this is the "why does this matter" section for README)

This is a research-acceleration tool, not a trading signal generator.
It automates the first-pass statistical screening a junior quant analyst
would otherwise do manually. In Phase 2 it will also package output with
economic context so a human researcher can decide what's worth investigating
further — but the statistical screening itself is the Phase 1 deliverable
and already stands on its own.

Four concrete use cases to document in README with examples from the actual
output once built:

1. Risk management / hedging timing — if oil price moves statistically
   precede airline stock moves by N days, a risk desk can time hedges
   proactively rather than react after the fact. Lead-lag relationships are
   tradeable information even without proof of causation.

2. Hidden concentration risk in portfolios — reveals when "diversified"
   holdings are actually all driven by the same underlying shock
   (e.g., 3 holdings all sensitive to oil price moves), which standard
   correlation matrices often obscure.

3. Macro narrative validation for research desks — analysts often assume a
   causal story (e.g. "rate hikes are hurting financials") without testing
   it. This gives a fast, rigorous first-pass check before a research note
   is written.

4. Regime-change early warning — flags WHEN a previously reliable
   relationship breaks down, which is exactly what risk models need to catch
   before they fail (correlations breaking down was a core failure mode in
   2008 and 2020).

Known limitations to state explicitly in README (this is a strength, not a
weakness, when documented):
- Multiple-comparisons problem when testing many asset pairs simultaneously —
  apply FDR or Bonferroni correction, document that you did
- LLM plausibility checks can rationalize spurious findings convincingly
  (Phase 2 caveat — name this risk now in the README's "known limitations"
  section even before Phase 2 is built, so it's documented as a design
  decision rather than discovered later by an interviewer)
- This is not a trading system and must never be framed as one

---

## Tech Stack — Phase 1 (no Ollama)
- Python 3.11+
- statsmodels → Granger causality testing
- causal-learn → causal graph discovery (PC algorithm)
- hmmlearn → regime change detection
- yfinance → market data (equities, commodities, currency pairs)
- Polars → data processing (NOT pandas)
- NetworkX → causal graph data structure
- FastAPI → REST API
- Streamlit + streamlit-agraph (or pyvis) → interactive graph dashboard
- Pydantic → all data validation
- SQLite → store analysis runs and hypothesis cards (table created now,
  populated in Phase 2)

Deferred to Phase 2: Ollama, llama3.1:8b-instruct-q4_0, anything in `llm/`.

I'm not going to pin exact version numbers for causal-learn or hmmlearn in
requirements.txt here — I'm not confident those numbers would be current or
accurate, and a wrong pin is worse than no pin. Install latest, run
`pip freeze`, and lock versions yourself once it's working on your machine.

NOTE: causal-learn and hmmlearn APIs may have changed since my training data
cutoff. Before using either library, inspect the actually installed
version's API via `pip show` + reading source/docstrings rather than
assuming a remembered interface is current. I'd flag this even if you hadn't
asked — it's a real risk with smaller, faster-moving libraries like these.

---

## Fixed Asset Universe (12 assets)
- Commodities: Crude Oil (CL=F), Gold (GC=F), Natural Gas (NG=F)
- Currencies: EUR/USD, USD/JPY, USD/INR
- Equity Indices: S&P 500 (^GSPC), Nasdaq (^IXIC), Nifty 50 (^NSEI)
- Bonds/Rates: 10Y Treasury Yield (^TNX)
- Sector ETFs: Energy (XLE), Financials (XLF), Airlines (JETS)

Verify all tickers are currently valid on yfinance before building — ticker
symbols and availability change over time and I can't confirm current
validity from here. Your Step 1 task list below includes this check
explicitly.

---

## Architecture — Phase 1 Scope

### Layer 1 — Statistical Causal Discovery (full build now)
1. Pairwise Granger causality tests across all asset pairs (both directions)
2. Apply multiple-comparisons correction (FDR or Bonferroni) since testing
   12 assets pairwise creates many simultaneous tests
3. PC algorithm to build a directed causal graph across the full asset set
4. Time-lagged cross-correlation to detect lead-lag relationships
5. HMM-based regime detection — flag WHEN a relationship started/stopped
   holding, not just whether it exists overall

Output: `CausalCandidate` objects — asset_a, asset_b, direction, lag,
granger_p_value, corrected_p_value, correlation_strength, regime_periods,
statistical_confidence

### Layer 2 — LLM Validation & Explanation — **DEFERRED TO PHASE 2**
Stub only in Phase 1: define the `HypothesisCard` Pydantic model and the
`llm/` directory with empty/placeholder modules so the import structure
exists, but write no prompts and make no Ollama calls yet.

### Layer 3 — Dashboard (Phase 1 builds everything except hypothesis-card-dependent views)
Static views — **build in Phase 1**:
- Full causal graph (all assets, all significant edges, corrected p-values shown)
- Regime timeline (when relationships were active/inactive)

Static views — **deferred to Phase 2** (depend on hypothesis cards):
- Hypothesis card feed (sorted by confidence)
- Business use-case panel

Interactive exploration — **build in Phase 1**:
- Click any asset node → see all its causal in/out edges
- Click any edge → see the full statistical evidence and regime history for
  that specific relationship (the "full hypothesis card" portion of this
  view is added in Phase 2 once cards exist — for now the edge click shows
  raw statistics only)
- Filter by: minimum confidence, asset class (the "plausibility flag" filter
  is added in Phase 2)

---

## Project Structure
```
causal-discovery-engine/
├── data/
│   ├── fetcher.py
│   └── preprocessor.py
├── causal/
│   ├── granger.py
│   ├── graph_discovery.py
│   ├── regime_detection.py
│   ├── correction.py        # FDR/Bonferroni multiple-comparisons correction
│   └── models.py
├── llm/                      # PHASE 2 — stub only, no logic yet
│   ├── validator.py          # empty placeholder
│   ├── prompts.py            # empty placeholder
│   └── models.py             # HypothesisCard Pydantic model lives here now
├── api/
│   └── main.py               # endpoints that don't depend on hypothesis cards
├── dashboard/
│   └── app.py                # static + interactive views, minus card feed/use-case panel
├── db/
│   └── storage.py            # full schema created now, hypothesis_cards table populated in Phase 2
├── tests/
│   ├── test_granger.py       # validate against known textbook relationships
│   └── test_correction.py
├── requirements.txt
└── README.md                 # full business case + honesty framing required
```

---

## Build Order — Phase 1 (7 Steps)
1. Project scaffolding — directory structure, requirements.txt, SQLite schema
   (including the `hypothesis_cards` table definition, even though it stays
   empty until Phase 2)
2. `data/fetcher.py` + `preprocessor.py` — pull and clean all 12 assets
3. `causal/models.py` — all Pydantic dataclasses, including a stub
   `HypothesisCard` model in `llm/models.py` so Phase 2 has a stable
   contract to build against
4. `causal/granger.py` — pairwise Granger tests, validated against a known
   textbook example (oil → airlines) before trusting on full dataset
5. `causal/correction.py` — multiple-comparisons correction
6. `causal/graph_discovery.py` — PC algorithm, build NetworkX graph
7. `causal/regime_detection.py` — HMM regime detection per asset pair

Then:
8. `api/main.py` — all endpoints that serve Layer 1 output (causal
   candidates, graph data, regime timelines). Skip/stub any endpoint that
   would serve hypothesis cards.
9. `dashboard/app.py` — static graph view, regime timeline, interactive
   node/edge exploration on raw statistics. Skip the hypothesis card feed
   and business use-case panel.
10. `README.md` — architecture, business case (use cases can reference
    Layer 1 findings; note Phase 2 will add LLM-generated mechanism
    explanations), honesty/limitations section, demo instructions, and an
    explicit "Current Status: Phase 1 complete, Phase 2 (LLM layer) pending
    model download" note so anyone reading it understands the project's
    real state.

**Stop after Step 10 and review before considering Phase 2.**

---

## Hard Rules — Phase 1
- No pandas — Polars only
- Every statistical claim must include its corrected p-value or confidence
  interval alongside it — never show a causal arrow without the underlying
  statistic
- README must explicitly state the Granger-causality-is-not-causation caveat
  AND the multiple-comparisons caveat AND the "not a trading system" caveat,
  even before Phase 2 exists
- Regime detection results must show time-bound validity, not permanent claims
- Use Pydantic models for all data validation
- This is the premier resume project — prioritize code clarity and
  documentation quality over speed of completion
- Do not write any Ollama client code, async or otherwise, in Phase 1 — keep
  `llm/` as inert stubs only, so there's no half-finished LLM integration
  sitting in the codebase while Phase 2 is pending

---

## Your First Task
1. Verify all 12 yfinance tickers are currently valid (some may need
   updating) — I can't confirm this from my end, so this needs an actual
   check against yfinance on your machine before you build on top of it.
2. Start with Step 1 — project scaffolding:
   - Create full directory structure (including the inert `llm/` stubs)
   - Write requirements.txt with Phase 1 dependencies only (no `ollama`
     package); leave versions unpinned until you've confirmed what
     installs cleanly, then pin via `pip freeze`
   - Initialize SQLite schema (analysis_runs, causal_candidates,
     hypothesis_cards — the last one created but unused until Phase 2)
   - Write barebones README.md with business case, honesty framing, and a
     "Current Status: Phase 1" section drafted (even if findings sections
     are placeholder until analysis is run)
3. After Step 1, stop and show the structure before proceeding to Step 2

---

## Picking Up Phase 2 Later

When llama3.1:8b-instruct-q4_0 is pulled and Ollama is running, the original
project doc's Layer 2 section (LLM validation & explanation), Step 8 (now
renumbered) for `llm/validator.py` + `prompts.py`, and the deferred dashboard
panels are the remaining scope. At that point also re-verify:
- Ollama is running and the model is actually available locally
- The `causal-learn` / `hmmlearn` API surfaces you built against in Phase 1
  haven't drifted if time has passed and you've updated dependencies
