# Deployment — Live read-only demo (Layer 1 + dashboard)

This document covers deploying a **live, read-only public demo** of the
Cross-Asset Causal Discovery Engine: the FastAPI service (Layer 1 findings +
the already-recorded Layer 2 cards) plus the Streamlit dashboard. It is a
**demo of a pre-recorded run**, not a re-run-everything deployment.

> The analysis logic is unchanged. Everything here is deployment plumbing.

---

## 1. What "demo mode" is, and why

The intended host (e.g. **Render free tier**) has **no GPU, ~512 MB RAM, no
Ollama instance, and an ephemeral filesystem** that is not guaranteed to persist
across restarts. None of the heavy, stateful work the engine does locally can
run there:

- `POST /analyze` runs the full network-bound Layer-1 pipeline (yfinance fetch →
  Granger → PC → HMM) and **writes** a new run to the database.
- `POST /runs/{id}/validate` needs a **local Ollama** model (`llama3.1:8b-instruct-q4_0`)
  to generate Layer-2 hypothesis cards — there is no LLM on the host.
- `POST /monitor` re-runs the pipeline and **writes** flip-tracking state.

So the demo runs in **`DEMO_MODE`**: a read-only snapshot server.

- It serves the **committed** database `db/causal_engine.db` — the recorded run
  **`run_20260619_133653_683874bb`** (156 candidates, 106 significant, 20
  in-graph edges, 2143 regime windows, **106 LLM hypothesis cards** already
  generated locally: 98 likely-spurious / 3 known-mechanism / 5 novel).
- `db/causal_engine.db` is committed to git **on purpose** (a `.gitignore`
  exception; ~1.3 MB) so a fresh clone has data to serve. This is *not* a policy
  change to track databases generally — only this one snapshot is whitelisted.
- The three pipeline/LLM/monitor endpoints return a clear **`503`** with an
  explanatory message instead of hanging or erroring.
- `GET /llm/health` returns `ollama_available: false` **immediately** (no
  connection attempt, no hang). Outside demo mode the probe still fast-fails on
  a 2 s timeout.
- Every **GET** endpoint (`/runs`, `/runs/{id}/candidates|graph|regimes|cards`,
  `/flips`) works normally against the snapshot.
- Because nothing is ever written, the ephemeral filesystem losing the DB on
  restart is harmless — the snapshot is restored from the repo on each deploy.

`DEMO_MODE` is read from an environment variable in `config.py` and defaults to
**`False`**, so a normal local run is completely unaffected.

### NOT available in the hosted demo (run locally instead)

> **Layer 2 LLM card *generation* and live regime-flip monitoring are NOT
> available in this hosted demo.** They require a local Ollama instance and a
> writable pipeline. The demo shows the *already-generated* 106 cards
> read-only; to generate fresh cards, run a new analysis, or run a monitor
> cycle, **clone the repo and run it locally** (see the main `README.md`).

---

## 2. Render — FastAPI service

Repository layout note: the engine lives in the subdirectory
`Cross-Asset Causal Discovery Engine/`, which is why `render.yaml` sets
`rootDir` accordingly.

### Option A — Blueprint (`render.yaml`, already in the repo)

```yaml
services:
  - type: web
    name: causal-discovery-api
    runtime: python
    rootDir: Cross-Asset Causal Discovery Engine
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn api.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: DEMO_MODE
        value: "true"
      - key: CAUSAL_ENGINE_DB
        value: db/causal_engine.db
```

Render → **New → Blueprint** → point at this repo. The blueprint creates the
service with the env vars below already set.

### Option B — Manual web service

| Field | Value |
|-------|-------|
| Root directory | `Cross-Asset Causal Discovery Engine` |
| Build command | `pip install -r requirements.txt` |
| Start command | `uvicorn api.main:app --host 0.0.0.0 --port $PORT` |
| Environment | Python |

**Environment variables:**

| Key | Value | Why |
|-----|-------|-----|
| `DEMO_MODE` | `true` | Read-only snapshot mode (gates analyze/validate/monitor). |
| `CAUSAL_ENGINE_DB` | `db/causal_engine.db` | Serve the committed recorded run. |
| `PYTHON_VERSION` | `3.11.9` | Match the pinned wheels (built/verified on 3.11). |

Notes:

- **Port:** the app never hardcodes a port. `uvicorn ... --port $PORT` binds
  Render's injected `$PORT`. Do not add a port anywhere in `api/main.py`.
- **Python version:** `requirements.txt` is pinned (`==`) to the versions
  verified on Python **3.11.9**. Pin Render to 3.11 (via `PYTHON_VERSION` or a
  `runtime.txt` containing `3.11.9`) so the same wheels resolve.
- **Dependencies install cleanly on Linux:** no Windows-only or GPU-specific
  packages. `causal-learn`, `hmmlearn`, `polars`, `statsmodels`, `scipy`,
  `numpy`, `matplotlib` all ship manylinux wheels.
- **No system Graphviz needed (verified by tracing the call path, not inferred
  from the pip dependency tree):** `causal-learn` transitively installs the
  pure-Python `graphviz`/`pydot` bindings, but nothing ever invokes the system
  `dot` executable. The live request paths for `GET /graph` and `GET
  /candidates` are raw SQLite `SELECT`s (`db/storage.py::load_graph` /
  `load_candidates`) → Pydantic — no graph library is on the path at all. The PC
  algorithm (`pc()` in `causal/graph_discovery.py`) runs **only** inside
  `/analyze`, which is 503-gated in demo mode, so it is never even reached on the
  host; and even it builds an in-memory NetworkX graph with no `to_pydot`/render
  call. The offline `graph.png` (a static `results/` artifact, regenerated only
  by `scripts/record_validation_run.py`, never by the API) uses
  `nx.spring_layout` + matplotlib, **not** `graphviz_layout`. So the build needs
  no apt packages / Aptfile.
- **Memory (~512 MB):** the API imports the full scientific stack at startup
  (it shares modules with the pipeline) even though demo mode never *runs* it.
  This fits in 512 MB for read-only serving but is not generous; if the free
  tier OOMs at boot, bump to the next instance size — no code change needed.
- **Health check path:** `/health` (returns `{"status":"ok","database":true,"demo_mode":true}`).

Verify after deploy:

```bash
curl https://<your-api>.onrender.com/health
# {"status":"ok","database":true,"demo_mode":true}
curl -X POST https://<your-api>.onrender.com/analyze
# 503 {"detail":"This endpoint requires a local Ollama instance ..."}
curl https://<your-api>.onrender.com/runs/run_20260619_133653_683874bb/cards | head
# 106 cards served read-only
```

---

## 3. Streamlit Cloud — dashboard

The dashboard is a **thin HTTP client**; it never imports the pipeline. Point it
at the deployed API.

| Field | Value |
|-------|-------|
| Repository | this repo |
| Main file path | `Cross-Asset Causal Discovery Engine/dashboard/app.py` |
| Python version | 3.11 |

**Environment variables / secrets:**

| Key | Value | Why |
|-----|-------|-----|
| `API_BASE_URL` | `https://<your-api>.onrender.com` | Point the dashboard at the Render API. |
| `DEMO_MODE` | `true` *(optional)* | Fallback banner if the API is briefly offline. |

The dashboard detects demo mode primarily from the API's `/health`
(`demo_mode: true`), so it shows the read-only state correctly even though it
runs on a different host. `DEMO_MODE` on the Streamlit side is only a fallback
for when the API can't be reached. In demo mode the dashboard:

- shows a top banner: *"Live read-only demo — showing a pre-recorded 8-year
  analysis. The local-LLM layer and live monitoring require running this locally
  (see README)."*;
- replaces the **"Run a new analysis"** sidebar form with a note;
- disables Layer-2 **card generation** (the pre-recorded cards still render in
  the **Hypothesis cards** tab);
- marks the **Events** tab read-only (the recorded analysis run has no flips of
  its own; flips are diffs *between* monitor runs, which the demo does not run).

> The first request to a free-tier Render service after idle can take ~30–60 s
> to cold-start. The dashboard surfaces a clean "cannot reach the API" state
> rather than crashing if that times out — just retry.

---

## 4. Summary of what changed for deployment

| File | Change |
|------|--------|
| `config.py` | `DEMO_MODE` flag (env-driven, default `False`) + `_env_flag` helper. |
| `api/main.py` | `503` guard on `/analyze`, `/runs/{id}/validate`, `/monitor`; `/llm/health` fast-returns unavailable in demo mode; `/health` reports `demo_mode`. |
| `llm/validator.py` | `ollama_available` fast-fails on a 2 s timeout. |
| `dashboard/app.py` | Demo banner + read-only degradation of run-control, Layer-2 generation, and the Events tab. |
| `.gitignore` | Whitelist exception so `db/causal_engine.db` (the demo snapshot) is committed. |
| `render.yaml` | `DEMO_MODE=true` env var. |
| `tests/test_api.py` | Demo-mode test (writes → 503, reads still serve the snapshot). |

Full suite: **63 passed** (was 62; +1 demo-mode test). Run with the project venv:
`venv/Scripts/python -m pytest -q`.
