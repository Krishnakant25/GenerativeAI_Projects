"""FastAPI surface for the Cross-Asset Causal Discovery Engine — Layers 1 & 2.

Serves the statistical findings (Layer 1: causal *candidates*, the discovered
causal graph, time-bound regime windows) and the LLM plausibility layer
(Layer 2: ``POST /runs/{id}/validate`` to generate hypothesis cards,
``GET /runs/{id}/cards`` to read them, ``GET /llm/health`` for Ollama liveness).

Honesty framing (a project hard rule, enforced at the API boundary):
  * Every response that carries a causal edge also carries the statistic that
    justifies it — corrected p-value, lag and (where available) correlation.
    There is no endpoint that returns a bare directional arrow.
  * Outputs are "candidate causal hypotheses for human review", never proven
    causal claims. Granger causality is predictive precedence, not causation.

Tested against FastAPI 0.137.x / Pydantic 2.13.x (lifespan startup, typed
``response_model`` lists, ``Depends`` injection).

Run locally:
    uvicorn api.main:app --reload
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

import config
from causal.models import (
    AnalysisRun,
    AnalyzeRequest,
    CausalCandidate,
    CausalGraph,
    PairRegimes,
    RegimeFlipEvent,
)
from causal.pipeline import InsufficientDataError, run_analysis
from data.fetcher import DataUnavailableError
from db import storage
from llm.models import HypothesisCard
from scripts.run_monitor import run_monitor_cycle
from llm.validator import (
    DEFAULT_MODEL_NAME,
    OllamaUnreachableError,
    OllamaValidator,
    ollama_available,
    summarize_flags,
)

logger = logging.getLogger("causal_engine.api")

API_DESCRIPTION = (
    "Surfaces **candidate** causal relationships across a fixed cross-asset "
    "universe (commodities, FX, equity indices, rates, sector ETFs).\n\n"
    "**This is a research-screening tool, not a trading system and not a proof "
    "of causation.** Granger causality measures predictive precedence; every "
    "edge is reported with its corrected p-value and is a hypothesis for human "
    "review. Multiple-comparisons correction is applied across all pairs.\n\n"
    "**Layer 2 (LLM):** the `/validate` and `/cards` endpoints attach a local "
    "LLM's economic-mechanism explanation and a *plausibility* flag to findings "
    "that already passed significance + correction. That plausibility check is a "
    "**heuristic filter, not validation** — an LLM can rationalise a spurious "
    "relationship — and it never upgrades a finding to 'causal'. Every card "
    "still carries the corrected statistic it explains."
)


class ValidateSummary(BaseModel):
    """Result of a Layer-2 validation pass over a run's significant candidates."""

    run_id: str
    model_name: str = Field(..., description="Local LLM that generated the cards.")
    n_candidates_validated: int
    counts: dict[str, int] = Field(
        ..., description="Card count per plausibility_flag (incl. parse_failed)."
    )


class LLMHealth(BaseModel):
    """Liveness of the local LLM backing Layer 2."""

    ollama_available: bool
    model_name: str


class MonitorSummary(BaseModel):
    """Result of one on-demand monitor cycle (Phase 3).

    A cycle persists a fresh run and advances regime-flip tracking against the
    prior run. ``confirmed`` flips are the real signal; ``still_pending`` flips
    are provisional — they have not yet survived the confirmation window — and
    ``reverted`` flips snapped back (rejected as data-revision/transient)."""

    run_id: str = Field(..., description="The run this cycle persisted.")
    window: str = Field(..., description="Data window analysed (start..end).")
    n_significant_candidates: int = Field(
        ..., description="Significant-after-correction candidates in this run."
    )
    n_flips_touched: int = Field(
        ..., description="Flip events created or advanced this cycle."
    )
    new_flips: int = Field(..., description="Flips first detected this cycle.")
    confirmed: int = Field(
        ..., description="Flips that reached confirmation this cycle (real signal)."
    )
    reverted: int = Field(..., description="Flips rejected (snapped back) this cycle.")
    still_pending: int = Field(
        ..., description="Flips still inside the confirmation window (provisional)."
    )
    flips: list[RegimeFlipEvent] = Field(
        default_factory=list,
        description="The touched flip events, each carrying its corrected p-value.",
    )


def get_db_path() -> Path:
    """Resolve the SQLite path. Overridable via ``CAUSAL_ENGINE_DB`` so the API
    and the test suite can point at a throwaway database."""
    return Path(os.environ.get("CAUSAL_ENGINE_DB", str(config.DB_PATH)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure the schema exists before the first request.
    path = storage.init_db(get_db_path())
    logger.info("SQLite schema ready at %s", path)
    yield


app = FastAPI(
    title="Cross-Asset Causal Discovery Engine",
    version="2.0.0",
    description=API_DESCRIPTION,
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# Liveness
# --------------------------------------------------------------------------


@app.get("/health", tags=["meta"])
def health(db_path: Path = Depends(get_db_path)) -> dict:
    """Liveness + DB-reachability check."""
    try:
        conn = storage.get_connection(db_path)
        try:
            conn.execute("SELECT 1").fetchone()
        finally:
            conn.close()
        db_ok = True
    except Exception:  # noqa: BLE001 - report degraded rather than crash
        logger.exception("Health check could not reach the database")
        db_ok = False
    return {"status": "ok" if db_ok else "degraded", "database": db_ok}


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


@app.post("/analyze", response_model=AnalysisRun, tags=["analysis"])
def analyze(
    request: AnalyzeRequest | None = None,
    db_path: Path = Depends(get_db_path),
) -> AnalysisRun:
    """Run the full Layer-1 pipeline over the asset universe and persist it.

    fetch -> preprocess -> Granger -> correction -> PC graph -> regimes.
    Returns the run metadata (including ``run_id``); fetch the findings via the
    ``/runs/{run_id}/...`` endpoints.
    """
    req = request or AnalyzeRequest()

    # Cheap up-front validation so an obviously bad window fails as 400, not 500.
    try:
        start = date.fromisoformat(req.start_date)
        end = date.fromisoformat(req.end_date) if req.end_date else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid date: {exc}") from exc
    if end is not None and end <= start:
        raise HTTPException(
            status_code=400, detail="end_date must be after start_date"
        )

    try:
        result = run_analysis(
            start_date=req.start_date,
            end_date=req.end_date,
            max_lag=req.max_lag,
            alpha=req.alpha,
            correction_method=req.correction_method,
            notes=req.notes,
        )
    except DataUnavailableError as exc:
        # Upstream market-data provider returned nothing usable.
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except InsufficientDataError as exc:
        # Data came back but is too short for Granger / PC / HMM.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - never leak a raw traceback
        logger.exception("Pipeline failed for request %s", req.model_dump())
        raise HTTPException(
            status_code=500, detail=f"Analysis failed: {exc}"
        ) from exc

    storage.persist_run(
        result.run, result.candidates, result.graph_meta, db_path=db_path
    )
    return result.run


# --------------------------------------------------------------------------
# Run retrieval
# --------------------------------------------------------------------------


def _require_run(run_id: str, db_path: Path) -> AnalysisRun:
    run = storage.load_run(run_id, db_path=db_path)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    return run


@app.get("/runs", response_model=list[AnalysisRun], tags=["runs"])
def list_runs_endpoint(db_path: Path = Depends(get_db_path)) -> list[AnalysisRun]:
    """All persisted runs, most recent first (by created_at).

    Returns the same metadata shape as ``GET /runs/{run_id}`` so clients can
    build a run picker without needing to know any run_id in advance.
    """
    return list(reversed(storage.list_runs(db_path=db_path)))


@app.get("/runs/{run_id}", response_model=AnalysisRun, tags=["runs"])
def get_run(run_id: str, db_path: Path = Depends(get_db_path)) -> AnalysisRun:
    """Run metadata (window, universe, correction method, alpha)."""
    return _require_run(run_id, db_path)


@app.get(
    "/runs/{run_id}/candidates",
    response_model=list[CausalCandidate],
    tags=["runs"],
)
def get_candidates(
    run_id: str,
    significant_only: bool = Query(
        default=False,
        description="Return only edges that passed the corrected alpha threshold.",
    ),
    db_path: Path = Depends(get_db_path),
) -> list[CausalCandidate]:
    """Corrected causal candidates with full statistics, most significant first.

    Each candidate carries both its raw and corrected p-value — a directional
    claim is never returned without its supporting statistic.
    """
    _require_run(run_id, db_path)
    return storage.load_candidates(
        run_id, significant_only=significant_only, db_path=db_path
    )


@app.get("/runs/{run_id}/graph", response_model=CausalGraph, tags=["runs"])
def get_graph(run_id: str, db_path: Path = Depends(get_db_path)) -> CausalGraph:
    """The discovered causal graph (node-link). Every edge carries its
    corrected p-value, lag, correlation and how it was oriented."""
    graph = storage.load_graph(run_id, db_path=db_path)
    if graph is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    return graph


@app.get(
    "/runs/{run_id}/regimes",
    response_model=list[PairRegimes],
    tags=["runs"],
)
def get_regimes(
    run_id: str, db_path: Path = Depends(get_db_path)
) -> list[PairRegimes]:
    """Time-bound regime windows per pair (computed for significant edges).

    Validity is always attached to an explicit date range — never asserted as a
    permanent relationship.
    """
    _require_run(run_id, db_path)
    return storage.load_regimes(run_id, db_path=db_path)


# --------------------------------------------------------------------------
# Layer 2 — LLM plausibility / explanation
# --------------------------------------------------------------------------


@app.get("/llm/health", response_model=LLMHealth, tags=["layer2"])
async def llm_health() -> LLMHealth:
    """Is the local Ollama model reachable? Lets clients disable the (slow)
    validate action and show a clear status instead of failing mid-request."""
    available = await ollama_available(model=DEFAULT_MODEL_NAME)
    return LLMHealth(ollama_available=available, model_name=DEFAULT_MODEL_NAME)


@app.post(
    "/runs/{run_id}/validate",
    response_model=ValidateSummary,
    tags=["layer2"],
)
async def validate_run_endpoint(
    run_id: str,
    limit: int | None = Query(
        default=None,
        ge=1,
        description=(
            "Validate only the first N significant candidates (by corrected "
            "p-value). Omit to validate them all — SLOW: a local 8B model takes "
            "~1 min per candidate."
        ),
    ),
    db_path: Path = Depends(get_db_path),
) -> ValidateSummary:
    """Run Layer 2 over a run's surviving (significant) candidates and persist a
    hypothesis card for each. Returns counts per plausibility flag.

    The LLM only *explains* and *rates plausibility* — it never re-tests or
    upgrades a finding to causal, and the underlying statistic passes through
    untouched. Degrades to **503** (clear message) if Ollama is unreachable,
    never a 500 traceback.
    """
    _require_run(run_id, db_path)

    validator = OllamaValidator()
    try:
        cards = await validator.validate_and_persist(
            run_id, db_path=db_path, limit=limit
        )
    except OllamaUnreachableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - never leak a raw traceback
        logger.exception("Layer-2 validation failed for run %s", run_id)
        raise HTTPException(
            status_code=500, detail=f"Validation failed: {exc}"
        ) from exc

    return ValidateSummary(
        run_id=run_id,
        model_name=validator.model,
        n_candidates_validated=len(cards),
        counts=summarize_flags(cards),
    )


@app.get(
    "/runs/{run_id}/cards",
    response_model=list[HypothesisCard],
    tags=["layer2"],
)
def get_cards(
    run_id: str, db_path: Path = Depends(get_db_path)
) -> list[HypothesisCard]:
    """Hypothesis cards for a run, most confident first. Each card embeds the
    full underlying ``candidate`` — per the project's hard rule, no card is ever
    returned without the corrected statistic that justifies it."""
    _require_run(run_id, db_path)
    return storage.load_hypothesis_cards(run_id=run_id, db_path=db_path)


# --------------------------------------------------------------------------
# Phase 3 — scheduled regime-flip monitoring
# --------------------------------------------------------------------------


@app.get("/flips", response_model=list[RegimeFlipEvent], tags=["monitor"])
def get_flips(
    status: str | None = Query(
        default=None,
        description="Filter by lifecycle state: 'pending' (provisional, still "
        "inside the confirmation window), 'confirmed' (survived it — the real "
        "signal), or 'reverted' (snapped back / rejected).",
    ),
    asset_a: str | None = Query(default=None, description="Filter by driver ticker."),
    asset_b: str | None = Query(default=None, description="Filter by affected ticker."),
    db_path: Path = Depends(get_db_path),
) -> list[RegimeFlipEvent]:
    """Detected regime-status flips, most recently detected first.

    A flip is a pair's coupling switching on/off between two monitor runs. Each
    event carries the corrected p-value of its new-run candidate (hard rule: no
    flip without its statistic) and a ``confirmed`` flag — gate "this is a real
    regime change" on ``confirmed``, treat ``pending`` as provisional. Only pairs
    that were significant after correction are ever surfaced (alert-fatigue gate).
    """
    if status is not None and status not in ("pending", "confirmed", "reverted"):
        raise HTTPException(
            status_code=400,
            detail="status must be 'pending', 'confirmed' or 'reverted'",
        )
    return storage.load_flip_events(
        asset_a=asset_a, asset_b=asset_b, status=status, db_path=db_path
    )


@app.post("/monitor", response_model=MonitorSummary, tags=["monitor"])
def run_monitor(
    request: AnalyzeRequest | None = None,
    db_path: Path = Depends(get_db_path),
) -> MonitorSummary:
    """Run ONE on-demand monitor cycle: persist a fresh run, then diff its
    per-pair regime status against the prior run and advance flip tracking.

    This is the same single cycle the Task Scheduler / CLI runs (scheduling still
    belongs to the OS — see ``scripts/run_monitor.py``); the endpoint just lets a
    client trigger one now. SLOW: it runs the full network-bound Layer-1 pipeline
    synchronously. The first ever cycle only establishes a baseline (no prior run
    to diff), so it returns zero flips — that is expected, not an error.
    """
    req = request or AnalyzeRequest()
    try:
        touched = run_monitor_cycle(
            start_date=req.start_date,
            end_date=req.end_date,
            alpha=req.alpha,
            db_path=db_path,
        )
    except DataUnavailableError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except InsufficientDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - never leak a raw traceback
        logger.exception("Monitor cycle failed for request %s", req.model_dump())
        raise HTTPException(
            status_code=500, detail=f"Monitor cycle failed: {exc}"
        ) from exc

    # The cycle just persisted the newest run; identify it the same way the cycle
    # itself does (newest by created_at) rather than re-deriving it.
    runs = storage.list_runs(db_path=db_path)
    latest = runs[-1]
    n_significant = sum(
        1
        for c in storage.load_candidates(
            latest.run_id, significant_only=True, db_path=db_path
        )
    )

    confirmed = sum(1 for e in touched if e.status == "confirmed")
    reverted = sum(1 for e in touched if e.status == "reverted")
    pending = [e for e in touched if e.status == "pending"]
    new_flips = sum(1 for e in pending if e.new_run_id == latest.run_id)

    return MonitorSummary(
        run_id=latest.run_id,
        window=f"{latest.start_date}..{latest.end_date}",
        n_significant_candidates=n_significant,
        n_flips_touched=len(touched),
        new_flips=new_flips,
        confirmed=confirmed,
        reverted=reverted,
        still_pending=len(pending),
        flips=touched,
    )
