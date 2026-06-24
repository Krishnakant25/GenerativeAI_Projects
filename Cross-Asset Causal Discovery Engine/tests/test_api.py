"""API smoke tests using FastAPI's TestClient against a throwaway SQLite DB.

We do NOT run the real (network-bound) pipeline here: GET endpoints are tested
against a hand-inserted mock run, and POST /analyze is tested with the pipeline
monkeypatched to return a canned result, so the endpoint + persistence path are
exercised without touching yfinance.

The hard rule that "no edge is returned without its statistic" is asserted
directly on the candidate and graph payloads.
"""

from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient

from causal.models import (
    AnalysisRun,
    CausalCandidate,
    Direction,
    RegimePeriod,
)


def _mock_run() -> tuple[AnalysisRun, list[CausalCandidate], dict]:
    run = AnalysisRun(
        run_id="run_test",
        created_at="2026-01-01T00:00:00+00:00",
        start_date="2020-01-01",
        end_date="2021-01-01",
        asset_universe=["AAA", "BBB"],
        max_lag=5,
        correction_method="fdr_bh",
        alpha=0.05,
    )
    sig = CausalCandidate(
        candidate_id="run_test:AAA->BBB",
        run_id="run_test",
        asset_a="AAA",
        asset_b="BBB",
        direction=Direction.A_CAUSES_B,
        lag=2,
        granger_p_value=0.001,
        corrected_p_value=0.01,
        correlation_strength=0.6,
        statistical_confidence=0.79,
        is_significant=True,
        regime_periods=[
            RegimePeriod(
                start=date(2020, 1, 1),
                end=date(2020, 6, 1),
                active=True,
                mean_correlation=0.6,
            )
        ],
    )
    nonsig = CausalCandidate(
        candidate_id="run_test:BBB->AAA",
        run_id="run_test",
        asset_a="BBB",
        asset_b="AAA",
        direction=Direction.A_CAUSES_B,
        lag=1,
        granger_p_value=0.4,
        corrected_p_value=0.8,
        correlation_strength=0.05,
        statistical_confidence=0.2,
        is_significant=False,
    )
    graph_meta = {sig.candidate_id: ("directed", "pc")}
    return run, [sig, nonsig], graph_meta


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient bound to a throwaway DB via the CAUSAL_ENGINE_DB env var."""
    db = tmp_path / "api_test.db"
    monkeypatch.setenv("CAUSAL_ENGINE_DB", str(db))

    from db import storage
    import api.main as main

    storage.init_db(db)
    with TestClient(main.app) as c:
        yield c, main, db


def test_health(client):
    c, _, _ = client
    resp = c.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["database"] is True


def test_list_runs(client):
    c, _, db = client
    from db import storage

    # Empty before any run is persisted.
    resp = c.get("/runs")
    assert resp.status_code == 200
    assert resp.json() == []

    # After persisting two runs, both appear most-recent-first.
    run1, cands1, meta1 = _mock_run()
    run2 = run1.model_copy(update={
        "run_id": "run_test_2",
        "created_at": "2026-06-01T00:00:00+00:00",
    })
    storage.persist_run(run1, cands1, meta1, db_path=db)
    storage.persist_run(run2, [], {}, db_path=db)

    resp = c.get("/runs")
    assert resp.status_code == 200
    runs = resp.json()
    assert len(runs) == 2
    # Most recent (run_test_2, 2026-06-01) must be first.
    assert runs[0]["run_id"] == "run_test_2"
    assert runs[1]["run_id"] == "run_test"
    # Every entry carries the fields a run picker needs.
    for r in runs:
        assert "run_id" in r and "start_date" in r and "end_date" in r


def test_get_run_and_candidates(client):
    c, _, db = client
    from db import storage

    run, candidates, graph_meta = _mock_run()
    storage.persist_run(run, candidates, graph_meta, db_path=db)

    # Run metadata
    resp = c.get("/runs/run_test")
    assert resp.status_code == 200
    assert resp.json()["run_id"] == "run_test"

    # All candidates, most significant first; every one carries its statistic.
    resp = c.get("/runs/run_test/candidates")
    assert resp.status_code == 200
    cands = resp.json()
    assert len(cands) == 2
    for cand in cands:
        assert "corrected_p_value" in cand and cand["corrected_p_value"] is not None
        assert "granger_p_value" in cand
    assert cands[0]["corrected_p_value"] <= cands[1]["corrected_p_value"]

    # significant_only filter
    resp = c.get("/runs/run_test/candidates", params={"significant_only": True})
    assert resp.status_code == 200
    only = resp.json()
    assert len(only) == 1 and only[0]["is_significant"] is True


def test_get_graph_carries_statistics(client):
    c, _, db = client
    from db import storage

    run, candidates, graph_meta = _mock_run()
    storage.persist_run(run, candidates, graph_meta, db_path=db)

    resp = c.get("/runs/run_test/graph")
    assert resp.status_code == 200
    graph = resp.json()
    assert graph["nodes"] == ["AAA", "BBB"]
    assert len(graph["edges"]) == 1
    edge = graph["edges"][0]
    assert edge["source"] == "AAA" and edge["target"] == "BBB"
    assert edge["orientation_source"] == "pc"
    # Honesty rule: the edge must carry its supporting statistic.
    assert edge["corrected_p_value"] is not None


def test_get_regimes_are_time_bound(client):
    c, _, db = client
    from db import storage

    run, candidates, graph_meta = _mock_run()
    storage.persist_run(run, candidates, graph_meta, db_path=db)

    resp = c.get("/runs/run_test/regimes")
    assert resp.status_code == 200
    pairs = resp.json()
    assert len(pairs) == 1  # only the significant candidate had regimes
    period = pairs[0]["regime_periods"][0]
    assert period["start"] and period["end"]
    assert period["active"] is True


def test_unknown_run_is_404(client):
    c, _, _ = client
    assert c.get("/runs/nope/candidates").status_code == 404
    assert c.get("/runs/nope/graph").status_code == 404
    assert c.get("/runs/nope/regimes").status_code == 404


def test_analyze_persists_and_returns_run(client, monkeypatch):
    c, main, db = client
    from causal.pipeline import AnalysisResult

    run, candidates, graph_meta = _mock_run()
    # Rename so it doesn't collide with the GET fixtures' run_test id.
    run = run.model_copy(update={"run_id": "run_mock"})
    candidates = [
        cand.model_copy(
            update={
                "run_id": "run_mock",
                "candidate_id": cand.candidate_id.replace("run_test", "run_mock"),
            }
        )
        for cand in candidates
    ]
    graph_meta = {"run_mock:AAA->BBB": ("directed", "pc")}

    def fake_run_analysis(**kwargs):
        return AnalysisResult(
            run=run,
            candidates=candidates,
            graph_meta=graph_meta,
            missing_tickers=[],
        )

    monkeypatch.setattr(main, "run_analysis", fake_run_analysis)

    resp = c.post("/analyze", json={})
    assert resp.status_code == 200
    assert resp.json()["run_id"] == "run_mock"

    # The run was actually persisted and is now retrievable end-to-end.
    follow = c.get("/runs/run_mock/candidates")
    assert follow.status_code == 200
    assert len(follow.json()) == 2


def test_analyze_rejects_bad_dates(client):
    c, _, _ = client
    # Unparseable date -> 400, not 500.
    assert c.post("/analyze", json={"start_date": "not-a-date"}).status_code == 400
    # end before start -> 400.
    bad = {"start_date": "2021-01-01", "end_date": "2020-01-01"}
    assert c.post("/analyze", json=bad).status_code == 400


# --------------------------------------------------------------------------
# Phase 3 — regime-flip monitoring endpoints
# --------------------------------------------------------------------------


def _run_with_regime(run_id: str, created_at: str, active: bool):
    """A minimal one-pair run whose single significant candidate's current
    regime window is `active`. Used to manufacture a flip via storage."""
    run = AnalysisRun(
        run_id=run_id,
        created_at=created_at,
        start_date="2020-01-01",
        end_date="2021-01-01",
        asset_universe=["AAA", "BBB"],
        max_lag=5,
        correction_method="fdr_bh",
        alpha=0.05,
    )
    cand = CausalCandidate(
        candidate_id=f"{run_id}:AAA->BBB",
        run_id=run_id,
        asset_a="AAA",
        asset_b="BBB",
        direction=Direction.A_CAUSES_B,
        lag=2,
        granger_p_value=0.001,
        corrected_p_value=0.01,
        correlation_strength=0.6,
        statistical_confidence=0.79,
        is_significant=True,
        regime_periods=[
            RegimePeriod(
                start=date(2020, 1, 1),
                end=date(2020, 6, 1),
                active=active,
                mean_correlation=0.6 if active else 0.05,
            )
        ],
    )
    return run, [cand], {}


def test_flips_empty_then_surfaced(client):
    c, _, db = client
    from db import storage

    # No flips recorded yet -> empty list, not an error.
    resp = c.get("/flips")
    assert resp.status_code == 200 and resp.json() == []

    # Two runs where AAA->BBB flips from coupled (active) to decoupled.
    r1, c1, _ = _run_with_regime("run_a", "2026-01-01T00:00:00+00:00", active=True)
    r2, c2, _ = _run_with_regime("run_b", "2026-01-02T00:00:00+00:00", active=False)
    storage.persist_run(r1, c1, {}, db_path=db)
    storage.persist_run(r2, c2, {}, db_path=db)
    storage.record_and_update_flips("run_a", "run_b", db_path=db)

    resp = c.get("/flips")
    assert resp.status_code == 200
    flips = resp.json()
    assert len(flips) == 1
    ev = flips[0]
    assert ev["asset_a"] == "AAA" and ev["asset_b"] == "BBB"
    # Hard rule: a flip never travels without its statistic.
    assert ev["corrected_p_value"] is not None
    # First detection is provisional, not yet the real signal.
    assert ev["status"] == "pending" and ev["confirmed"] is False

    # Filtering by a non-matching status yields nothing; bad status -> 400.
    assert c.get("/flips", params={"status": "confirmed"}).json() == []
    assert c.get("/flips", params={"status": "bogus"}).status_code == 400


def test_monitor_endpoint_runs_cycle(client, monkeypatch):
    c, main, db = client
    from causal.pipeline import AnalysisResult

    # Patch the (network-bound) pipeline the monitor cycle calls so the endpoint
    # is exercised without touching yfinance. Each call returns a fresh run id.
    state = {"n": 0}

    def fake_run_analysis(**kwargs):
        state["n"] += 1
        run, cands, meta = _run_with_regime(
            f"run_mon_{state['n']}",
            f"2026-02-0{state['n']}T00:00:00+00:00",
            active=True,
        )
        return AnalysisResult(
            run=run, candidates=cands, graph_meta=meta, missing_tickers=[]
        )

    # run_monitor_cycle imports run_analysis from causal.pipeline into its own
    # module namespace; patch it there.
    import scripts.run_monitor as monitor
    monkeypatch.setattr(monitor, "run_analysis", fake_run_analysis)

    # First cycle: baseline only, nothing to diff -> zero flips, still 200.
    resp = c.post("/monitor", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == "run_mon_1"
    assert body["n_flips_touched"] == 0
    assert body["n_significant_candidates"] == 1


# --------------------------------------------------------------------------
# Read-only demo mode (hosted deployment, no Ollama, ephemeral filesystem)
# --------------------------------------------------------------------------


def test_demo_mode_gates_writes_but_serves_reads(client, monkeypatch):
    """With DEMO_MODE on, the pipeline/LLM-triggering endpoints return a clear
    503 while every GET keeps serving the pre-populated snapshot, and
    /llm/health degrades to unavailable without hanging."""
    c, _, db = client
    import config
    from db import storage
    from llm.models import HypothesisCard, PlausibilityFlag

    # Pre-populate the throwaway DB with a run + one card — this stands in for
    # the committed db/causal_engine.db demo snapshot.
    run, candidates, graph_meta = _mock_run()
    storage.persist_run(run, candidates, graph_meta, db_path=db)
    sig = candidates[0]
    card = HypothesisCard(
        card_id="",
        candidate=sig,
        in_graph=True,
        mechanism_explanation="A pre-recorded explanation.",
        mechanism_channel=None,
        plausibility_flag=PlausibilityFlag.LIKELY_SPURIOUS,
        llm_confidence=0.5,
        caveats=["pre-recorded"],
        addresses_pc_rejection=False,
        model_name="test-model",
        raw_response="{}",
    )
    storage.replace_hypothesis_cards("run_test", [card], db_path=db)

    # Flip demo mode on at request time (independent of import order / env).
    monkeypatch.setattr(config, "DEMO_MODE", True)

    # Pipeline / LLM / monitor endpoints are gated with an explanatory 503.
    gated = {
        "POST /analyze": c.post("/analyze", json={}),
        "POST /validate": c.post("/runs/run_test/validate"),
        "POST /monitor": c.post("/monitor", json={}),
    }
    for label, resp in gated.items():
        assert resp.status_code == 503, f"{label} should be 503 in demo mode"
        detail = resp.json()["detail"].lower()
        assert "demo" in detail and "locally" in detail, label

    # Reads still work normally against the pre-populated DB.
    assert c.get("/runs/run_test/candidates").status_code == 200
    assert c.get("/runs/run_test/graph").status_code == 200
    assert c.get("/runs/run_test/regimes").status_code == 200
    cards_resp = c.get("/runs/run_test/cards")
    assert cards_resp.status_code == 200
    cards = cards_resp.json()
    assert len(cards) == 1
    # Hard rule survives demo mode: the card still carries its statistic.
    assert cards[0]["candidate"]["corrected_p_value"] is not None

    # /llm/health degrades cleanly to unavailable (no connection attempt/hang).
    health = c.get("/llm/health")
    assert health.status_code == 200
    assert health.json()["ollama_available"] is False

    # /health advertises demo mode so a client (the dashboard) can detect it.
    h = c.get("/health")
    assert h.status_code == 200 and h.json()["demo_mode"] is True
