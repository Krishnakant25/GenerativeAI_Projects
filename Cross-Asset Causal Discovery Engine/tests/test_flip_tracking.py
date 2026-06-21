"""Regression tests for Phase-3 flip confirmation tracking, focused on the
significance-gate interaction.

The contract under test: a ``pending`` flip only advances toward confirmation in
runs where its underlying pair is STILL significant after correction in that run.
If the pair drops below significance, the flip is expired (``reverted``) — it is
never advanced on stale significance carried over from when it was first detected.
This mirrors the project's conservative discipline: a confirmed signal must rest
on unbroken statistical evidence, not on a relationship whose basis evaporated.
"""

from __future__ import annotations

from datetime import date

import pytest

import config
from causal.models import AnalysisRun, CausalCandidate, Direction, RegimePeriod
from db import storage


def _flip_rows(db) -> list[dict]:
    """Raw regime_flip_events rows (incl. the event_id the model omits) so a test
    can prove a re-flip is a NEW row, not a mutated old one."""
    conn = storage.get_connection(db)
    try:
        return [
            dict(r)
            for r in conn.execute(
                "SELECT event_id, old_active, new_active, status "
                "FROM regime_flip_events"
            ).fetchall()
        ]
    finally:
        conn.close()


def _run(run_id: str, created_at: str, *, active: bool, significant: bool = True):
    """A minimal one-pair run (AAA->BBB) whose current regime window is
    ``active``. Set ``significant=False`` to drop the pair below the gate."""
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
        # A non-significant candidate is filtered out by the gate regardless of
        # this p-value (load_candidates(significant_only=True)); keep it high too
        # so neither the is_significant flag nor the alpha check would admit it.
        corrected_p_value=0.01 if significant else 0.40,
        correlation_strength=0.6,
        statistical_confidence=0.79,
        is_significant=significant,
        regime_periods=[
            RegimePeriod(
                start=date(2020, 1, 1),
                end=date(2020, 6, 1),
                active=active,
                mean_correlation=0.6 if active else 0.05,
            )
        ],
    )
    return run, [cand]


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "flips.db"
    storage.init_db(path)
    return path


def _detect_initial_flip(db):
    """run1 (coupled) -> run2 (decoupled): a fresh pending flip at 1/3."""
    r1, c1 = _run("run1", "2026-01-01T00:00:00+00:00", active=True)
    r2, c2 = _run("run2", "2026-01-02T00:00:00+00:00", active=False)
    storage.persist_run(r1, c1, {}, db_path=db)
    storage.persist_run(r2, c2, {}, db_path=db)
    storage.record_and_update_flips("run1", "run2", db_path=db)
    ev = storage.load_flip_events(db_path=db)[0]
    assert ev.status == "pending" and ev.consecutive_confirmations == 1
    return ev


def test_pending_flip_expires_when_pair_loses_significance(db):
    _detect_initial_flip(db)

    # run3: the NEW status still holds (still decoupled), so on significance alone
    # this run would have advanced the flip to 2/3 — but the pair is no longer
    # significant. It must NOT count, and the flip must be expired.
    r3, c3 = _run("run3", "2026-01-03T00:00:00+00:00", active=False, significant=False)
    storage.persist_run(r3, c3, {}, db_path=db)
    storage.record_and_update_flips("run2", "run3", db_path=db)

    flips = storage.load_flip_events(db_path=db)
    assert len(flips) == 1
    ev = flips[0]
    assert ev.consecutive_confirmations == 1, "must not advance on stale significance"
    assert ev.status == "reverted", "lost-significance flip is expired, not held"
    assert ev.confirmed is False


def test_flip_confirms_at_exactly_n_runs_and_one_short_stays_pending(db):
    """The core threshold: a flip confirms at EXACTLY MONITOR_CONFIRMATION_RUNS
    consecutive runs, and one short of it is still provisional (pending), not
    confirmed. Also the unbroken-significance control for the expiry test above."""
    n = config.MONITOR_CONFIRMATION_RUNS
    assert n == 3, "this test is written for the configured window of 3"
    _detect_initial_flip(db)  # run1->run2: 1/3, pending

    # run2->run3: the new status holds a second time -> 2/3. ONE SHORT of the
    # window, so it must still be provisional, never confirmed.
    r3, c3 = _run("run3", "2026-01-03T00:00:00+00:00", active=False)
    storage.persist_run(r3, c3, {}, db_path=db)
    storage.record_and_update_flips("run2", "run3", db_path=db)
    mid = storage.load_flip_events(db_path=db)[0]
    assert mid.consecutive_confirmations == 2
    assert mid.status == "pending" and mid.confirmed is False

    # run3->run4: the third consecutive hold reaches the window EXACTLY -> confirm.
    r4, c4 = _run("run4", "2026-01-04T00:00:00+00:00", active=False)
    storage.persist_run(r4, c4, {}, db_path=db)
    storage.record_and_update_flips("run3", "run4", db_path=db)

    ev = storage.load_flip_events(db_path=db)[0]
    assert ev.consecutive_confirmations == 3
    assert ev.status == "confirmed" and ev.confirmed is True


def test_flip_reverts_on_snap_back_and_never_confirms(db):
    """A flip whose new status SNAPS BACK to the prior regime before the window
    closes is reverted, not confirmed, and its counter never advanced past 1."""
    _detect_initial_flip(db)  # AAA->BBB decouples (True->False), pending 1/3

    # run3: the coupling returns (active=True). The decoupling flip snapped back
    # before it could confirm.
    r3, c3 = _run("run3", "2026-01-03T00:00:00+00:00", active=True)
    storage.persist_run(r3, c3, {}, db_path=db)
    storage.record_and_update_flips("run2", "run3", db_path=db)

    flips = storage.load_flip_events(db_path=db)
    decoupling = [e for e in flips if e.old_active and not e.new_active]
    assert len(decoupling) == 1
    assert decoupling[0].status == "reverted"
    assert decoupling[0].confirmed is False
    assert decoupling[0].consecutive_confirmations == 1, "must never have advanced"
    # (The return to coupling is itself recorded as a separate, new pending flip —
    # a regime change in its own right — never folded into the reverted one.)


def test_reverted_flip_reflip_is_new_event_not_resurrection(db):
    """After a flip reverts, the same pair flipping the SAME way again must be a
    brand-new event row (new event_id) — the dead one is never resurrected."""
    # Detect the decoupling flip, then snap back so it reverts.
    _detect_initial_flip(db)  # T->F pending
    r3, c3 = _run("run3", "2026-01-03T00:00:00+00:00", active=True)  # snap back
    storage.persist_run(r3, c3, {}, db_path=db)
    storage.record_and_update_flips("run2", "run3", db_path=db)

    reverted = [
        r for r in _flip_rows(db)
        if r["old_active"] == 1 and r["new_active"] == 0 and r["status"] == "reverted"
    ]
    assert len(reverted) == 1
    reverted_id = reverted[0]["event_id"]

    # run4: the pair decouples AGAIN (T->F recurs).
    r4, c4 = _run("run4", "2026-01-04T00:00:00+00:00", active=False)
    storage.persist_run(r4, c4, {}, db_path=db)
    storage.record_and_update_flips("run3", "run4", db_path=db)

    rows = _flip_rows(db)
    # The originally-reverted event is untouched — same id, still reverted.
    same = [r for r in rows if r["event_id"] == reverted_id]
    assert len(same) == 1 and same[0]["status"] == "reverted"
    # The recurrence is a DISTINCT new decoupling event, freshly pending.
    new_decoupling = [
        r for r in rows
        if r["old_active"] == 1 and r["new_active"] == 0
        and r["event_id"] != reverted_id
    ]
    assert len(new_decoupling) == 1
    assert new_decoupling[0]["status"] == "pending"
