"""Unit tests for the dashboard's Phase-3 flip rendering helpers.

These are pure string builders (no Streamlit context needed). They lock two
honesty guarantees of the Events panel: a pending flip is visibly marked
provisional with its confirmation progress, and every event carries its
underlying statistic — the hard rule, carried into Phase 3.
"""

from __future__ import annotations

import config
from dashboard import app


def _event(**overrides) -> dict:
    base = {
        "asset_a": "CL=F",
        "asset_b": "XLE",
        "corrected_p_value": 1.26e-05,
        "lag": 3,
        "correlation_strength": 0.151,
        "status": "pending",
        "consecutive_confirmations": 2,
        "direction": "coupled -> decoupled (deactivated)",
        "confirmed": False,
    }
    base.update(overrides)
    return base


def test_pending_badge_shows_provisional_and_progress():
    html = app.flip_badge_html(_event(status="pending", consecutive_confirmations=2))
    assert "PROVISIONAL" in html
    # Progress toward confirmation is in the badge itself, not a missable label.
    assert f"2/{config.MONITOR_CONFIRMATION_RUNS}" in html
    # Distinct colour from confirmed so the two can't be confused at a glance.
    assert app.FLIP_STATUS_STYLE["pending"]["color"] in html
    assert app.FLIP_STATUS_STYLE["pending"]["color"] != (
        app.FLIP_STATUS_STYLE["confirmed"]["color"]
    )


def test_confirmed_and_reverted_badges_are_distinct():
    confirmed = app.flip_badge_html(_event(status="confirmed", confirmed=True))
    reverted = app.flip_badge_html(_event(status="reverted"))
    assert "CONFIRMED" in confirmed and "PROVISIONAL" not in confirmed
    assert "REVERTED" in reverted
    assert app.FLIP_STATUS_STYLE["confirmed"]["color"] in confirmed
    assert app.FLIP_STATUS_STYLE["reverted"]["color"] in reverted


def test_every_flip_carries_its_statistic():
    line = app.flip_stat_line(_event())
    assert "corrected p" in line
    assert "1.26e-05" in line          # the corrected p-value
    assert "lag 3d" in line            # the lag
    assert "+0.151" in line            # the correlation


def test_stat_line_tolerates_missing_correlation():
    # Hard rule still holds when correlation is absent — p and lag always shown.
    line = app.flip_stat_line(_event(correlation_strength=None))
    assert "corrected p = 1.26e-05" in line
    assert "r = n/a" in line
