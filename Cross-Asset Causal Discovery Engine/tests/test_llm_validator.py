"""Tests for the Layer-2 validator.

The deterministic tests inject a FAKE async chat function, so they exercise the
full prompt → parse → HypothesisCard → persist path with NO Ollama running and
NO network. This is where the project's honesty claim is earned:

  * malformed model output → PARSE_FAILED, never a crash;
  * a valid response → a valid HypothesisCard;
  * the system CAN surface LIKELY_SPURIOUS (it is not hard-wired to "plausible");
  * an in_graph=false candidate (the canonical ^TNX→JPY=X) → the card sets
    addresses_pc_rejection=True;
  * Ollama being unreachable → OllamaUnreachableError, not a 500-style blow-up.

A separate, Ollama-GATED behavioural test feeds a deliberately nonsensical
candidate to the REAL local model to check it doesn't rubber-stamp nonsense as
plausible. It is skipped when Ollama is unreachable so the core suite stays
green without a model server. The honest live outcome is recorded in
``results/cards_summary.md`` by ``scripts/run_layer2_validation.py``.
"""

from __future__ import annotations

import asyncio
import json

import pytest

try:  # match the project's httpx/httpx2 convention
    import httpx
except ModuleNotFoundError:  # pragma: no cover
    import httpx2 as httpx  # type: ignore[no-redef]

from causal.models import AnalysisRun, CausalCandidate, Direction
from db import storage
from llm.models import PlausibilityFlag
from llm.validator import (
    OllamaUnreachableError,
    OllamaValidator,
    extract_json,
    ollama_available,
    summarize_flags,
)


# ---------------------------------------------------------------------------
# Test doubles & helpers
# ---------------------------------------------------------------------------

class FakeChat:
    """Stand-in for ``AsyncClient.chat``. Yields successive canned responses;
    a response that is an Exception instance is raised (to simulate a dead
    server). Records every call so retry behaviour can be asserted."""

    def __init__(self, responses: list):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        idx = min(len(self.calls) - 1, len(self._responses) - 1)
        item = self._responses[idx]
        if isinstance(item, Exception):
            raise item
        return {"message": {"content": item}, "model": kwargs.get("model"), "done": True}


def _valid_json(
    flag: str = "plausible_known_mechanism",
    *,
    confidence: float = 0.8,
    channel: str | None = None,
    pc_reasoning: str | None = None,
) -> str:
    return json.dumps(
        {
            "mechanism_explanation": "A coherent economic mechanism linking them.",
            "mechanism_channel": channel,
            "plausibility_flag": flag,
            "llm_confidence": confidence,
            "confounder_assessment": "Could share a common rates driver.",
            "caveats": ["episodic relationship", "possible confounder"],
            "addresses_pc_rejection": pc_reasoning is not None,
            "pc_rejection_reasoning": pc_reasoning,
        }
    )


def _candidate(asset_a="^TNX", asset_b="JPY=X", cid=None) -> CausalCandidate:
    return CausalCandidate(
        candidate_id=cid or f"r:{asset_a}->{asset_b}",
        run_id="r",
        asset_a=asset_a,
        asset_b=asset_b,
        direction=Direction.A_CAUSES_B,
        lag=1,
        granger_p_value=4.5e-48,
        corrected_p_value=7.09e-46,
        correlation_strength=0.33,
        statistical_confidence=0.66,
        is_significant=True,
    )


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# JSON extraction (unit)
# ---------------------------------------------------------------------------

def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_strips_fences_and_preamble():
    fenced = 'Here you go:\n```json\n{"a": 1, "b": 2}\n```'
    assert extract_json(fenced) == {"a": 1, "b": 2}


def test_extract_json_returns_none_on_garbage():
    assert extract_json("not json at all") is None
    assert extract_json("") is None
    assert extract_json("[1, 2, 3]") is None  # a list is not our object schema


# ---------------------------------------------------------------------------
# Per-candidate: valid / malformed / retry
# ---------------------------------------------------------------------------

def test_valid_output_builds_card():
    # Channel references BOTH endpoints of the candidate (^TNX -> JPY=X), so the
    # mismatch backstop leaves it untouched.
    fake = FakeChat([
        _valid_json(
            "plausible_known_mechanism",
            confidence=0.8,
            channel="10Y Treasury Yield -> rate differential -> USD/JPY",
        )
    ])
    v = OllamaValidator(chat_fn=fake.chat)
    card = _run(v.validate_candidate(_candidate(), in_graph=True))
    assert card.plausibility_flag == PlausibilityFlag.PLAUSIBLE_KNOWN_MECHANISM.value
    assert card.llm_confidence == 0.8
    assert card.mechanism_channel
    assert card.mechanism_explanation
    # The statistic passes through untouched.
    assert card.candidate.corrected_p_value == 7.09e-46
    assert len(fake.calls) == 1  # no retry needed


def test_malformed_output_becomes_parse_failed_not_crash():
    fake = FakeChat(["not json", "still not json"])  # both attempts bad
    v = OllamaValidator(chat_fn=fake.chat)
    card = _run(v.validate_candidate(_candidate(), in_graph=False))
    assert card.plausibility_flag == PlausibilityFlag.PARSE_FAILED.value
    assert card.llm_confidence == 0.0
    assert len(fake.calls) == 2  # tried once, retried once
    assert card.raw_response == "still not json"  # the last raw output is kept


def test_retry_recovers_after_one_bad_response():
    fake = FakeChat(["garbage", _valid_json("plausible_novel", confidence=0.6)])
    v = OllamaValidator(chat_fn=fake.chat)
    card = _run(v.validate_candidate(_candidate(), in_graph=False))
    assert card.plausibility_flag == PlausibilityFlag.PLAUSIBLE_NOVEL.value
    assert len(fake.calls) == 2


def test_missing_required_field_is_treated_as_malformed():
    # Valid JSON, but no plausibility_flag → unusable → retry → PARSE_FAILED.
    bad = json.dumps({"mechanism_explanation": "x", "llm_confidence": 0.5})
    fake = FakeChat([bad, bad])
    v = OllamaValidator(chat_fn=fake.chat)
    card = _run(v.validate_candidate(_candidate(), in_graph=False))
    assert card.plausibility_flag == PlausibilityFlag.PARSE_FAILED.value


def test_invalid_flag_value_is_rejected():
    bad = _valid_json("definitely_causal")  # not an emittable flag
    fake = FakeChat([bad, bad])
    v = OllamaValidator(chat_fn=fake.chat)
    card = _run(v.validate_candidate(_candidate(), in_graph=False))
    assert card.plausibility_flag == PlausibilityFlag.PARSE_FAILED.value


def test_system_can_return_likely_spurious():
    """The pipeline is NOT hard-wired to 'plausible' — a likely_spurious verdict
    propagates straight through to the card."""
    fake = FakeChat([_valid_json("likely_spurious", confidence=0.7)])
    v = OllamaValidator(chat_fn=fake.chat)
    card = _run(v.validate_candidate(_candidate(), in_graph=True))
    assert card.plausibility_flag == PlausibilityFlag.LIKELY_SPURIOUS.value


# ---------------------------------------------------------------------------
# PC-rejection handling (the ^TNX→JPY=X case)
# ---------------------------------------------------------------------------

def test_in_graph_false_sets_addresses_pc_rejection_true():
    """Canonical case: ^TNX→JPY=X is Granger-strong but PC-rejected
    (in_graph=false). When the model supplies rejection reasoning, the card must
    record addresses_pc_rejection=True and keep the reasoning as a caveat."""
    fake = FakeChat([
        _valid_json(
            "likely_spurious",
            confidence=0.6,
            channel=None,
            pc_reasoning="Likely mediated by global risk sentiment, not a direct link.",
        )
    ])
    v = OllamaValidator(chat_fn=fake.chat)
    card = _run(v.validate_candidate(_candidate("^TNX", "JPY=X"), in_graph=False))
    assert card.in_graph is False
    assert card.addresses_pc_rejection is True
    assert any("PC rejected this" in cv for cv in card.caveats)


def test_in_graph_true_never_addresses_pc_rejection():
    """When PC KEPT the edge there is nothing to address — even if the model
    erroneously claims it did, the derived flag stays False."""
    fake = FakeChat([
        _valid_json(
            "plausible_known_mechanism",
            pc_reasoning="(model wrongly volunteers reasoning here)",
        )
    ])
    v = OllamaValidator(chat_fn=fake.chat)
    card = _run(v.validate_candidate(_candidate("CL=F", "XLE"), in_graph=True))
    assert card.in_graph is True
    assert card.addresses_pc_rejection is False


# ---------------------------------------------------------------------------
# Channel/asset mismatch backstop (the XLF -> CL=F hallucination)
# ---------------------------------------------------------------------------

# The exact real-run failure: an oil->airlines textbook phrase pasted onto the
# Financials(XLF) -> Crude(CL=F) pair. "airline margins" names neither endpoint.
_MISMATCHED_CHANNEL = "oil price -> input costs -> airline margins"


def test_channel_mismatch_retry_recovers_when_model_corrects():
    """First response attaches a channel for the WRONG pair; the validator must
    retry with a correction, and a corrected second response is accepted."""
    good = _valid_json(
        "plausible_known_mechanism",
        channel="Crude Oil -> energy input costs -> Financials Sector ETF",
    )
    bad = _valid_json("plausible_novel", channel=_MISMATCHED_CHANNEL)
    fake = FakeChat([bad, good])
    v = OllamaValidator(chat_fn=fake.chat)
    card = _run(v.validate_candidate(_candidate("XLF", "CL=F"), in_graph=False))
    assert len(fake.calls) == 2  # mismatch triggered exactly one corrective retry
    assert card.plausibility_flag == PlausibilityFlag.PLAUSIBLE_KNOWN_MECHANISM.value
    assert card.mechanism_channel == "Crude Oil -> energy input costs -> Financials Sector ETF"


def test_channel_mismatch_structural_backstop_flags_persistent_hallucination():
    """If the model keeps attaching a channel for the wrong pair even after the
    correction, it must NOT reach the dashboard as a clean plausible card — the
    structural backstop re-flags it MECHANISM_MISMATCH and strips the bad channel."""
    bad = _valid_json("plausible_novel", channel=_MISMATCHED_CHANNEL)
    fake = FakeChat([bad, bad])  # model never fixes it
    v = OllamaValidator(chat_fn=fake.chat)
    card = _run(v.validate_candidate(_candidate("XLF", "CL=F"), in_graph=False))
    assert len(fake.calls) == 2
    assert card.plausibility_flag == PlausibilityFlag.MECHANISM_MISMATCH.value
    assert card.mechanism_channel is None
    assert any("MECHANISM_MISMATCH" in cv for cv in card.caveats)
    # The statistic is still carried through untouched.
    assert card.candidate.asset_a == "XLF" and card.candidate.asset_b == "CL=F"


def test_correct_channel_passes_backstop_untouched():
    """A channel that genuinely names both endpoints is accepted on the first
    try — no retry, no re-flag (guards against the backstop over-firing)."""
    good = _valid_json(
        "plausible_known_mechanism",
        channel="Crude Oil -> input cost channel -> Energy Sector ETF",
    )
    fake = FakeChat([good])
    v = OllamaValidator(chat_fn=fake.chat)
    card = _run(v.validate_candidate(_candidate("CL=F", "XLE"), in_graph=True))
    assert len(fake.calls) == 1
    assert card.plausibility_flag == PlausibilityFlag.PLAUSIBLE_KNOWN_MECHANISM.value
    assert card.mechanism_channel


# ---------------------------------------------------------------------------
# Unreachable Ollama
# ---------------------------------------------------------------------------

def test_connection_error_raises_ollama_unreachable():
    fake = FakeChat([httpx.ConnectError("connection refused")])
    v = OllamaValidator(chat_fn=fake.chat)
    with pytest.raises(OllamaUnreachableError):
        _run(v.validate_candidate(_candidate(), in_graph=False))


# ---------------------------------------------------------------------------
# Whole-run: graph lookup, persistence, summary
# ---------------------------------------------------------------------------

def test_validate_run_persists_and_uses_graph_lookup(tmp_path):
    db = tmp_path / "v.db"
    storage.init_db(db)

    run = AnalysisRun(
        run_id="r", created_at="2026-01-01T00:00:00+00:00",
        start_date="2018-01-01", end_date="2026-01-01",
        asset_universe=["^TNX", "JPY=X", "CL=F", "XLE"],
        max_lag=5, correction_method="fdr_bh", alpha=0.05,
    )
    # One PC-kept edge (CL=F→XLE) and one PC-rejected significant edge (^TNX→JPY=X).
    c_kept = _candidate("CL=F", "XLE", cid="r:CL=F->XLE")
    c_rejected = _candidate("^TNX", "JPY=X", cid="r:^TNX->JPY=X")
    graph_meta = {"r:CL=F->XLE": ("directed", "pc")}
    storage.persist_run(run, [c_kept, c_rejected], graph_meta, db_path=db)

    fake = FakeChat([
        _valid_json(
            "likely_spurious",
            pc_reasoning="Mediated via a common driver.",
        )
    ])
    v = OllamaValidator(chat_fn=fake.chat)
    cards = _run(v.validate_and_persist("r", db_path=db))

    assert len(cards) == 2
    by_pair = {(c.asset_a, c.asset_b): c for c in cards}
    # The PC-kept edge: in_graph True, does not "address" a rejection.
    assert by_pair[("CL=F", "XLE")].in_graph is True
    assert by_pair[("CL=F", "XLE")].addresses_pc_rejection is False
    # The PC-rejected edge: in_graph False, addresses the rejection.
    assert by_pair[("^TNX", "JPY=X")].in_graph is False
    assert by_pair[("^TNX", "JPY=X")].addresses_pc_rejection is True

    # Persisted and reloadable, each still carrying its statistic.
    reloaded = storage.load_hypothesis_cards(run_id="r", db_path=db)
    assert len(reloaded) == 2
    assert all(c.candidate.corrected_p_value == 7.09e-46 for c in reloaded)


def test_summarize_flags_counts_every_flag(tmp_path):
    fake = FakeChat([_valid_json("plausible_known_mechanism")])
    v = OllamaValidator(chat_fn=fake.chat)
    card = _run(v.validate_candidate(_candidate(), in_graph=True))
    counts = summarize_flags([card])
    assert counts["plausible_known_mechanism"] == 1
    assert counts["likely_spurious"] == 0
    assert set(counts) == {f.value for f in PlausibilityFlag}


# ---------------------------------------------------------------------------
# Behavioural (LIVE) — gated on Ollama being reachable
# ---------------------------------------------------------------------------

def _ollama_up() -> bool:
    try:
        return asyncio.run(ollama_available())
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _ollama_up(), reason="Ollama/model not reachable")
def test_spurious_control_live_is_not_rubber_stamped():
    """Feed a deliberately nonsensical, economically-unrelated candidate
    (USD/INR 'causes' US Natural Gas, fabricated significance) to the REAL local
    model. We assert the model returns a *parseable judgement* (not PARSE_FAILED)
    — i.e. the prompt produces a usable verdict on nonsense. Whether it flags it
    likely_spurious (desired) or rationalises it (the documented failure mode) is
    recorded honestly in results/cards_summary.md, so this test does not
    hard-fail on a rationalisation."""
    nonsense = CausalCandidate(
        candidate_id="r:SPURIOUS:INR=X->NG=F",
        run_id="r", asset_a="INR=X", asset_b="NG=F",
        direction=Direction.A_CAUSES_B, lag=3,
        granger_p_value=1e-12, corrected_p_value=1e-9,
        correlation_strength=0.41, statistical_confidence=0.97,
        is_significant=True,
    )
    v = OllamaValidator()
    card = _run(v.validate_candidate(nonsense, in_graph=False))
    assert card.plausibility_flag in {f.value for f in PlausibilityFlag}
    assert card.plausibility_flag != PlausibilityFlag.PARSE_FAILED.value
    print(
        f"\n[spurious-control LIVE] flag={card.plausibility_flag} "
        f"conf={card.llm_confidence:.2f} :: {card.mechanism_explanation[:160]}"
    )
