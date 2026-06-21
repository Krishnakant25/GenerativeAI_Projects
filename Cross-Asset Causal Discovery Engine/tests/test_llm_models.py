"""Pydantic-contract tests for the Layer-2 HypothesisCard.

These are pure data-model tests — no Ollama, no DB. They lock down the
guarantees the rest of Layer 2 relies on: a bounded confidence, a strict
plausibility enum, caveats coercion, and the embedded statistic that makes
"no card without its stat" structural.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from causal.models import CausalCandidate, Direction
from llm.models import HypothesisCard, PlausibilityFlag


def _candidate() -> CausalCandidate:
    return CausalCandidate(
        candidate_id="r:^TNX->JPY=X",
        run_id="r",
        asset_a="^TNX",
        asset_b="JPY=X",
        direction=Direction.A_CAUSES_B,
        lag=1,
        granger_p_value=4.5e-48,
        corrected_p_value=7.09e-46,
        correlation_strength=0.33,
        statistical_confidence=0.66,
        is_significant=True,
    )


def _card(**overrides) -> HypothesisCard:
    base = dict(
        card_id="",
        candidate=_candidate(),
        in_graph=False,
        mechanism_explanation="A mechanism.",
        plausibility_flag=PlausibilityFlag.LIKELY_SPURIOUS,
        llm_confidence=0.5,
        caveats=["a caveat"],
        addresses_pc_rejection=True,
    )
    base.update(overrides)
    return HypothesisCard(**base)


def test_card_embeds_its_statistic():
    """A card carries the corrected p-value it explains — the honesty rule made
    structural. The convenience accessors read off the embedded candidate."""
    card = _card()
    assert card.candidate.corrected_p_value == 7.09e-46
    assert card.candidate_id == "r:^TNX->JPY=X"
    assert card.asset_a == "^TNX"
    assert card.asset_b == "JPY=X"


def test_confidence_is_bounded():
    with pytest.raises(ValidationError):
        _card(llm_confidence=1.5)
    with pytest.raises(ValidationError):
        _card(llm_confidence=-0.1)


def test_plausibility_flag_is_strict_enum():
    with pytest.raises(ValidationError):
        _card(plausibility_flag="totally_made_up")
    # PARSE_FAILED is a valid flag (the validator's marker).
    assert _card(plausibility_flag=PlausibilityFlag.PARSE_FAILED).plausibility_flag == (
        PlausibilityFlag.PARSE_FAILED.value
    )


def test_caveats_coercion_from_string_and_none():
    """A near-miss model output (single string or null) is coerced, not rejected."""
    assert _card(caveats="just one").caveats == ["just one"]
    assert _card(caveats=None).caveats == []
    assert _card(caveats=["a", "", "  ", "b"]).caveats == ["a", "b"]


def test_use_enum_values_serializes_flag_as_string():
    card = _card(plausibility_flag=PlausibilityFlag.PLAUSIBLE_KNOWN_MECHANISM)
    dumped = card.model_dump()
    assert dumped["plausibility_flag"] == "plausible_known_mechanism"
    # And the embedded statistic survives a round-trip.
    assert dumped["candidate"]["corrected_p_value"] == 7.09e-46
