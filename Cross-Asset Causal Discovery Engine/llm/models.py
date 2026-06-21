"""Pydantic contract for the Layer-2 LLM layer.

A :class:`HypothesisCard` is the human-facing artifact of Layer 2: it wraps a
statistical :class:`~causal.models.CausalCandidate` — one that already passed
significance *and* multiple-comparisons correction in Layer 1 — with an
LLM-generated economic mechanism, a plausibility judgement, and explicit
caveats.

NON-NEGOTIABLE FRAMING (see README "Layer 2"):
    The LLM NEVER asserts causality on its own. It only *explains* and *rates
    the plausibility* of a statistical candidate the discovery layer already
    produced. Granger causality is predictive precedence, not proof of
    causation; this plausibility flag is a HEURISTIC FILTER, not validation.
    The named, designed-for risk: an LLM will happily generate a confident,
    fluent economic mechanism for a statistically SPURIOUS relationship. The
    ``LIKELY_SPURIOUS`` flag and the ``caveats`` field exist so the model has a
    sanctioned way to push back instead of rationalising everything.

Honesty rule made structural: a card embeds the *full* ``CausalCandidate`` it
explains (``candidate``), so a card can never be shown without the corrected
p-value / lag / correlation that justifies it. The validator only *attaches*
narrative — it never creates or mutates the underlying statistic, which passes
through untouched.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator

from causal.models import CausalCandidate

DEFAULT_MODEL_NAME = "llama3.1:8b-instruct-q4_0"


class PlausibilityFlag(str, Enum):
    """The LLM's heuristic judgement of whether a credible economic mechanism
    links the two assets in the stated direction. A *filter*, not validation.

    ``PARSE_FAILED`` and ``MECHANISM_MISMATCH`` are NOT LLM judgements — they are
    the validator's own markers. ``PARSE_FAILED`` records that the model's output
    could not be parsed into this schema even after a retry. ``MECHANISM_MISMATCH``
    records that the model named a transmission channel whose endpoints do not
    correspond to the two assets in the candidate (it pattern-matched a memorised
    textbook phrase onto a superficially similar pair) — caught structurally so a
    mismatched-asset explanation never reaches the UI labelled as a clean result.
    """

    PLAUSIBLE_KNOWN_MECHANISM = "plausible_known_mechanism"  # a textbook channel exists
    PLAUSIBLE_NOVEL = "plausible_novel"                      # coherent but non-textbook
    LIKELY_SPURIOUS = "likely_spurious"                      # no credible mechanism
    PARSE_FAILED = "parse_failed"                            # output unparseable (not a judgement)
    MECHANISM_MISMATCH = "mechanism_mismatch"                # channel references the wrong assets (not a judgement)


# Flags the LLM itself is allowed to emit. PARSE_FAILED is reserved for the
# validator and must never appear in raw model output.
LLM_EMITTABLE_FLAGS: frozenset[str] = frozenset(
    {
        PlausibilityFlag.PLAUSIBLE_KNOWN_MECHANISM.value,
        PlausibilityFlag.PLAUSIBLE_NOVEL.value,
        PlausibilityFlag.LIKELY_SPURIOUS.value,
    }
)


class HypothesisCard(BaseModel):
    """Layer-2 output: a statistical candidate + an LLM economic narrative.

    A *candidate hypothesis for human review*, never a proven causal claim.
    """

    # --- Identity ---------------------------------------------------------
    card_id: str = Field(
        ..., description="Stable card identifier (uuid hex, assigned on persist)."
    )

    # --- The statistic this card explains (passes through UNTOUCHED) ------
    candidate: CausalCandidate = Field(
        ...,
        description=(
            "The full Layer-1 CausalCandidate this card explains. Embedded so a "
            "card is never shown without its corrected p-value / lag / "
            "correlation. The validator never mutates these statistics."
        ),
    )
    in_graph: bool = Field(
        ...,
        description=(
            "Whether PC kept this as a DIRECT edge. False means PC's "
            "conditional-independence test rejected it as direct even though "
            "Granger flagged it — the card must reason about that tension."
        ),
    )

    # --- LLM-generated narrative (Layer 2) --------------------------------
    mechanism_explanation: str = Field(
        ...,
        description=(
            "Plain-English explanation of the economic mechanism (if any) that "
            "could link asset_a to asset_b at this lag. A candidate explanation "
            "for human review, not an assertion of causation."
        ),
    )
    mechanism_channel: str | None = Field(
        default=None,
        description=(
            "Named textbook transmission channel if one exists "
            "(e.g. 'oil price -> input costs -> airline margins'); null if the "
            "model could not name a recognised channel."
        ),
    )
    plausibility_flag: PlausibilityFlag = Field(
        ...,
        description="Heuristic plausibility judgement. A filter, not proof.",
    )
    llm_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "LLM-reported confidence in the proposed mechanism, 0-1. DISTINCT "
            "from the statistical_confidence carried on the candidate — this is "
            "the model's own degree of belief, not a statistical quantity."
        ),
    )
    caveats: list[str] = Field(
        default_factory=list,
        description=(
            "Explicit caveats: confounders, alternative explanations, why the "
            "relationship might be spurious. The model's sanctioned channel for "
            "pushing back instead of rationalising."
        ),
    )
    addresses_pc_rejection: bool = Field(
        default=False,
        description=(
            "True iff this card engaged with an in_graph=false candidate — i.e. "
            "the model reasoned about WHY a Granger-strong signal was rejected "
            "by PC as a direct edge (likely mediated/confounded). Always False "
            "when in_graph is True (nothing to address)."
        ),
    )

    # --- Provenance -------------------------------------------------------
    model_name: str = Field(
        default=DEFAULT_MODEL_NAME,
        description="Local model that generated this card.",
    )
    raw_response: str | None = Field(
        default=None,
        description=(
            "Raw model output, retained for debugging and for inspecting "
            "PARSE_FAILED cards. Not shown in the primary UI."
        ),
    )
    created_at: str | None = Field(
        default=None, description="ISO-8601 UTC creation timestamp (set on persist)."
    )

    model_config = {"use_enum_values": True}

    # --- Convenience accessors (keep the API/dashboard terse) -------------
    @property
    def candidate_id(self) -> str:
        return self.candidate.candidate_id

    @property
    def asset_a(self) -> str:
        return self.candidate.asset_a

    @property
    def asset_b(self) -> str:
        return self.candidate.asset_b

    @field_validator("caveats", mode="before")
    @classmethod
    def _coerce_caveats(cls, v: object) -> list[str]:
        """The prompt asks for a JSON list, but tolerate a model that returns a
        single string (wrap it) or null (empty list) so a near-miss output still
        validates instead of becoming a PARSE_FAILED."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v.strip() else []
        if isinstance(v, (list, tuple)):
            return [str(item) for item in v if str(item).strip()]
        return [str(v)]
