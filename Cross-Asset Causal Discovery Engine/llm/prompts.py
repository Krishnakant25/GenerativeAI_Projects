"""LLM prompt templates for Layer 2 (plausibility / economic-mechanism layer).

These prompts hand the local model ONE statistical candidate that has ALREADY
passed Granger significance *and* multiple-comparisons (FDR/Bonferroni)
correction in Layer 1, with its full statistics and regime history, and ask it
to explain — not discover — a possible economic mechanism and rate the
candidate's plausibility.

Design rules baked into the wording (do not relax these without re-reading the
README "Layer 2" framing):

  * The model is told, explicitly and repeatedly, that it is EXPLAINING a
    statistical finding, NOT discovering causation, and that it must not upgrade
    a candidate to "causal" on economic intuition alone.
  * It is required to actively consider whether the relationship is SPURIOUS or
    confounded by a common driver, and is given a sanctioned flag
    (``likely_spurious``) for saying so — this guards against the known failure
    mode where an LLM fabricates a fluent mechanism for noise.
  * The ``in_graph`` flag is fed in. When PC's conditional-independence test
    REJECTED the edge as direct (``in_graph=false``) even though Granger flagged
    it, the prompt forces the model to reason about that tension (most likely
    mediated or confounded) rather than ignore it. The canonical case is
    ``^TNX -> JPY=X``: Granger-strong, stationarity-clean, yet PC-rejected.
  * Output MUST be strict, parseable JSON only — no prose preamble, no markdown
    fences. Every field is specified below.

``build_system_prompt`` and ``build_user_prompt`` are pure string builders so
the validator (and tests) can construct the exact payload deterministically.
"""

from __future__ import annotations

from causal.models import CausalCandidate
from config import asset_name

# ---------------------------------------------------------------------------
# Output schema. This is the single source of truth for what the model must
# emit; it is embedded verbatim into the user prompt AND mirrored by
# llm.models.HypothesisCard so the contract can't silently drift.
# ---------------------------------------------------------------------------

def _output_schema(a_name: str, b_name: str) -> str:
    """The output schema, with the channel field's required format bound to the
    TWO ACTUAL assets of this candidate. Binding the real names in (rather than a
    static example) is a structural guard against the model pasting a memorised
    textbook phrase for a different, only-superficially-similar pair."""
    return (
        "{\n"
        '  "mechanism_explanation": string,   // 2-4 sentences, plain English. The economic\n'
        "                                     // mechanism (if any) that could link driver->affected\n"
        "                                     // at this lag. Explain a STATISTICAL finding; do not\n"
        "                                     // assert proven causation.\n"
        '  "mechanism_channel": string|null,  // a short, named transmission channel that MUST start\n'
        f'                                     // at "{a_name}" and end at "{b_name}" — the exact two\n'
        "                                     // assets of THIS candidate — in the format\n"
        f'                                     //   "{a_name} -> <intermediate step(s)> -> {b_name}".\n'
        f'                                     // Both endpoints must literally name {a_name} and {b_name}.\n'
        "                                     // Set this to null (and prefer likely_spurious) if you\n"
        "                                     // cannot construct a specific channel for THESE two\n"
        "                                     // assets. NEVER reuse a channel built for a different\n"
        "                                     // pair just because the topic feels similar.\n"
        '  "plausibility_flag": string,       // EXACTLY one of:\n'
        '                                     //   "plausible_known_mechanism" - a recognised textbook\n'
        "                                     //        channel plausibly links THESE TWO assets in this\n"
        "                                     //        direction/lag (mechanism_channel must be non-null)\n"
        '                                     //   "plausible_novel" - coherent economic story specific\n'
        "                                     //        to THESE TWO assets but not a standard textbook channel\n"
        '                                     //   "likely_spurious" - no credible mechanism for THIS pair;\n'
        "                                     //        more likely noise, coincidence, a common-driver\n"
        "                                     //        confound, OR no clean mechanism links these two\n"
        "                                     //        specific assets (use this rather than inventing one)"
        + _SCHEMA_TAIL
    )


_SCHEMA_TAIL = """
  "llm_confidence": number,          // 0.0-1.0, YOUR confidence in the proposed mechanism.
                                     // This is your belief, NOT a statistical quantity. Be
                                     // calibrated: low when the story is thin or speculative.
  "confounder_assessment": string,   // 1-3 sentences: could a common driver (rates, the dollar,
                                     // broad risk sentiment, a sector factor) produce this
                                     // statistical relationship without a direct economic link?
  "caveats": [string],               // 1-4 short caveats / alternative explanations / reasons it
                                     // could be spurious. Never empty.
  "addresses_pc_rejection": boolean, // true ONLY if the candidate was PC-rejected as a direct
                                     // edge (in_graph=false) AND you explained why a strong
                                     // predictive signal can still be rejected as direct
                                     // (mediation / confounding). false otherwise.
  "pc_rejection_reasoning": string|null  // if in_graph=false, 1-3 sentences on why PC likely
                                     // rejected a Granger-strong edge as a DIRECT link
                                     // (e.g. a third asset mediates it). null if in_graph=true.
}\
"""


def build_system_prompt() -> str:
    """The system message: fixes the model's role and the honesty framing."""
    return (
        "You are a markets economist assisting a research-screening tool. The "
        "tool has already run a rigorous statistical pipeline: Granger-causality "
        "testing with multiple-comparisons (FDR) correction, a PC causal-graph "
        "discovery step, and HMM regime detection. You are Layer 2: you EXPLAIN "
        "and CONTEXTUALISE one statistical finding that already survived that "
        "pipeline. You do not run statistics and you do not discover causation.\n"
        "\n"
        "Hard rules you must obey:\n"
        "1. Granger causality is PREDICTIVE PRECEDENCE, not proof of causation. "
        "Never describe the relationship as proven or causal. Your job is to "
        "rate the PLAUSIBILITY of an economic mechanism, which is a heuristic "
        "filter, not validation.\n"
        "2. Do NOT upgrade a candidate to 'causal' on economic intuition. A "
        "fluent story is not evidence.\n"
        "3. A statistically significant relationship can still be SPURIOUS or "
        "driven by a common confounder. You must actively consider this and you "
        "are expected to answer 'likely_spurious' when no credible mechanism "
        "exists. Inventing a confident mechanism for noise is the single worst "
        "thing you can do here.\n"
        "4. Respond with STRICT JSON ONLY. No prose before or after, no markdown "
        "code fences. Emit exactly the specified fields and nothing else."
    )


def _format_regime_history(candidate: CausalCandidate, max_windows: int = 8) -> str:
    """Compact, token-bounded summary of the candidate's regime history."""
    periods = candidate.regime_periods
    if not periods:
        return "No regime windows were recorded for this pair."

    active = sum(1 for p in periods if p.active)
    total = len(periods)
    span = f"{periods[0].start.isoformat()} to {periods[-1].end.isoformat()}"

    # Show the most recent few windows so the model sees that activity is
    # episodic / time-bound (the relationship is not a standing claim).
    tail = periods[-max_windows:]
    lines = []
    for p in tail:
        state = "ACTIVE (coupled)" if p.active else "inactive (decoupled)"
        mc = f", mean_corr={p.mean_correlation:+.3f}" if p.mean_correlation is not None else ""
        lines.append(
            f"    - {p.start.isoformat()} to {p.end.isoformat()}: {state}{mc}"
        )
    shown = "\n".join(lines)
    return (
        f"{total} windows over {span}; {active} active / {total - active} inactive. "
        f"This relationship is EPISODIC, not permanent. Most recent windows:\n{shown}"
    )


def build_user_prompt(
    candidate: CausalCandidate,
    *,
    in_graph: bool,
    edge_type: str | None = None,
    orientation_source: str | None = None,
) -> str:
    """Build the per-candidate user prompt.

    Parameters mirror what Layer 1 persists: the candidate carries the
    statistics; ``in_graph`` / ``edge_type`` / ``orientation_source`` come from
    the discovered PC graph (a candidate may be Granger-significant yet absent
    from the graph — that tension is exactly what the model must address).
    """
    a, b = candidate.asset_a, candidate.asset_b
    a_name, b_name = asset_name(a), asset_name(b)
    corr = (
        f"{candidate.correlation_strength:+.3f}"
        if candidate.correlation_strength is not None
        else "n/a"
    )

    if in_graph:
        pc_block = (
            f"PC GRAPH STATUS: KEPT as a direct edge "
            f"(edge_type={edge_type or 'n/a'}, oriented by "
            f"{orientation_source or 'n/a'}). PC's conditional-independence test "
            f"did NOT screen this relationship off via any third asset, so it is "
            f"consistent with a direct link.\n"
            f"Set addresses_pc_rejection=false and pc_rejection_reasoning=null."
        )
    else:
        pc_block = (
            "PC GRAPH STATUS: REJECTED as a direct edge (in_graph=false). Granger "
            "flagged this as significant, but PC's conditional-independence test "
            "found it became independent once other assets were conditioned on — "
            "so PC dropped it from the discovered graph. This is a TENSION you "
            "MUST address: a strong predictive signal that is NOT a direct edge "
            "usually means the relationship is MEDIATED (runs through a third "
            "asset) or CONFOUNDED (a common driver moves both). Explain the most "
            "likely such pathway for THIS pair.\n"
            "Set addresses_pc_rejection=true and fill pc_rejection_reasoning."
        )

    return (
        "Explain and rate the plausibility of the following statistical "
        "candidate. It has ALREADY passed Granger significance and FDR "
        "multiple-comparisons correction — your task is NOT to re-test it but to "
        "judge whether a real economic mechanism could explain it, or whether it "
        "is more likely spurious/confounded.\n"
        "\n"
        "CANDIDATE (directional, predictive precedence):\n"
        f"  Driver (precedes):   {a_name}  [{a}]\n"
        f"  Affected (follows):  {b_name}  [{b}]\n"
        f"  Direction:           {a_name} Granger-precedes {b_name}\n"
        f"  Lag:                 {candidate.lag} trading day(s)\n"
        f"  Raw Granger p:       {candidate.granger_p_value:.3e}\n"
        f"  Corrected p (FDR):   {candidate.corrected_p_value:.3e}\n"
        f"  Peak lead-lag corr:  {corr}\n"
        f"  Statistical conf.:   {candidate.statistical_confidence:.3f}\n"
        "\n"
        f"REGIME HISTORY: {_format_regime_history(candidate)}\n"
        "\n"
        f"{pc_block}\n"
        "\n"
        "Now decide:\n"
        "  (a) Is there a KNOWN economic transmission mechanism that plausibly "
        "links these two assets in THIS direction at THIS lag? Name the textbook "
        "channel if one exists.\n"
        "  (b) Could this instead be SPURIOUS or confounded by a common driver "
        "(e.g. rates, the US dollar, broad risk sentiment)? Say so if so.\n"
        "  (c) Do not assert causation; you are explaining a statistical finding.\n"
        "\n"
        "SELF-CHECK before you answer (this is mandatory):\n"
        f"  - If you name a mechanism_channel, its first endpoint MUST be {a_name} "
        f"and its last endpoint MUST be {b_name}. Re-read your channel: if either end "
        "names a DIFFERENT asset (e.g. you wrote about airlines, gold, or another "
        f"sector that is not {a_name} or {b_name}), you have pattern-matched a "
        "memorised phrase onto the wrong pair — delete it and either build a channel "
        f"that genuinely connects {a_name} to {b_name}, or set mechanism_channel=null "
        "and flag likely_spurious.\n"
        f"  - Do NOT force a channel. If no specific, honest mechanism links {a_name} "
        f"to {b_name}, that is a valid finding: null channel + likely_spurious.\n"
        "\n"
        "Respond with STRICT JSON ONLY matching exactly this schema "
        "(no markdown, no commentary):\n"
        f"{_output_schema(a_name, b_name)}"
    )
