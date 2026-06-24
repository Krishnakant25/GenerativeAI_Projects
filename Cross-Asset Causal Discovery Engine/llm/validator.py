"""Layer-2 validator — async, local-LLM plausibility / mechanism layer.

Takes the ``CausalCandidate``s that survived Layer 1 (Granger significance +
FDR/Bonferroni correction), asks the local model for an economic mechanism and
a plausibility judgement, parses the strict-JSON response into a
``HypothesisCard``, and (optionally) persists the batch.

CRITICAL FRAMING (repeated here on purpose): the plausibility flag is a
HEURISTIC FILTER, not validation. An LLM can — and, given a spurious input,
sometimes will — produce a fluent, confident economic mechanism for a
relationship that is actually noise. This module is built to *survive* that
(the model is given a sanctioned ``likely_spurious`` flag and a caveats field),
not to assume it away. The validator NEVER creates or modifies the underlying
statistic; ``card.candidate`` is the Layer-1 object passed through untouched.

Engineering hard rules honoured here:
  * ALL Ollama calls are async (``ollama.AsyncClient``).
  * Output is forced to JSON (``format="json"``) and parsed robustly: stray
    fences stripped, one retry on malformed output, then the card is flagged
    ``PARSE_FAILED`` rather than crashing the batch.
  * Deterministic where possible: temperature pinned to 0.0 and a fixed seed,
    so re-runs reproduce.
  * Ollama being unreachable raises ``OllamaUnreachableError`` (a clean signal
    the API turns into a 503), never a leaked traceback.

Probed Ollama interface (ollama-python 0.6.2, model llama3.1:8b-instruct-q4_0):
    resp = await AsyncClient(host).chat(
        model=..., messages=[{role, content}, ...],
        format="json", options={"temperature": 0.0, "seed": 7, "num_predict": N},
    )
    resp is a ChatResponse (pydantic); content at resp["message"]["content"].
    format="json" returns a clean JSON object with no markdown fences.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

try:  # the project ships httpx2 (an httpx-compatible fork) in some envs
    import httpx
except ModuleNotFoundError:  # pragma: no cover - depends on the local env
    import httpx2 as httpx  # type: ignore[no-redef]

from ollama import AsyncClient

from causal.models import CausalCandidate
from config import DB_PATH as DEFAULT_DB_PATH
from config import asset_name
from db import storage
from llm.models import (
    DEFAULT_MODEL_NAME,
    LLM_EMITTABLE_FLAGS,
    HypothesisCard,
    PlausibilityFlag,
)
from llm.prompts import build_system_prompt, build_user_prompt

logger = logging.getLogger("causal_engine.llm.validator")

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 7
# JSON output is ~7 short fields; 800 tokens is comfortable headroom and the
# model stops early when the object closes under format="json".
DEFAULT_NUM_PREDICT = 800

# A chat callable matches ``AsyncClient.chat``: kwargs in, awaitable response
# out. Injecting one lets tests supply canned / malformed output without Ollama.
ChatFn = Callable[..., Awaitable[Any]]

# Connection-level failures that mean "Ollama isn't there", as opposed to the
# model returning something unparseable (which is a per-card PARSE_FAILED).
_CONNECTION_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    ConnectionError,
    OSError,
)


class OllamaUnreachableError(RuntimeError):
    """Raised when the local Ollama server cannot be reached. The message is
    safe to surface to a caller / API client."""


# ---------------------------------------------------------------------------
# Robust JSON extraction
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def extract_json(text: str) -> dict | None:
    """Best-effort parse of a model response into a JSON object.

    ``format="json"`` usually returns a clean object, but this stays defensive:
    strips markdown fences, then falls back to the first ``{`` … last ``}``
    span. Returns ``None`` if no JSON object can be recovered (caller decides
    whether to retry or flag PARSE_FAILED).
    """
    if not text or not text.strip():
        return None

    cleaned = _FENCE_RE.sub("", text).strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: grab the outermost {...} span and try again.
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(cleaned[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None
    return None


# Tokens that appear in many asset names and so carry no disambiguating power —
# matching on them would let a channel for the wrong pair look "correct". "usd"
# is dropped because all three currency pairs share it (the distinctive token is
# jpy / inr / eur).
_GENERIC_ASSET_TOKENS = frozenset(
    {"etf", "sector", "index", "fund", "the", "and", "for", "usd"}
)


def _asset_tokens(ticker: str) -> set[str]:
    """Distinctive lowercase tokens that identify an asset in free text: words
    from its human name plus its cleaned ticker, minus generic filler."""
    raw = f"{asset_name(ticker)} {ticker}"
    tokens = {t for t in re.split(r"[^a-z0-9]+", raw.lower()) if len(t) >= 3}
    return tokens - _GENERIC_ASSET_TOKENS


def _channel_references_both_assets(channel: str, candidate: CausalCandidate) -> bool:
    """True iff the named channel text references BOTH endpoints of the candidate.

    This is the structural backstop for the rationalisation failure mode where
    the model pastes a memorised textbook phrase (e.g. an oil->airlines channel)
    onto a pair it does not actually describe (e.g. Financials->Crude). A channel
    that fails to mention one of its own two endpoints is treated as mismatched.
    """
    text = channel.lower()
    a_tokens = _asset_tokens(candidate.asset_a)
    b_tokens = _asset_tokens(candidate.asset_b)
    a_hit = any(tok in text for tok in a_tokens)
    b_hit = any(tok in text for tok in b_tokens)
    return a_hit and b_hit


def _coerce_flag(value: object) -> PlausibilityFlag | None:
    """Map a model-emitted flag string to a PlausibilityFlag, tolerating case /
    whitespace. Returns None if it is not a flag the LLM is allowed to emit
    (PARSE_FAILED is never accepted from the model)."""
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v in LLM_EMITTABLE_FLAGS:
        return PlausibilityFlag(v)
    return None


def _coerce_confidence(value: object) -> float | None:
    """Parse and clamp the LLM confidence to [0, 1]; None if not a number."""
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, f))


class OllamaValidator:
    """Async Layer-2 validator over the local Ollama model.

    Inject ``chat_fn`` (or a whole ``client`` exposing ``.chat``) to test
    without a live server. By default it talks to ``llama3.1:8b-instruct-q4_0``
    at ``localhost:11434`` with temperature 0 and a fixed seed.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL_NAME,
        host: str = DEFAULT_OLLAMA_HOST,
        temperature: float = DEFAULT_TEMPERATURE,
        seed: int = DEFAULT_SEED,
        num_predict: int = DEFAULT_NUM_PREDICT,
        chat_fn: ChatFn | None = None,
        concurrency: int = 1,
    ) -> None:
        self.model = model
        self.host = host
        self.temperature = temperature
        self.seed = seed
        self.num_predict = num_predict
        self.concurrency = max(1, concurrency)
        # ``chat_fn`` short-circuits client construction (used by tests).
        self._chat_fn: ChatFn = chat_fn or AsyncClient(host=host).chat

    # -- low-level call ----------------------------------------------------

    async def _chat(self, messages: list[dict], *, seed: int) -> str:
        """One async chat call. Returns the raw message content string.

        Raises ``OllamaUnreachableError`` on connection-level failures so a dead
        server is never mistaken for a malformed response."""
        try:
            resp = await self._chat_fn(
                model=self.model,
                messages=messages,
                format="json",
                options={
                    "temperature": self.temperature,
                    "seed": seed,
                    "num_predict": self.num_predict,
                },
            )
        except _CONNECTION_ERRORS as exc:
            raise OllamaUnreachableError(
                f"Cannot reach Ollama at {self.host}. Is it running "
                f"(`ollama serve`) and is the model '{self.model}' pulled? [{exc}]"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            # ollama.ResponseError (e.g. model not found) and anything else that
            # isn't a clean response. Treat as "service problem", clean message.
            raise OllamaUnreachableError(
                f"Ollama call failed for model '{self.model}': {exc}"
            ) from exc

        # ChatResponse supports both attribute and mapping access.
        try:
            return resp["message"]["content"]
        except (KeyError, TypeError):
            return resp.message.content  # type: ignore[union-attr]

    # -- per-candidate -----------------------------------------------------

    async def validate_candidate(
        self,
        candidate: CausalCandidate,
        *,
        in_graph: bool,
        edge_type: str | None = None,
        orientation_source: str | None = None,
    ) -> HypothesisCard:
        """Produce one HypothesisCard for one already-significant candidate.

        Calls the model, parses strict JSON, retries once on malformed output,
        then falls back to a PARSE_FAILED card. Never raises on bad model output
        — only ``OllamaUnreachableError`` (connection) propagates."""
        system = build_system_prompt()
        user = build_user_prompt(
            candidate,
            in_graph=in_graph,
            edge_type=edge_type,
            orientation_source=orientation_source,
        )

        raw = ""
        correction: str | None = None  # set when a retry needs a specific nudge
        for attempt in range(2):  # one initial try + one retry
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            if correction is not None:
                # Nudge the retry with the SPECIFIC reason it failed, and vary the
                # seed so it doesn't deterministically reproduce the same output.
                messages.append({"role": "user", "content": correction})
            raw = await self._chat(messages, seed=self.seed + attempt)
            parsed = extract_json(raw)
            card = self._build_card(candidate, in_graph, parsed, raw)

            if card is None:
                correction = (
                    "Your previous response could not be parsed as JSON. "
                    "Respond again with STRICT JSON ONLY — no prose, no markdown "
                    "fences — using exactly the specified fields."
                )
                logger.warning(
                    "Unparseable LLM output for %s (attempt %d/2)",
                    candidate.candidate_id,
                    attempt + 1,
                )
                continue

            # Structural backstop: a plausible card whose named channel does not
            # reference both of THIS candidate's assets has pattern-matched a
            # memorised phrase onto the wrong pair. Retry once with an explicit
            # correction; if it still mismatches, flag it rather than passing a
            # mislabelled "plausible" card to the dashboard.
            if self._is_channel_mismatch(card):
                if attempt == 0:
                    a_name, b_name = asset_name(candidate.asset_a), asset_name(
                        candidate.asset_b
                    )
                    correction = (
                        f"Your mechanism_channel '{card.mechanism_channel}' does not "
                        f"connect the two assets in THIS candidate. It must start at "
                        f"'{a_name}' and end at '{b_name}'. You referenced different "
                        "assets — you pattern-matched a channel built for another pair. "
                        f"Either give a channel that genuinely links {a_name} to "
                        f"{b_name}, or set mechanism_channel to null and use "
                        "likely_spurious. Respond again with STRICT JSON ONLY."
                    )
                    logger.warning(
                        "Channel/asset mismatch for %s: %r (retrying)",
                        candidate.candidate_id,
                        card.mechanism_channel,
                    )
                    continue
                logger.warning(
                    "Channel/asset mismatch persisted for %s: %r -> MECHANISM_MISMATCH",
                    candidate.candidate_id,
                    card.mechanism_channel,
                )
                return self._mechanism_mismatch_card(card)

            return card

        # Both attempts failed to yield a valid card.
        return self._parse_failed_card(candidate, in_graph, raw)

    def _is_channel_mismatch(self, card: HypothesisCard) -> bool:
        """A 'plausible' card carrying a non-null channel that does not reference
        both of its candidate's assets. likely_spurious cards are exempt — they
        make no mechanism claim worth policing — as are null-channel cards."""
        plausible = card.plausibility_flag in {
            PlausibilityFlag.PLAUSIBLE_KNOWN_MECHANISM.value,
            PlausibilityFlag.PLAUSIBLE_NOVEL.value,
        }
        channel = card.mechanism_channel
        if not plausible or not channel:
            return False
        return not _channel_references_both_assets(channel, card.candidate)

    def _mechanism_mismatch_card(self, card: HypothesisCard) -> HypothesisCard:
        """Re-flag a card whose channel references the wrong assets. The original
        narrative is preserved (with a leading marker) so the failure is visible
        and auditable rather than silently dropped or rubber-stamped as plausible."""
        a_name, b_name = asset_name(card.asset_a), asset_name(card.asset_b)
        note = (
            f"MECHANISM_MISMATCH: the named channel '{card.mechanism_channel}' does "
            f"not connect {a_name} to {b_name}; the model attached a channel for a "
            "different pair. Not accepted as a plausible mechanism."
        )
        return card.model_copy(
            update={
                "plausibility_flag": PlausibilityFlag.MECHANISM_MISMATCH.value,
                "mechanism_channel": None,
                "caveats": [note, *card.caveats],
            }
        )

    def _build_card(
        self,
        candidate: CausalCandidate,
        in_graph: bool,
        parsed: dict | None,
        raw: str,
    ) -> HypothesisCard | None:
        """Construct a HypothesisCard from parsed JSON, or None if the output is
        missing required fields / has an unusable flag (→ caller retries)."""
        if parsed is None:
            return None

        flag = _coerce_flag(parsed.get("plausibility_flag"))
        confidence = _coerce_confidence(parsed.get("llm_confidence"))
        mechanism = parsed.get("mechanism_explanation")

        # Core contract: a usable flag, a numeric confidence, and a non-empty
        # mechanism string. Anything else is a malformed response.
        if flag is None or confidence is None or not isinstance(mechanism, str) or not mechanism.strip():
            return None

        caveats = parsed.get("caveats", [])  # HypothesisCard coerces str/None
        channel = parsed.get("mechanism_channel")
        if isinstance(channel, str) and not channel.strip():
            channel = None

        # addresses_pc_rejection is derived, not blindly trusted: it is only
        # meaningful when PC actually rejected the edge (in_graph=false). When it
        # did, we accept the model's engagement if it either set the flag or
        # supplied rejection reasoning — which we then surface as a caveat so the
        # tension is never lost.
        addresses = False
        pc_reasoning = parsed.get("pc_rejection_reasoning")
        if not in_graph:
            has_reasoning = isinstance(pc_reasoning, str) and bool(pc_reasoning.strip())
            addresses = has_reasoning or bool(parsed.get("addresses_pc_rejection"))
            if has_reasoning:
                caveats = (list(caveats) if isinstance(caveats, list) else [caveats]) + [
                    f"PC rejected this as a *direct* edge; {pc_reasoning.strip()}"
                ]

        # The confounder assessment, when present, becomes an explicit caveat —
        # the whole point is to keep the "could be spurious" reasoning visible.
        confounder = parsed.get("confounder_assessment")
        if isinstance(confounder, str) and confounder.strip():
            caveats = (list(caveats) if isinstance(caveats, list) else [caveats]) + [
                f"Confounder check: {confounder.strip()}"
            ]

        try:
            return HypothesisCard(
                card_id="",  # assigned on persist
                candidate=candidate,
                in_graph=in_graph,
                mechanism_explanation=mechanism.strip(),
                mechanism_channel=channel,
                plausibility_flag=flag,
                llm_confidence=confidence,
                caveats=caveats,
                addresses_pc_rejection=addresses,
                model_name=self.model,
                raw_response=raw,
            )
        except Exception:  # noqa: BLE001 - validation edge cases → treat as malformed
            logger.exception("HypothesisCard validation failed; treating as malformed")
            return None

    def _parse_failed_card(
        self, candidate: CausalCandidate, in_graph: bool, raw: str
    ) -> HypothesisCard:
        """A non-crashing placeholder card recording that the model's output
        could not be parsed after a retry. PARSE_FAILED is the validator's
        marker, not an LLM plausibility judgement."""
        return HypothesisCard(
            card_id="",
            candidate=candidate,
            in_graph=in_graph,
            mechanism_explanation=(
                "[parse failed] The model did not return parseable JSON matching "
                "the required schema after one retry. No mechanism was recorded."
            ),
            mechanism_channel=None,
            plausibility_flag=PlausibilityFlag.PARSE_FAILED,
            llm_confidence=0.0,
            caveats=["LLM output could not be parsed into the required schema."],
            addresses_pc_rejection=False,
            model_name=self.model,
            raw_response=raw,
        )

    # -- whole run ---------------------------------------------------------

    async def validate_run(
        self,
        run_id: str,
        *,
        db_path: Path | str = DEFAULT_DB_PATH,
        limit: int | None = None,
        progress: Callable[[int, int, HypothesisCard], None] | None = None,
    ) -> list[HypothesisCard]:
        """Validate every significant candidate of ``run_id`` and return the
        cards (NOT persisted here — the caller persists). Raises
        ``OllamaUnreachableError`` if the server is down (fails fast on the first
        candidate rather than producing a half-batch)."""
        candidates = storage.load_candidates(
            run_id, significant_only=True, db_path=db_path
        )
        if limit is not None:
            candidates = candidates[:limit]

        graph = storage.load_graph(run_id, db_path=db_path)
        graph_lookup: dict[tuple[str, str], tuple[str | None, str | None]] = {
            (e.source, e.target): (e.edge_type, e.orientation_source)
            for e in (graph.edges if graph else [])
        }

        total = len(candidates)
        cards: list[HypothesisCard] = [None] * total  # type: ignore[list-item]
        sem = asyncio.Semaphore(self.concurrency)
        done = 0
        done_lock = asyncio.Lock()

        async def _one(idx: int, cand: CausalCandidate) -> None:
            nonlocal done
            meta = graph_lookup.get((cand.asset_a, cand.asset_b))
            in_graph = meta is not None
            edge_type, orientation_source = meta if meta else (None, None)
            async with sem:
                card = await self.validate_candidate(
                    cand,
                    in_graph=in_graph,
                    edge_type=edge_type,
                    orientation_source=orientation_source,
                )
            cards[idx] = card
            if progress is not None:
                async with done_lock:
                    done += 1
                    progress(done, total, card)

        if total:
            await asyncio.gather(
                *(_one(i, c) for i, c in enumerate(candidates))
            )
        return cards

    async def validate_and_persist(
        self,
        run_id: str,
        *,
        db_path: Path | str = DEFAULT_DB_PATH,
        limit: int | None = None,
        progress: Callable[[int, int, HypothesisCard], None] | None = None,
    ) -> list[HypothesisCard]:
        """``validate_run`` + atomic persistence (replaces any prior cards for
        the run). Returns the cards written."""
        cards = await self.validate_run(
            run_id, db_path=db_path, limit=limit, progress=progress
        )
        storage.replace_hypothesis_cards(run_id, cards, db_path=db_path)
        return cards


def summarize_flags(cards: list[HypothesisCard]) -> dict[str, int]:
    """Count cards per plausibility flag (enum value → count), including any
    that are PARSE_FAILED. Stable key order for reporting."""
    counts = {f.value: 0 for f in PlausibilityFlag}
    for card in cards:
        # use_enum_values means plausibility_flag is already the str value.
        counts[card.plausibility_flag] = counts.get(card.plausibility_flag, 0) + 1
    return counts


# Liveness probes must FAIL FAST: if no Ollama is listening (e.g. a hosted demo
# host), we want a quick "unavailable" rather than a hang on the default socket
# timeout. The probe is a cheap local call, so a couple of seconds is ample.
OLLAMA_HEALTH_TIMEOUT_SECONDS = 2.0


async def ollama_available(
    host: str = DEFAULT_OLLAMA_HOST,
    model: str = DEFAULT_MODEL_NAME,
    timeout: float = OLLAMA_HEALTH_TIMEOUT_SECONDS,
) -> bool:
    """True if Ollama is reachable and ``model`` is present. Used by the API to
    degrade gracefully and by tests to gate the live behavioural check.

    Fast-fails on ``timeout`` seconds so a nonexistent Ollama (e.g. on a hosted
    demo host) returns False quickly instead of hanging on a connection attempt.
    """
    try:
        resp = await AsyncClient(host=host, timeout=timeout).list()
    except Exception:  # noqa: BLE001 - any failure (incl. timeout) means "not available"
        return False
    names = {m.get("model") or m.get("name") for m in resp.get("models", [])}
    return model in names
