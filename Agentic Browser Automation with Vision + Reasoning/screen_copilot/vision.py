"""llava:13b vision wrapper — describes active window content for the copilot."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import os
from datetime import datetime
from typing import Optional

import httpx

from screen_copilot.models import Observation
from screen_copilot.ocr import extract_text

logger = logging.getLogger("copilot.vision")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VISION_MODEL = os.getenv("VISION_MODEL", "llava:7b")
VISION_TIMEOUT = 90.0
MAX_RETRIES = 2

_vision_cache: dict = {}

_PROMPT_TEMPLATE = """\
You are an AI assistant analyzing a screenshot to understand exactly
what the person is reading, working on, or looking at — be specific,
not generic.

Active window: {app_name}
Window title: {window_title}
{ocr_section}
Use the EXACT TEXT above as your primary source of truth for what's
on screen — it is more reliable than visual interpretation. Use the
image only for layout, visual context, and confirming what app/page this is.

Identify the SPECIFIC subject matter from the extracted text, not
just the app category.

Respond in this EXACT format — no other text:

ACTIVITY: [specific phrase based on the actual text content]
APP_TYPE: [coding | writing | browsing | email | spreadsheet | video | documentation | other]
TOPIC: [the specific subject/title/heading from the extracted text]
KEY_DETAIL: [one specific fact, term, or detail quoted/paraphrased from the extracted text]
FOCUS_ITEM: [the most prominent heading or item from the extracted text]
CONTEXT_SUMMARY: [one sentence describing what the person appears to be learning or doing, referencing the TOPIC]
SUGGESTION: [one specific, actionable suggestion DIRECTLY RELATED to the TOPIC]
CONFIDENCE: [high | medium | low]\
"""

_DEFAULTS = {
    "activity": "unknown",
    "app_type": "other",
    "topic": "",
    "key_detail": "",
    "focus_item": "",
    "context_summary": "Could not analyze screen",
    "suggestion": "",
    "confidence": "low",
    "raw_response": "",
}


def encode_image(image_path: str) -> Optional[str]:
    """Read an image file and return its base64-encoded UTF-8 string."""
    try:
        with open(image_path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("utf-8")
    except Exception as exc:
        logger.warning("encode_image failed for %s: %s", image_path, exc)
        return None


async def analyze_window(
    image_path: str,
    window_title: str,
    app_name: str,
) -> dict:
    """Send screenshot to llava:13b and return a structured analysis dict.

    Args:
        image_path: Path to the captured PNG.
        window_title: Title bar text of the active window.
        app_name: Application name derived from the title.

    Returns:
        Dict with keys: activity, app_type, focus_item, context_summary,
        suggestion, confidence, raw_response.
    """
    # Cache key from file contents so identical frames are not re-analysed
    try:
        with open(image_path, "rb") as fh:
            cache_key = hashlib.md5(fh.read()).hexdigest()
    except Exception:
        cache_key = None

    if cache_key and cache_key in _vision_cache:
        logger.debug("vision: cache hit for %s", image_path)
        return _vision_cache[cache_key]

    image_b64 = encode_image(image_path)
    if image_b64 is None:
        return {**_DEFAULTS, "raw_response": ""}

    ocr_text = extract_text(image_path)
    ocr_section = (
        f"\n\nEXACT TEXT EXTRACTED FROM SCREEN (via OCR — this is precise, trust it):\n{ocr_text}\n"
        if ocr_text else "\n\nNo text could be extracted from this screen.\n"
    )

    prompt = _PROMPT_TEMPLATE.format(
        app_name=app_name, window_title=window_title, ocr_section=ocr_section
    )
    raw = await _call_llava(image_b64, prompt)
    result = _parse_response(raw)

    if cache_key and result["confidence"] != "low":
        if len(_vision_cache) >= 30:
            # Drop the oldest entry
            oldest = next(iter(_vision_cache))
            del _vision_cache[oldest]
        _vision_cache[cache_key] = result

    return result


async def _call_llava(image_b64: str, prompt: str) -> Optional[str]:
    """POST to Ollama /api/generate and return the model response text."""
    payload = {
        "model": VISION_MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 320},
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=VISION_TIMEOUT) as client:
                resp = await client.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json=payload,
                )
                resp.raise_for_status()
                return resp.json().get("response", "")
        except Exception as exc:
            logger.warning("_call_llava attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                await asyncio.sleep(1.5)

    return None


def _parse_response(raw: Optional[str]) -> dict:
    """Parse llava's structured KEY: value response into a dict."""
    result = {**_DEFAULTS, "raw_response": raw or ""}

    if not raw:
        return result

    try:
        key_map = {
            "ACTIVITY": "activity",
            "APP_TYPE": "app_type",
            "TOPIC": "topic",
            "KEY_DETAIL": "key_detail",
            "FOCUS_ITEM": "focus_item",
            "CONTEXT_SUMMARY": "context_summary",
            "SUGGESTION": "suggestion",
            "CONFIDENCE": "confidence",
        }
        for line in raw.splitlines():
            line = line.strip()
            for prefix, field in key_map.items():
                if line.startswith(f"{prefix}:"):
                    value = line[len(prefix) + 1:].strip()
                    value = value.strip('"').strip("'")
                    result[field] = value
                    break
    except Exception as exc:
        logger.warning("_parse_response failed: %s", exc)
        return {**_DEFAULTS, "raw_response": raw}

    return result


async def build_observation(
    image_path: str,
    window_title: str,
    app_name: str,
    session_id: str,
) -> Observation:
    """Combine vision analysis and metadata into a single Observation.

    Args:
        image_path: Path to the captured screenshot.
        window_title: Active window title.
        app_name: Application name.
        session_id: Current copilot session ID (not stored here, passed to storage).

    Returns:
        A fully populated Observation (vision_summary, suggested_action, confidence set).
    """
    result = await analyze_window(image_path, window_title, app_name)

    return Observation(
        timestamp=datetime.utcnow(),
        window_title=window_title,
        app_name=app_name,
        screenshot_path=image_path,
        vision_summary=result["context_summary"],
        suggested_action=result["suggestion"],
        confidence=result["confidence"],
        topic=result.get("topic", ""),
        key_detail=result.get("key_detail", ""),
    )
