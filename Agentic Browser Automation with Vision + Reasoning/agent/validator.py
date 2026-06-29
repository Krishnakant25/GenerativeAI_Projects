"""Moondream2 validation layer — fast second-opinion vision check before committing to a click."""

import logging
import os

import httpx

from browser.screenshot import encode_to_base64

logger = logging.getLogger("agent.validator")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VALIDATION_MODEL = "moondream"
VALIDATION_TIMEOUT = 30.0


async def validate_action(image_path: str, proposed_action: str, proposed_target: str) -> dict:
    """
    Quick binary check: does the proposed action make sense given what's visible on screen?
    Returns: {"valid": bool, "confidence": str, "reason": str}
    """
    try:
        image_b64 = encode_to_base64(image_path)
    except Exception as exc:
        logger.warning("validation skipped: %s", exc)
        return {"valid": True, "confidence": "low", "reason": f"validation error: {exc}"}
    if not image_b64:
        return {"valid": True, "confidence": "low", "reason": "could not encode image — skipping validation"}

    prompt = f"""Look at this browser screenshot.
Proposed action: {proposed_action}
Proposed target: {proposed_target}

Is this action reasonable given what you see on screen?
Answer in this EXACT format:
VALID: yes or no
REASON: one sentence"""

    try:
        async with httpx.AsyncClient(timeout=VALIDATION_TIMEOUT) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": VALIDATION_MODEL,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 64},
                },
            )
            if resp.status_code != 200:
                return {"valid": True, "confidence": "low", "reason": "validation model unavailable"}
            raw = resp.json().get("response", "")
            valid = "yes" in raw.lower().split("valid:")[-1][:10]
            reason_line = [l for l in raw.splitlines() if "REASON:" in l]
            reason = reason_line[0].split("REASON:")[-1].strip() if reason_line else raw[:80]
            return {"valid": valid, "confidence": "high", "reason": reason}
    except Exception as exc:
        logger.warning("validation skipped: %s", exc)
        return {"valid": True, "confidence": "low", "reason": f"validation error: {exc}"}
