"""Vision module — LLaVA:13b wrapper for screenshot analysis only (no reasoning/planning)."""

import asyncio
import base64
import copy
import hashlib
import logging
import os
import re
from typing import Optional

import httpx

from agent.models import VisionOutput
from browser.screenshot import (
    encode_to_base64,
    capture_region,
    resize_if_needed,
    preprocess_for_vision,
)
from browser.som import (
    generate_regions_from_elements,
    generate_regions_from_viewport,
    draw_marks,
    pick_region,
    parse_som_response,
    SoMRegion,
)

logger = logging.getLogger("agent.vision")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VISION_MODEL = os.getenv("VISION_MODEL", "llava:13b")
VISION_TIMEOUT = 120.0  # seconds — llava:13b is slow on RTX 2060
MAX_RETRIES = 3

_COORD_PATTERN = re.compile(r"\((\d+),\s*(\d+)\)")

# In-memory, process-scoped cache keyed by md5 of the (preprocessed) screenshot bytes.
_VISION_CACHE_MAX = 50
_vision_cache: dict[str, VisionOutput] = {}

_VISION_PROMPT_TEMPLATE = """You are a precise browser UI analyzer. Study this screenshot carefully.

Current goal: {goal}
Step: {step_number}

IMPORTANT RULES:
- Only report elements you can actually see in the image
- Coordinates must be pixel positions within a 1280x800 viewport
- If you cannot see a search box, report SEARCH_BOX: no
- Do not guess or invent elements

Respond in this EXACT format — no other text:

PAGE_TYPE: [search_results | article | form | homepage | error | other]
VISIBLE_TEXT: [key text visible on page, comma separated, max 10 items]
CLICKABLE_ELEMENTS: [list as "label:(x,y)", comma separated, max 8 items]
SEARCH_BOX: [yes:(x,y) | no]
CURRENT_URL_VISIBLE: [yes | no]
SUGGESTED_ACTION: [click | type | scroll | navigate | extract | finish]
SUGGESTED_TARGET: [element label or (x,y) or text string]
CONFIDENCE: [high | medium | low]
REASONING: [one sentence — what you see and why this action]"""


async def analyze_screenshot(
    image_path: str,
    goal: str,
    step_number: int,
    controller=None,
    task_id: Optional[str] = None,
) -> VisionOutput:
    """Send a screenshot to llava:13b and return structured UI analysis.

    Never raises — returns a VisionOutput with success=False on any failure.
    When `controller` and `task_id` are supplied, a low-confidence result
    triggers an automatic 300x150 cropped-region retry around the first
    clickable element. analyze_region's internal call back into this
    function omits them, so the retry can never recurse.
    """
    try:
        resize_if_needed(image_path)
        preprocess_for_vision(image_path)

        with open(image_path, "rb") as f:
            cache_key = hashlib.md5(f.read()).hexdigest()
        if cache_key in _vision_cache:
            logger.info("vision: cache hit for step %s", step_number)
            return _vision_cache[cache_key]

        image_b64 = encode_to_base64(image_path)
        if image_b64 is None:
            logger.error("analyze_screenshot: encoding failed for %s", image_path)
            return VisionOutput(success=False, raw_response="encoding failed")

        prompt = _VISION_PROMPT_TEMPLATE.format(goal=goal, step_number=step_number)
        raw_response = await _call_llava(image_b64, prompt)
        vision_output = _parse_llava_response(raw_response)

        if (
            vision_output.confidence == "low"
            and vision_output.clickable_elements
            and controller is not None
            and task_id is not None
        ):
            first = vision_output.clickable_elements[0]
            region_output = await analyze_region(
                controller,
                int(first.get("x", 0)),
                int(first.get("y", 0)),
                300,
                150,
                task_id,
                step_number,
                goal,
            )
            if region_output.confidence != "low":
                logger.info(
                    "vision: low confidence on full screenshot — region retry succeeded"
                )
                vision_output = region_output

        if vision_output.success:  # never cache failed analyses — they deserve a re-run
            _vision_cache[cache_key] = vision_output
            if len(_vision_cache) > _VISION_CACHE_MAX:
                _vision_cache.pop(next(iter(_vision_cache)))

        return vision_output
    except Exception as exc:  # noqa: BLE001 — public function must never raise
        logger.error("analyze_screenshot failed for %s: %s", image_path, exc)
        return VisionOutput(success=False, raw_response=f"analyze_screenshot error: {exc}")


async def analyze_region(
    controller,
    x: int,
    y: int,
    width: int,
    height: int,
    task_id: str,
    step_number: int,
    goal: str,
) -> VisionOutput:
    """Fallback analysis on a cropped region when full-screenshot analysis is weak."""
    try:
        region_path = await capture_region(
            controller, x, y, width, height, task_id, step_number, suffix="retry"
        )
        if region_path is None:
            logger.error("analyze_region: capture_region returned None")
            return VisionOutput(success=False, raw_response="region capture failed")

        return await analyze_screenshot(region_path, goal, step_number)
    except Exception as exc:  # noqa: BLE001 — public function must never raise
        logger.error("analyze_region failed: %s", exc)
        return VisionOutput(success=False, raw_response=f"analyze_region error: {exc}")


async def _call_llava(image_b64: str, prompt: str) -> Optional[str]:
    """POST to Ollama /api/generate with retries. Returns the response text or None."""
    payload = {
        "model": VISION_MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 512},
    }
    url = f"{OLLAMA_BASE_URL}/api/generate"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=VISION_TIMEOUT) as client:
                response = await client.post(url, json=payload)
                if response.status_code != 200:
                    raise httpx.HTTPStatusError(
                        f"status {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                return response.json()["response"]
        except Exception as exc:  # noqa: BLE001 — retry on any failure
            logger.warning("LLaVA call attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                await asyncio.sleep(2.0)

    logger.error("LLaVA call exhausted all %d retries", MAX_RETRIES)
    return None


def _parse_llava_response(raw: Optional[str]) -> VisionOutput:
    """Parse llava's structured text response into a VisionOutput, line by line."""
    if not raw:
        return VisionOutput(success=False, raw_response="no response")

    try:
        result = VisionOutput(raw_response=raw)
        fields_parsed = 0
        suggested_action_present = False

        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped or ":" not in stripped:
                continue
            prefix, _, value = stripped.partition(":")
            key = prefix.strip().upper()
            value = value.strip()

            if key == "PAGE_TYPE":
                result.page_type = value or "unknown"
                fields_parsed += 1
            elif key == "VISIBLE_TEXT":
                result.visible_text = [
                    item.strip() for item in value.split(",") if item.strip()
                ]
                fields_parsed += 1
            elif key == "CLICKABLE_ELEMENTS":
                result.clickable_elements = _parse_clickable_elements(value)
                fields_parsed += 1
            elif key == "SEARCH_BOX":
                present, coords = _parse_search_box(value)
                result.search_box_present = present
                result.search_box_coords = coords
                fields_parsed += 1
            elif key == "SUGGESTED_ACTION":
                if value:
                    result.suggested_action = value
                    suggested_action_present = True
                fields_parsed += 1
            elif key == "SUGGESTED_TARGET":
                result.suggested_target = value
                fields_parsed += 1
            elif key == "CONFIDENCE":
                result.confidence = value.lower() if value.lower() in (
                    "high",
                    "medium",
                    "low",
                ) else "low"
                fields_parsed += 1
            elif key == "REASONING":
                result.reasoning = value
                fields_parsed += 1

        result.success = suggested_action_present
        if fields_parsed < 4:
            logger.warning(
                "LLaVA response parsed only %d fields — marking unsuccessful", fields_parsed
            )
            result.success = False

        return result
    except Exception as exc:  # noqa: BLE001 — never raise out of parser
        logger.error("_parse_llava_response failed: %s", exc)
        return VisionOutput(success=False, raw_response=raw)


def _parse_clickable_elements(value: str) -> list[dict]:
    """Extract [{"label": str, "x": int, "y": int}, ...] from a comma-separated list."""
    elements: list[dict] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        match = _COORD_PATTERN.search(item)
        if not match:
            continue
        label = item[: match.start()].rstrip(": ").strip()
        elements.append(
            {"label": label, "x": int(match.group(1)), "y": int(match.group(2))}
        )
    return elements


def _parse_search_box(value: str) -> tuple[bool, Optional[tuple[int, int]]]:
    """Parse the SEARCH_BOX field: 'yes:(x,y)' -> (True, (x, y)); else (False, None)."""
    if value.lower().startswith("yes"):
        match = _COORD_PATTERN.search(value)
        if match:
            return True, (int(match.group(1)), int(match.group(2)))
        return True, None
    return False, None


_SOM_PROMPT_TEMPLATE = """You are controlling a web browser. Numbered boxes are drawn on this screenshot.
Current goal: {goal}

Step: {step_number}
Available numbered regions:

{region_list}
Which numbered region should be interacted with to make progress toward the goal?
Respond in this EXACT format — no other text:
REGION: [number from the image]

ACTION: [click | type | scroll | extract | finish | navigate]

VALUE: [text to type if action is type, otherwise empty]

CONFIDENCE: [high | medium | low]

REASONING: [one sentence explaining your choice]"""


async def analyze_screenshot_som(
    image_path: str,
    goal: str,
    step_number: int,
    controller=None,
    task_id: Optional[str] = None,
) -> tuple[VisionOutput, list]:
    """SoM-enhanced screenshot analysis — draw numbered boxes, ask llava to pick one.

    Returns (VisionOutput, list[SoMRegion]).  Falls back to the plain VisionOutput
    with an empty region list on any failure so the caller always gets a usable result.
    Never raises.
    """
    try:
        # Step 1: standard vision pass to get initial element hints
        initial_vision = await analyze_screenshot(
            image_path, goal, step_number, controller=controller, task_id=task_id
        )
        if not initial_vision.success:
            return initial_vision, []

        # Step 2: build regions from detected elements or fall back to grid
        if initial_vision.clickable_elements:
            regions = generate_regions_from_elements(initial_vision.clickable_elements)
        else:
            regions = generate_regions_from_viewport()

        # Step 3: draw marks onto a new file
        base, ext = os.path.splitext(image_path)
        marked_path = base + "_som" + ext
        draw_marks(image_path, regions, output_path=marked_path)

        # Step 4: encode marked image
        marked_b64 = encode_to_base64(marked_path)
        if marked_b64 is None:
            logger.warning("analyze_screenshot_som: encoding marked image failed — using initial result")
            return initial_vision, regions

        # Step 5: build and send SoM prompt
        region_list = "\n".join(
            f"{r.number}: {r.label} at ({r.center_x},{r.center_y})" for r in regions
        )
        som_prompt = _SOM_PROMPT_TEMPLATE.format(
            goal=goal, step_number=step_number, region_list=region_list
        )
        raw = await _call_llava(marked_b64, som_prompt)
        if raw is None:
            logger.warning("analyze_screenshot_som: llava call failed — using initial result")
            return initial_vision, regions

        # Step 6: parse SoM response line by line
        parsed_action = ""
        parsed_value = ""
        parsed_confidence = "low"
        parsed_reasoning = ""
        region_number: Optional[int] = None

        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped or ":" not in stripped:
                continue
            prefix, _, value = stripped.partition(":")
            key = prefix.strip().upper()
            value = value.strip()
            if key == "REGION":
                region_number = parse_som_response(value)
            elif key == "ACTION":
                parsed_action = value.lower()
            elif key == "VALUE":
                parsed_value = value
            elif key == "CONFIDENCE":
                parsed_confidence = value.lower() if value.lower() in ("high", "medium", "low") else "low"
            elif key == "REASONING":
                parsed_reasoning = value

        if region_number is None:
            logger.warning("analyze_screenshot_som: no region number parsed — using initial result")
            return initial_vision, regions

        region = pick_region(region_number, regions)
        if region is None:
            logger.warning("analyze_screenshot_som: region %d not in region list — using initial result", region_number)
            return initial_vision, regions

        # Step 7: build enhanced VisionOutput from initial fields + SoM overrides
        enhanced = VisionOutput(
            success=True,
            page_type=initial_vision.page_type,
            visible_text=initial_vision.visible_text,
            clickable_elements=initial_vision.clickable_elements,
            search_box_present=initial_vision.search_box_present,
            search_box_coords=initial_vision.search_box_coords,
            suggested_action=parsed_action or initial_vision.suggested_action,
            suggested_target=f"({region.center_x},{region.center_y})",
            confidence=parsed_confidence,
            reasoning=parsed_reasoning,
            raw_response=raw,
        )

        logger.info(
            "vision SoM: selected region %d → (%d,%d) confidence=%s",
            region_number, region.center_x, region.center_y, parsed_confidence,
        )
        return enhanced, regions

    except Exception as exc:  # noqa: BLE001 — public function must never raise
        logger.error("analyze_screenshot_som failed: %s", exc)
        return VisionOutput(success=False, raw_response=f"analyze_screenshot_som error: {exc}"), []
