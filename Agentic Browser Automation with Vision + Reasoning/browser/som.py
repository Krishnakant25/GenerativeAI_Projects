"""Set-of-Mark prompting helpers — draw numbered boxes onto screenshots for llava."""

import logging
import re
from dataclasses import dataclass
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("browser.som")

VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 800
_GRID_COLS = 4
_GRID_ROWS = 3
_PADDING = 4
_BOX_WIDTH = 120
_BOX_HEIGHT = 50
_CIRCLE_RADIUS = 12

_GRID_LABELS = {
    1: "top-left", 2: "top-center-left", 3: "top-center-right", 4: "top-right",
    5: "middle-left", 6: "middle-center-left", 7: "middle-center-right", 8: "middle-right",
    9: "bottom-left", 10: "bottom-center-left", 11: "bottom-center-right", 12: "bottom-right",
}


@dataclass
class SoMRegion:
    number: int
    x: int        # top-left x
    y: int        # top-left y
    width: int
    height: int
    label: str    # descriptive label e.g. "search box", "button"

    @property
    def center_x(self) -> int:
        return self.x + self.width // 2

    @property
    def center_y(self) -> int:
        return self.y + self.height // 2


def generate_regions_from_viewport(
    viewport_width: int = VIEWPORT_WIDTH,
    viewport_height: int = VIEWPORT_HEIGHT,
) -> list[SoMRegion]:
    """Return a 4×3 grid of SoMRegions covering the full viewport."""
    cell_w = viewport_width // _GRID_COLS
    cell_h = viewport_height // _GRID_ROWS
    regions: list[SoMRegion] = []
    number = 1
    for row in range(_GRID_ROWS):
        for col in range(_GRID_COLS):
            regions.append(
                SoMRegion(
                    number=number,
                    x=col * cell_w + _PADDING,
                    y=row * cell_h + _PADDING,
                    width=cell_w - 2 * _PADDING,
                    height=cell_h - 2 * _PADDING,
                    label=_GRID_LABELS[number],
                )
            )
            number += 1
    return regions


def generate_regions_from_elements(elements: list[dict]) -> list[SoMRegion]:
    """Convert clickable_elements dicts (label, x, y) into SoMRegions.

    Each element is a box of fixed size centred on (x, y), clamped to the viewport.
    Falls back to generate_regions_from_viewport() when elements is empty.
    """
    if not elements:
        return generate_regions_from_viewport()

    regions: list[SoMRegion] = []
    for i, elem in enumerate(elements, start=1):
        cx = int(elem.get("x", 0))
        cy = int(elem.get("y", 0))
        x = max(0, cx - _BOX_WIDTH // 2)
        y = max(0, cy - _BOX_HEIGHT // 2)
        # Clamp right/bottom edges to viewport
        x = min(x, VIEWPORT_WIDTH - _BOX_WIDTH)
        y = min(y, VIEWPORT_HEIGHT - _BOX_HEIGHT)
        regions.append(
            SoMRegion(
                number=i,
                x=x,
                y=y,
                width=_BOX_WIDTH,
                height=_BOX_HEIGHT,
                label=str(elem.get("label", f"element-{i}")),
            )
        )
    return regions


def draw_marks(
    image_path: str,
    regions: list[SoMRegion],
    output_path: Optional[str] = None,
) -> str:
    """Draw numbered red boxes onto a screenshot and save the result.

    Returns output_path on success, or image_path unchanged on any failure.
    """
    if output_path is None:
        output_path = image_path
    try:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except (IOError, OSError):
            font = ImageFont.load_default()

        for region in regions:
            # Red border rectangle
            draw.rectangle(
                [region.x, region.y, region.x + region.width, region.y + region.height],
                outline="#FF0000",
                width=2,
            )
            # Filled red circle at top-left corner of the box
            cx, cy = region.x, region.y
            draw.ellipse(
                [cx - _CIRCLE_RADIUS, cy - _CIRCLE_RADIUS,
                 cx + _CIRCLE_RADIUS, cy + _CIRCLE_RADIUS],
                fill="#FF0000",
            )
            # White number centred in the circle
            text = str(region.number)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((cx - tw // 2, cy - th // 2), text, fill="white", font=font)

        image.save(output_path)
        logger.info("draw_marks: drew %d marks onto %s", len(regions), output_path)
    except Exception as exc:  # noqa: BLE001 — drawing must never break the agent loop
        logger.error("draw_marks failed for %s: %s", image_path, exc)
        return image_path
    return output_path


def pick_region(number: int, regions: list[SoMRegion]) -> Optional[SoMRegion]:
    """Return the SoMRegion with the given number, or None if not found."""
    for region in regions:
        if region.number == number:
            return region
    return None


def parse_som_response(raw: Optional[str]) -> Optional[int]:
    """Extract the first integer from a SoM llava response (the chosen region number)."""
    if not raw:
        return None
    match = re.search(r"\b(\d+)\b", raw)
    return int(match.group(1)) if match else None
