"""Screenshot utilities — captures, crops, encodes, and resizes browser screenshots."""

import asyncio
import base64
import io
import logging
import os
import pathlib
import time
from typing import Optional

from PIL import Image, ImageEnhance

from browser.controller import BrowserController

logger = logging.getLogger("browser.screenshot")

SCREENSHOT_DIR = os.getenv("SCREENSHOT_DIR", "screenshots")
MAX_DIMENSION = 1280  # resize if larger
JPEG_QUALITY = 85


async def capture(
    controller: BrowserController, task_id: str, step_number: int
) -> Optional[str]:
    """Capture a viewport screenshot to screenshots/{task_id}/step_{n:03d}.png."""
    if not controller.is_running or controller.page is None:
        logger.warning("capture called but browser is not running")
        return None
    try:
        task_dir = pathlib.Path(SCREENSHOT_DIR) / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        full_path = str(task_dir / f"step_{step_number:03d}.png")
        await controller.page.screenshot(path=full_path, full_page=False)
        logger.info("Screenshot saved: %s", full_path)
        return full_path
    except Exception as exc:
        logger.error("capture failed for task %s step %d: %s", task_id, step_number, exc)
        return None


async def capture_region(
    controller: BrowserController,
    x: int,
    y: int,
    width: int,
    height: int,
    task_id: str,
    step_number: int,
    suffix: str = "crop",
) -> Optional[str]:
    """Capture a full screenshot, crop to (x, y, x+width, y+height), and save it."""
    if not controller.is_running or controller.page is None:
        logger.warning("capture_region called but browser is not running")
        return None
    try:
        raw = await controller.page.screenshot(type="png")
        image = Image.open(io.BytesIO(raw))
        cropped = image.crop((x, y, x + width, y + height))

        task_dir = pathlib.Path(SCREENSHOT_DIR) / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        full_path = str(task_dir / f"step_{step_number:03d}_{suffix}.png")
        cropped.save(full_path)
        logger.info("Region screenshot saved: %s", full_path)
        return full_path
    except Exception as exc:
        logger.error(
            "capture_region failed for task %s step %d: %s", task_id, step_number, exc
        )
        return None


def encode_to_base64(image_path: str) -> Optional[str]:
    """Read an image file and return its base64-encoded contents as a UTF-8 string."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as exc:
        logger.error("encode_to_base64 failed for %s: %s", image_path, exc)
        return None


def preprocess_for_vision(image_path: str) -> str:
    """Enhance contrast and sharpness in place to improve LLaVA's UI readability.

    Returns the path unchanged whether or not preprocessing succeeded.
    """
    try:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")  # handles RGBA PNGs
        enhanced = ImageEnhance.Contrast(rgb).enhance(1.3)
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.5)
        enhanced.save(image_path)
        logger.info("Preprocessed %s for vision (contrast+sharpness)", image_path)
    except Exception as exc:
        logger.error("preprocess_for_vision failed for %s: %s", image_path, exc)
    return image_path


def resize_if_needed(image_path: str) -> str:
    """Resize an image in place if either dimension exceeds MAX_DIMENSION.

    Returns the path unchanged whether or not a resize was performed.
    """
    try:
        with Image.open(image_path) as image:
            width, height = image.size
            if width <= MAX_DIMENSION and height <= MAX_DIMENSION:
                return image_path

            scale = MAX_DIMENSION / max(width, height)
            new_size = (int(width * scale), int(height * scale))
            resized = image.resize(new_size, Image.LANCZOS)
            resized.save(image_path, quality=JPEG_QUALITY)
            logger.info(
                "Resized %s from %dx%d to %dx%d",
                image_path,
                width,
                height,
                new_size[0],
                new_size[1],
            )
    except Exception as exc:
        logger.error("resize_if_needed failed for %s: %s", image_path, exc)
    return image_path
