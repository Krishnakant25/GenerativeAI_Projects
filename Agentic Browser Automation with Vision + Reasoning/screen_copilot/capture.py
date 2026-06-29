"""Active window screenshot capture using mss and pygetwindow."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Optional

import mss
import mss.tools
from PIL import Image, ImageEnhance
import pygetwindow as gw

logger = logging.getLogger("copilot.capture")

SCREENSHOT_DIR = "screen_copilot/screenshots"

_last_screenshot_hash = {"value": None}

BLACKLISTED_APPS = ["1Password", "Bitwarden", "KeePass", "LastPass",
                    "Dashlane", "Banking", "Wallet", "Keychain",
                    "Authenticator"]
BLACKLIST_KEYWORDS = ["password", "login", "sign in", "credit card",
                      "bank", "ssn", "social security", "private key",
                      "seed phrase"]


def is_blacklisted(title: str) -> bool:
    title_lower = title.lower()
    for app in BLACKLISTED_APPS:
        if app.lower() in title_lower:
            return True
    for kw in BLACKLIST_KEYWORDS:
        if kw in title_lower:
            return True
    return False


def has_screen_changed(image_path: str, threshold: float = 0.05) -> bool:
    """
    Compares current screenshot to the previous one using average hash.
    Returns True if the screen changed significantly, False if nearly identical.
    Used to skip expensive vision calls when nothing changed.
    """
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            # downscale to 16x16 grayscale for fast comparison
            small = img.convert("L").resize((16, 16))
            pixels = list(small.getdata())
            avg = sum(pixels) / len(pixels)
            # build a simple hash: bit per pixel above/below average
            bits = "".join("1" if p > avg else "0" for p in pixels)

        prev = _last_screenshot_hash["value"]
        _last_screenshot_hash["value"] = bits

        if prev is None:
            return True  # first capture always counts as changed

        # hamming distance between bit strings
        diff = sum(c1 != c2 for c1, c2 in zip(prev, bits))
        change_ratio = diff / len(bits)
        return change_ratio > threshold
    except Exception as e:
        logger.warning("has_screen_changed failed: %s", e)
        return True  # on error, assume changed (safe default — still analyzes)


def get_active_window_info() -> dict:
    """Return title, app name, and pixel bounds of the currently active window."""
    try:
        window = gw.getActiveWindow()
        if window is None:
            return {"title": "Unknown", "app": "Unknown", "bounds": None}

        title = (window.title or "").strip()

        # Skip self-capture: if the overlay itself is the active window,
        # avoid feeding the copilot's own tip back into its reasoning.
        if "Screen Copilot" in title:
            logger.info("capture: active window is the overlay itself — skip cycle")
            return {"title": "SKIP_SELF", "app": "SKIP_SELF", "bounds": None}

        if is_blacklisted(title):
            logger.info("capture: BLACKLISTED window detected — skipping for privacy")
            return {"title": "BLACKLISTED", "app": "BLACKLISTED", "bounds": None}

        # Derive app name from the last segment of the title bar
        for sep in (" — ", " - "):
            if sep in title:
                app = title.split(sep)[-1].strip()
                break
        else:
            app = title[:20].strip()

        bounds = {
            "left": max(0, window.left),
            "top": max(0, window.top),
            "width": max(100, window.width),
            "height": max(100, window.height),
        }

        return {"title": title, "app": app, "bounds": bounds}

    except Exception as exc:
        logger.warning("get_active_window_info failed: %s", exc)
        return {"title": "Unknown", "app": "Unknown", "bounds": None}


async def capture_active_window(session_id: str, index: int) -> Optional[str]:
    """Capture the active window (or full screen) and save as PNG.

    Args:
        session_id: Unique session identifier used to namespace the save directory.
        index: Sequential capture index used in the filename.

    Returns:
        Absolute path to the saved PNG, or None on failure.
    """
    try:
        window_info = get_active_window_info()
        bounds = window_info.get("bounds")
        use_fullscreen = (
            bounds is None
            or bounds["width"] < 100
            or bounds["height"] < 100
        )

        if use_fullscreen:
            logger.info("capture: no valid window bounds — falling back to full screen")

        save_dir = Path(SCREENSHOT_DIR) / session_id
        save_dir.mkdir(parents=True, exist_ok=True)

        filename = f"obs_{index:04d}.png"
        full_path = str(save_dir / filename)

        with mss.mss() as sct:
            if use_fullscreen:
                monitor = sct.monitors[1]
            else:
                monitor = {
                    "left": bounds["left"],
                    "top": bounds["top"],
                    "width": bounds["width"],
                    "height": bounds["height"],
                }

            sct_img = sct.grab(monitor)
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            img.save(full_path)

        logger.info(
            "capture: saved %s (%s — %s)",
            full_path,
            window_info["app"],
            window_info["title"][:40],
        )
        return full_path

    except Exception as exc:
        logger.error("capture_active_window failed: %s", exc)
        return None


def preprocess(image_path: str) -> str:
    """Enhance screenshot contrast and sharpness; resize if oversized.

    Args:
        image_path: Path to the PNG to process (modified in-place).

    Returns:
        The same path (unchanged on any failure).
    """
    try:
        img = Image.open(image_path).convert("RGB")

        img = ImageEnhance.Contrast(img).enhance(1.2)
        img = ImageEnhance.Sharpness(img).enhance(1.3)

        w, h = img.size
        if w > 1280 or h > 1280:
            scale = 1280 / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        img.save(image_path)
    except Exception as exc:
        logger.warning("preprocess failed for %s: %s", image_path, exc)

    return image_path
