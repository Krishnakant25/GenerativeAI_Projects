"""Tesseract OCR pre-pass — extracts exact on-screen text before vision analysis."""

import logging
import os

import pytesseract
from PIL import Image

logger = logging.getLogger("copilot.ocr")

# Windows often needs explicit path — set via env var with sensible default
TESSERACT_PATH = os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def extract_text(image_path: str, max_chars: int = 1500) -> str:
    """
    Extract visible text from a screenshot using Tesseract OCR.
    Returns empty string on failure — never raises.
    """
    try:
        img = Image.open(image_path)
        raw_text = pytesseract.image_to_string(img)
        # Clean up: remove excessive blank lines, strip
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        cleaned = "\n".join(lines)
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars] + "... [truncated]"
        logger.info("ocr: extracted %d chars from %s", len(cleaned), image_path)
        return cleaned
    except Exception as e:
        logger.warning("ocr: extraction failed for %s: %s", image_path, e)
        return ""


def is_tesseract_available() -> bool:
    """Check if Tesseract is properly configured and working."""
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception as e:
        logger.warning("ocr: tesseract not available: %s", e)
        return False
