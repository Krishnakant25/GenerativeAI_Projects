"""Screen Copilot — entry point.

Usage:
    python -m screen_copilot.main
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("screen_copilot/copilot.log", encoding="utf-8"),
    ],
)

from pynput import keyboard

from screen_copilot.models import CopilotConfig
from screen_copilot.overlay import CopilotOverlay
from screen_copilot.loop import CopilotLoop


def setup_hotkey(copilot):
    """Register Ctrl+Shift+Space to force an immediate suggestion."""
    def on_activate():
        logging.getLogger("main").info("hotkey: manual trigger pressed")
        copilot.force_suggestion = True

    hotkey = keyboard.GlobalHotKeys({
        '<ctrl>+<shift>+<space>': on_activate
    })
    hotkey.start()
    return hotkey


def setup_selection_hotkey(copilot):
    """Register Ctrl+Alt+S (analyze selection) and Ctrl+Alt+D (detail).

    Uses Alt-based combos to avoid conflicts with common browser shortcuts.
    """
    def on_analyze_selection():
        logging.getLogger("main").info("hotkey: analyze selection pressed")
        copilot.analyze_selection_requested = True

    def on_detail():
        logging.getLogger("main").info("hotkey: detail requested")
        copilot.detail_requested = True

    hk = keyboard.GlobalHotKeys({
        '<ctrl>+<alt>+s': on_analyze_selection,
        '<ctrl>+<alt>+d': on_detail
    })
    hk.start()
    return hk


def run_async_loop(loop_instance: CopilotLoop) -> None:
    """Run the async copilot loop in a background daemon thread."""
    asyncio.run(loop_instance.run())


def main() -> None:
    """Bootstrap overlay on main thread and async loop on a daemon thread."""
    config = CopilotConfig()

    overlay = CopilotOverlay(config)
    overlay.build()

    from screen_copilot.ocr import is_tesseract_available
    if not is_tesseract_available():
        logging.getLogger("main").warning(
            "Tesseract OCR not found — vision quality will be reduced. "
            "Install from https://github.com/UB-Mannheim/tesseract/wiki "
            "and set TESSERACT_PATH env var if needed."
        )

    copilot = CopilotLoop(config, overlay)
    overlay.on_more_requested = lambda: setattr(copilot, 'detail_requested', True)
    hotkey_listener = setup_hotkey(copilot)
    selection_hotkey = setup_selection_hotkey(copilot)
    bg_thread = threading.Thread(
        target=run_async_loop,
        args=(copilot,),
        daemon=True,
    )
    bg_thread.start()
    logging.getLogger("main").info("Screen Copilot started")

    try:
        overlay.run()
    except KeyboardInterrupt:
        pass
    finally:
        copilot.stop()
        logging.getLogger("main").info("Screen Copilot stopped")


if __name__ == "__main__":
    main()
