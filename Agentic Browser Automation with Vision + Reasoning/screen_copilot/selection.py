"""Capture the user's current text selection via clipboard."""
import logging
import time

logger = logging.getLogger("copilot.selection")

def get_selected_text() -> str:
    """
    Copies the current selection to clipboard (simulates Ctrl+C),
    reads it, and returns the text. Returns empty string on failure.
    """
    try:
        import pyperclip
        from pynput.keyboard import Key, Controller

        keyboard = Controller()
        # Save current clipboard to restore later
        try:
            original = pyperclip.paste()
        except Exception:
            original = ""

        # Simulate Ctrl+C to copy current selection
        keyboard.press(Key.ctrl)
        keyboard.press('c')
        keyboard.release('c')
        keyboard.release(Key.ctrl)
        time.sleep(0.15)  # give the OS time to update clipboard

        selected = pyperclip.paste()

        # If nothing new was selected, selected == original
        if selected == original:
            logger.info("selection: no new text selected")
            return ""

        logger.info("selection: captured %d chars", len(selected))
        return selected.strip()
    except Exception as e:
        logger.warning("selection: capture failed: %s", e)
        return ""
