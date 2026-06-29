"""Main capture → analyse → suggest → display async loop."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Optional

from screen_copilot.models import CopilotConfig, Suggestion
from screen_copilot.capture import (
    capture_active_window,
    get_active_window_info,
    has_screen_changed,
    preprocess,
)
from screen_copilot.vision import build_observation
from screen_copilot.context import ContextManager
from screen_copilot.suggester import generate_suggestion, generate_app_switch_message
from screen_copilot.suggester import generate_detailed_explanation
from screen_copilot.storage import init_db, save_observation, save_suggestion
from screen_copilot.overlay import CopilotOverlay

logger = logging.getLogger("copilot.loop")


class CopilotLoop:
    """Orchestrates the periodic capture-analyse-suggest cycle."""

    def __init__(self, config: CopilotConfig, overlay: CopilotOverlay) -> None:
        self.config = config
        self.overlay = overlay
        self.context = ContextManager(max_size=config.max_context_size)
        self.session_id = str(uuid.uuid4())
        self.capture_index = 0
        self.last_suggestion_time: Optional[datetime] = None
        self.is_running = False
        self.force_suggestion: bool = False
        self.detail_requested: bool = False
        self.analyze_selection_requested: bool = False
        self.last_suggestion: Optional[Suggestion] = None
        self._last_tick_had_change = True
        from screen_copilot.audit import AuditLogger
        self.audit = AuditLogger(self.session_id)

    async def run(self) -> None:
        """Main async loop — runs until is_running is set False."""
        await init_db()
        self.is_running = True
        logger.info("copilot loop started — session %s", self.session_id)

        watcher_task = asyncio.create_task(self._detail_watcher())

        while self.is_running:
            try:
                await self._tick()
            except Exception as exc:
                logger.error("loop tick error: %s", exc)
            interval = (self.config.active_interval if self._last_tick_had_change
                        else self.config.idle_interval)
            await asyncio.sleep(interval)

        watcher_task.cancel()

    async def _detail_watcher(self) -> None:
        """Continuously watches for detail requests — runs independently
        of the capture cycle so the button always responds."""
        while self.is_running:
            if self.detail_requested:
                self.detail_requested = False
                if self.last_suggestion is not None:
                    self.overlay.set_status("● getting more detail...")
                    try:
                        detail = await generate_detailed_explanation(
                            self.context, self.last_suggestion
                        )
                        self.overlay.show_detail(detail)
                    except Exception as e:
                        logger.error("detail generation failed: %s", e)
                        self.overlay.show_detail("Could not generate detail right now.")
            if self.analyze_selection_requested:
                self.analyze_selection_requested = False
                self.overlay.set_status("● analyzing selection...")
                from screen_copilot.selection import get_selected_text
                from screen_copilot.suggester import analyze_selection
                selected = get_selected_text()
                suggestion = await analyze_selection(selected)
                self.last_suggestion = suggestion
                self.overlay.update_suggestion(suggestion)
                await save_suggestion(suggestion, self.session_id)
                self.audit.log_event("selection", {"chars_analyzed": len(selected)})
            await asyncio.sleep(0.3)

    async def _tick(self) -> None:
        """Single capture → analyse → suggest cycle."""

        # 1. Active window info
        win_info = get_active_window_info()

        # Skip the cycle entirely when the overlay itself was the active window,
        # to prevent a self-capture feedback loop.
        if win_info["app"] == "SKIP_SELF":
            logger.info("tick: skipping — overlay was the active window")
            return

        if win_info["app"] == "BLACKLISTED":
            logger.info("tick: skipping — blacklisted app (privacy protection)")
            self.overlay.set_status("● paused (sensitive app)")
            self.audit.log_event("blacklist_skip", {"app": win_info.get("app", "")})
            return

        window_title = win_info["title"]
        app_name = win_info["app"]

        # 2. Capture
        self.overlay.set_status("● capturing...")
        screenshot_path = await capture_active_window(
            self.session_id, self.capture_index
        )
        self.capture_index += 1

        if screenshot_path is None:
            logger.warning("tick: screenshot capture failed — skipping")
            self.overlay.set_status("● capture failed")
            return

        # 3. Preprocess
        preprocess(screenshot_path)

        # 3b. Skip expensive vision when the screen is static (unless manually triggered)
        if not self.force_suggestion and not has_screen_changed(screenshot_path):
            logger.info("tick: screen unchanged — skipping vision analysis")
            self.overlay.set_status(f"● idle ({self.context.current_app})")
            self._last_tick_had_change = False
            return
        self._last_tick_had_change = True

        # 4. Vision analysis
        self.overlay.set_status("● analyzing...")
        obs = await build_observation(
            screenshot_path, window_title, app_name, self.session_id
        )

        # 5. Add to context and detect app switch
        switched = self.context.add_observation(obs)
        await save_observation(obs, self.session_id)
        self.audit.log_event("analysis", {"app": app_name, "topic": obs.topic})

        # 6. Generate suggestion when needed
        should_suggest = (
            switched
            or self.force_suggestion
            or self.context.should_update_suggestion(self.last_suggestion_time)
        )

        if not should_suggest:
            self.overlay.set_status(
                f"● active ({self.context.get_time_on_task()}s on {app_name})"
            )
            return

        manual = self.force_suggestion
        if manual:
            self.overlay.set_status("● manual trigger...")
            self.force_suggestion = False  # reset after handling

        if switched and not manual:
            old_app = (
                self.context.window.observations[-2].app_name
                if len(self.context.window.observations) >= 2
                else "previous app"
            )
            self.overlay.set_status("● app switch detected...")
            suggestion = await generate_app_switch_message(old_app, app_name)
        else:
            self.overlay.set_status("● thinking...")
            suggestion = await generate_suggestion(self.context)

        self.last_suggestion_time = datetime.utcnow()
        self.last_suggestion = suggestion
        self.context.add_suggestion_to_history(suggestion.suggestion_text)

        # 7. Persist and display
        await save_suggestion(suggestion, self.session_id)
        self.overlay.update_suggestion(suggestion)
        self.audit.log_event("suggestion", {"category": suggestion.category})
        logger.info(
            "tick: suggestion delivered [%s] confidence=%s",
            suggestion.category,
            suggestion.confidence,
        )

    def stop(self) -> None:
        """Signal the loop to exit after the current tick completes."""
        self.is_running = False
