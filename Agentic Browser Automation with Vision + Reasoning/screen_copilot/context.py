"""Rolling context window manager and app-switch detection."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from screen_copilot.models import ContextWindow, Observation

logger = logging.getLogger("copilot.context")


class ContextManager:
    """Wraps ContextWindow with app-switch tracking and suggestion timing."""

    def __init__(self, max_size: int = 5) -> None:
        self.window = ContextWindow(max_size=max_size)
        self.current_app: str = ""
        self.current_task: str = ""
        self.task_start_time: datetime = datetime.utcnow()
        self.switch_count: int = 0
        self.recent_suggestions: list[str] = []  # last 5 suggestion texts
        self.max_suggestion_history: int = 5

    def add_suggestion_to_history(self, suggestion_text: str) -> None:
        """Track recent suggestion texts so the suggester can avoid repeats."""
        self.recent_suggestions.append(suggestion_text)
        if len(self.recent_suggestions) > self.max_suggestion_history:
            self.recent_suggestions.pop(0)

    def get_recent_suggestions_string(self) -> str:
        """Return recent suggestions as a bullet list for prompt injection."""
        if not self.recent_suggestions:
            return "None yet."
        return "\n".join(f"- {s}" for s in self.recent_suggestions)

    def add_observation(self, obs: Observation) -> bool:
        """Add observation to the rolling window.

        Returns True if an app switch was detected, False otherwise.
        """
        switched = False
        if self.current_app and obs.app_name != self.current_app:
            logger.info(
                "context: app switch detected %s -> %s",
                self.current_app,
                obs.app_name,
            )
            self.switch_count += 1
            switched = True
            self.task_start_time = datetime.utcnow()

        self.current_app = obs.app_name
        self.current_task = obs.vision_summary
        self.window.add(obs)
        return switched

    def get_context_string(self) -> str:
        """Return formatted rolling context for the suggester."""
        return self.window.get_context_string()

    def get_time_on_task(self) -> int:
        """Return seconds spent on the current app/task."""
        return int((datetime.utcnow() - self.task_start_time).total_seconds())

    def get_summary(self) -> dict:
        """Return current context state as a plain dict."""
        return {
            "current_app": self.current_app,
            "current_task": self.current_task,
            "observations_count": len(self.window.observations),
            "time_on_task_seconds": self.get_time_on_task(),
            "switch_count": self.switch_count,
        }

    def clear(self) -> None:
        """Reset context — call when starting a new session."""
        self.window = ContextWindow(max_size=self.window.max_size)
        self.current_app = ""
        self.current_task = ""
        self.task_start_time = datetime.utcnow()
        self.switch_count = 0

    def should_update_suggestion(
        self,
        last_suggestion_time: Optional[datetime],
        min_interval: int = 20,
    ) -> bool:
        """Return True if enough time has passed to generate a new suggestion.

        Args:
            last_suggestion_time: Timestamp of the last generated suggestion,
                or None to force an immediate update.
            min_interval: Minimum seconds between suggestions (default 20 —
                slightly less than capture_interval so a suggestion is ready
                almost every cycle).
        """
        if last_suggestion_time is None:
            return True
        elapsed = (datetime.utcnow() - last_suggestion_time).total_seconds()
        return elapsed >= min_interval
