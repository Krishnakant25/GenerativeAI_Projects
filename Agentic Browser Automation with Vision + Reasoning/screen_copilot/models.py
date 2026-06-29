"""Pydantic data models for Screen Copilot."""

import uuid
from datetime import datetime
from pydantic import BaseModel, Field


class Observation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime
    window_title: str
    app_name: str
    screenshot_path: str
    vision_summary: str = ""
    suggested_action: str = ""
    confidence: str = "low"
    topic: str = ""
    key_detail: str = ""


class ContextWindow(BaseModel):
    observations: list[Observation] = Field(default_factory=list)
    max_size: int = 5

    def add(self, obs: Observation) -> None:
        self.observations.append(obs)
        if len(self.observations) > self.max_size:
            self.observations.pop(0)

    def get_context_string(self) -> str:
        if not self.observations:
            return "No observations yet."
        lines = []
        for o in self.observations:
            topic_part = f" | Topic: {o.topic}" if o.topic else ""
            lines.append(
                f"[{o.timestamp.strftime('%H:%M:%S')}] "
                f"{o.app_name} — {o.window_title}: {o.vision_summary}{topic_part}"
            )
        return "\n".join(lines)


class Suggestion(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    suggestion_text: str
    category: str = "general"  # general | warning | tip | info
    confidence: str = "low"
    based_on_observations: int = 0


class CopilotConfig(BaseModel):
    capture_interval: int = 22
    idle_interval: int = 35      # slower polling when screen is static
    active_interval: int = 22    # normal polling when active
    max_context_size: int = 5
    ollama_url: str = "http://localhost:11434"
    vision_model: str = "llava:7b"
    reasoning_model: str = "llama3.1:8b-instruct-q4_0"
    screenshot_dir: str = "screen_copilot/screenshots"
    db_path: str = "screen_copilot/copilot.db"
    overlay_opacity: float = 0.88
    overlay_width: int = 380
    overlay_height: int = 220
    blacklisted_apps: list[str] = Field(default_factory=lambda: [
        "1Password", "Bitwarden", "KeePass", "LastPass", "Dashlane",
        "Banking", "Wallet", "Keychain", "Authenticator",
    ])
    blacklist_keywords: list[str] = Field(default_factory=lambda: [
        "password", "login", "sign in", "credit card", "bank",
        "ssn", "social security", "private key", "seed phrase",
    ])
