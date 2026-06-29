"""Pydantic models and TypedDicts shared across the agent package."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class Action(BaseModel):
    type: Literal["navigate", "click", "type", "scroll", "wait", "extract"]
    selector: Optional[str] = None
    url: Optional[str] = None
    text: Optional[str] = None
    direction: Optional[Literal["up", "down"]] = None
    amount: Optional[int] = None
    ms: Optional[int] = None


class ActionResult(BaseModel):
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None


class VisionOutput(BaseModel):
    success: bool = False
    page_type: str = "unknown"
    visible_text: list[str] = Field(default_factory=list)
    clickable_elements: list[dict] = Field(default_factory=list)
    search_box_present: bool = False
    search_box_coords: Optional[tuple[int, int]] = None
    suggested_action: str = "navigate"
    suggested_target: str = ""
    confidence: str = "low"
    reasoning: str = ""
    raw_response: str = ""


class Plan(BaseModel):
    success: bool = False
    action: str = "navigate"
    target: str = ""
    value: str = ""
    reasoning: str = ""
    confidence: str = "low"
    goal_complete: bool = False
    raw_response: str = ""


class StepRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    step_number: int
    screenshot_path: Optional[str] = None
    llava_output: Optional[str] = None
    llama_reasoning: Optional[str] = None
    action_taken: Optional[str] = None
    action_result: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TaskConfig(BaseModel):
    goal: str
    max_steps: int = 25
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class AgentState(BaseModel):
    task_id: str
    goal: str
    step_number: int = 0
    max_steps: int = 25
    status: str = "running"  # running | complete | partial | failed
    current_screenshot: Optional[str] = None
    action_history: list[dict] = Field(default_factory=list)
    extracted_texts: list[str] = Field(default_factory=list)
    task_memory: list[str] = Field(default_factory=list)  # rolling fact log, capped at 10 in act_node
    last_vision: Optional[VisionOutput] = None
    last_plan: Optional[Plan] = None
    final_report: Optional[dict] = None
    error_message: Optional[str] = None
    browser_controller: Any = None  # holds the live BrowserController instance
    screenshot_paths: list = Field(default_factory=list)
    last_som_regions: list = Field(default_factory=list)
