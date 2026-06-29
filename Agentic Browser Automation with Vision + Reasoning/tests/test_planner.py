"""Tests for the planner module — plan parsing, loop detection, and fact extraction."""

import pytest
from unittest.mock import AsyncMock, patch
from agent.planner import (
    _parse_plan_response,
    check_loop_detection,
    extract_key_facts,
)
from agent.models import VisionOutput, Plan


class TestParsePlanResponse:

    def test_valid_full_response(self):
        raw = """ACTION: navigate
TARGET: https://anthropic.com
VALUE:
REASONING: Starting at the homepage to find funding information.
CONFIDENCE: high
GOAL_COMPLETE: no"""
        result = _parse_plan_response(raw)
        assert result.success is True
        assert result.action == "navigate"
        assert result.target == "https://anthropic.com"
        assert result.confidence == "high"
        assert result.goal_complete is False

    def test_none_input_returns_default_plan(self):
        result = _parse_plan_response(None)
        assert result.success is False
        assert result.action == "navigate"

    def test_unknown_action_defaults_to_navigate(self):
        raw = """ACTION: teleport
TARGET: somewhere
VALUE:
REASONING: test
CONFIDENCE: high
GOAL_COMPLETE: no"""
        result = _parse_plan_response(raw)
        assert result.action == "navigate"
        assert result.success is False

    def test_goal_complete_yes(self):
        raw = """ACTION: finish
TARGET: goal_complete
VALUE:
REASONING: All information extracted.
CONFIDENCE: high
GOAL_COMPLETE: yes"""
        result = _parse_plan_response(raw)
        assert result.goal_complete is True
        assert result.action == "finish"

    def test_all_valid_actions_recognised(self):
        valid_actions = ["navigate", "click", "click_text", "type",
                         "scroll", "extract", "finish"]
        for action in valid_actions:
            raw = f"""ACTION: {action}
TARGET: test
VALUE:
REASONING: test
CONFIDENCE: medium
GOAL_COMPLETE: no"""
            result = _parse_plan_response(raw)
            assert result.action == action
            assert result.success is True


class TestCheckLoopDetection:

    @pytest.mark.asyncio
    async def test_detects_loop_same_action_and_target(self):
        history = [
            {"action_taken": "click", "target": "(340,220)", "step_number": 1, "action_result": "success"},
            {"action_taken": "click", "target": "(340,220)", "step_number": 2, "action_result": "success"},
            {"action_taken": "click", "target": "(340,220)", "step_number": 3, "action_result": "success"},
        ]
        result = await check_loop_detection(history, "click", "(340,220)")
        assert result is True

    @pytest.mark.asyncio
    async def test_no_loop_different_targets(self):
        history = [
            {"action_taken": "click", "target": "(100,100)", "step_number": 1, "action_result": "success"},
            {"action_taken": "click", "target": "(200,200)", "step_number": 2, "action_result": "success"},
            {"action_taken": "click", "target": "(300,300)", "step_number": 3, "action_result": "success"},
        ]
        result = await check_loop_detection(history, "click", "(400,400)")
        assert result is False

    @pytest.mark.asyncio
    async def test_no_loop_fewer_than_3_entries(self):
        history = [
            {"action_taken": "click", "target": "(100,100)", "step_number": 1, "action_result": "success"},
            {"action_taken": "click", "target": "(100,100)", "step_number": 2, "action_result": "success"},
        ]
        result = await check_loop_detection(history, "click", "(100,100)")
        assert result is False

    @pytest.mark.asyncio
    async def test_empty_history_returns_false(self):
        result = await check_loop_detection([], "click", "(100,100)")
        assert result is False


class TestExtractKeyFacts:

    @pytest.mark.asyncio
    async def test_parses_fact_lines(self):
        mock_response = """FACT: Anthropic raised $7.3B in Series E
FACT: Lead investor was Google
FACT: Funding announced in 2024"""
        with patch("agent.planner._call_llama", new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            facts = await extract_key_facts("some content", "funding research")
        assert len(facts) == 3
        assert "Anthropic raised $7.3B in Series E" in facts

    @pytest.mark.asyncio
    async def test_ignores_non_fact_lines(self):
        mock_response = """Here are the facts:
FACT: One real fact
Some commentary line
FACT: Another real fact"""
        with patch("agent.planner._call_llama", new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            facts = await extract_key_facts("content", "goal")
        assert len(facts) == 2

    @pytest.mark.asyncio
    async def test_llama_failure_returns_empty_list(self):
        with patch("agent.planner._call_llama", new_callable=AsyncMock) as mock:
            mock.return_value = None
            facts = await extract_key_facts("content", "goal")
        assert facts == []
