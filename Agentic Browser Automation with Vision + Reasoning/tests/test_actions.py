"""Tests for the actions dispatcher — execute_action routing and helper functions."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agent.actions import execute_action, _do_click, _do_finish


class TestExecuteAction:

    @pytest.mark.asyncio
    async def test_unknown_action_returns_error_dict(self):
        mock_controller = MagicMock()
        result = await execute_action(
            mock_controller, "teleport", "mars", "", "task1", 1
        )
        assert result["action_taken"] == "teleport"
        assert "unknown action" in result["action_result"]
        assert result["step_number"] == 1
        assert result["target"] == "mars"

    @pytest.mark.asyncio
    async def test_finish_action_returns_complete_dict(self):
        mock_controller = MagicMock()
        result = await execute_action(
            mock_controller, "finish", "", "", "task1", 5
        )
        assert result["action_taken"] == "finish"
        assert result["action_result"] == "agent signalled task complete"
        assert result["target"] == "goal_complete"

    @pytest.mark.asyncio
    async def test_result_dict_always_has_four_keys(self):
        mock_controller = MagicMock()
        result = await execute_action(
            mock_controller, "finish", "", "", "task1", 1
        )
        assert "step_number" in result
        assert "action_taken" in result
        assert "target" in result
        assert "action_result" in result


class TestDoClick:

    @pytest.mark.asyncio
    async def test_bad_coordinates_returns_failed_dict(self):
        mock_controller = MagicMock()
        result = await _do_click(mock_controller, "not-coords", "task1", 1)
        assert "failed" in result["action_result"]
        assert result["action_taken"] == "click"

    @pytest.mark.asyncio
    async def test_valid_coordinates_calls_controller(self):
        mock_controller = AsyncMock()
        mock_controller.is_running = True
        mock_controller.click = AsyncMock(return_value=True)
        mock_controller.page = AsyncMock()
        with patch("agent.actions.capture", new_callable=AsyncMock) as mock_cap:
            mock_cap.return_value = "/fake/path.png"
            result = await _do_click(mock_controller, "(340, 220)", "task1", 1)
        mock_controller.click.assert_called_once_with(340, 220)
        assert result["action_result"] == "success"


class TestDoFinish:

    @pytest.mark.asyncio
    async def test_returns_correct_schema(self):
        result = await _do_finish("task_abc", 10)
        assert result["step_number"] == 10
        assert result["action_taken"] == "finish"
        assert result["target"] == "goal_complete"
        assert result["action_result"] == "agent signalled task complete"
