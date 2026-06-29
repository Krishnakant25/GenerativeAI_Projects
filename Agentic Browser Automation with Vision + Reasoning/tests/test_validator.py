import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestValidateAction:

    @pytest.mark.asyncio
    async def test_returns_valid_true_on_model_unavailable(self):
        from agent.validator import validate_action
        with patch("agent.validator.encode_to_base64", return_value=None):
            result = await validate_action("/fake/path.png", "click", "(340,220)")
        assert result["valid"] is True

    @pytest.mark.asyncio
    async def test_parses_yes_response(self):
        from agent.validator import validate_action
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "VALID: yes\nREASON: Button is visible"}
        with patch("agent.validator.encode_to_base64", return_value="fakebase64"):
            with patch("httpx.AsyncClient") as mock_client:
                mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_resp)
                result = await validate_action("/fake/path.png", "click", "(340,220)")
        assert result["valid"] is True

    @pytest.mark.asyncio
    async def test_parses_no_response(self):
        from agent.validator import validate_action
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "VALID: no\nREASON: No button visible at those coordinates"}
        with patch("agent.validator.encode_to_base64", return_value="fakebase64"):
            with patch("httpx.AsyncClient") as mock_client:
                mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_resp)
                result = await validate_action("/fake/path.png", "click", "(340,220)")
        assert result["valid"] is False
        assert "No button" in result["reason"]

    @pytest.mark.asyncio
    async def test_exception_returns_valid_true(self):
        from agent.validator import validate_action
        with patch("agent.validator.encode_to_base64", side_effect=Exception("disk error")):
            result = await validate_action("/fake/path.png", "click", "(340,220)")
        assert result["valid"] is True


class TestTaskTemplates:

    def test_all_templates_exist(self):
        from agent.task_templates import TEMPLATES
        assert "news_aggregation" in TEMPLATES
        assert "multi_site_research" in TEMPLATES
        assert "competitive_intelligence" in TEMPLATES

    def test_get_template_fills_topic(self):
        from agent.task_templates import get_template
        t = get_template("news_aggregation", topic="OpenAI")
        assert "OpenAI" in t.goal

    def test_get_template_invalid_key_raises(self):
        from agent.task_templates import get_template
        with pytest.raises(ValueError):
            get_template("nonexistent_template", topic="test")

    def test_competitive_intelligence_fills_company(self):
        from agent.task_templates import get_template
        t = get_template("competitive_intelligence", topic="Anthropic", company="Anthropic")
        assert "Anthropic" in t.goal
        assert t.max_steps == 30
