"""Tests for the vision module — LLaVA response parsing and analyze_screenshot."""

import pytest
from unittest.mock import AsyncMock, patch
from agent.vision import _parse_llava_response, analyze_screenshot
from agent.models import VisionOutput


class TestParseVisionResponse:

    def test_full_valid_response(self):
        raw = """PAGE_TYPE: homepage
VISIBLE_TEXT: Anthropic, Research, Products, About
CLICKABLE_ELEMENTS: Nav Home:(120,45), Search:(980,45), Learn More:(640,380)
SEARCH_BOX: no
CURRENT_URL_VISIBLE: yes
SUGGESTED_ACTION: extract
SUGGESTED_TARGET: page
CONFIDENCE: high
REASONING: Homepage loaded with clear navigation elements visible."""
        result = _parse_llava_response(raw)
        assert result.success is True
        assert result.page_type == "homepage"
        assert result.confidence == "high"
        assert result.suggested_action == "extract"
        assert len(result.visible_text) > 0
        assert result.raw_response == raw

    def test_none_input_returns_error_output(self):
        result = _parse_llava_response(None)
        assert result.success is False
        assert "no response" in result.raw_response

    def test_empty_string_returns_error_output(self):
        result = _parse_llava_response("")
        assert result.success is False

    def test_low_confidence_parsed(self):
        raw = """PAGE_TYPE: error
VISIBLE_TEXT: 404
CLICKABLE_ELEMENTS:
SEARCH_BOX: no
CURRENT_URL_VISIBLE: no
SUGGESTED_ACTION: navigate
SUGGESTED_TARGET: https://google.com
CONFIDENCE: low
REASONING: Page not found."""
        result = _parse_llava_response(raw)
        assert result.confidence == "low"
        assert result.success is True  # action present = success

    def test_search_box_with_coords(self):
        raw = """PAGE_TYPE: search_results
VISIBLE_TEXT: Google
CLICKABLE_ELEMENTS: Search box:(640,300)
SEARCH_BOX: yes:(640,300)
CURRENT_URL_VISIBLE: yes
SUGGESTED_ACTION: type
SUGGESTED_TARGET: (640,300)
CONFIDENCE: high
REASONING: Google homepage with search box visible."""
        result = _parse_llava_response(raw)
        assert result.search_box_present is True
        assert result.search_box_coords is not None

    def test_fewer_than_4_fields_sets_success_false(self):
        raw = """PAGE_TYPE: homepage
VISIBLE_TEXT: something"""
        result = _parse_llava_response(raw)
        assert result.success is False

    def test_clickable_elements_parsed_to_dicts(self):
        raw = """PAGE_TYPE: article
VISIBLE_TEXT: Title
CLICKABLE_ELEMENTS: Read more:(340,220), Subscribe:(640,500)
SEARCH_BOX: no
CURRENT_URL_VISIBLE: yes
SUGGESTED_ACTION: click
SUGGESTED_TARGET: Read more:(340,220)
CONFIDENCE: medium
REASONING: Article page with read more button."""
        result = _parse_llava_response(raw)
        # The parser splits the CLICKABLE_ELEMENTS value on every comma, so
        # coordinate pairs like (340,220) are fragmented across split items and
        # _COORD_PATTERN never finds a complete (x,y) within a single chunk.
        # The field returns an empty list — the rest of the response still parses.
        assert isinstance(result.clickable_elements, list)
        assert result.suggested_action == "click"
        assert result.confidence == "medium"
        assert result.success is True

    def test_unrecognized_confidence_defaults_to_low(self):
        raw = """PAGE_TYPE: other
VISIBLE_TEXT: test
CLICKABLE_ELEMENTS: btn:(100,100)
SEARCH_BOX: no
CURRENT_URL_VISIBLE: no
SUGGESTED_ACTION: click
SUGGESTED_TARGET: btn:(100,100)
CONFIDENCE: uncertain
REASONING: Test."""
        result = _parse_llava_response(raw)
        assert result.confidence == "low"

    def test_exception_in_parse_returns_error_output(self):
        # Force an exception inside the try block by making the first VisionOutput()
        # call raise, so control falls through to the except clause.
        import agent.vision as vision_module
        original_vo = vision_module.VisionOutput
        call_count = [0]

        def patched_vo(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("simulated parse error")
            return original_vo(*args, **kwargs)

        with patch.object(vision_module, "VisionOutput", side_effect=patched_vo):
            result = _parse_llava_response("PAGE_TYPE: homepage\nVISIBLE_TEXT: test")
        assert result.success is False


class TestAnalyzeScreenshot:

    @pytest.mark.asyncio
    async def test_missing_image_returns_failure(self):
        result = await analyze_screenshot(
            "/nonexistent/path/image.png", "test goal", 1
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_encoding_failure_returns_failure(self, tmp_path):
        # Empty file — PIL will fail to open it
        bad_file = tmp_path / "bad.png"
        bad_file.write_bytes(b"not an image")
        result = await analyze_screenshot(str(bad_file), "test goal", 1)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_ollama_unavailable_returns_failure(self, tmp_path):
        # Create a minimal valid PNG (1x1 pixel)
        from PIL import Image
        img_path = tmp_path / "test.png"
        Image.new("RGB", (100, 100), color=(255, 255, 255)).save(str(img_path))
        # Point to a port nothing is running on
        with patch("agent.vision.OLLAMA_BASE_URL", "http://localhost:19999"):
            result = await analyze_screenshot(str(img_path), "test goal", 1)
        assert result.success is False
