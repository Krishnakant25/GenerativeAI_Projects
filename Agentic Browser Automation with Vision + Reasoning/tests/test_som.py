"""Tests for the Set-of-Mark (SoM) prompting helpers in browser/som.py."""

import os
import pytest
from PIL import Image

from browser.som import (
    SoMRegion,
    generate_regions_from_viewport,
    generate_regions_from_elements,
    draw_marks,
    pick_region,
    parse_som_response,
)


class TestSoMRegion:

    def test_center_coordinates(self):
        region = SoMRegion(number=1, x=100, y=200, width=120, height=50, label="button")
        assert region.center_x == 160
        assert region.center_y == 225


class TestGenerateRegions:

    def test_viewport_grid_generates_12_regions(self):
        regions = generate_regions_from_viewport()
        assert len(regions) == 12

    def test_viewport_regions_numbered_1_to_12(self):
        regions = generate_regions_from_viewport()
        numbers = [r.number for r in regions]
        assert numbers == list(range(1, 13))

    def test_regions_from_elements(self):
        elements = [
            {"label": "search", "x": 640, "y": 300},
            {"label": "button", "x": 340, "y": 220},
        ]
        regions = generate_regions_from_elements(elements)
        assert len(regions) == 2
        assert regions[0].label == "search"
        assert regions[0].center_x == 640
        assert regions[0].center_y == 300

    def test_empty_elements_falls_back_to_grid(self):
        regions = generate_regions_from_elements([])
        assert len(regions) == 12

    def test_regions_clamped_to_viewport(self):
        elements = [{"label": "edge", "x": 5, "y": 5}]
        regions = generate_regions_from_elements(elements)
        assert regions[0].x >= 0
        assert regions[0].y >= 0


class TestDrawMarks:

    def test_creates_output_file(self, tmp_path):
        img_path = str(tmp_path / "test.png")
        Image.new("RGB", (1280, 800), color=(255, 255, 255)).save(img_path)
        regions = generate_regions_from_viewport()
        output = str(tmp_path / "test_som.png")
        result = draw_marks(img_path, regions, output_path=output)
        assert os.path.exists(result)

    def test_bad_image_path_returns_original(self):
        result = draw_marks("/nonexistent/image.png", [], output_path="/tmp/out.png")
        assert result == "/nonexistent/image.png"


class TestPickRegion:

    def test_finds_correct_region(self):
        regions = generate_regions_from_viewport()
        region = pick_region(5, regions)
        assert region is not None
        assert region.number == 5

    def test_returns_none_for_missing_number(self):
        regions = generate_regions_from_viewport()
        assert pick_region(99, regions) is None


class TestParseSoMResponse:

    def test_extracts_number_from_region_line(self):
        raw = "REGION: 7\nACTION: click\nCONFIDENCE: high\nREASONING: test"
        assert parse_som_response(raw) == 7

    def test_returns_none_for_no_number(self):
        assert parse_som_response("no numbers here") is None

    def test_handles_none_input(self):
        assert parse_som_response(None) is None
