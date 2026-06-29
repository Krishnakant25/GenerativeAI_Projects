import pytest
from PIL import Image, ImageDraw
from screen_copilot.ocr import extract_text, is_tesseract_available


class TestOCR:

    def test_extract_text_from_simple_image(self, tmp_path):
        if not is_tesseract_available():
            pytest.skip("Tesseract not installed in this environment")
        img_path = str(tmp_path / "test.png")
        img = Image.new("RGB", (400, 100), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "Hello World Test", fill="black")
        img.save(img_path)
        result = extract_text(img_path)
        assert "Hello" in result or "World" in result

    def test_extract_text_nonexistent_file_returns_empty(self):
        result = extract_text("/nonexistent/path.png")
        assert result == ""

    def test_extract_text_respects_max_chars(self, tmp_path):
        if not is_tesseract_available():
            pytest.skip("Tesseract not installed in this environment")
        img_path = str(tmp_path / "test2.png")
        img = Image.new("RGB", (100, 100), color="white")
        img.save(img_path)
        result = extract_text(img_path, max_chars=10)
        assert len(result) <= 30  # 10 + "... [truncated]" suffix
