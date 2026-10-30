"""
Unit Tests for Streamlit Application Helpers and Page Handlers
AI-Based Food Recognition and Nutrition Tracker Using Deep Learning
"""

import io
from pathlib import Path
import pytest
from PIL import Image

from app import (
    validate_uploaded_image,
    format_nutrient_val,
    render_page_about,
    render_page_dashboard,
    render_page_recognize_food,
    render_page_food_search,
    render_page_meal_history,
)


class DummyUploadedFile:
    """Mock Streamlit UploadedFile object."""
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def test_validate_uploaded_image_valid():
    """Verify valid PNG/JPG passes validation."""
    buf = io.BytesIO()
    img = Image.new("RGB", (100, 100), color=(100, 200, 50))
    img.save(buf, format="PNG")
    mock_file = DummyUploadedFile("salad.png", buf.getvalue())

    is_valid, err_msg, pil_img = validate_uploaded_image(mock_file)
    assert is_valid is True
    assert err_msg is None
    assert pil_img is not None
    assert pil_img.size == (100, 100)


def test_validate_uploaded_image_invalid_extension():
    """Verify unsupported extensions are rejected."""
    mock_file = DummyUploadedFile("notes.txt", b"hello world")
    is_valid, err_msg, pil_img = validate_uploaded_image(mock_file)
    assert is_valid is False
    assert "not supported" in err_msg
    assert pil_img is None


def test_validate_uploaded_image_corrupted():
    """Verify corrupt images are detected."""
    mock_file = DummyUploadedFile("fake_photo.jpg", b"corrupted bytes here")
    is_valid, err_msg, pil_img = validate_uploaded_image(mock_file)
    assert is_valid is False
    assert "Corrupted or invalid" in err_msg


def test_validate_uploaded_image_size_exceeded():
    """Verify file sizes exceeding 10 MB are rejected."""
    oversized_bytes = b"0" * (11 * 1024 * 1024)
    mock_file = DummyUploadedFile("large.png", oversized_bytes)
    is_valid, err_msg, pil_img = validate_uploaded_image(mock_file)
    assert is_valid is False
    assert "exceeds maximum limit" in err_msg


def test_format_nutrient_val():
    """Verify nutrient formatting handles None, NaN, and floats."""
    assert format_nutrient_val(None) == "N/A"
    assert format_nutrient_val(float("nan")) == "N/A"
    assert format_nutrient_val(25.46, "g") == "25.5 g"
    assert format_nutrient_val(350.0, "kcal") == "350.0 kcal"


def test_page_functions_exist_and_callable():
    """Verify all 5 page render functions are defined and callable."""
    assert callable(render_page_dashboard)
    assert callable(render_page_recognize_food)
    assert callable(render_page_food_search)
    assert callable(render_page_meal_history)
    assert callable(render_page_about)


# ---------------------------------------------------------------------------
# Edge-case: validate_uploaded_image with JPEG bytes
# ---------------------------------------------------------------------------

def test_validate_uploaded_image_jpeg_valid():
    """Verify valid JPEG bytes pass validation with .jpg extension."""
    buf = io.BytesIO()
    img = Image.new("RGB", (80, 80), color=(255, 128, 0))
    img.save(buf, format="JPEG")
    mock_file = DummyUploadedFile("burger.jpg", buf.getvalue())

    is_valid, err_msg, pil_img = validate_uploaded_image(mock_file)
    assert is_valid is True
    assert err_msg is None
    assert pil_img is not None


def test_validate_uploaded_image_empty_bytes():
    """Verify an empty file (zero bytes) with valid extension is rejected as corrupted."""
    mock_file = DummyUploadedFile("empty.png", b"")
    is_valid, err_msg, pil_img = validate_uploaded_image(mock_file)
    assert is_valid is False
    assert pil_img is None


def test_format_nutrient_val_zero():
    """Zero is a valid nutrient value and should format, not show N/A."""
    result = format_nutrient_val(0.0, "g")
    assert result != "N/A"
    assert "0" in result


def test_format_nutrient_val_integer():
    """Integer inputs should be accepted and formatted."""
    result = format_nutrient_val(100, "kcal")
    assert "100" in result
    assert "N/A" not in result

