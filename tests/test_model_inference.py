"""
Unit Tests for Model Inference Module
AI-Based Food Recognition and Nutrition Tracker Using Deep Learning
"""

import pytest
import numpy as np
from PIL import Image

from model_inference import FoodClassifier, ModelInferenceError, get_food_classifier
from config import MODEL_PATH, CLASS_NAMES_PATH


@pytest.fixture
def classifier():
    """Fixture providing initialized FoodClassifier."""
    return get_food_classifier(MODEL_PATH, CLASS_NAMES_PATH)


@pytest.fixture
def sample_pil_image():
    """Create a dummy 300x300 RGB image for testing."""
    return Image.new("RGB", (300, 300), color=(200, 100, 50))


def test_classifier_loaded(classifier):
    """Verify model and class names load properly."""
    assert classifier.is_ready is True
    assert len(classifier.class_names) == 101


def test_image_preprocessing(classifier, sample_pil_image):
    """Verify preprocessing produces expected shape (1, 224, 224, 3)."""
    batch = classifier.preprocess_image(sample_pil_image)
    assert isinstance(batch, np.ndarray)
    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32


def test_image_preprocessing_rgba_conversion(classifier):
    """Verify non-RGB images (e.g. RGBA with transparency) convert properly."""
    rgba_image = Image.new("RGBA", (150, 150), color=(100, 150, 200, 255))
    batch = classifier.preprocess_image(rgba_image)
    assert batch.shape == (1, 224, 224, 3)


def test_prediction_returns_top_3(classifier, sample_pil_image):
    """Verify prediction returns top 3 items with valid probabilities."""
    results = classifier.predict(sample_pil_image, top_k=3)
    assert len(results) == 3
    for label, conf in results:
        assert isinstance(label, str)
        assert 0.0 <= conf <= 1.0
        assert label in classifier.class_names


def test_invalid_image_raises_error(classifier):
    """Ensure invalid input raises ModelInferenceError."""
    with pytest.raises(ModelInferenceError):
        classifier.predict("not_an_image")


# ---------------------------------------------------------------------------
# Edge-case: missing model artifact (no trained .keras file)
# ---------------------------------------------------------------------------

def test_missing_model_is_not_ready(tmp_path):
    """FoodClassifier with nonexistent paths sets is_ready=False."""
    c = FoodClassifier(
        model_path=str(tmp_path / "missing.keras"),
        class_names_path=str(tmp_path / "missing.json"),
    )
    assert c.is_ready is False
    assert c.model is None
    assert c.class_names == []


def test_missing_model_predict_raises(tmp_path):
    """predict() on a classifier with no model raises ModelInferenceError."""
    c = FoodClassifier(
        model_path=str(tmp_path / "missing.keras"),
        class_names_path=str(tmp_path / "missing.json"),
    )
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    with pytest.raises(ModelInferenceError, match="not loaded"):
        c.predict(img)


def test_preprocess_image_does_not_need_model():
    """preprocess_image is a static method and works without a loaded model."""
    img = Image.new("L", (100, 100), color=200)  # Grayscale
    arr = FoodClassifier.preprocess_image(img)
    assert arr.shape == (1, 224, 224, 3)
    assert arr.dtype == np.float32


def test_pixel_range_float32(sample_pil_image):
    """Preprocessed pixels must be in [0, 255] float32 range (MobileNetV2 raw input)."""
    arr = FoodClassifier.preprocess_image(sample_pil_image)
    assert arr.min() >= 0.0
    assert arr.max() <= 255.0


def test_class_names_only_no_model(tmp_path):
    """When only class_names.json exists (model missing), class_names loads but is_ready is False."""
    import json
    names_path = tmp_path / "class_names.json"
    names_path.write_text(json.dumps([f"food_{i}" for i in range(101)]))

    c = FoodClassifier(
        model_path=str(tmp_path / "missing.keras"),
        class_names_path=str(names_path),
    )
    assert c.class_names == [f"food_{i}" for i in range(101)]
    assert c.model is None
    assert c.is_ready is False  # model missing → not ready

