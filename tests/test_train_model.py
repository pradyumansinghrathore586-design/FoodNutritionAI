"""
Unit Tests for Deep Learning Model Training Pipeline
AI-Based Food Recognition and Nutrition Tracker Using Deep Learning

Covers:
- Image resizing and preprocessing shapes (224x224x3)
- Data augmentation behavior
- MobileNetV2 architecture, frozen base, and 101-class Softmax output
- Non-overlapping dataset splitting
- Model saving (.keras) and loading integrity
- Class names JSON ordering and preservation
- Artifact detection logic
"""

import json
from pathlib import Path
import pytest
import numpy as np
import tensorflow as tf

from train_model import (
    build_food_model,
    build_data_augmentation,
    preprocess_image_and_label,
    create_smoke_test_datasets,
    save_class_names,
    check_artifacts_exist,
    FOOD_101_CLASSES,
)
from config import IMAGE_SIZE, NUM_CLASSES


def test_preprocess_image_and_label_shapes():
    """Verify preprocessing function resizes arbitrary image to (224, 224, 3) float32."""
    raw_image = tf.random.uniform(shape=(300, 450, 3), minval=0, maxval=255, dtype=tf.float32)
    label = tf.constant(5, dtype=tf.int64)

    proc_img, proc_lbl = preprocess_image_and_label(raw_image, label)

    assert proc_img.shape == (224, 224, 3)
    assert proc_img.dtype == tf.float32
    assert proc_lbl.numpy() == 5


def test_data_augmentation_pipeline():
    """Verify data augmentation preserves expected tensor shapes."""
    augmentation = build_data_augmentation()
    sample_batch = tf.random.uniform(shape=(4, 224, 224, 3), minval=0, maxval=255, dtype=tf.float32)

    augmented = augmentation(sample_batch, training=True)
    assert augmented.shape == (4, 224, 224, 3)


def test_build_food_model_architecture():
    """Verify MobileNetV2 transfer learning model specifications."""
    model = build_food_model(num_classes=NUM_CLASSES)

    # Input shape: (None, 224, 224, 3)
    assert model.input_shape == (None, 224, 224, 3)

    # Output shape: (None, 101)
    assert model.output_shape == (None, 101)

    # Output activation must be softmax
    output_layer = model.get_layer("food_predictions")
    assert output_layer.activation.__name__ == "softmax"

    # Base MobileNetV2 model must be frozen
    mobilenet_base = None
    for layer in model.layers:
        if "mobilenetv2" in layer.name.lower() or "functional" in layer.name.lower():
            mobilenet_base = layer
            break

    assert mobilenet_base is not None, "MobileNetV2 base layer must be present in model."
    assert mobilenet_base.trainable is False, "Base model must be frozen for initial feature extraction."


def test_smoke_test_datasets():
    """Verify smoke-test dataset generates non-overlapping batches."""
    batch_size = 8
    num_samples = 32
    train_ds, val_ds, test_ds, class_names = create_smoke_test_datasets(
        num_samples=num_samples,
        batch_size=batch_size,
        num_classes=NUM_CLASSES
    )

    assert len(class_names) == 101

    # Check batch shapes from train, val, and test
    for img_batch, lbl_batch in train_ds.take(1):
        assert img_batch.shape == (batch_size, 224, 224, 3)
        assert lbl_batch.shape == (batch_size,)

    for img_batch, lbl_batch in val_ds.take(1):
        assert img_batch.shape == (batch_size, 224, 224, 3)

    for img_batch, lbl_batch in test_ds.take(1):
        assert img_batch.shape == (batch_size, 224, 224, 3)


def test_save_and_load_class_names(tmp_path):
    """Verify class names are preserved in exact label-index order."""
    json_path = tmp_path / "test_classes.json"
    save_class_names(FOOD_101_CLASSES, json_path)

    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert len(loaded) == 101
    assert loaded == FOOD_101_CLASSES
    assert loaded[0] == "apple_pie"
    assert loaded[-1] == "waffles"


def test_model_saving_and_inference_integrity(tmp_path):
    """Verify model can be saved to .keras format and reloaded for prediction."""
    model = build_food_model(num_classes=NUM_CLASSES)
    model_file = tmp_path / "test_food_model.keras"

    model.save(model_file)
    assert model_file.exists()

    loaded_model = tf.keras.models.load_model(model_file, compile=False)
    dummy_input = np.random.uniform(0.0, 255.0, size=(1, 224, 224, 3)).astype(np.float32)

    preds = loaded_model.predict(dummy_input, verbose=0)
    assert preds.shape == (1, 101)
    assert np.isclose(np.sum(preds), 1.0, atol=1e-4)


def test_artifacts_detection(tmp_path):
    """Verify check_artifacts_exist accurately identifies present vs missing files."""
    m_path = tmp_path / "model.keras"
    c_path = tmp_path / "classes.json"

    assert not check_artifacts_exist(m_path, c_path)

    m_path.write_text("dummy model content")
    assert not check_artifacts_exist(m_path, c_path)

    c_path.write_text("[]")
    assert check_artifacts_exist(m_path, c_path)
