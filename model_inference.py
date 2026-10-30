"""
Deep Learning Model Inference Module
AI-Based Food Recognition and Nutrition Tracker Using Deep Learning

Handles image validation, preprocessing, and Top-K Food-101 category prediction
using the MobileNetV2 transfer learning model.
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
from PIL import Image

# Suppress verbose TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from config import (
    CLASS_NAMES_PATH,
    IMAGE_SIZE,
    MODEL_PATH,
    NUM_CLASSES,
)

logger = logging.getLogger("FoodClassifierInference")


class ModelInferenceError(Exception):
    """Raised when model loading or prediction encounters an error."""
    pass


class FoodClassifier:
    """Manages loading the Food-101 MobileNetV2 model and performing predictions."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        class_names_path: Optional[Union[str, Path]] = None
    ):
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.class_names_path = Path(class_names_path) if class_names_path else CLASS_NAMES_PATH
        self.model: Optional[tf.keras.Model] = None
        self.class_names: List[str] = []
        self._load_class_names()
        self._load_model()

    def _load_class_names(self) -> None:
        """Load Food-101 class labels from JSON."""
        if not self.class_names_path.exists():
            logger.warning(f"Class names file not found at {self.class_names_path}")
            self.class_names = []
            return

        try:
            with open(self.class_names_path, "r", encoding="utf-8") as f:
                self.class_names = json.load(f)
            logger.info(f"Loaded {len(self.class_names)} food categories.")
        except Exception as exc:
            logger.error(f"Failed to read class names from {self.class_names_path}: {exc}")
            self.class_names = []

    def _load_model(self) -> None:
        """Load the trained Keras model."""
        if not self.model_path.exists():
            logger.warning(f"Model file does not exist at {self.model_path}")
            self.model = None
            return

        try:
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            logger.info(f"Successfully loaded model from {self.model_path}")
        except Exception as exc:
            logger.error(f"Failed to load model from {self.model_path}: {exc}")
            self.model = None

    @property
    def is_ready(self) -> bool:
        """Check if both model and class labels are loaded and operational."""
        return self.model is not None and len(self.class_names) > 0

    @staticmethod
    def preprocess_image(image: Image.Image) -> np.ndarray:
        """
        Validate and preprocess a PIL Image for MobileNetV2 input.
        
        Args:
            image: PIL Image instance.
            
        Returns:
            Preprocessed numpy array of shape (1, 224, 224, 3) with float32 values.
            
        Raises:
            ModelInferenceError: If image is invalid or cannot be processed.
        """
        if not isinstance(image, Image.Image):
            raise ModelInferenceError("Input must be a valid PIL Image instance.")

        try:
            # Convert non-RGB (RGBA, Grayscale, etc.) to RGB
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Resize to expected dimensions
            resized = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)

            # Convert to numpy array
            arr = np.array(resized, dtype=np.float32)

            # MobileNetV2 expects shape (1, 224, 224, 3)
            # MobileNetV2 preprocessing layer is included in the model or scaled:
            batch_arr = np.expand_dims(arr, axis=0)
            return batch_arr
        except Exception as exc:
            raise ModelInferenceError(f"Image preprocessing failed: {exc}") from exc

    def predict(
        self,
        image: Image.Image,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Predict Top-K food categories and confidence scores for an image.
        
        Args:
            image: PIL Image of the food.
            top_k: Number of highest-probability categories to return (default 3).
            
        Returns:
            List of tuples: [(class_name, confidence_score_float), ...]
            where confidence_score is between 0.0 and 1.0.
            
        Raises:
            ModelInferenceError: If model is not loaded or prediction fails.
        """
        if not self.is_ready:
            raise ModelInferenceError(
                "Model is not loaded. Train the model using train_model.py or export starter model."
            )

        preprocessed = self.preprocess_image(image)

        try:
            # Model output: probabilities across 101 classes
            predictions = self.model.predict(preprocessed, verbose=0)
            probabilities = predictions[0]

            # Top-K indices sorted in descending order
            top_indices = np.argsort(probabilities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                class_label = (
                    self.class_names[idx]
                    if idx < len(self.class_names)
                    else f"class_{idx}"
                )
                score = float(probabilities[idx])
                results.append((class_label, round(score, 4)))

            return results
        except Exception as exc:
            raise ModelInferenceError(f"Inference execution failed: {exc}") from exc


# Singleton instance helper
_default_classifier: Optional[FoodClassifier] = None


def get_food_classifier(
    model_path: Optional[Union[str, Path]] = None,
    class_names_path: Optional[Union[str, Path]] = None
) -> FoodClassifier:
    """Retrieve or initialize the FoodClassifier instance."""
    global _default_classifier
    if _default_classifier is None or model_path is not None:
        _default_classifier = FoodClassifier(model_path, class_names_path)
    return _default_classifier

