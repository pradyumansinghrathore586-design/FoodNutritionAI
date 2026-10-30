
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

from config import (
    ACTIVE_MODEL,
    CLASS_NAMES_PATH,
    EFFICIENTNET_IMAGE_SIZE,
    IMAGE_SIZE,
    MODEL_PATH,
    NUM_CLASSES,
)

logger = logging.getLogger("FoodClassifierInference")


class ModelInferenceError(Exception):
    """Raised when food classification cannot be completed."""


class FoodClassifier:
    def __init__(
        self,
        model_path=None,
        class_names_path=None,
    ):
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.class_names_path = (
            Path(class_names_path) if class_names_path else CLASS_NAMES_PATH
        )

        self.model = None
        self.class_names = []
        self.image_size = (
            EFFICIENTNET_IMAGE_SIZE
            if ACTIVE_MODEL.lower() == "efficientnetb3"
            else IMAGE_SIZE
        )

        self._load_class_names()
        self._load_model()

    def _load_class_names(self):
        if not self.class_names_path.exists():
            logger.error(
                "Class names file not found: %s",
                self.class_names_path,
            )
            self.class_names = []
            return

        try:
            with open(self.class_names_path, "r", encoding="utf-8") as file:
                self.class_names = json.load(file)

            if not isinstance(self.class_names, list):
                raise ValueError("Class names must be a JSON list.")

            if len(self.class_names) != NUM_CLASSES:
                raise ValueError(
                    f"Expected {NUM_CLASSES} classes, "
                    f"found {len(self.class_names)}."
                )

        except Exception as exc:
            logger.exception("Could not load class names: %s", exc)
            self.class_names = []

    def _load_model(self):
        if not self.model_path.exists():
            logger.error("Model file not found: %s", self.model_path)
            self.model = None
            return

        try:
            self.model = tf.keras.models.load_model(
                self.model_path,
                compile=False,
            )

            input_shape = self.model.input_shape

            if isinstance(input_shape, list):
                raise ValueError("Multiple model inputs are not supported.")

            if len(input_shape) != 4:
                raise ValueError(
                    f"Expected a 4D image input, got {input_shape}."
                )

            height, width, channels = input_shape[1:]

            if channels != 3:
                raise ValueError(
                    f"Expected 3 RGB channels, got {channels}."
                )

            if height is None or width is None:
                raise ValueError(
                    "Model must have fixed image dimensions."
                )

            self.image_size = (int(width), int(height))

            output_shape = self.model.output_shape

            if isinstance(output_shape, list):
                raise ValueError("Multiple model outputs are not supported.")

            if output_shape[-1] != NUM_CLASSES:
                raise ValueError(
                    f"Expected {NUM_CLASSES} output classes, "
                    f"got {output_shape[-1]}."
                )

            logger.info("Model loaded: %s", self.model_path)
            logger.info("Model input shape: %s", input_shape)
            logger.info("Model output shape: %s", output_shape)

        except Exception as exc:
            logger.exception("Could not load model: %s", exc)
            self.model = None

    @property
    def is_ready(self) -> bool:
        return (
            self.model is not None
            and len(self.class_names) == NUM_CLASSES
        )

    @staticmethod
    def preprocess_image(
        image: Image.Image,
        image_size: Tuple[int, int] = IMAGE_SIZE,
    ) -> np.ndarray:
        if not isinstance(image, Image.Image):
            raise ModelInferenceError(
                "Input must be a PIL image."
            )

        try:
            image = image.convert("RGB")
            image = image.resize(
                image_size,
                Image.Resampling.BILINEAR,
            )

            image_array = np.asarray(
                image,
                dtype=np.float32,
            )

            image_array = np.expand_dims(image_array, axis=0)

            return image_array

        except Exception as exc:
            raise ModelInferenceError(
                f"Image preprocessing failed: {exc}"
            ) from exc

    def predict(
        self,
        image: Image.Image,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        if not self.is_ready:
            raise ModelInferenceError(
                "Food classifier is not ready. "
                "Check the model and class names files."
            )

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        try:
            preprocessed_image = self.preprocess_image(
                image,
                self.image_size,
            )

            predictions = self.model.predict(
                preprocessed_image,
                verbose=0,
            )

            probabilities = np.asarray(predictions[0])

            if probabilities.shape != (NUM_CLASSES,):
                raise ModelInferenceError(
                    "Model returned an unexpected prediction shape."
                )

            if not np.all(np.isfinite(probabilities)):
                raise ModelInferenceError(
                    "Model returned invalid prediction values."
                )

            top_indices = np.argsort(probabilities)[::-1][
                :min(top_k, NUM_CLASSES)
            ]

            results = []

            for index in top_indices:
                results.append(
                    (
                        self.class_names[index],
                        round(float(probabilities[index]), 4),
                    )
                )

            return results

        except ModelInferenceError:
            raise

        except Exception as exc:
            logger.exception("Prediction failed: %s", exc)
            raise ModelInferenceError(
                f"Food prediction failed: {exc}"
            ) from exc


_default_classifier: Optional[FoodClassifier] = None


def get_food_classifier(
    model_path=None,
    class_names_path=None,
) -> FoodClassifier:
    global _default_classifier

    if _default_classifier is None or model_path is not None:
        _default_classifier = FoodClassifier(
            model_path=model_path,
            class_names_path=class_names_path,
        )

    return _default_classifier
