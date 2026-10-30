
"""
Project Configuration Module
AI-Based Food Recognition and Nutrition Tracker Using Deep Learning
Supports MobileNetV2 and EfficientNetB3 models.
"""

from pathlib import Path

# ==========================================
# PROJECT DIRECTORIES
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
TESTS_DIR = BASE_DIR / "tests"
TFDS_DIR = DATA_DIR / "tfds"


# ==========================================
# DATA FILE PATHS
# ==========================================

NUTRITION_CSV_PATH = DATA_DIR / "nutrition_dataset.csv"
DATABASE_PATH = DATA_DIR / "meals.db"
LOGO_PATH = ASSETS_DIR / "logo.png"


# ==========================================
# MOBILENETV2 MODEL (ORIGINAL BACKUP)
# ==========================================

MOBILENET_MODEL_PATH = (
    MODELS_DIR / "food_classifier.keras"
)

MOBILENET_CLASS_NAMES_PATH = (
    MODELS_DIR / "class_names.json"
)


# ==========================================
# EFFICIENTNETB3 MODEL (NEW CANDIDATE)
# ==========================================

EFFICIENTNET_MODEL_PATH = (
    MODELS_DIR / "food_classifier_efficientnetb3_candidate.keras"
)

EFFICIENTNET_CLASS_NAMES_PATH = (
    MODELS_DIR / "class_names_efficientnetb3_candidate.json"
)


# ==========================================
# ACTIVE MODEL CONFIGURATION
# ==========================================

# Keep MobileNetV2 active until EfficientNetB3
# passes all integration and application tests.

ACTIVE_MODEL = "efficientnetb3"

if ACTIVE_MODEL == "efficientnetb3":
    MODEL_PATH = EFFICIENTNET_MODEL_PATH
    CLASS_NAMES_PATH = EFFICIENTNET_CLASS_NAMES_PATH
else:
    MODEL_PATH = MOBILENET_MODEL_PATH
    CLASS_NAMES_PATH = MOBILENET_CLASS_NAMES_PATH


# ==========================================
# IMAGE PROCESSING CONFIGURATION
# ==========================================

# MobileNetV2 input dimensions
IMAGE_SIZE = (224, 224)
IMAGE_SHAPE = (224, 224, 3)

# EfficientNetB3 input dimensions
EFFICIENTNET_IMAGE_SIZE = (300, 300)
EFFICIENTNET_IMAGE_SHAPE = (300, 300, 3)

ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}

MAX_IMAGE_SIZE_MB = 10


# ==========================================
# MACHINE LEARNING CONFIGURATION
# ==========================================

NUM_CLASSES = 101

DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 1e-4

FINE_TUNE_EPOCHS = 5
FINE_TUNE_LEARNING_RATE = 1e-5


# ==========================================
# MEAL LOGGING CONFIGURATION
# ==========================================

MEAL_TYPES = [
    "Breakfast",
    "Lunch",
    "Dinner",
    "Snack"
]


# ==========================================
# DAILY NUTRITION GOALS
# ==========================================

# Standard adult 2000 kcal reference baseline.
# These are configurable reference values, not
# individualized dietary recommendations.

DEFAULT_DAILY_GOALS = {
    "calories": 2000.0,
    "protein": 75.0,
    "carbs": 250.0,
    "fat": 70.0,
    "fiber": 30.0,
    "sugar": 50.0,
    "sodium_mg": 2300.0
}


# ==========================================
# MACRONUTRIENT CALORIE VALUES
# ==========================================

CALORIES_PER_MACRO = {
    "protein": 4.0,
    "carbs": 4.0,
    "fat": 9.0
}


# ==========================================
# DATA ATTRIBUTION
# ==========================================

NUTRITION_DATA_SOURCE = (
    "USDA FoodData Central / Standard Reference Database"
)


# ==========================================
# DIRECTORY INITIALIZATION
# ==========================================

def ensure_directories() -> None:
    """Ensure all required project directories exist."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    TFDS_DIR.mkdir(parents=True, exist_ok=True)
    TESTS_DIR.mkdir(parents=True, exist_ok=True)


# Automatically verify directories upon import
ensure_directories()
