"""
Project Configuration Module
AI-Based Food Recognition and Nutrition Tracker Using Deep Learning
"""

from pathlib import Path

# Project Root Directory
BASE_DIR = Path(__file__).resolve().parent

# Core Subdirectories
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
TESTS_DIR = BASE_DIR / "tests"
TFDS_DIR = DATA_DIR / "tfds"

# Core File Paths
NUTRITION_CSV_PATH = DATA_DIR / "nutrition_dataset.csv"
MODEL_PATH = MODELS_DIR / "food_classifier.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
DATABASE_PATH = DATA_DIR / "meals.db"
LOGO_PATH = ASSETS_DIR / "logo.png"

# Image Processing Configuration
IMAGE_SIZE = (224, 224)
IMAGE_SHAPE = (224, 224, 3)
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGE_SIZE_MB = 10

# Machine Learning Configuration
NUM_CLASSES = 101
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 1e-4
FINE_TUNE_EPOCHS = 5
FINE_TUNE_LEARNING_RATE = 1e-5

# Meal Logging Configuration
MEAL_TYPES = ["Breakfast", "Lunch", "Dinner", "Snack"]

# Standard Nutritional Reference Daily Values (Adult 2000 kcal baseline)
DEFAULT_DAILY_GOALS = {
    "calories": 2000.0,
    "protein": 75.0,     # grams
    "carbs": 250.0,      # grams
    "fat": 70.0,         # grams
    "fiber": 30.0,       # grams
    "sugar": 50.0,       # grams
    "sodium_mg": 2300.0  # milligrams
}

# Macro Caloric Values (kcal per gram)
CALORIES_PER_MACRO = {
    "protein": 4.0,
    "carbs": 4.0,
    "fat": 9.0
}

# Attribution and Citation
NUTRITION_DATA_SOURCE = "USDA FoodData Central / Standard Reference Database"


def ensure_directories() -> None:
    """Ensure all required project directories exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    TFDS_DIR.mkdir(parents=True, exist_ok=True)
    TESTS_DIR.mkdir(parents=True, exist_ok=True)


# Automatically verify directories upon import
ensure_directories()

