"""
Nutrition Data Subsystem Module
AI-Based Food Recognition and Nutrition Tracker Using Deep Learning

Features:
- Loads official CSV dataset from data/nutrition_dataset.csv
- Required schema: food_name,calories_kcal,protein_g,carbs_g,fat_g,fiber_g,source
- Values represent 100g baseline portions
- Conservative matching: exact normalized matches preferred, never silently substitutes a different food
- Missing nutrient values are preserved as None/NaN (never silently zeroed) and reported clearly
- Serving calculations scaled via: nutrient_for_serving = nutrient_per_100g * serving_grams / 100
"""

import logging
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from config import NUTRITION_CSV_PATH

logger = logging.getLogger("NutritionSubsystem")

REQUIRED_COLUMNS = [
    "food_name",
    "calories_kcal",
    "protein_g",
    "carbs_g",
    "fat_g",
    "fiber_g",
    "source",
]

NUTRIENT_FIELDS = [
    "calories_kcal",
    "protein_g",
    "carbs_g",
    "fat_g",
    "fiber_g",
]

# Module-level cache for loaded dataset
_CACHED_DF: Optional[pd.DataFrame] = None
_CACHED_CSV_PATH: Optional[Path] = None


class NutritionError(Exception):
    """Base exception for nutrition subsystem errors."""
    pass


class NutritionDatasetError(NutritionError):
    """Raised when dataset CSV cannot be found, read, or is malformed."""
    pass


class FoodNotFoundError(NutritionError, KeyError):
    """
    Raised when requested food cannot be found via exact normalized match.
    Carries suggestions if fuzzy/partial matches were discovered.
    """
    def __init__(self, food_name: str, suggestions: Optional[List[str]] = None):
        self.food_name = food_name
        self.suggestions = suggestions or []
        msg = f"Food item '{food_name}' was not found in the nutrition dataset."
        if self.suggestions:
            msg += f" Did you mean: {', '.join(self.suggestions[:5])}?"
        super().__init__(msg)


class InvalidServingSizeError(NutritionError, ValueError):
    """Raised when serving size is non-positive or non-numeric."""
    pass


def normalize_food_name(food_name: Any) -> str:
    """
    Normalize food name by stripping whitespace, lowercasing, and
    converting underscores, hyphens, and multi-spaces into single spaces.
    
    Args:
        food_name: Any input representation of food name.
        
    Returns:
        Canonical normalized string.
        
    Examples:
        normalize_food_name("  Apple_Pie  ") -> "apple pie"
        normalize_food_name("Caesar-Salad") -> "caesar salad"
        normalize_food_name("PIZZA") -> "pizza"
    """
    if food_name is None:
        return ""
    
    s = str(food_name).strip().lower()
    # Replace underscores, hyphens, and slashes with spaces
    s = re.sub(r"[_\-/]+", " ", s)
    # Remove any unwanted symbols keeping alphanumeric and spaces
    s = re.sub(r"[^\w\s]", " ", s)
    # Collapse multiple consecutive whitespace characters
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_nutrition_dataset(
    csv_path: Optional[Union[str, Path]] = None,
    reload: bool = False
) -> pd.DataFrame:
    """
    Load, validate, clean, and deduplicate the nutrition CSV dataset.
    
    Args:
        csv_path: Optional path to the CSV file. Defaults to config.NUTRITION_CSV_PATH.
        reload: If True, forces re-reading from disk even if cached.
        
    Returns:
        Validated pandas DataFrame with cleaned columns and normalized keys.
        
    Raises:
        NutritionDatasetError: If file is missing, empty, or missing required columns.
    """
    global _CACHED_DF, _CACHED_CSV_PATH

    target_path = Path(csv_path) if csv_path is not None else NUTRITION_CSV_PATH

    if not reload and _CACHED_DF is not None and _CACHED_CSV_PATH == target_path:
        return _CACHED_DF

    if not target_path.exists():
        raise NutritionDatasetError(
            f"Nutrition dataset not found at {target_path}. "
            "Please ensure data/nutrition_dataset.csv exists."
        )

    try:
        df = pd.read_csv(target_path, encoding="utf-8")
    except Exception as exc:
        raise NutritionDatasetError(f"Failed to read CSV at {target_path}: {exc}") from exc

    if df.empty:
        raise NutritionDatasetError(f"Nutrition dataset at {target_path} is empty.")

    # Validate required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise NutritionDatasetError(
            f"Nutrition dataset at {target_path} is missing required columns: {missing_cols}. "
            f"Required schema: {REQUIRED_COLUMNS}"
        )

    # Clean text columns
    df["food_name"] = df["food_name"].astype(str).str.strip()
    df["source"] = df["source"].astype(str).str.strip()

    # Drop rows where food_name is empty or null
    df = df[df["food_name"].str.len() > 0].copy()

    # Create normalized lookup key
    df["_normalized_name"] = df["food_name"].apply(normalize_food_name)
    df = df[df["_normalized_name"].str.len() > 0].copy()

    # Handle duplicates: keep first occurrence and log if duplicates are present
    dup_mask = df.duplicated(subset=["_normalized_name"], keep="first")
    num_dups = int(dup_mask.sum())
    if num_dups > 0:
        logger.warning(
            f"Dataset at {target_path} contained {num_dups} duplicate food entries. "
            "Deduplicating by keeping the first occurrence."
        )
        df = df.drop_duplicates(subset=["_normalized_name"], keep="first").copy()

    # Clean numeric columns without silently converting missing values to 0!
    for col in NUTRIENT_FIELDS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Invalidate negative numbers: nutrients cannot be negative
        df.loc[df[col] < 0, col] = np.nan

    _CACHED_DF = df
    _CACHED_CSV_PATH = target_path
    return _CACHED_DF


def search_foods(
    query: str,
    limit: int = 10,
    csv_path: Optional[Union[str, Path]] = None
) -> List[Dict[str, Any]]:
    """
    Search for foods matching the query text. Offers exact, prefix, and partial
    matches as suggestions, ranked by relevance.
    
    Args:
        query: Search string.
        limit: Maximum results to return (default 10).
        csv_path: Optional custom path to dataset CSV.
        
    Returns:
        List of structured food dictionaries with match metadata.
    """
    if not query or not str(query).strip():
        return []

    df = load_nutrition_dataset(csv_path)
    norm_query = normalize_food_name(query)
    if not norm_query:
        return []

    results = []

    # 1. Exact normalized matches
    m_exact = df["_normalized_name"] == norm_query
    for _, row in df[m_exact].iterrows():
        results.append(_row_to_nutrient_dict(row, match_type="exact"))

    # 2. Prefix matches (starts with query)
    m_prefix = df["_normalized_name"].str.startswith(norm_query) & (~m_exact)
    for _, row in df[m_prefix].iterrows():
        results.append(_row_to_nutrient_dict(row, match_type="prefix"))

    # 3. Substring matches (contains query)
    m_contains = (
        df["_normalized_name"].str.contains(norm_query, na=False, regex=False) &
        (~m_exact) &
        (~m_prefix)
    )
    for _, row in df[m_contains].iterrows():
        results.append(_row_to_nutrient_dict(row, match_type="contains"))

    # 4. Token matches (all words in query are in the food name)
    query_tokens = norm_query.split()
    if len(query_tokens) > 1:
        existing_indices = set(df[m_exact | m_prefix | m_contains].index)
        for idx, row in df.iterrows():
            if idx in existing_indices:
                continue
            name_norm = row["_normalized_name"]
            if all(token in name_norm for token in query_tokens):
                results.append(_row_to_nutrient_dict(row, match_type="token"))

    return results[:limit]


def get_nutrition(
    food_name: str,
    csv_path: Optional[Union[str, Path]] = None,
    raise_on_not_found: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Retrieve nutritional information for a food item.
    
    Food matching is strictly conservative:
    Only exact normalized matches are accepted. It will NEVER silently select
    a different food. If unmatched, it searches for suggestions.
    
    Args:
        food_name: Name of the food item (e.g. 'apple pie', 'apple_pie', 'Apple Pie').
        csv_path: Optional path to dataset CSV.
        raise_on_not_found: If True, raises FoodNotFoundError on unmatched food.
                           If False, returns a structured error dictionary.
                           
    Returns:
        Structured dictionary containing nutrient values per 100g, completeness flag,
        and missing nutrient report.
        
    Raises:
        FoodNotFoundError: If food is not found and raise_on_not_found is True.
    """
    if not food_name or not str(food_name).strip():
        if raise_on_not_found:
            raise FoodNotFoundError("Empty food name provided.")
        return {
            "found": False,
            "food_name": "",
            "normalized_name": "",
            "error": "Food name cannot be empty.",
            "suggestions": [],
            "calories_kcal": None,
            "protein_g": None,
            "carbs_g": None,
            "fat_g": None,
            "fiber_g": None,
            "source": None,
            "is_complete": False,
            "missing_nutrients": list(NUTRIENT_FIELDS),
        }

    df = load_nutrition_dataset(csv_path)
    norm_target = normalize_food_name(food_name)

    match = df[df["_normalized_name"] == norm_target]

    if match.empty:
        # Search suggestions to help user without silently selecting one
        suggestions = [
            r["food_name"] for r in search_foods(food_name, limit=5, csv_path=csv_path)
        ]
        if raise_on_not_found:
            raise FoodNotFoundError(food_name, suggestions=suggestions)
        return {
            "found": False,
            "food_name": str(food_name),
            "normalized_name": norm_target,
            "error": f"Food '{food_name}' was not found in the nutrition dataset.",
            "suggestions": suggestions,
            "calories_kcal": None,
            "protein_g": None,
            "carbs_g": None,
            "fat_g": None,
            "fiber_g": None,
            "source": None,
            "is_complete": False,
            "missing_nutrients": list(NUTRIENT_FIELDS),
        }

    row = match.iloc[0]
    return _row_to_nutrient_dict(row, match_type="exact")


def calculate_serving_nutrition(
    food_name: str,
    serving_grams: float,
    csv_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Calculate nutritional values scaled for a specific portion size in grams.
    Formula: nutrient_for_serving = nutrient_per_100g * serving_grams / 100
    
    Missing nutrient values are preserved as None and never silently zeroed.
    
    Args:
        food_name: Food item name.
        serving_grams: Portion size in grams (strictly > 0).
        csv_path: Optional custom CSV path.
        
    Returns:
        Structured dictionary containing scaled nutrients, portion size,
        completeness report, and source attribution.
        
    Raises:
        InvalidServingSizeError: If serving_grams <= 0, NaN, or non-numeric.
        FoodNotFoundError: If food_name is not in the dataset.
    """
    try:
        serving_val = float(serving_grams)
    except (ValueError, TypeError) as exc:
        raise InvalidServingSizeError(
            f"Serving size must be a valid numeric value. Received: {serving_grams}"
        ) from exc

    if math.isnan(serving_val) or math.isinf(serving_val) or serving_val <= 0:
        raise InvalidServingSizeError(
            f"Serving size must be strictly positive and greater than 0 grams. Received: {serving_grams}"
        )

    base_nutrition = get_nutrition(food_name, csv_path=csv_path, raise_on_not_found=True)

    factor = serving_val / 100.0
    scaled: Dict[str, Any] = {
        "food_name": base_nutrition["food_name"],
        "serving_grams": round(serving_val, 1),
        "source": base_nutrition["source"],
        "match_type": base_nutrition.get("match_type", "exact"),
    }

    missing_nutrients = []
    warnings = []

    for field in NUTRIENT_FIELDS:
        val = base_nutrition.get(field)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            scaled[field] = None
            missing_nutrients.append(field)
        else:
            scaled[field] = round(float(val) * factor, 2)

    is_complete = len(missing_nutrients) == 0
    if not is_complete:
        warnings.append(
            f"Nutritional profile is incomplete for '{base_nutrition['food_name']}': "
            f"missing values for {', '.join(missing_nutrients)}."
        )

    scaled["is_complete"] = is_complete
    scaled["missing_nutrients"] = missing_nutrients
    scaled["warnings"] = warnings

    # Add convenience aliases for database and UI compatibility
    scaled["calories"] = scaled["calories_kcal"]
    scaled["protein"] = scaled["protein_g"]
    scaled["carbs"] = scaled["carbs_g"]
    scaled["fat"] = scaled["fat_g"]
    scaled["fiber"] = scaled["fiber_g"]

    scaled["per_100g_reference"] = {
        field: base_nutrition[field] for field in NUTRIENT_FIELDS
    }

    return scaled


def _row_to_nutrient_dict(row: pd.Series, match_type: str = "exact") -> Dict[str, Any]:
    """Helper to convert a pandas row to structured nutrient dictionary."""
    missing_nutrients = []
    warnings = []

    data: Dict[str, Any] = {
        "found": True,
        "food_name": str(row["food_name"]),
        "normalized_name": str(row.get("_normalized_name", normalize_food_name(row["food_name"]))),
        "source": str(row["source"]),
        "match_type": match_type,
    }

    for field in NUTRIENT_FIELDS:
        val = row.get(field)
        if pd.isna(val):
            data[field] = None
            missing_nutrients.append(field)
        else:
            data[field] = round(float(val), 2)

    is_complete = len(missing_nutrients) == 0
    if not is_complete:
        warnings.append(f"Incomplete nutrient information: missing {missing_nutrients}")

    data["is_complete"] = is_complete
    data["missing_nutrients"] = missing_nutrients
    data["warnings"] = warnings

    # Backwards-compatible aliases for legacy callers
    data["calories"] = data["calories_kcal"]
    data["protein"] = data["protein_g"]
    data["carbs"] = data["carbs_g"]
    data["fat"] = data["fat_g"]
    data["fiber"] = data["fiber_g"]

    return data
