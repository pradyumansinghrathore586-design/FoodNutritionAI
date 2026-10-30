"""
Unit Tests for Nutrition Subsystem
AI-Based Food Recognition and Nutrition Tracker Using Deep Learning

Covers:
- Exact normalized matches
- Missing foods and conservative matching (no silent fuzzy substitution)
- Invalid serving sizes
- Serving scaling calculations (per 100g * serving_grams / 100)
- Missing nutrient handling (must NOT silently zero missing values)
- Duplicate handling, whitespace, and capitalization
"""

import math
from pathlib import Path
import pytest
import pandas as pd

from nutrition import (
    load_nutrition_dataset,
    normalize_food_name,
    search_foods,
    get_nutrition,
    calculate_serving_nutrition,
    NutritionDatasetError,
    FoodNotFoundError,
    InvalidServingSizeError,
)
from config import NUTRITION_CSV_PATH


def test_normalize_food_name():
    """Verify name normalization strips whitespace, lowercases, and cleans separators."""
    assert normalize_food_name("  Apple_Pie  ") == "apple pie"
    assert normalize_food_name("PIZZA") == "pizza"
    assert normalize_food_name("Caesar-Salad") == "caesar salad"
    assert normalize_food_name("French   Fries") == "french fries"
    assert normalize_food_name("Macaroni_and_Cheese") == "macaroni and cheese"
    assert normalize_food_name(None) == ""


def test_load_nutrition_dataset():
    """Verify dataset loads successfully with required schema."""
    df = load_nutrition_dataset(NUTRITION_CSV_PATH, reload=True)
    assert not df.empty
    assert len(df) >= 100
    for col in ["food_name", "calories_kcal", "protein_g", "carbs_g", "fat_g", "fiber_g", "source"]:
        assert col in df.columns


def test_exact_matches():
    """Verify exact normalized matches across cases, underscores, and spacing."""
    # Standard spacing
    pizza = get_nutrition("pizza")
    assert pizza["found"] is True
    assert pizza["food_name"] == "pizza"
    assert pizza["calories_kcal"] > 0
    assert pizza["protein_g"] > 0

    # Food-101 underscore format
    apple_pie = get_nutrition("apple_pie")
    assert apple_pie["found"] is True
    assert apple_pie["food_name"] == "apple pie"

    # Capitalization and trailing whitespace
    salad = get_nutrition("  Caesar Salad  ")
    assert salad["found"] is True
    assert salad["food_name"] == "caesar salad"


def test_missing_foods_raises_food_not_found_error():
    """Verify missing foods raise FoodNotFoundError and provide suggestions."""
    with pytest.raises(FoodNotFoundError) as exc_info:
        get_nutrition("unobtainium_space_fruit_999")

    # Suggestions should be populated without selecting anything silently
    assert "unobtainium_space_fruit_999" in str(exc_info.value)

    # When raise_on_not_found=False, returns structured error dict
    res = get_nutrition("unobtainium_space_fruit_999", raise_on_not_found=False)
    assert res["found"] is False
    assert "error" in res
    assert res["calories_kcal"] is None


def test_conservative_matching_no_silent_substitution():
    """
    Verify conservative matching:
    A partial search like 'apple' must NOT silently return 'apple pie'.
    """
    with pytest.raises(FoodNotFoundError) as exc_info:
        # 'apple' is not an exact match for 'apple pie'
        # (unless fresh apple is in dataset, but here we test unmatched dish)
        get_nutrition("pizz")

    assert "pizz" in str(exc_info.value)
    # Suggestions should propose 'pizza'
    assert any("pizza" in s.lower() for s in exc_info.value.suggestions)


def test_serving_calculation_scaling():
    """Verify formula: nutrient_for_serving = nutrient_per_100g * serving_grams / 100."""
    base = get_nutrition("pizza")

    # 100 grams portion should match base 100g values exactly
    s_100 = calculate_serving_nutrition("pizza", 100.0)
    assert s_100["serving_grams"] == 100.0
    assert s_100["calories_kcal"] == base["calories_kcal"]
    assert s_100["protein_g"] == base["protein_g"]
    assert s_100["carbs_g"] == base["carbs_g"]
    assert s_100["fat_g"] == base["fat_g"]
    assert s_100["fiber_g"] == base["fiber_g"]

    # 200 grams portion should double the nutrients
    s_200 = calculate_serving_nutrition("pizza", 200.0)
    assert round(s_200["calories_kcal"], 1) == round(base["calories_kcal"] * 2.0, 1)
    assert round(s_200["protein_g"], 1) == round(base["protein_g"] * 2.0, 1)

    # 50 grams portion should halve the nutrients
    s_50 = calculate_serving_nutrition("pizza", 50.0)
    assert round(s_50["calories_kcal"], 1) == round(base["calories_kcal"] * 0.5, 1)


def test_invalid_serving_sizes():
    """Verify invalid portion sizes raise InvalidServingSizeError."""
    with pytest.raises(InvalidServingSizeError):
        calculate_serving_nutrition("pizza", 0)

    with pytest.raises(InvalidServingSizeError):
        calculate_serving_nutrition("pizza", -100)

    with pytest.raises(InvalidServingSizeError):
        calculate_serving_nutrition("pizza", "not_a_number")

    with pytest.raises(InvalidServingSizeError):
        calculate_serving_nutrition("pizza", float("nan"))


def test_missing_nutrient_values_not_zeroed(tmp_path):
    """
    Ensure missing nutrient values in CSV are NOT silently converted to zero,
    and incomplete nutritional profiles are clearly reported.
    """
    test_csv = tmp_path / "incomplete_nutrition.csv"
    test_csv.write_text(
        "food_name,calories_kcal,protein_g,carbs_g,fat_g,fiber_g,source\n"
        "mystery_soup,85.0,4.2,,1.5,,Test Lab\n",
        encoding="utf-8"
    )

    df = load_nutrition_dataset(test_csv, reload=True)
    assert pd.isna(df.loc[0, "carbs_g"])
    assert pd.isna(df.loc[0, "fiber_g"])

    item = get_nutrition("mystery_soup", csv_path=test_csv)
    assert item["found"] is True
    assert item["calories_kcal"] == 85.0
    assert item["protein_g"] == 4.2
    assert item["carbs_g"] is None  # NOT zero!
    assert item["fiber_g"] is None  # NOT zero!
    assert item["is_complete"] is False
    assert "carbs_g" in item["missing_nutrients"]
    assert "fiber_g" in item["missing_nutrients"]

    # Test serving calculation also preserves None and reports completeness
    scaled = calculate_serving_nutrition("mystery_soup", 150.0, csv_path=test_csv)
    assert scaled["calories_kcal"] == round(85.0 * 1.5, 2)
    assert scaled["carbs_g"] is None
    assert scaled["is_complete"] is False
    assert len(scaled["warnings"]) > 0


def test_duplicate_and_whitespace_handling(tmp_path):
    """Verify duplicate food rows are deduplicated and whitespace is trimmed."""
    test_csv = tmp_path / "duplicates.csv"
    test_csv.write_text(
        "food_name,calories_kcal,protein_g,carbs_g,fat_g,fiber_g,source\n"
        "  Tacos  ,226.0,10.0,20.0,11.0,2.0,Lab 1\n"
        "tacos,230.0,11.0,21.0,12.0,3.0,Lab 2\n",
        encoding="utf-8"
    )

    df = load_nutrition_dataset(test_csv, reload=True)
    assert len(df) == 1  # Deduplicated to first occurrence
    item = get_nutrition("tacos", csv_path=test_csv)
    assert item["food_name"] == "Tacos"
    assert item["calories_kcal"] == 226.0


def test_search_foods():
    """Verify search finds exact, prefix, and partial matches."""
    matches = search_foods("salad", limit=10)
    assert len(matches) >= 3
    names = [m["food_name"] for m in matches]
    assert any("salad" in name.lower() for name in names)


def test_missing_csv_raises_dataset_error(tmp_path):
    """Verify missing CSV file raises NutritionDatasetError."""
    nonexistent = tmp_path / "does_not_exist.csv"
    with pytest.raises(NutritionDatasetError):
        load_nutrition_dataset(nonexistent, reload=True)
