"""
Unit Tests for Database Module
AI-Based Food Recognition and Nutrition Tracker Using Deep Learning

Covers:
- Database initialization and table creation
- Meal insertion via keyword arguments and dictionary payloads
- Meal retrieval with and without date filtering
- Daily totals aggregation and handling of empty database
- Meal history date range queries and continuous date generation
- Deletion of meals
- Validation of invalid inputs: serving sizes, dates, meal types, and numeric values
"""

from datetime import date, timedelta
from pathlib import Path
import pytest

from database import (
    init_db,
    add_meal,
    get_meals,
    get_daily_totals,
    delete_meal,
    get_meal_history,
    clear_all_meals,
    DatabaseValidationError,
    DatabaseError,
)


@pytest.fixture
def temp_db(tmp_path):
    """Fixture providing a fresh isolated SQLite database path."""
    db_file = tmp_path / "test_meals.db"
    init_db(db_file)
    return db_file


@pytest.fixture
def sample_meal_kwargs():
    """Fixture for sample meal insertion keyword arguments."""
    return {
        "food_name": "pizza",
        "serving_grams": 200.0,
        "calories_kcal": 532.0,
        "protein_g": 22.8,
        "carbs_g": 66.6,
        "fat_g": 19.6,
        "fiber_g": 4.6,
        "meal_type": "lunch",
        "meal_date": "2026-10-29",
        "prediction_confidence": 0.94,
        "nutrition_source": "USDA FoodData Central",
    }


def test_init_db(temp_db):
    """Verify database initialization creates the meals table."""
    assert temp_db.exists()
    meals = get_meals(db_path=temp_db)
    assert meals == []


def test_add_meal_and_get_meals(temp_db, sample_meal_kwargs):
    """Test inserting a meal and retrieving it by date and without date."""
    meal_id = add_meal(db_path=temp_db, **sample_meal_kwargs)
    assert meal_id > 0

    # Retrieve with date filter
    meals_today = get_meals(date_string="2026-10-29", db_path=temp_db)
    assert len(meals_today) == 1
    m = meals_today[0]
    assert m["id"] == meal_id
    assert m["food_name"] == "pizza"
    assert m["serving_grams"] == 200.0
    assert m["calories_kcal"] == 532.0
    assert m["protein_g"] == 22.8
    assert m["prediction_confidence"] == 0.94
    assert m["meal_type"] == "lunch"

    # Retrieve all meals
    all_meals = get_meals(db_path=temp_db)
    assert len(all_meals) == 1


def test_add_meal_dict_payload(temp_db, sample_meal_kwargs):
    """Test inserting a meal using a dictionary payload."""
    meal_id = add_meal(sample_meal_kwargs, db_path=temp_db)
    assert meal_id > 0

    meals = get_meals(date_string=sample_meal_kwargs["meal_date"], db_path=temp_db)
    assert len(meals) == 1
    assert meals[0]["food_name"] == "pizza"


def test_get_daily_totals(temp_db, sample_meal_kwargs):
    """Test daily totals aggregation and empty database handling."""
    # Empty date returns 0 totals
    empty_totals = get_daily_totals("2026-10-29", db_path=temp_db)
    assert empty_totals["meal_count"] == 0
    assert empty_totals["calories_kcal"] == 0.0

    # Add meal 1
    add_meal(db_path=temp_db, **sample_meal_kwargs)

    # Add meal 2 on the same date
    meal_2 = sample_meal_kwargs.copy()
    meal_2["calories_kcal"] = 250.0
    meal_2["protein_g"] = 10.0
    meal_2["carbs_g"] = 30.0
    meal_2["fat_g"] = 5.0
    meal_2["fiber_g"] = 2.0
    meal_2["meal_type"] = "dinner"
    add_meal(db_path=temp_db, **meal_2)

    totals = get_daily_totals("2026-10-29", db_path=temp_db)
    assert totals["meal_count"] == 2
    assert totals["calories_kcal"] == 782.0
    assert totals["protein_g"] == 32.8
    assert totals["carbs_g"] == 96.6
    assert totals["fat_g"] == 24.6
    assert totals["fiber_g"] == 6.6


def test_delete_meal(temp_db, sample_meal_kwargs):
    """Test deleting an existing meal."""
    meal_id = add_meal(db_path=temp_db, **sample_meal_kwargs)
    assert len(get_meals(db_path=temp_db)) == 1

    # Delete existing meal
    success = delete_meal(meal_id, db_path=temp_db)
    assert success is True
    assert len(get_meals(db_path=temp_db)) == 0

    # Deleting nonexistent ID returns False
    success_again = delete_meal(meal_id, db_path=temp_db)
    assert success_again is False


def test_get_meal_history(temp_db, sample_meal_kwargs):
    """Test retrieving meal history over date range filling missing days with zeroes."""
    add_meal(db_path=temp_db, **sample_meal_kwargs)

    history = get_meal_history("2026-10-27", "2026-10-29", db_path=temp_db)
    assert len(history) == 3

    # Day 1 & Day 2 (unlogged)
    assert history[0]["meal_date"] == "2026-10-27"
    assert history[0]["meal_count"] == 0
    assert history[0]["calories_kcal"] == 0.0

    assert history[1]["meal_date"] == "2026-10-28"
    assert history[1]["meal_count"] == 0

    # Day 3 (logged meal)
    assert history[2]["meal_date"] == "2026-10-29"
    assert history[2]["meal_count"] == 1
    assert history[2]["calories_kcal"] == 532.0


def test_validation_invalid_serving_sizes(temp_db, sample_meal_kwargs):
    """Verify non-positive or non-numeric serving sizes raise DatabaseValidationError."""
    bad_serving_1 = sample_meal_kwargs.copy()
    bad_serving_1["serving_grams"] = 0.0
    with pytest.raises(DatabaseValidationError):
        add_meal(db_path=temp_db, **bad_serving_1)

    bad_serving_2 = sample_meal_kwargs.copy()
    bad_serving_2["serving_grams"] = -50.0
    with pytest.raises(DatabaseValidationError):
        add_meal(db_path=temp_db, **bad_serving_2)

    bad_serving_3 = sample_meal_kwargs.copy()
    bad_serving_3["serving_grams"] = "invalid_number"
    with pytest.raises(DatabaseValidationError):
        add_meal(db_path=temp_db, **bad_serving_3)


def test_validation_invalid_meal_types(temp_db, sample_meal_kwargs):
    """Verify invalid meal types raise DatabaseValidationError."""
    bad_type = sample_meal_kwargs.copy()
    bad_type["meal_type"] = "midnight_dessert"
    with pytest.raises(DatabaseValidationError):
        add_meal(db_path=temp_db, **bad_type)


def test_validation_invalid_dates(temp_db, sample_meal_kwargs):
    """Verify malformed date strings raise DatabaseValidationError."""
    bad_date = sample_meal_kwargs.copy()
    bad_date["meal_date"] = "29-10-2026"  # Not ISO format
    with pytest.raises(DatabaseValidationError):
        add_meal(db_path=temp_db, **bad_date)

    with pytest.raises(DatabaseValidationError):
        get_meals(date_string="not_a_date", db_path=temp_db)

    with pytest.raises(DatabaseValidationError):
        get_meal_history("2026-10-30", "2026-10-25", db_path=temp_db)  # start after end


def test_validation_negative_nutrients(temp_db, sample_meal_kwargs):
    """Verify negative nutrient numbers raise DatabaseValidationError."""
    bad_cal = sample_meal_kwargs.copy()
    bad_cal["calories_kcal"] = -100.0
    with pytest.raises(DatabaseValidationError):
        add_meal(db_path=temp_db, **bad_cal)


def test_validation_empty_food_name(temp_db, sample_meal_kwargs):
    """Verify empty food name raises DatabaseValidationError."""
    bad_food = sample_meal_kwargs.copy()
    bad_food["food_name"] = "   "
    with pytest.raises(DatabaseValidationError):
        add_meal(db_path=temp_db, **bad_food)


def test_clear_all_meals(temp_db, sample_meal_kwargs):
    """Test clearing all meals from database."""
    add_meal(db_path=temp_db, **sample_meal_kwargs)
    assert len(get_meals(db_path=temp_db)) == 1

    assert clear_all_meals(db_path=temp_db) is True
    assert len(get_meals(db_path=temp_db)) == 0
