"""
Database Management Module
AI-Based Food Recognition and Nutrition Tracker Using Deep Learning

Provides thread-safe SQLite persistence for user meal records,
daily aggregations, and nutritional trend tracking using parameterized SQL queries.

Database: data/meals.db
Table: meals
Columns:
- id: primary key (INTEGER)
- meal_date: ISO-format date (TEXT YYYY-MM-DD)
- created_at: timestamp (TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
- food_name: text (TEXT)
- meal_type: breakfast, lunch, dinner, or snack (TEXT)
- serving_grams: positive numeric value (REAL)
- calories_kcal: numeric (REAL)
- protein_g: numeric (REAL)
- carbs_g: numeric (REAL)
- fat_g: numeric (REAL)
- fiber_g: numeric (REAL)
- prediction_confidence: optional numeric (REAL)
- nutrition_source: text (TEXT)
"""

from datetime import date, datetime, timedelta
import math
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Union

from config import DATABASE_PATH

ALLOWED_MEAL_TYPES = {"breakfast", "lunch", "dinner", "snack"}


class DatabaseError(Exception):
    """Base exception for database operation failures."""
    pass


class DatabaseValidationError(DatabaseError, ValueError):
    """Raised when meal input validation fails."""
    pass


def get_db_connection(db_path: Optional[Union[str, Path]] = None) -> sqlite3.Connection:
    """
    Establish a connection to the SQLite database with Row factory.
    
    Args:
        db_path: Path to the SQLite database file. Defaults to config.DATABASE_PATH.
        
    Returns:
        sqlite3.Connection instance with row_factory set to sqlite3.Row.
    """
    target_path = Path(db_path) if db_path else DATABASE_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        conn = sqlite3.connect(str(target_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as exc:
        raise DatabaseError(f"Failed to connect to database at {target_path}: {exc}") from exc


def init_db(db_path: Optional[Union[str, Path]] = None) -> None:
    """
    Initialize SQLite database tables, schema, and indexes.
    
    Args:
        db_path: Optional path to the SQLite database.
    """
    create_meals_table = """
    CREATE TABLE IF NOT EXISTS meals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        meal_date TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        food_name TEXT NOT NULL,
        meal_type TEXT NOT NULL,
        serving_grams REAL NOT NULL,
        calories_kcal REAL NOT NULL,
        protein_g REAL NOT NULL,
        carbs_g REAL NOT NULL,
        fat_g REAL NOT NULL,
        fiber_g REAL DEFAULT 0.0,
        prediction_confidence REAL DEFAULT NULL,
        nutrition_source TEXT DEFAULT NULL
    );
    """

    create_index = """
    CREATE INDEX IF NOT EXISTS idx_meals_meal_date ON meals (meal_date);
    """

    try:
        conn = get_db_connection(db_path)
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(create_meals_table)
                cursor.execute(create_index)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise DatabaseError(f"Database initialization failed: {exc}") from exc


def _validate_date_string(date_str: str, field_name: str = "date") -> str:
    """Validate that date_str is a non-empty ISO-format date YYYY-MM-DD."""
    if not date_str or not isinstance(date_str, str):
        raise DatabaseValidationError(f"{field_name} must be a valid ISO-format date string (YYYY-MM-DD).")
    clean_date = date_str.strip()
    try:
        datetime.strptime(clean_date, "%Y-%m-%d")
        return clean_date
    except ValueError as exc:
        raise DatabaseValidationError(
            f"Invalid {field_name} '{date_str}'. Expected ISO format YYYY-MM-DD."
        ) from exc


def _validate_numeric(val: Any, name: str, allow_zero: bool = True) -> float:
    """Validate that val is a valid non-negative float."""
    try:
        f_val = float(val)
    except (TypeError, ValueError) as exc:
        raise DatabaseValidationError(f"{name} must be a valid numeric value. Received: {val}") from exc

    if math.isnan(f_val) or math.isinf(f_val):
        raise DatabaseValidationError(f"{name} cannot be NaN or infinity. Received: {val}")

    if allow_zero and f_val < 0.0:
        raise DatabaseValidationError(f"{name} cannot be negative. Received: {val}")
    elif not allow_zero and f_val <= 0.0:
        raise DatabaseValidationError(f"{name} must be strictly positive (> 0). Received: {val}")

    return f_val


def add_meal(
    food_name: Optional[Union[str, Dict[str, Any]]] = None,
    serving_grams: Optional[float] = None,
    calories_kcal: Optional[float] = None,
    protein_g: Optional[float] = None,
    carbs_g: Optional[float] = None,
    fat_g: Optional[float] = None,
    fiber_g: Optional[float] = 0.0,
    meal_type: str = "lunch",
    meal_date: Optional[str] = None,
    prediction_confidence: Optional[float] = None,
    nutrition_source: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
    **kwargs: Any
) -> int:
    """
    Insert a confirmed meal entry into the database using parameterized queries.
    
    Accepts individual keyword arguments or a single payload dictionary.
    
    Args:
        food_name: Food name string, or dictionary containing all meal fields.
        serving_grams: Serving size in grams (> 0).
        calories_kcal: Energy content in kcal (>= 0).
        protein_g: Protein in grams (>= 0).
        carbs_g: Carbohydrates in grams (>= 0).
        fat_g: Total fat in grams (>= 0).
        fiber_g: Dietary fiber in grams (>= 0, default 0.0).
        meal_type: 'breakfast', 'lunch', 'dinner', or 'snack'.
        meal_date: ISO date string YYYY-MM-DD (defaults to today).
        prediction_confidence: Optional AI model confidence (0.0 to 1.0).
        nutrition_source: Optional attribution text.
        db_path: Optional path to SQLite database.
        
    Returns:
        The integer ID of the newly inserted row.
        
    Raises:
        DatabaseValidationError: If any input field fails validation.
        DatabaseError: If insertion or database transaction fails.
    """
    # Handle dictionary payload passed as first positional argument
    if isinstance(food_name, dict):
        payload = food_name
        food_name = payload.get("food_name") or payload.get("display_name")
        serving_grams = payload.get("serving_grams")
        calories_kcal = payload.get("calories_kcal", payload.get("calories"))
        protein_g = payload.get("protein_g", payload.get("protein"))
        carbs_g = payload.get("carbs_g", payload.get("carbs"))
        fat_g = payload.get("fat_g", payload.get("fat"))
        fiber_g = payload.get("fiber_g", payload.get("fiber", 0.0))
        meal_type = payload.get("meal_type", "lunch")
        meal_date = payload.get("meal_date", payload.get("logged_date"))
        prediction_confidence = payload.get("prediction_confidence", payload.get("confidence"))
        nutrition_source = payload.get("nutrition_source", payload.get("source"))

    # Fallback to kwargs aliases if arguments were provided as aliases
    if calories_kcal is None and "calories" in kwargs:
        calories_kcal = kwargs["calories"]
    if protein_g is None and "protein" in kwargs:
        protein_g = kwargs["protein"]
    if carbs_g is None and "carbs" in kwargs:
        carbs_g = kwargs["carbs"]
    if fat_g is None and "fat" in kwargs:
        fat_g = kwargs["fat"]
    if fiber_g is None and "fiber" in kwargs:
        fiber_g = kwargs["fiber"]
    if meal_date is None and "logged_date" in kwargs:
        meal_date = kwargs["logged_date"]
    if prediction_confidence is None and "confidence" in kwargs:
        prediction_confidence = kwargs["confidence"]
    if nutrition_source is None and "source" in kwargs:
        nutrition_source = kwargs["source"]

    # 1. Validate food_name
    if not food_name or not str(food_name).strip():
        raise DatabaseValidationError("food_name must be a non-empty string.")
    food_name_clean = str(food_name).strip()

    # 2. Validate serving_grams (must be strictly > 0)
    serving_grams_val = _validate_numeric(serving_grams, "serving_grams", allow_zero=False)

    # 3. Validate meal_type (breakfast, lunch, dinner, or snack)
    if not meal_type or not str(meal_type).strip():
        raise DatabaseValidationError(
            f"meal_type cannot be empty. Must be one of: {', '.join(sorted(ALLOWED_MEAL_TYPES))}"
        )
    meal_type_clean = str(meal_type).strip().lower()
    if meal_type_clean not in ALLOWED_MEAL_TYPES:
        raise DatabaseValidationError(
            f"Invalid meal_type '{meal_type}'. Must be one of: {', '.join(sorted(ALLOWED_MEAL_TYPES))}"
        )

    # 4. Validate meal_date (ISO format YYYY-MM-DD)
    if not meal_date:
        meal_date_clean = date.today().isoformat()
    else:
        meal_date_clean = _validate_date_string(str(meal_date), "meal_date")

    # 5. Validate numeric nutrients
    calories_val = _validate_numeric(calories_kcal if calories_kcal is not None else 0.0, "calories_kcal")
    protein_val = _validate_numeric(protein_g if protein_g is not None else 0.0, "protein_g")
    carbs_val = _validate_numeric(carbs_g if carbs_g is not None else 0.0, "carbs_g")
    fat_val = _validate_numeric(fat_g if fat_g is not None else 0.0, "fat_g")
    fiber_val = _validate_numeric(fiber_g if fiber_g is not None else 0.0, "fiber_g")

    # 6. Validate prediction_confidence (optional 0.0 to 1.0)
    conf_val: Optional[float] = None
    if prediction_confidence is not None:
        try:
            conf_val = float(prediction_confidence)
            if conf_val < 0.0 or conf_val > 1.0 or math.isnan(conf_val):
                raise DatabaseValidationError("prediction_confidence must be between 0.0 and 1.0.")
            conf_val = round(conf_val, 4)
        except (ValueError, TypeError) as exc:
            raise DatabaseValidationError(
                f"prediction_confidence must be a valid float between 0.0 and 1.0. Received: {prediction_confidence}"
            ) from exc

    # 7. Validate nutrition_source
    source_clean = str(nutrition_source).strip() if nutrition_source else None

    insert_query = """
    INSERT INTO meals (
        meal_date, food_name, meal_type, serving_grams,
        calories_kcal, protein_g, carbs_g, fat_g, fiber_g,
        prediction_confidence, nutrition_source
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """

    values = (
        meal_date_clean,
        food_name_clean,
        meal_type_clean,
        serving_grams_val,
        calories_val,
        protein_val,
        carbs_val,
        fat_val,
        fiber_val,
        conf_val,
        source_clean,
    )

    try:
        conn = get_db_connection(db_path)
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(insert_query, values)
                return cursor.lastrowid
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise DatabaseError(f"Failed to insert meal into database: {exc}") from exc


def get_meals(
    date_string: Optional[str] = None,
    limit: Optional[int] = None,
    db_path: Optional[Union[str, Path]] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve logged meals, optionally filtered by ISO date (YYYY-MM-DD).
    
    Args:
        date_string: Optional ISO date string (YYYY-MM-DD).
        limit: Optional maximum number of meals to return.
        db_path: Optional database path.
        
    Returns:
        List of meal dictionaries ordered chronologically.
    """
    if date_string is not None:
        valid_date = _validate_date_string(date_string, "date_string")
        query = """
        SELECT id, meal_date, created_at, food_name, meal_type, serving_grams,
               calories_kcal, protein_g, carbs_g, fat_g, fiber_g,
               prediction_confidence, nutrition_source
        FROM meals
        WHERE meal_date = ?
        ORDER BY created_at ASC, id ASC;
        """
        params: Tuple[Any, ...] = (valid_date,)
    else:
        query = """
        SELECT id, meal_date, created_at, food_name, meal_type, serving_grams,
               calories_kcal, protein_g, carbs_g, fat_g, fiber_g,
               prediction_confidence, nutrition_source
        FROM meals
        ORDER BY meal_date DESC, created_at DESC, id DESC
        """
        if limit is not None:
            query += " LIMIT ?;"
            params = (int(limit),)
        else:
            query += ";"
            params = ()

    try:
        conn = get_db_connection(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            results = []
            for row in rows:
                item = dict(row)
                # Add convenience aliases for UI and backward compatibility
                item["calories"] = item["calories_kcal"]
                item["protein"] = item["protein_g"]
                item["carbs"] = item["carbs_g"]
                item["fat"] = item["fat_g"]
                item["fiber"] = item["fiber_g"]
                item["confidence"] = item["prediction_confidence"]
                item["display_name"] = item["food_name"].title()
                item["logged_date"] = item["meal_date"]
                item["logged_at"] = item["created_at"]
                results.append(item)
            return results
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise DatabaseError(f"Failed to fetch meals: {exc}") from exc


def get_daily_totals(
    date_string: str,
    db_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Calculate aggregated nutritional totals for a specific date (YYYY-MM-DD).
    Handles empty database or non-existent dates gracefully by returning zeroes.
    
    Args:
        date_string: ISO date string (YYYY-MM-DD).
        db_path: Optional database path.
        
    Returns:
        Dictionary containing meal_count, calories_kcal, protein_g, carbs_g, fat_g, fiber_g.
    """
    valid_date = _validate_date_string(date_string, "date_string")

    query = """
    SELECT 
        COUNT(id) AS meal_count,
        COALESCE(SUM(calories_kcal), 0.0) AS calories_kcal,
        COALESCE(SUM(protein_g), 0.0) AS protein_g,
        COALESCE(SUM(carbs_g), 0.0) AS carbs_g,
        COALESCE(SUM(fat_g), 0.0) AS fat_g,
        COALESCE(SUM(fiber_g), 0.0) AS fiber_g
    FROM meals
    WHERE meal_date = ?;
    """

    try:
        conn = get_db_connection(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(query, (valid_date,))
            row = cursor.fetchone()
            count = int(row["meal_count"]) if row else 0
            cal = round(float(row["calories_kcal"]), 2) if row else 0.0
            prot = round(float(row["protein_g"]), 2) if row else 0.0
            carbs = round(float(row["carbs_g"]), 2) if row else 0.0
            fat = round(float(row["fat_g"]), 2) if row else 0.0
            fiber = round(float(row["fiber_g"]), 2) if row else 0.0

            return {
                "meal_date": valid_date,
                "date": valid_date,
                "meal_count": count,
                "total_meals": count,
                "calories_kcal": cal,
                "calories": cal,
                "protein_g": prot,
                "protein": prot,
                "carbs_g": carbs,
                "carbs": carbs,
                "fat_g": fat,
                "fat": fat,
                "fiber_g": fiber,
                "fiber": fiber,
            }
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise DatabaseError(f"Failed to calculate daily totals for {date_string}: {exc}") from exc


def get_meal_history(
    start_date: str,
    end_date: str,
    db_path: Optional[Union[str, Path]] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve daily aggregated nutritional totals between start_date and end_date (inclusive).
    Guarantees an entry for every calendar date in the range, filling unlogged days with zeroes.
    
    Args:
        start_date: ISO date string YYYY-MM-DD.
        end_date: ISO date string YYYY-MM-DD.
        db_path: Optional database path.
        
    Returns:
        List of daily summary dictionaries ordered from oldest to newest.
    """
    start_clean = _validate_date_string(start_date, "start_date")
    end_clean = _validate_date_string(end_date, "end_date")

    d_start = datetime.strptime(start_clean, "%Y-%m-%d").date()
    d_end = datetime.strptime(end_clean, "%Y-%m-%d").date()

    if d_start > d_end:
        raise DatabaseValidationError(
            f"start_date ({start_date}) cannot be after end_date ({end_date})."
        )

    query = """
    SELECT 
        meal_date,
        COUNT(id) AS meal_count,
        COALESCE(SUM(calories_kcal), 0.0) AS calories_kcal,
        COALESCE(SUM(protein_g), 0.0) AS protein_g,
        COALESCE(SUM(carbs_g), 0.0) AS carbs_g,
        COALESCE(SUM(fat_g), 0.0) AS fat_g,
        COALESCE(SUM(fiber_g), 0.0) AS fiber_g
    FROM meals
    WHERE meal_date BETWEEN ? AND ?
    GROUP BY meal_date;
    """

    try:
        conn = get_db_connection(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(query, (start_clean, end_clean))
            rows = cursor.fetchall()
            db_map = {row["meal_date"]: dict(row) for row in rows}
        finally:
            conn.close()

        total_days = (d_end - d_start).days + 1
        history: List[Dict[str, Any]] = []

        for i in range(total_days):
            current_day_str = (d_start + timedelta(days=i)).isoformat()
            if current_day_str in db_map:
                entry = db_map[current_day_str]
                m_count = int(entry["meal_count"])
                cal = round(float(entry["calories_kcal"]), 2)
                prot = round(float(entry["protein_g"]), 2)
                carbs = round(float(entry["carbs_g"]), 2)
                fat = round(float(entry["fat_g"]), 2)
                fiber = round(float(entry["fiber_g"]), 2)
            else:
                m_count = 0
                cal = 0.0
                prot = 0.0
                carbs = 0.0
                fat = 0.0
                fiber = 0.0

            history.append({
                "meal_date": current_day_str,
                "date": current_day_str,
                "meal_count": m_count,
                "total_meals": m_count,
                "calories_kcal": cal,
                "calories": cal,
                "protein_g": prot,
                "protein": prot,
                "carbs_g": carbs,
                "carbs": carbs,
                "fat_g": fat,
                "fat": fat,
                "fiber_g": fiber,
                "fiber": fiber,
            })

        return history
    except sqlite3.Error as exc:
        raise DatabaseError(f"Failed to fetch meal history: {exc}") from exc


def delete_meal(meal_id: int, db_path: Optional[Union[str, Path]] = None) -> bool:
    """
    Delete a specific meal entry by ID using a parameterized query.
    
    Args:
        meal_id: The integer ID of the meal to delete.
        db_path: Optional database path.
        
    Returns:
        True if a meal was deleted, False if ID was not found.
    """
    try:
        id_val = int(meal_id)
    except (ValueError, TypeError) as exc:
        raise DatabaseValidationError(f"meal_id must be a valid integer. Received: {meal_id}") from exc

    query = "DELETE FROM meals WHERE id = ?;"

    try:
        conn = get_db_connection(db_path)
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(query, (id_val,))
                return cursor.rowcount > 0
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise DatabaseError(f"Failed to delete meal ID {meal_id}: {exc}") from exc


def clear_all_meals(db_path: Optional[Union[str, Path]] = None) -> bool:
    """
    Delete all meal entries from the meals table.
    
    Args:
        db_path: Optional database path.
        
    Returns:
        True upon successful clearance.
    """
    try:
        conn = get_db_connection(db_path)
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM meals;")
                return True
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise DatabaseError(f"Failed to clear meals table: {exc}") from exc


# Backward compatibility aliases
def get_meals_by_date(date_str: str, db_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """Backward compatibility alias for get_meals(date_string=date_str)."""
    return get_meals(date_string=date_str, db_path=db_path)


def get_daily_summary(date_str: str, db_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Backward compatibility alias for get_daily_totals(date_string=date_str)."""
    return get_daily_totals(date_string=date_str, db_path=db_path)


def get_nutrition_trends(
    days: int = 7,
    end_date: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None
) -> List[Dict[str, Any]]:
    """Backward compatibility alias for get_meal_history over the last N days."""
    end = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else date.today()
    start = end - timedelta(days=days - 1)
    return get_meal_history(start.isoformat(), end.isoformat(), db_path=db_path)


def get_recent_meals(limit: int = 50, db_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """Backward compatibility alias for get_meals(limit=limit)."""
    return get_meals(limit=limit, db_path=db_path)


# Automatically initialize the database on import
init_db()
