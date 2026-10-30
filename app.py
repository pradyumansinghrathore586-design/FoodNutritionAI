"""
FoodNutritionAI - AI-Based Food Recognition and Nutrition Tracker
Main Streamlit Application

Pages:
1. Dashboard: Daily calories, macros, Plotly meal-type breakdown & macronutrient donut chart
2. Recognize Food: Image upload (JPG/PNG), MobileNetV2 Top-3 classification, serving scaling & logging
3. Food Search: Search nutrition CSV, view per 100g and custom portions, log meals directly
4. Meal History: Chronological table of saved meals, date/type filtering, explicit deletion
5. About: Architecture, model pipeline, dataset roles, calculation formulas, and medical disclaimer
"""

from datetime import date, datetime, timedelta
import io
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image, UnidentifiedImageError
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    ALLOWED_IMAGE_EXTENSIONS,
    ASSETS_DIR,
    CALORIES_PER_MACRO,
    CLASS_NAMES_PATH,
    DEFAULT_DAILY_GOALS,
    LOGO_PATH,
    MAX_IMAGE_SIZE_MB,
    MEAL_TYPES,
    MODEL_PATH,
    NUTRITION_CSV_PATH,
    NUTRITION_DATA_SOURCE,
)
from database import (
    add_meal,
    delete_meal,
    get_daily_totals,
    get_meal_history,
    get_meals,
    init_db,
    DatabaseValidationError,
)
from model_inference import FoodClassifier, ModelInferenceError, get_food_classifier
from nutrition import (
    calculate_serving_nutrition,
    get_nutrition,
    load_nutrition_dataset,
    normalize_food_name,
    search_foods,
    FoodNotFoundError,
    InvalidServingSizeError,
    NutritionDatasetError,
)

# Page configuration
st.set_page_config(
    page_title="FoodNutritionAI - Food Recognition & Nutrition Tracker",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_styles() -> None:
    """Inject custom CSS for polished cards, typography, and badges."""
    st.markdown(
        """
        <style>
        /* Main card styling */
        .dash-card {
            background-color: #FFFFFF;
            border-radius: 10px;
            padding: 18px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06), 0 1px 2px rgba(0, 0, 0, 0.04);
            border: 1px solid #E2E8F0;
            margin-bottom: 14px;
        }
        .dash-metric-title {
            font-size: 0.85rem;
            color: #64748B;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
        }
        .dash-metric-val {
            font-size: 1.8rem;
            font-weight: 700;
            color: #0F172A;
            margin: 0;
            line-height: 1.2;
        }
        .dash-metric-sub {
            font-size: 0.8rem;
            color: #64748B;
            margin-top: 4px;
        }
        /* Confidence and Source badges */
        .source-tag {
            display: inline-block;
            background-color: #F1F5F9;
            color: #475569;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.78rem;
            border: 1px solid #CBD5E1;
            margin-bottom: 8px;
        }
        .conf-badge {
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.82rem;
            display: inline-block;
        }
        .conf-high { background-color: #DCFCE7; color: #166534; }
        .conf-med  { background-color: #FEF9C3; color: #854D0E; }
        .conf-low  { background-color: #FEE2E2; color: #991B1B; }
        .disclaimer-box {
            background-color: #FFFBEB;
            border-left: 4px solid #F59E0B;
            padding: 12px 16px;
            border-radius: 4px;
            font-size: 0.85rem;
            color: #92400E;
            margin: 12px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_cached_nutrition_data() -> pd.DataFrame:
    """Load and cache the official nutrition CSV."""
    return load_nutrition_dataset(NUTRITION_CSV_PATH)


@st.cache_resource(show_spinner=False)
def load_cached_classifier() -> FoodClassifier:
    """Load and cache the MobileNetV2 food classifier."""
    return get_food_classifier(MODEL_PATH, CLASS_NAMES_PATH)


def validate_uploaded_image(uploaded_file) -> Tuple[bool, Optional[str], Optional[Image.Image]]:
    """Validate uploaded image format, size, and integrity."""
    if uploaded_file is None:
        return False, "No file provided.", None

    file_bytes = uploaded_file.getvalue()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        return False, f"File size ({size_mb:.1f} MB) exceeds maximum limit of {MAX_IMAGE_SIZE_MB} MB.", None

    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        return False, f"Format '{suffix}' not supported. Allowed formats: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}", None

    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()
        # Re-open stream since verify() invalidates the image handle
        img = Image.open(io.BytesIO(file_bytes))
        return True, None, img
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return False, f"Corrupted or invalid image file: {exc}", None


def format_nutrient_val(val: Optional[float], unit: str = "g") -> str:
    """Format nutrient value, returning 'N/A' if missing."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    return f"{val:.1f} {unit}"


def render_sidebar() -> str:
    """Render the sidebar with navigation, logo, and daily targets."""
    with st.sidebar:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=75)
        st.title("FoodNutritionAI")
        st.caption("Deep Learning Food Recognition & Nutrition Tracker")

        st.markdown("---")
        page_choice = st.radio(
            "Go to Page",
            [
                "📊 Dashboard",
                "📷 Recognize Food",
                "🔍 Food Search",
                "📋 Meal History",
                "ℹ️ About"
            ],
            index=0
        )

        st.markdown("---")
        st.subheader("🎯 Daily Targets")
        with st.expander("Customize Goals", expanded=False):
            c_cal = st.number_input("Calories (kcal)", min_value=800, max_value=5000, value=2000, step=50)
            c_prot = st.number_input("Protein (g)", min_value=10, max_value=300, value=75, step=5)
            c_carb = st.number_input("Carbohydrates (g)", min_value=20, max_value=600, value=250, step=10)
            c_fat = st.number_input("Total Fat (g)", min_value=10, max_value=250, value=70, step=5)
            c_fiber = st.number_input("Dietary Fiber (g)", min_value=5, max_value=100, value=30, step=2)

            st.session_state["daily_goals"] = {
                "calories": float(c_cal),
                "protein": float(c_prot),
                "carbs": float(c_carb),
                "fat": float(c_fat),
                "fiber": float(c_fiber),
            }

        st.markdown("---")
        st.markdown(
            f"""
            <div style='font-size: 0.76rem; color: #64748B;'>
                <b>Dataset:</b> Food-101 (101 categories)<br>
                <b>Nutrient Baseline:</b> 100g USDA reference<br>
                <b>Database:</b> SQLite (<code>data/meals.db</code>)
            </div>
            """,
            unsafe_allow_html=True
        )

    return page_choice


# ==========================================
# PAGE 1: DASHBOARD
# ==========================================
def render_page_dashboard() -> None:
    """Page 1: Real-time nutritional analytics based on actual database records."""
    st.header("📊 Daily Nutrition Dashboard")
    st.write(
        "View daily caloric intake, macronutrient distributions, and meal-type breakdowns "
        "calculated from actual database records."
    )

    col_date, col_action = st.columns([2, 1])
    with col_date:
        selected_date = st.date_input("Select Date:", value=date.today(), key="dash_date")
    with col_action:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()

    date_str = selected_date.isoformat()
    daily_goals = st.session_state.get("daily_goals", DEFAULT_DAILY_GOALS)

    # Fetch daily totals and meal items
    totals = get_daily_totals(date_str)
    meals = get_meals(date_string=date_str)

    # Top Metric KPI Cards
    cal_pct = (totals["calories_kcal"] / daily_goals["calories"]) * 100 if daily_goals["calories"] > 0 else 0
    prot_pct = (totals["protein_g"] / daily_goals["protein"]) * 100 if daily_goals["protein"] > 0 else 0
    carb_pct = (totals["carbs_g"] / daily_goals["carbs"]) * 100 if daily_goals["carbs"] > 0 else 0
    fat_pct = (totals["fat_g"] / daily_goals["fat"]) * 100 if daily_goals["fat"] > 0 else 0
    fiber_pct = (totals["fiber_g"] / daily_goals["fiber"]) * 100 if daily_goals["fiber"] > 0 else 0

    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    with kpi1:
        st.metric("Calories", f"{totals['calories_kcal']:.0f} kcal", f"{cal_pct:.1f}% goal")
    with kpi2:
        st.metric("Protein", f"{totals['protein_g']:.1f} g", f"{prot_pct:.1f}% goal")
    with kpi3:
        st.metric("Carbs", f"{totals['carbs_g']:.1f} g", f"{carb_pct:.1f}% goal")
    with kpi4:
        st.metric("Fat", f"{totals['fat_g']:.1f} g", f"{fat_pct:.1f}% goal")
    with kpi5:
        st.metric("Fiber", f"{totals['fiber_g']:.1f} g", f"{fiber_pct:.1f}% goal")
    with kpi6:
        st.metric("Meals Logged", f"{totals['meal_count']}", f"{date_str}")

    st.markdown("---")

    # Real Plotly Charts based strictly on actual database records
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Macronutrient Calorie Breakdown")
        prot_cal = totals["protein_g"] * CALORIES_PER_MACRO["protein"]
        carb_cal = totals["carbs_g"] * CALORIES_PER_MACRO["carbs"]
        fat_cal = totals["fat_g"] * CALORIES_PER_MACRO["fat"]
        total_macro_cal = prot_cal + carb_cal + fat_cal

        if total_macro_cal > 0:
            labels = ["Protein", "Carbohydrates", "Fat"]
            values = [prot_cal, carb_cal, fat_cal]
            colors = ["#0D9488", "#F59E0B", "#EF4444"]

            fig_donut = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors),
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Calories: %{value:.1f} kcal<br>Share: %{percent}<extra></extra>"
            )])
            fig_donut.update_layout(
                margin=dict(l=15, r=15, t=25, b=15),
                height=320,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("No meals logged for this date yet. Macronutrient chart will appear once a meal is recorded.")

    with col_chart2:
        st.subheader("Calories by Meal Type")
        if meals:
            meal_breakdown: Dict[str, float] = {}
            for m in meals:
                m_type = m.get("meal_type", "lunch").title()
                meal_breakdown[m_type] = meal_breakdown.get(m_type, 0.0) + float(m["calories_kcal"])

            df_type = pd.DataFrame({
                "Meal Type": list(meal_breakdown.keys()),
                "Calories (kcal)": list(meal_breakdown.values())
            })
            fig_bar = px.bar(
                df_type,
                x="Meal Type",
                y="Calories (kcal)",
                color="Meal Type",
                color_discrete_sequence=["#14B8A6", "#3B82F6", "#8B5CF6", "#F97316"],
                text_auto=".1f"
            )
            fig_bar.update_layout(
                margin=dict(l=15, r=15, t=25, b=15),
                height=320,
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No meals recorded for this date yet. Meal-type breakdown will appear once meals are saved.")

    # 7-Day Trend
    st.markdown("---")
    st.subheader("📈 7-Day Caloric Trajectory")
    start_7d = (selected_date - timedelta(days=6)).isoformat()
    history = get_meal_history(start_7d, date_str)
    df_hist = pd.DataFrame(history)

    if not df_hist.empty:
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=df_hist["meal_date"],
            y=df_hist["calories_kcal"],
            mode="lines+markers",
            name="Intake (kcal)",
            line=dict(color="#0D9488", width=3),
            marker=dict(size=8, color="#0D9488")
        ))
        fig_trend.add_hline(
            y=daily_goals["calories"],
            line_dash="dash",
            line_color="#EF4444",
            annotation_text=f"Target: {daily_goals['calories']:.0f} kcal",
            annotation_position="top left"
        )
        fig_trend.update_layout(
            xaxis_title="Date",
            yaxis_title="Calories (kcal)",
            margin=dict(l=15, r=15, t=25, b=15),
            height=300
        )
        st.plotly_chart(fig_trend, use_container_width=True)


# ==========================================
# PAGE 2: RECOGNIZE FOOD
# ==========================================
def render_page_recognize_food() -> None:
    """Page 2: Image upload, MobileNetV2 classification, serving scaling, and meal logging."""
    st.header("📷 AI Food Recognition & Serving Logger")
    st.write(
        "Upload a food image to identify the dish using MobileNetV2 transfer learning, "
        "confirm the prediction, scale your portion size in grams, and record the meal to your database."
    )

    # Load nutrition dataset
    try:
        nutrition_df = load_cached_nutrition_data()
    except Exception as exc:
        st.error(f"⚠️ Nutrition dataset missing or invalid: {exc}")
        return

    # Check model classifier readiness
    classifier = load_cached_classifier()
    if not classifier.is_ready:
        st.warning(
            "⚠️ **Trained model artifact not found (`models/food_classifier.keras`).**\n\n"
            "To generate the transfer learning model, run this in your terminal:\n"
            "```powershell\npython train_model.py --smoke-test\n```\n"
            "*Note: Food Search, Daily Dashboard, and Meal History remain fully functional.*"
        )

    # Upload options
    col_upload, col_sample = st.columns([2, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Choose Food Photo (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Select an image file up to 10 MB."
        )

    pil_image = None
    if uploaded_file is not None:
        is_valid, err_msg, img = validate_uploaded_image(uploaded_file)
        if not is_valid:
            st.error(f"⚠️ Invalid image: {err_msg}")
            return
        pil_image = img

    if pil_image is None:
        st.info("👆 Upload a food image or choose a sample to initiate deep learning classification.")
        return

    # Image Preview & Inference
    col_preview, col_pred = st.columns([1, 1])
    with col_preview:
        st.image(pil_image, caption="Uploaded Dish Preview", use_container_width=True)

    predictions: List[Tuple[str, float]] = []
    with col_pred:
        st.subheader("🔍 MobileNetV2 Predictions")
        if not classifier.is_ready:
            st.info("Model not loaded. You can select food manually from the list below.")
        else:
            with st.spinner("Extracting features with MobileNetV2..."):
                try:
                    predictions = classifier.predict(pil_image, top_k=3)
                except ModelInferenceError as exc:
                    st.error(f"Prediction failed: {exc}")

            if predictions:
                st.write("**Top-3 Food-101 Predictions:**")
                for rank, (cat_label, score) in enumerate(predictions, start=1):
                    score_pct = score * 100.0
                    b_class = "conf-high" if score_pct >= 60 else ("conf-med" if score_pct >= 25 else "conf-low")
                    display_lbl = cat_label.replace("_", " ").title()
                    st.markdown(
                        f"**{rank}. {display_lbl}** — <span class='conf-badge {b_class}'>{score_pct:.1f}% confidence</span>",
                        unsafe_allow_html=True
                    )
                    st.progress(min(max(score, 0.0), 1.0))

                st.markdown(
                    """
                    <div class='disclaimer-box'>
                        <b>Note on Confidence Scores:</b> Confidence represents the model's Softmax output probability across Food-101 visual classes. It is an algorithmic estimation based on pixel likeness, not absolute culinary certainty.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown("---")

    # Selection & Confirmation
    st.subheader("✅ Confirm or Select Dish")
    all_names = sorted(nutrition_df["food_name"].dropna().unique().tolist())

    choice_labels = []
    label_to_food = {}
    for cat_label, score in predictions:
        lbl = f"{cat_label.replace('_', ' ').title()} ({score*100:.1f}%)"
        choice_labels.append(lbl)
        label_to_food[lbl] = cat_label.replace("_", " ")

    choice_labels.append("🔍 Search other food...")
    user_choice = st.radio("Selected food source:", choice_labels, horizontal=True)

    if user_choice.startswith("🔍"):
        selected_food_name = st.selectbox("Search & select food:", all_names)
    else:
        selected_food_name = label_to_food[user_choice]

    # Look up per-100g nutrition from CSV
    food_profile = get_nutrition(selected_food_name, raise_on_not_found=False)
    if not food_profile or not food_profile.get("found"):
        st.error(f"⚠️ Food '{selected_food_name}' was not found in data/nutrition_dataset.csv.")
        return

    st.markdown(f"#### Nutritional Profile for **{food_profile['food_name'].title()}** (Per 100g)")
    st.markdown(f"<span class='source-tag'>Source: {food_profile.get('source', 'CSV Dataset')}</span>", unsafe_allow_html=True)

    if not food_profile.get("is_complete", True):
        st.warning(
            f"Note: Incomplete nutritional data for {', '.join(food_profile.get('missing_nutrients', []))}. "
            "Values are preserved as unavailable and not assumed to be zero."
        )

    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("Calories", format_nutrient_val(food_profile.get("calories_kcal"), "kcal"))
    b2.metric("Protein", format_nutrient_val(food_profile.get("protein_g"), "g"))
    b3.metric("Carbs", format_nutrient_val(food_profile.get("carbs_g"), "g"))
    b4.metric("Fat", format_nutrient_val(food_profile.get("fat_g"), "g"))
    b5.metric("Fiber", format_nutrient_val(food_profile.get("fiber_g"), "g"))

    st.markdown("---")

    # Serving Size & Portion Scaling
    st.subheader("⚖️ Serving Size & Portion Calculation")
    c_serv, c_type, c_dt = st.columns([1, 1, 1])
    with c_serv:
        portion_g = st.number_input("Portion in Grams (g):", min_value=1.0, max_value=3000.0, value=150.0, step=10.0)
    with c_type:
        selected_meal_type = st.selectbox("Meal Type:", MEAL_TYPES, index=1)
    with c_dt:
        selected_log_date = st.date_input("Date Logged:", value=date.today())

    # Calculate serving nutrition
    try:
        serving_data = calculate_serving_nutrition(food_profile["food_name"], portion_g)
    except (InvalidServingSizeError, FoodNotFoundError) as exc:
        st.error(f"Serving calculation error: {exc}")
        return

    st.markdown(f"##### Scaled Breakdown for **{portion_g:.0f}g** Portion")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Calories", format_nutrient_val(serving_data.get("calories_kcal"), "kcal"))
    s2.metric("Protein", format_nutrient_val(serving_data.get("protein_g"), "g"))
    s3.metric("Carbs", format_nutrient_val(serving_data.get("carbs_g"), "g"))
    s4.metric("Fat", format_nutrient_val(serving_data.get("fat_g"), "g"))
    s5.metric("Fiber", format_nutrient_val(serving_data.get("fiber_g"), "g"))

    # Explicit Add Meal Action Form
    st.markdown("---")
    with st.form(key="confirm_meal_form", clear_on_submit=False):
        top_conf = predictions[0][1] if predictions else None
        st.write("Press **Add Meal** to persist this entry into your SQLite meal log:")
        submit_add = st.form_submit_button("➕ Add Meal", type="primary", use_container_width=True)

        if submit_add:
            # Prevent duplicate insertion on Streamlit reruns
            sig = f"{food_profile['food_name']}_{portion_g}_{selected_meal_type}_{selected_log_date}_{serving_data.get('calories_kcal')}"
            if st.session_state.get("last_submission_sig") == sig:
                st.warning("⚠️ This meal has already been logged! Modify portion or food to log another.")
            else:
                try:
                    add_meal(
                        food_name=food_profile["food_name"],
                        serving_grams=portion_g,
                        calories_kcal=serving_data.get("calories_kcal") or 0.0,
                        protein_g=serving_data.get("protein_g") or 0.0,
                        carbs_g=serving_data.get("carbs_g") or 0.0,
                        fat_g=serving_data.get("fat_g") or 0.0,
                        fiber_g=serving_data.get("fiber_g") or 0.0,
                        meal_type=selected_meal_type.lower(),
                        meal_date=selected_log_date.isoformat(),
                        prediction_confidence=top_conf,
                        nutrition_source=food_profile.get("source"),
                    )
                    st.session_state["last_submission_sig"] = sig
                    st.success(
                        f"🎉 Successfully recorded **{food_profile['food_name'].title()}** "
                        f"({portion_g:.0f}g, {serving_data.get('calories_kcal', 0):.0f} kcal) "
                        f"under **{selected_meal_type}** for {selected_log_date}!"
                    )
                    st.balloons()
                except Exception as exc:
                    st.error(f"Database error: {exc}")


# ==========================================
# PAGE 3: FOOD SEARCH
# ==========================================
def render_page_food_search() -> None:
    """Page 3: Search available foods in the CSV, inspect per 100g, and scale custom portions."""
    st.header("🔍 Nutrition Dataset Search")
    st.write(
        "Search all food items in `data/nutrition_dataset.csv`. "
        "View official reference values per 100g, calculate custom portions, and log meals directly."
    )

    try:
        df = load_cached_nutrition_data()
    except Exception as exc:
        st.error(f"Failed to load dataset: {exc}")
        return

    query = st.text_input("Search foods by name:", placeholder="e.g. pizza, salad, curry, chicken, apple...")

    if query and query.strip():
        matches = search_foods(query.strip(), limit=20)
    else:
        matches = [get_nutrition(name, raise_on_not_found=False) for name in sorted(df["food_name"].unique())[:20]]

    if not matches:
        st.info(f"No matching foods found for '{query}'. Try a different keyword.")
        return

    st.write(f"Displaying **{len(matches)}** food items:")

    # Detailed Food Card Selector
    food_names = [m["food_name"].title() for m in matches if m.get("found")]
    chosen_food_title = st.selectbox("Select food to inspect portion nutrition:", food_names)

    selected_item = get_nutrition(chosen_food_title.lower(), raise_on_not_found=False)
    if selected_item and selected_item.get("found"):
        st.markdown(f"### Nutritional Profile: **{selected_item['food_name'].title()}**")
        st.markdown(f"<span class='source-tag'>Source: {selected_item.get('source')}</span>", unsafe_allow_html=True)

        if not selected_item.get("is_complete", True):
            st.warning(f"Incomplete data: missing values for {', '.join(selected_item.get('missing_nutrients', []))}.")

        col_100g, col_calc = st.columns([1, 1])

        with col_100g:
            st.markdown("##### Reference per 100 Grams")
            st.write(f"• **Calories:** {format_nutrient_val(selected_item.get('calories_kcal'), 'kcal')}")
            st.write(f"• **Protein:** {format_nutrient_val(selected_item.get('protein_g'), 'g')}")
            st.write(f"• **Carbohydrates:** {format_nutrient_val(selected_item.get('carbs_g'), 'g')}")
            st.write(f"• **Total Fat:** {format_nutrient_val(selected_item.get('fat_g'), 'g')}")
            st.write(f"• **Dietary Fiber:** {format_nutrient_val(selected_item.get('fiber_g'), 'g')}")

        with col_calc:
            st.markdown("##### Portion Calculator & Quick Log")
            with st.form(key="search_log_form", clear_on_submit=False):
                calc_portion = st.number_input("Serving Size (g):", min_value=1.0, max_value=3000.0, value=150.0, step=10.0)
                calc_meal_type = st.selectbox("Meal Category:", MEAL_TYPES, index=1)
                calc_date = st.date_input("Log Date:", value=date.today())

                scaled = calculate_serving_nutrition(selected_item["food_name"], calc_portion)
                st.write(f"**Portion Energy:** {format_nutrient_val(scaled.get('calories_kcal'), 'kcal')}")
                st.write(f"**Portion Protein:** {format_nutrient_val(scaled.get('protein_g'), 'g')} | **Carbs:** {format_nutrient_val(scaled.get('carbs_g'), 'g')}")

                submit_search_log = st.form_submit_button("➕ Log This Food", type="primary", use_container_width=True)

            if submit_search_log:
                sig = f"{selected_item['food_name']}_{calc_portion}_{calc_meal_type}_{calc_date}"
                if st.session_state.get("last_submission_sig") == sig:
                    st.warning("This meal was already logged.")
                else:
                    try:
                        add_meal(
                            food_name=selected_item["food_name"],
                            serving_grams=calc_portion,
                            calories_kcal=scaled.get("calories_kcal") or 0.0,
                            protein_g=scaled.get("protein_g") or 0.0,
                            carbs_g=scaled.get("carbs_g") or 0.0,
                            fat_g=scaled.get("fat_g") or 0.0,
                            fiber_g=scaled.get("fiber_g") or 0.0,
                            meal_type=calc_meal_type.lower(),
                            meal_date=calc_date.isoformat(),
                            nutrition_source=selected_item.get("source"),
                        )
                        st.session_state["last_submission_sig"] = sig
                        st.success(f"Added {selected_item['food_name'].title()} ({calc_portion:.0f}g) to {calc_date} log!")
                    except Exception as exc:
                        st.error(f"Error saving meal: {exc}")


# ==========================================
# PAGE 4: MEAL HISTORY
# ==========================================
def render_page_meal_history() -> None:
    """Page 4: Saved meals, date filtering, and explicit deletion."""
    st.header("📋 Meal History & Log Records")
    st.write(
        "Inspect all saved meal entries from `data/meals.db`. "
        "Filter records by date range or meal category, and remove entries after explicit confirmation."
    )

    # Filter controls
    col_f1, col_f2, col_f3 = st.columns([1, 1, 1])
    with col_f1:
        start_d = st.date_input("From Date:", value=date.today() - timedelta(days=7), key="hist_start")
    with col_f2:
        end_d = st.date_input("To Date:", value=date.today(), key="hist_end")
    with col_f3:
        type_filter = st.selectbox("Filter Meal Type:", ["All Types"] + [t.title() for t in MEAL_TYPES])

    if start_d > end_d:
        st.error("⚠️ 'From Date' cannot be after 'To Date'.")
        return

    # Fetch meals in date range
    all_records = get_meals()
    filtered = [
        m for m in all_records
        if start_d.isoformat() <= m.get("meal_date", "") <= end_d.isoformat()
    ]

    if type_filter != "All Types":
        filtered = [m for m in filtered if m.get("meal_type", "").lower() == type_filter.lower()]

    if not filtered:
        st.info("No meals found matching the selected date range and filter criteria.")
        return

    st.write(f"Showing **{len(filtered)}** meal entries:")

    # Summary metrics for filtered range
    tot_cal = sum(float(m.get("calories_kcal", 0.0)) for m in filtered)
    tot_prot = sum(float(m.get("protein_g", 0.0)) for m in filtered)
    tot_carbs = sum(float(m.get("carbs_g", 0.0)) for m in filtered)
    tot_fat = sum(float(m.get("fat_g", 0.0)) for m in filtered)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Energy", f"{tot_cal:.0f} kcal")
    m2.metric("Total Protein", f"{tot_prot:.1f} g")
    m3.metric("Total Carbs", f"{tot_carbs:.1f} g")
    m4.metric("Total Fat", f"{tot_fat:.1f} g")

    st.markdown("---")

    # Meal entry cards with explicit deletion buttons
    for meal in filtered:
        with st.container():
            c_info, c_nutr, c_meta, c_del = st.columns([3, 2, 2, 1])
            with c_info:
                st.write(f"**{meal.get('food_name', '').title()}** ({meal.get('meal_type', '').title()})")
                st.caption(f"Portion: {meal.get('serving_grams', 0):.0f}g | Date: {meal.get('meal_date', '')}")
            with c_nutr:
                st.write(f"🔥 **{meal.get('calories_kcal', 0):.1f} kcal**")
                st.caption(f"Prot: {meal.get('protein_g', 0):.1f}g | Carbs: {meal.get('carbs_g', 0):.1f}g | Fat: {meal.get('fat_g', 0):.1f}g")
            with c_meta:
                st.write(f"Fiber: {meal.get('fiber_g', 0):.1f}g")
                if meal.get("prediction_confidence") is not None:
                    st.caption(f"Confidence: {float(meal['prediction_confidence'])*100:.1f}%")
            with c_del:
                if st.button("🗑️ Delete", key=f"del_meal_{meal['id']}", help="Delete this meal record permanently."):
                    delete_meal(meal["id"])
                    st.toast(f"Removed {meal.get('food_name', '').title()} from database.")
                    st.rerun()
            st.markdown("<hr style='margin: 6px 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)


# ==========================================
# PAGE 5: ABOUT
# ==========================================
def render_page_about() -> None:
    """Page 5: Architecture, model pipeline, dataset roles, calculation formulas, and disclaimer."""
    st.header("ℹ️ About FoodNutritionAI")

    st.markdown(
        """
        ### 🧠 Deep Learning Architecture & Image Recognition
        - **Convolutional Backbone:** MobileNetV2 pre-trained on ImageNet.
        - **Input Dimensions:** Images are resized to $(224 \times 224 \times 3)$ and normalized via `tf.keras.applications.mobilenet_v2.preprocess_input` into $[-1, 1]$.
        - **Transfer Learning Head:** GlobalAveragePooling2D $\to$ Dropout(0.2) $\to$ Dense(101, Softmax).
        - **Benchmark Dataset:** Trained on the canonical Food-101 dataset (101 food classes).
        - **Inference Principle:** Model outputs statistical Softmax probabilities across visual categories. The Top-3 predictions are presented to the user for confirmation.

        ### 🥗 Nutrition Calculation Engine
        - **Strict Source Sourcing:** Nutrition is never hallucinated or hardcoded in JSON. Data is read directly from `data/nutrition_dataset.csv`.
        - **Scaling Formula:**
        """
    )
    st.latex(r"\text{Nutrient}_{\text{serving}} = \text{Nutrient}_{\text{per 100g}} \times \frac{\text{Serving (grams)}}{100}")

    st.markdown(
        """
        - **Preservation of Missing Values:** Incomplete nutrient data in the CSV is preserved as `None` / `N/A` and clearly reported rather than silently zeroed.
        - **Conservative Matching:** Strictly exact normalized matching is required to retrieve nutrition data. Fuzzy matches are offered only as user suggestions.

        ### 🗄️ Database Architecture
        - **Relational Storage:** SQLite database at `data/meals.db`.
        - **Parameterized Queries:** All operations utilize parameterized statements (`?`) to prevent SQL injection.
        - **Rerun Protection:** Streamlit forms and unique session-state submission tokens prevent accidental entry duplication on page reruns.

        ### ⚠️ Medical & Dietary Disclaimer
        > **Important Disclaimer:** FoodNutritionAI is an engineering demonstration project designed for educational and informational dietary tracking. 
        > Model confidence scores reflect visual similarity distributions and do not verify recipe composition, allergens, or preparation methods. 
        > This tool is **not medical advice** and should not be used for medical diagnosis, treatment, or clinical nutrition planning. Consult a certified dietitian or physician for medical dietary guidance.
        """
    )


# ==========================================
# MAIN APPLICATION CONTROLLER
# ==========================================
def main() -> None:
    """Initialize database, inject styles, and route pages."""
    init_db()
    inject_custom_styles()
    chosen_page = render_sidebar()

    if chosen_page == "📊 Dashboard":
        render_page_dashboard()
    elif chosen_page == "📷 Recognize Food":
        render_page_recognize_food()
    elif chosen_page == "🔍 Food Search":
        render_page_food_search()
    elif chosen_page == "📋 Meal History":
        render_page_meal_history()
    elif chosen_page == "ℹ️ About":
        render_page_about()


if __name__ == "__main__":
    main()
