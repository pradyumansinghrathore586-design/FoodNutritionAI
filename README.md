# AI-Based Food Recognition and Nutrition Tracker Using Deep Learning

A deep learning and data-driven nutrition tracking web application built with **Python 3.11**, **TensorFlow/Keras (MobileNetV2)**, **Food-101**, **Streamlit**, **SQLite**, and **Plotly**.

---

## 1. 📌 Project Overview and Features

This college engineering capstone project bridges computer vision and dietary tracking. It allows users to upload a photograph of a food plate, automatically predicts the dish category using a transfer-learning convolutional neural network (**MobileNetV2** pre-trained on ImageNet), looks up the official nutritional profile from a verified CSV dataset per 100 grams, scales the values according to custom portion sizes, and records daily meals into a local SQLite database for real-time macronutrient visualization.

### Key Features
- **Transfer Learning Image Recognition:** Classifies food dishes across 101 canonical categories using MobileNetV2 and Food-101.
- **Top-3 Confidence Scoring:** Ranks the highest-probability categories with visual confidence percentage bars.
- **Conservative Food Confirmation:** Protects against false classifications by letting users confirm or search for alternative dishes.
- **Portion Scaling Engine:** Dynamically calculates energy and macronutrients for custom gram portions based on a 100g reference baseline.
- **Data Integrity (No Silent Zeroing):** Missing or unavailable nutrient values in the dataset are preserved as `N/A` rather than silently converted to zero.
- **Parameterized SQLite Persistence:** Stores meal history, timestamps, and portions in `data/meals.db` using parameterized SQL queries.
- **Rerun Protection:** Streamlit form guards prevent duplicate entries during page refreshes.
- **Interactive Plotly Visualizations:** Real-time macronutrient caloric share (donut chart), meal-type distribution (bar chart), and 7-day intake trends.
- **Self-Contained & Tested:** Includes 41 automated unit tests across all application subsystems.

---

## 2. 🔄 Architecture and Workflow

The system is organized into modular subsystems with a clean separation of concerns:

```
[ User Upload (JPG/PNG) ]
           │
           ▼
[ PIL Validation & Decode ] ──> Resize (224x224x3) & Check Format/Size (<10MB)
           │
           ▼
[ MobileNetV2 Transfer Model ] ──> Preprocessing [-1, 1] ──> Softmax Probabilities
           │
           ▼
[ Top-3 Predictions Displayed ] ──> User Confirms or Corrects via Search
           │
           ▼
[ Nutrition Module (nutrition.py) ] ──> Conservative Lookup in data/nutrition_dataset.csv
           │
           ▼
[ Portion Scaling Formula ] ──> Nutrient = Nutrient_100g * Grams / 100
           │
           ▼
[ SQLite Manager (database.py) ] ──> Parameterized INSERT into data/meals.db
           │
           ▼
[ Multi-Page Dashboard (app.py) ] ──> Real-time Plotly Charts & Daily Totals
```

### Subsystem Separation
1. **Model Inference (`model_inference.py`):** Preprocesses images, manages Keras model caching, and produces Top-3 prediction tuples.
2. **Model Training (`train_model.py`):** Handles TFDS Food-101 dataset loading, non-overlapping splits, data augmentation, MobileNetV2 transfer learning, early stopping, and held-out evaluation.
3. **Nutrition Subsystem (`nutrition.py`):** Reads `data/nutrition_dataset.csv`, cleans whitespace and duplicates, enforces conservative matching, and scales portion nutrients.
4. **Database Manager (`database.py`):** Manages SQLite connections, transactions, parameterized queries, and daily aggregations.
5. **Presentation Layer (`app.py`):** Modern multi-page Streamlit dashboard with responsive layouts, metric cards, and Plotly charts.

---

## 3. 📂 Project Directory Tree

```text
FoodNutritionAI/
├── app.py                      # Multi-page Streamlit application (5 pages)
├── train_model.py              # MobileNetV2 transfer learning & training pipeline
├── model_inference.py          # Image preprocessing & Top-3 inference engine
├── database.py                 # SQLite database manager with parameterized queries
├── nutrition.py                # CSV nutrition dataset lookup & serving calculator
├── config.py                   # Centralized configuration, paths, and constants
├── requirements.txt            # Python 3.11 compatible dependencies
├── README.md                   # Full documentation & setup instructions
├── .gitignore                  # Git exclusion rules for venv, DB, and cache
├── data/
│   ├── nutrition_dataset.csv   # Real CSV dataset (116 items, 100g baseline)
│   ├── meals.db                # SQLite user meal records (generated at runtime)
│   └── tfds/
│       └── .gitkeep            # Directory preservation for TFDS downloads
├── models/
│   ├── food_classifier.keras   # Trained MobileNetV2 model artifact
│   └── class_names.json        # Ordered Food-101 category labels (101 items)
├── assets/
│   ├── logo.png                # Application branding graphic
│   └── sample_images/          # Built-in demo food images
│       ├── .gitkeep
│       ├── sample_apple_pie.jpg
│       ├── sample_pizza.jpg
│       └── sample_salad.jpg
├── tests/
│   ├── test_app.py             # Unit tests for image validation & page handlers
│   ├── test_database.py        # Unit tests for SQLite queries & validations
│   ├── test_model_inference.py # Unit tests for model inference & image shapes
│   ├── test_nutrition.py       # Unit tests for nutrition lookup & portion scaling
│   └── test_train_model.py     # Unit tests for training pipeline & architectures
└── .streamlit/
    └── config.toml             # Streamlit theme & performance settings
```

---

## 4. 🐍 Python 3.11 Installation Check

This project is tested and verified on **Python 3.11 (64-bit)** on Windows. 

Open **Windows Command Prompt** (`cmd.exe`) and verify your Python version:
```cmd
python --version
```
Or if multiple Python versions are installed using the Windows Python Launcher:
```cmd
py -3.11 --version
```
Output should confirm `Python 3.11.x`.

> **Important Note on Python Versions:** TensorFlow 2.x releases for Windows natively provide pre-compiled wheels for Python 3.9 through 3.12. Newer releases such as **Python 3.14 do not yet support TensorFlow prebuilt wheels on Windows**. Ensure you use Python 3.11.

---

## 5. 💻 Windows Command Prompt Setup Commands

Navigate to the project root directory in Windows Command Prompt:
```cmd
cd C:\Users\Pradyuman\Desktop\food_nutrition_tracker
```

Verify the project files exist in the current folder:
```cmd
dir
```

---

## 6. 🌐 Virtual Environment Creation and Activation

### Using Standard Python `venv` (Command Prompt)
Create a dedicated virtual environment named `venv`:
```cmd
python -m venv venv
```
Activate the virtual environment:
```cmd
venv\Scripts\activate.bat
```
*(Your prompt will change to `(venv) C:\...` upon activation).*

### Using Conda / Miniconda (Alternative)
If you prefer Conda:
```cmd
conda create -n food_ai python=3.11 -y
conda activate food_ai
```

---

## 7. 📦 Dependency Installation

Upgrade `pip` and install all tested, mutually compatible dependencies from `requirements.txt`:
```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify that all key libraries import cleanly:
```cmd
python -c "import tensorflow as tf, tensorflow_datasets as tfds, streamlit, pandas, plotly, PIL, pytest; print('Environment verified successfully! TensorFlow version:', tf.__version__)"
```

---

## 8. 🥗 Nutrition CSV Preparation and Data-Source Attribution

### Required CSV Schema
The application strictly reads from `data/nutrition_dataset.csv` with these exact columns:
```csv
food_name,calories_kcal,protein_g,carbs_g,fat_g,fiber_g,source
```

All nutrient amounts represent **100 grams** of the food item:
- `food_name`: Canonical food or dish name (e.g. `apple pie`, `pizza`, `caesar salad`).
- `calories_kcal`: Energy in kilocalories per 100g.
- `protein_g`: Protein in grams per 100g.
- `carbs_g`: Carbohydrates in grams per 100g.
- `fat_g`: Total fat in grams per 100g.
- `fiber_g`: Dietary fiber in grams per 100g.
- `source`: Attribution string recording data origin.

### Demonstration Baseline Dataset
To permit immediate testing and grading without forcing multi-gigabyte downloads, the project ships with a curated baseline covering all 101 Food-101 categories plus staple reference items (116 total entries).
- **Attribution:** Sourced from standard reference averages in USDA FoodData Central (SR Legacy).
- **Labeling:** Explicitly identified in the `source` column as `Demonstration Baseline (USDA FoodData Central SR Legacy reference)`.

### Importing Complete USDA FoodData Central Bulk Releases
For clinical or dietetic applications, download official bulk datasets directly from the [USDA FoodData Central Download Page](https://fdc.nal.usda.gov/download-datasets.html) (files: `food.csv`, `nutrient.csv`, `food_nutrient.csv`).
Convert using the following Python script:
```python
import pandas as pd

# Load USDA CSV files
foods = pd.read_csv("food.csv", usecols=["fdc_id", "description"])
nutrients = pd.read_csv("food_nutrient.csv", usecols=["fdc_id", "nutrient_id", "amount"])

# Filter USDA standard nutrient IDs: Energy=1008, Protein=1003, Fat=1004, Carbs=1005, Fiber=1079
id_map = {1008: "calories_kcal", 1003: "protein_g", 1004: "fat_g", 1005: "carbs_g", 1079: "fiber_g"}
nutrients = nutrients[nutrients["nutrient_id"].isin(id_map.keys())]
nutrients["name"] = nutrients["nutrient_id"].map(id_map)

# Pivot and merge (100g basis)
pivoted = nutrients.pivot_table(index="fdc_id", columns="name", values="amount", aggfunc="mean").reset_index()
merged = pd.merge(foods, pivoted, on="fdc_id")
merged["food_name"] = merged["description"].str.lower().str.strip()
merged["source"] = "USDA FoodData Central (FDC ID: " + merged["fdc_id"].astype(str) + ")"

out_df = merged[["food_name", "calories_kcal", "protein_g", "carbs_g", "fat_g", "fiber_g", "source"]]
out_df.to_csv("data/nutrition_dataset.csv", index=False)
```

---

## 9. 🧠 Food-101 Download and Model Training

`train_model.py` trains MobileNetV2 with transfer learning on the Food-101 benchmark dataset.

To execute full training via TensorFlow Datasets (TFDS):
```cmd
python train_model.py --force-retrain --epochs 10 --batch-size 32
```

To train on a fast 10% subset (recommended for college lab presentations):
```cmd
python train_model.py --force-retrain --subset-percent 10 --epochs 5 --batch-size 32
```

### Artifact Protection Logic
If model artifacts already exist at `models/food_classifier.keras` and `models/class_names.json`, running:
```cmd
python train_model.py
```
will detect existing files and skip retraining, displaying instructions on how to use `--force-retrain`. The Streamlit application never triggers retraining on launch.

---

## 10. ⚡ Reduced-Data Smoke Test

To verify the complete training pipeline end-to-end (batching, MobileNetV2 transfer learning, gradient steps, ModelCheckpoint, held-out evaluation, and classification report) in under 10 seconds without downloading 5 GB of images:
```cmd
python train_model.py --smoke-test
```
This runs a 1-epoch pass on a synthetic dataset with non-overlapping splits, confirming that all TensorFlow, Keras, and Scikit-learn components function on your system.

---

## 11. 🚀 Running the Streamlit Application

Launch the web application:
```cmd
streamlit run app.py
```
Streamlit will launch locally. Open your browser at:
```
http://localhost:8501
```

### Application Pages
1. **📊 Dashboard:** Real-time calories, protein, carbs, fat, and fiber totals for the selected date; Plotly macronutrient caloric breakdown; meal-type distribution; 7-day intake trajectory vs target line.
2. **📷 Recognize Food:** Image uploader supporting JPG, JPEG, and PNG (with built-in sample images); MobileNetV2 Top-3 classification with confidence percentages; portion size scaling calculator in grams; duplicate-protected **Add Meal** button.
3. **🔍 Food Search:** Interactive search of the CSV dataset; per-100g reference values; custom portion calculator; direct logging capability.
4. **📋 Meal History:** Chronological table of saved meals from `data/meals.db`; date range and meal-type filters; explicit meal deletion.
5. **ℹ️ About:** Pipeline specifications, serving formula, architecture diagram, and medical/dietary disclaimers.

---

## 12. 🧪 Running Automated Tests

Run the complete test suite across all 41 test cases:
```cmd
pytest -v tests/
```

To run individual test modules:
```cmd
pytest -v tests/test_app.py
pytest -v tests/test_database.py
pytest -v tests/test_nutrition.py
pytest -v tests/test_train_model.py
pytest -v tests/test_model_inference.py
```

---

## 13. 🛠️ Common Errors and Fixes

| Issue / Error Message | Cause | Resolution |
| :--- | :--- | :--- |
| `No matching distribution found for tensorflow` | Using an unsupported Python version (e.g. Python 3.14). | Install Python 3.11 and create your virtual environment with `py -3.11 -m venv venv`. |
| `Model artifact not found (models/food_classifier.keras)` | Model has not been trained or exported yet. | Run `python train_model.py --smoke-test` to generate starter model artifacts immediately. |
| `TensorFlow GPU support is not available on native Windows` | TensorFlow $\ge 2.11$ discontinued native Windows CUDA support. | This is a normal warning. TensorFlow operates on CPU on Windows; for GPU acceleration, run inside WSL2 (Ubuntu). |
| `Address already in use: 8501` | Another Streamlit instance or process is using port 8501. | Run on another port: `streamlit run app.py --server.port 8502`. |
| `Invalid meal_type: ...` | Meal type was not one of the allowed categories. | Allowed categories are strictly: `breakfast`, `lunch`, `dinner`, or `snack`. |
| `Serving size must be strictly positive` | Entered serving size was zero, negative, or non-numeric. | Enter a valid portion in grams greater than 0 (e.g., 150). |
| `WinError 206: The filename or extension is too long` | Windows 260-character MAX_PATH limit exceeded during TFDS download. | Enable Long Paths in Windows: run PowerShell as Admin: `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force`. |

---

## 14. ⏱️ Hardware, Disk Space, and Training-Time Limitations

- **Dataset Size:** The official Food-101 download via `tensorflow_datasets` is approximately **5.0 GB compressed** and uncompresses to ~10–12 GB during processing. Ensure at least 15 GB of free disk space in `data/tfds/`.
- **CPU Training Time:** On a modern multi-core CPU (e.g., Intel Core i5/i7, AMD Ryzen 5/7), training MobileNetV2 on all 75,750 Food-101 images takes **25 to 45 minutes per epoch**. 10 full epochs will take several hours.
- **Recommendations for Presentations / Grading:**
  - Use `--smoke-test` for immediate verification in under 10 seconds.
  - Use `--subset-percent 10` for a fast 10-minute training demonstration.
- **Memory (RAM):** Minimum 8 GB RAM required; 16 GB recommended.

---

## 15. ⚖️ Model Limitations, Nutrition Limitations, and Ethical Considerations

### Model Visual Limitations
- **Hidden Ingredients:** Convolutional neural networks evaluate external pixel features. The model cannot detect hidden sugars, cooking oils, sodium, butter, dairy, or gluten.
- **Visual Ambiguity:** Visually similar dishes (e.g., chocolate cake vs chocolate mousse, or regular soup vs low-sodium soup) may share high visual probability despite having distinct nutritional profiles.
- **Confidence Scores:** The model's Softmax score indicates statistical visual likeness across the Food-101 benchmark dataset, not biochemical or culinary certainty. Always verify the predicted dish before logging.

### Nutrition Data Limitations
- **Reference Averages:** Values in `data/nutrition_dataset.csv` represent USDA standard baseline averages per 100 grams. Home-cooked meals and restaurant preparations will vary significantly based on cooking methods and portion sizes.
- **Missing Nutrients:** If a food entry lacks a specific nutrient in the CSV, it is labeled `N/A`. The system deliberately avoids assuming zero to prevent misleading dietary totals.

### Medical & Ethical Disclaimer
> **Medical Disclaimer:** FoodNutritionAI is an academic demonstration and dietary awareness software tool. It is **not a medical device** and does not provide clinical nutrition advice, metabolic diagnosis, or food allergen detection. Users with medical conditions (e.g., diabetes, hypertension, celiac disease, severe food allergies) must consult a certified dietitian or physician for individualized nutrition guidance.
