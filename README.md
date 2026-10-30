# 🍽️ FoodNutritionAI

**AI-powered food recognition and nutrition tracking using deep learning and Streamlit.**

FoodNutritionAI is a web application that identifies food from uploaded images and provides estimated nutritional information. It combines an EfficientNetB3 image-classification model trained on the Food-101 dataset with nutrition lookup, serving-size adjustment, and meal tracking.

## ✨ Features

- **AI Food Recognition:** Identify food images across 101 Food-101 categories.
- **Top-3 Predictions:** View the model's three most likely food classifications.
- **Nutrition Information:** View estimated calories, protein, carbohydrates, and fats.
- **Serving Size Adjustment:** Adjust the serving amount and recalculate nutrition values.
- **Meal Logging:** Save recognized meals and food-search entries.
- **Dashboard:** View daily nutrition summaries and visual charts.
- **Food Search:** Search the available nutrition database.
- **Meal History:** Review, filter, and delete logged meals.

## 🤖 Machine Learning Model

The current application uses **EfficientNetB3**, pretrained on ImageNet and fine-tuned for food classification.

| Property | Details |
|---|---|
| Architecture | EfficientNetB3 |
| Learning approach | Transfer learning and fine-tuning |
| Dataset | Food-101 |
| Number of classes | 101 |
| Input image size | 300 × 300 × 3 |
| Input format | RGB, float32, 0–255 |
| Framework | TensorFlow / Keras |
| Output | 101 food-class probabilities |

### Model Performance

The EfficientNetB3 model achieved the following results on the official Food-101 validation split:

| Metric | Result |
|---|---:|
| Top-1 validation accuracy | **80.91%** |
| Top-5 validation accuracy | **95.58%** |
| Validation loss | 0.6916 |

These are validation results, not results from an independent test set.

### Previous Model

The project also includes a MobileNetV2 baseline, which achieved 68.08% validation accuracy. It is retained as a backup model.

## 🖥️ Application Pages

1. **Dashboard** – Daily nutrition summaries, macro distribution, and meal trends.
2. **Recognize Food** – Upload an image, view predictions, adjust serving size, and log meals.
3. **Food Search** – Search foods and calculate nutrition for selected portions.
4. **Meal History** – Filter, review, and delete meal records.
5. **About** – Learn about the application, model, dataset, and nutrition calculations.

## 🛠️ Tech Stack

- Python
- Streamlit
- TensorFlow / Keras
- EfficientNetB3
- Food-101 dataset
- NumPy
- Pandas
- Pillow
- Plotly
- SQLite

## 📁 Project Structure

```text
FoodNutritionAI/
│
├── app.py
├── config.py
├── model_inference.py
├── nutrition.py
├── database.py
├── requirements.txt
│
├── models/
│   ├── food_classifier_efficientnetb3_candidate.keras
│   ├── class_names_efficientnetb3_candidate.json
│   ├── food_classifier.keras
│   └── class_names.json
│
├── data/
│   └── meals.db
│
├── tests/
│
└── README.md
```

*The exact contents may vary depending on the version of the repository. Do not commit local virtual environments, caches, or large training datasets.*

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/pradyumansinghrathore586-design/FoodNutritionAI.git
cd FoodNutritionAI
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure model files

Place the trained EfficientNetB3 model and its matching class-name JSON file in the `models/` directory.

The active model configuration in `config.py` should select:

```python
ACTIVE_MODEL = "efficientnetb3"
```

Make sure the configured model and class-name paths match the actual filenames.

### 5. Start the application

```bash