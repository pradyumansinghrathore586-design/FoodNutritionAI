# 🍎 FoodNutritionAI – AI-Powered Food Recognition & Nutrition Tracker

FoodNutritionAI is an AI-powered food recognition and nutrition tracking application built using **Python, TensorFlow, MobileNetV2, and Streamlit**.

The application allows users to upload food images, identify food categories using a deep learning model, view estimated nutritional information, calculate nutrition based on serving size, and maintain a meal history.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [System Architecture](#-system-architecture)
- [Machine Learning Model](#-machine-learning-model)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Running the Application](#-running-the-application)
- [Application Pages](#-application-pages)
- [Nutrition Calculation](#-nutrition-calculation)
- [Testing](#-testing)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Disclaimer](#-disclaimer)
- [Author](#-author)

---

## 📖 Project Overview

FoodNutritionAI combines computer vision and nutrition data to simplify food recognition and meal tracking.

Users can upload a food photograph, and the trained deep learning model predicts its food category. The application then retrieves corresponding nutritional information from a nutrition dataset.

The project demonstrates the practical application of:

- Deep Learning
- Image Classification
- Computer Vision
- Nutrition Data Processing
- Web Application Development
- Database Management

---

## ✨ Features

### 1. AI Food Recognition
- Upload food images in JPG, JPEG, or PNG format.
- Predict food categories using a trained MobileNetV2 model.
- Display the top three prediction results.
- View the uploaded food image and prediction results.

### 2. Nutrition Information
- Display estimated calories.
- Show protein, carbohydrates, and fat information.
- Calculate nutrition according to serving size.
- Scale nutrition values based on the selected portion.

### 3. Food Search
- Search for food items in the nutrition dataset.
- View nutritional information.
- Calculate nutrition for different serving quantities.

### 4. Meal Tracking
- Save recognized and searched food items.
- Record meals in the database.
- View previous meal entries.
- Filter meal history by date and meal type.
- Delete saved meal entries.

### 5. Interactive Dashboard
- View daily calorie information.
- Monitor macronutrient distribution.
- Explore meal-type summaries.
- View recent nutrition trends using interactive charts.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow / Keras | Deep learning and model inference |
| MobileNetV2 | Food image classification |
| Streamlit | Interactive web application |
| NumPy | Numerical operations |
| Pandas | Nutrition dataset processing |
| Pillow | Image processing |
| Plotly | Interactive visualizations |
| SQLite | Meal history storage |
| Scikit-learn | Machine learning utilities |
| Pytest | Application testing |

---

## 🏗️ System Architecture

The application follows a modular architecture.

```text
                 USER
                   |
                   v
           Streamlit Interface
                   |
          +--------+--------+
          |                 |
          v                 v
     Food Image         Food Search
       Upload           & Nutrition
          |                 |
          v                 |
    Image Validation        |
          |                 |
          v                 |
    Image Preprocessing     |
          |                 |
          v                 |
    MobileNetV2 Model       |
          |                 |
          v                 |
    Top-3 Predictions       |
          |                 |
          +--------+--------+
                   |
                   v
          Nutrition Dataset
                   |
                   v
          Portion Calculation
                   |
                   v
             Meal Logging
                   |
                   v
             SQLite Database
                   |
                   v
           Dashboard & History
```

---

## 🧠 Machine Learning Model

FoodNutritionAI uses **MobileNetV2**, a lightweight convolutional neural network architecture originally developed for efficient image classification.

### Model Configuration

| Parameter | Value |
|---|---|
| Architecture | MobileNetV2 |
| Pretrained Weights | ImageNet |
| Input Image Size | 224 × 224 |
| Input Channels | 3 (RGB) |
| Number of Classes | 101 |
| Output | Food category predictions |

### Model Performance

The original MobileNetV2 model achieved:

**68.08% validation accuracy**

Evaluation was performed on the official Food-101 validation split containing 25,250 images.

This result represents the project's baseline model performance. Actual predictions may vary depending on image quality, lighting, presentation, and food similarity.

### Prediction Process

1. The user uploads a food image.
2. The image is validated and converted to RGB.
3. The image is resized to 224 × 224 pixels.
4. The image is passed to the trained model.
5. The model generates class predictions.
6. The application displays the top three predictions.
7. The predicted food category is matched with the nutrition dataset.

---

## 📚 Dataset

### Food-101 Dataset

The image classification model is based on the **Food-101 dataset**.

- 101 food categories.
- 101,000 total images.
- 750 training images per category.
- 250 validation images per category.

Dataset information: [Food-101 – ETH Zurich](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)

### Nutrition Dataset

The application also uses a nutrition CSV dataset to retrieve nutritional information for recognized and searched food items.

The nutrition information includes:

- Calories
- Protein
- Carbohydrates
- Fat

The classification model identifies food categories, while the nutrition dataset supplies the corresponding nutrition information.

---

## 📁 Project Structure

```text
FoodNutritionAI/
│
├── app.py
├── config.py
├── database.py
├── model_inference.py
├── nutrition.py
├── train_model.py
├── train_model_efficientnetb3.py
│
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── assets/
│   ├── logo.png
│   └── sample_images/
│
├── data/
│   └── nutrition_dataset.csv
│
├── models/
│   ├── food_classifier.keras
│   └── class_names.json
│
└── tests/
    ├── test_app.py
    ├── test_database.py
    ├── test_model_inference.py
    ├── test_nutrition.py
    └── test_train_model.py
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.12 recommended
- Git
- Visual Studio Code (optional)

### Step 1: Clone the Repository

```bash
git clone https://github.com/pradyumansinghrathore586-design/FoodNutritionAI.git
```

### Step 2: Navigate to the Project

```bash
cd FoodNutritionAI
```

### Step 3: Create a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Verify Project Files

Make sure these essential files are present:

```text
models/food_classifier.keras
models/class_names.json
data/nutrition_dataset.csv
```

The application requires these files for food classification and nutrition lookup.

---

## 🚀 Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

If the browser does not open automatically, visit:

```text
http://localhost:8501
```

---

## 🖥️ Application Pages

### Dashboard
Provides an overview of calorie intake, macronutrients, meal distribution, and recent trends.

### Recognize Food
Allows users to upload food photographs and receive AI-based food predictions with estimated nutrition information.

### Food Search
Allows users to search the nutrition database and calculate nutritional values for selected portions.

### Meal History
Displays previously saved meals and provides date and meal-type filtering.

### About
Provides information about the application, model architecture, dataset, and nutrition calculations.

---

## 🧮 Nutrition Calculation

The application scales nutritional values according to the selected serving quantity.

The general calculation is:

\[
\text{Adjusted Nutrition}
=
\text{Nutrition per 100g}
\times
\frac{\text{Serving Size in grams}}{100}
\]

For example, if a food contains 200 calories per 100 grams, a 150-gram serving is estimated as:

\[
200 \times \frac{150}{100}
= 300 \text{ calories}
\]

This is an estimate based on the nutrition dataset, not a laboratory measurement of the uploaded food.

---

## 🧪 Testing

The project includes automated tests using Pytest.

Run the complete test suite:

```bash
pytest -v
```

Run an individual test file:

```bash
pytest tests/test_database.py -v
```

Available test modules cover:

- Application functionality
- Database operations
- Model inference
- Nutrition calculations
- Model training utilities

---

## ⚠️ Limitations

- Food recognition accuracy depends on image quality and food appearance.
- Similar-looking food categories can be misclassified.
- Nutrition values are estimates retrieved from a dataset.
- The model predicts food categories rather than exact ingredients.
- The application does not determine exact portion sizes from photographs.
- Meal records are not currently separated by user account.

---

## 🔮 Future Improvements

Possible future improvements include:

- Improving classification accuracy through additional training.
- Evaluating advanced architectures such as EfficientNet.
- Adding user authentication.
- Implementing individual user meal histories.
- Adding persistent cloud database storage.
- Supporting more food categories.
- Improving nutrition data coverage.
- Adding detailed nutrition reports.
- Developing a mobile-friendly experience.

---

## ⚖️ Disclaimer

FoodNutritionAI is an educational and informational project.

Food classifications and nutritional values are estimates and may not represent the exact contents of a meal.

This application is not intended to provide medical, dietary, or clinical advice.

---

## 👨‍💻 Author

**Pradyuman Singh Rathore**

GitHub: [@pradyumansinghrathore586-design](https://github.com/pradyumansinghrathore586-design)

Project Repository: [FoodNutritionAI](https://github.com/pradyumansinghrathore586-design/FoodNutritionAI)

---

⭐ If you find this project interesting, consider giving the repository a star!
