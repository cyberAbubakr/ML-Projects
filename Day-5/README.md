
# 🚕 NYC Taxi Trip Duration Prediction

## Machine Learning Regression Project — Day 5

---

## 📌 Project Overview

This project builds a complete end-to-end machine learning regression system to predict **NYC Yellow Taxi trip duration** using historical taxi trip records.

The project follows a professional machine learning workflow:

- Data ingestion
- Data validation
- Data cleaning
- Exploratory data analysis
- Feature engineering
- Feature preprocessing
- Model development
- Hyperparameter tuning
- Model evaluation
- Error analysis
- Model selection
- Model saving for deployment

The final goal is to build a regression model that accurately predicts taxi trip duration based on trip characteristics, time information, and location-based features.

---

# 🎯 Business Problem

Taxi companies need accurate trip duration estimates for:

- Better passenger experience
- Improved route planning
- Driver allocation
- ETA prediction
- Operational efficiency

This project solves this problem by learning patterns from millions of NYC taxi trips.

---

# 📊 Dataset Description

## NYC Yellow Taxi Trip Records

Dataset Source:

**NYC Taxi & Limousine Commission (TLC)**

Used Data:

| Month | Dataset |
|---|---|
| January 2023 | Yellow Taxi Trips |
| February 2023 | Yellow Taxi Trips |
| March 2023 | Yellow Taxi Trips |
| April 2023 | Yellow Taxi Trips |
| May 2023 | Yellow Taxi Trips |
| June 2023 | Yellow Taxi Trips |
| July 2023 | Yellow Taxi Trips |

The dataset contains millions of taxi trip records.

---

# 📥 Original Features

Important raw columns include:

- VendorID
- Passenger count
- Trip distance
- Rate code
- Store and forward flag
- Pickup location ID
- Dropoff location ID
- Pickup timestamp
- Dropoff timestamp
- Congestion surcharge
- Airport fee


---

# 🎯 Target Variable

The project predicts:

```

log_trip_duration

```

The logarithmic transformation reduces the impact of extreme trip duration outliers and improves regression performance.

---

# 🏗️ Project Architecture

```

Day-5
│
├── data
│   ├── raw
│   ├── interim
│   └── processed
│
├── models
│   ├── champ
│   ├── linear
│   └── tree
│
├── notebooks
│   │
│   ├── 01_data_audit.ipynb
│   ├── 02_data_prerocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_linear_models.ipynb
│   ├── 05_tree_based_models.ipynb
│   ├── 06_hyperparameter_tuning.ipynb
│   ├── 07_model_evaluation.ipynb
│   └── 08_error_analysis_and_explainability.ipynb
│
├── reports
│   ├── model_comparison_results.csv
│   └── tree_model_results.csv
│
├── src
│   │
│   ├── data
│   │   ├── load_data.py
│   │   ├── build_dataset.py
│   │   └── validate_data.py
│   │
│   ├── features
│   │   ├── build_features.py
│   │   └── preprocessor.py
│   │
│   └── models
│       ├── train.py
│       ├── evaluate.py
│       └── utils.py
│
├── requirements.txt
└── README.md

```

---

# ⚙️ Data Processing Pipeline

The data pipeline includes:

## Data Loading

Implemented in:

```

src/data/load_data.py

```

Responsibilities:

- Load monthly taxi files
- Load taxi zone lookup data
- Combine multiple months

---

## Data Validation

Implemented in:

```

src/data/validate_data.py

```

Checks:

- Missing values
- Invalid values
- Data types
- Dataset consistency


---

## Feature Building

Implemented in:

```

src/features/build_features.py

```

Creates machine-learning-ready features.

---

# 🧠 Feature Engineering

## Time Features

Created:

- pickup_year
- pickup_month
- pickup_day
- pickup_hour
- pickup_dayofweek
- pickup_week
- pickup_weekend
- pickup_quarter
- rush_hour


---

## Cyclic Time Encoding

To capture repeating patterns:

Created:

- hour_sin
- hour_cos
- day_sin
- day_cos


Examples:

- Morning vs evening traffic
- Weekday vs weekend behavior


---

## Distance Features

Created:

- distance_squared
- log_trip_distance
- distance_per_passenger


These capture nonlinear distance relationships.

---

## Location Features

Created:

- same_borough
- same_location
- is_airport_trip


These help models understand geographical relationships.

---

# 🔄 Preprocessing Pipeline

Implemented in:

```

src/features/preprocessor.py

```

The preprocessing pipeline handles:

---

## Numerical Features

Operations:

- Missing value handling
- Scaling


---

## Categorical Features

Operations:

- Missing value handling
- One-hot encoding


The trained preprocessing pipeline is saved for reuse.

---

# 🤖 Machine Learning Models

The project implements all required regression models.

---

# Baseline Model

## Dummy Median Regressor

Purpose:

Provides a simple benchmark to compare against machine learning models.

---

# Linear Regression Models

Implemented:

## 1. Multiple Linear Regression

Baseline linear model.

---

## 2. Ridge Regression

Adds L2 regularization.

---

## 3. Lasso Regression

Adds L1 regularization.

---

## 4. Elastic Net Regression

Combines L1 and L2 regularization.

---

# Tree-Based Regression Models

Implemented:

## 1. Decision Tree Regressor

Captures nonlinear relationships.

---

## 2. Random Forest Regressor

Uses multiple decision trees for improved generalization.

---

## 3. Gradient Boosting Regressor

Sequentially improves weak learners.

---

# 🔧 Hyperparameter Tuning

Performed for:

- Decision Tree
- Random Forest
- Gradient Boosting


Optimization objective:

```

Minimize RMSE

```

Tuned models:

```

tuned_decision_tree
tuned_random_forest
tuned_gradient_boosting

```

---

# 📈 Model Evaluation

Models were evaluated using:

---

## MAE

Mean Absolute Error

Measures average prediction error.

---

## RMSE

Root Mean Squared Error

Penalizes large prediction errors.

---

## Median Absolute Error

Shows typical model performance.

---

## R² Score

Measures explained variance.

---

# 🏆 Final Champion Model

After comparing all models:

## Selected Model

```

Tuned Random Forest Regressor

```

---

## Performance

| Metric | Score |
|---|---:|
| MAE | 0.195 |
| RMSE | 0.261 |
| Median AE | 0.155 |
| R² Score | 0.849 |


---

## Saved Model

Location:

```

models/champ/champion_random_forest.joblib

```

The model can be loaded for future predictions.

---

# 🔍 Error Analysis & Explainability

Notebook:

```

08_error_analysis_and_explainability.ipynb

````

Analysis performed:

- Prediction error distribution
- Residual analysis
- Worst prediction cases
- Feature importance
- Model behavior investigation


Purpose:

Understand:

- Where the model performs well
- Where predictions fail
- Which features influence predictions most

---

# 🚀 Installation Guide

## Clone Repository

```bash
git clone <repository-url>
````

Move into project:

```bash
cd Day-5
```

---

## Create Virtual Environment

```bash
python -m venv .venv
```

Activate:

Windows:

```bash
.venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running The Project

Run notebooks in order:

```
01_data_audit.ipynb

02_data_prerocessing.ipynb

03_feature_engineering.ipynb

04_linear_models.ipynb

05_tree_based_models.ipynb

06_hyperparameter_tuning.ipynb

07_model_evaluation.ipynb

08_error_analysis_and_explainability.ipynb
```

---

# 🛠️ Technologies Used

## Programming Language

* Python

## Machine Learning

* Scikit-learn

## Data Processing

* Pandas
* NumPy
* PyArrow

## Visualization

* Matplotlib
* Seaborn

## Model Management

* Joblib

## Development Tools

* Jupyter Notebook
* Git
* Virtual Environment

---

# 📚 Key Learnings

This project demonstrates:

✅ Real-world dataset handling
✅ Large-scale data processing
✅ Feature engineering techniques
✅ Regression model development
✅ Model comparison framework
✅ Hyperparameter optimization
✅ ML pipeline creation
✅ Model evaluation
✅ Explainability workflow
✅ Production-style project structure

---

# 👨‍💻 Author

**Abubakr**

Machine Learning Internship Project

## NYC Taxi Trip Duration Prediction

---


