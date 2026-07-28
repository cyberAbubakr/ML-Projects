# 🧠 Adult Income Prediction using Machine Learning

## Project Overview

This project builds an end-to-end Machine Learning pipeline to predict whether an individual's income exceeds **$50K/year** using demographic, education, employment, and financial information.

The project demonstrates a complete ML engineering workflow:

* Data loading
* Data preprocessing
* Missing value handling
* Duplicate removal
* Feature engineering
* Feature transformation
* Machine Learning pipeline creation
* Model training
* Model evaluation
* Model serialization

---

# 🎯 Problem Statement

Given information about an individual's:

* Age
* Education
* Occupation
* Work hours
* Capital gains/losses
* Relationship status
* Other demographic attributes

the goal is to predict:

```
Income > 50K ?
```

This is a **Binary Classification Problem**.

Target Variable:

```
income
```

Classes:

```
<=50K
>50K
```

---

# 📂 Dataset

Dataset:

**Adult Census Income Dataset**

Source:

UCI Machine Learning Repository

Dataset contains:

* 32,537 samples
* 15 original features
* Additional engineered features

---

# 📁 Project Structure

```
Day-4/
│
├── datasets/
│   └── adult/
│       ├── adult.data
│       ├── adult_processed.csv
│       └── adult_feature_engineered.csv
│
├── notebooks/
│   │
│   ├── 04_data_preprocessing.ipynb
│   ├── 05_feature_engineering.ipynb
│   └── 06_ml_pipeline_demo.ipynb
│
├── src/
│   │
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── pipeline.py
│   └── utils.py
│
├── models/
│   └── adult_income_pipeline.pkl
│
├── reports/
│   │
│   ├── feature_engineering_summary.csv
│   ├── model_evaluation.csv
│   ├── final_model_report.csv
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── requirements.txt
└── README.md
```

---

# 🔄 Machine Learning Workflow

```
Raw Dataset
     |
     ↓
Data Loading
     |
     ↓
Data Cleaning
     |
     ↓
Missing Value Handling
     |
     ↓
Duplicate Removal
     |
     ↓
Feature Engineering
     |
     ↓
Feature Transformation
     |
     ↓
ML Pipeline
     |
     ↓
Model Training
     |
     ↓
Evaluation
     |
     ↓
Saved Model
```

---

# 🧹 Data Preprocessing

Performed operations:

## Missing Values

Handled missing values represented as:

```
?
```

Techniques:

* Categorical → Most frequent value
* Numerical → Median value

---

## Duplicate Removal

Duplicate records were identified and removed.

---

## Feature Transformation

### Numerical Features

Applied:

```
StandardScaler
```

Features:

* Age
* Final Weight
* Education Number
* Capital Gain
* Capital Loss
* Work Hours

---

### Categorical Features

Applied:

```
OneHotEncoder
```

Features:

* Workclass
* Education
* Occupation
* Relationship
* Gender
* Country

---

# ⚙️ Feature Engineering

Created additional meaningful features.

## 1. Age Group

Converted numerical age into categories:

```
Young Adult
Adult
Middle Aged
Senior
```

---

## 2. Capital Net

Created:

```
CapitalNet = Capital Gain - Capital Loss
```

---

## 3. Work Hours Category

Grouped working hours:

```
Part-Time
Full-Time
Overtime
```

---

## 4. Education Level

Mapped education categories:

```
School
High School
College
Bachelor
Postgraduate
```

---

## 5. Income Per Hour

Created:

```
IncomePerHour =
Capital Gain / Hours Per Week
```

---

# 🤖 Machine Learning Pipeline

Implemented using:

```
Scikit-Learn Pipeline
```

Pipeline contains:

```
ColumnTransformer
        |
        |
 ┌───────────────┐
 | Numerical     |
 | Imputer       |
 | StandardScaler|
 └───────────────┘

 ┌───────────────┐
 | Categorical   |
 | Imputer       |
 | OneHotEncoder |
 └───────────────┘

        |
        ↓

Logistic Regression
```

---

# 🧪 Model

## Logistic Regression

Selected because:

* Strong baseline classifier
* Efficient for binary classification
* Provides probability predictions
* Works well with encoded features

---

# 📊 Model Evaluation

Metrics used:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

---

# 🏆 Results

## ROC-AUC Score

```
0.915
```

The model achieved excellent classification ability.

Interpretation:

The model can effectively distinguish between:

```
<=50K income
```

and

```
>50K income
```

---

# 📈 Evaluation Visualizations

Generated:

## ROC Curve

Shows model performance across different classification thresholds.

Saved:

```
reports/roc_curve.png
```

---

## Confusion Matrix

Shows:

* Correct predictions
* False positives
* False negatives

Saved:

```
reports/confusion_matrix.png
```

---

# 💾 Model Deployment Ready

The complete pipeline was saved:

```
models/adult_income_pipeline.pkl
```

The saved model contains:

* Preprocessing steps
* Feature transformations
* Trained classifier

It can be loaded directly for future predictions.

---

# 🛠️ Technologies Used

Python

Libraries:

* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Joblib

Development:

* VS Code
* Jupyter Notebook
* Git/GitHub

---

# 🚀 Future Improvements

Possible improvements:

* Test advanced models:

  * Random Forest
  * XGBoost
  * LightGBM

* Hyperparameter tuning

* Model explainability:

  * SHAP
  * Feature importance analysis

* Deployment using:

  * FastAPI
  * Streamlit
  * Docker

---

# 👨‍💻 Author

Syed Abubakr Kazmi

Machine Learning Engineering Practice Project

Built as part of an ML Engineering learning roadmap.
