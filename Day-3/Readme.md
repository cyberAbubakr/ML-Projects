# 🧹 Day 3 — Data Preprocessing & Feature Engineering

A complete Machine Learning preprocessing project demonstrating industry-standard data cleaning, feature engineering, preprocessing pipelines, and data leakage prevention using Scikit-Learn.

This project is part of my Machine Learning Engineering learning journey and focuses on building reusable preprocessing modules instead of writing all logic inside Jupyter notebooks.

---

# 📌 Project Overview

The objective of this project is to prepare real-world datasets for machine learning by performing comprehensive preprocessing and feature engineering.

The implementation follows ML engineering best practices by separating reusable code into the `src/` directory while using notebooks for experimentation and analysis.

---

# 📂 Datasets

## 1. House Prices (Regression)

**Dataset**
House Prices: Advanced Regression Techniques (Ames Housing Dataset)

**Target Variable**

```
SalePrice
```

**Kaggle**

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

---

## 2. Titanic (Classification)

**Dataset**

Titanic - Machine Learning from Disaster

**Target Variable**

```
Survived
```

**Kaggle**

https://www.kaggle.com/competitions/titanic

---

# 📁 Project Structure

```
Day-3-Data-Preprocessing/

│
├── README.md
├── requirements.txt
├── .gitignore
│
├── datasets/
│   ├── house_prices/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── sample_submission.csv
│   │   └── data_description.txt
│   │
│   └── titanic/
│       ├── train.csv
│       ├── test.csv
│       └── gender_submission.csv
│
├── notebooks/
│   ├── 01_house_preprocessing.ipynb
│   └── 02_titanic_preprocessing.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── pipeline.py
│   └── utils.py
│
├── reports/
│   ├── house_preprocessing_report.md
│   └── titanic_preprocessing_report.md
│
└── images/
```

---

# 🎯 Learning Objectives

- Handle missing values
- Clean messy datasets
- Detect and treat outliers
- Engineer meaningful features
- Encode categorical variables
- Scale numerical variables
- Prevent data leakage
- Build reusable preprocessing pipelines
- Follow ML engineering project architecture

---

# 🏠 House Prices Preprocessing

## Completed Tasks

### Data Exploration

- Dataset loading
- Shape inspection
- Data types inspection
- Statistical summaries
- Missing value visualization

---

### Data Quality Assessment

- Missing value analysis
- Missing value percentage calculation
- Duplicate detection
- Invalid value detection
- Category cleaning
- Data type verification

---

### Missing Value Treatment

Applied different strategies according to feature meaning.

| Feature | Strategy |
|----------|----------|
| PoolQC | Drop |
| MiscFeature | Drop |
| Alley | Drop |
| Fence | Drop |
| LotFrontage | Median |
| Electrical | Mode |
| Garage Features | NoGarage |
| Basement Features | NoBasement |
| FireplaceQu | NoFireplace |
| MasVnrType | None |
| MasVnrArea | 0 |
| GarageYrBlt | 0 |

---

### Imputation Comparison

Implemented comparison between

- Mean
- Median
- Mode
- KNN Imputation
- Iterative Imputation

---

### Outlier Detection

Implemented

- IQR Method
- Z-Score Method

Visualized using

- Boxplots

---

### Outlier Treatment

Compared

- Removal
- Winsorization (Capping)

Applied IQR-based capping on `SalePrice`.

---

### Feature Engineering

Created new features including

- HouseAge
- TotalBathrooms
- TotalArea
- TotalPorchArea

---

### Category Cleaning

- Removed leading/trailing spaces
- Standardized text to lowercase

---

### Scikit-Learn Pipeline

Created reusable preprocessing pipelines using

- Pipeline
- ColumnTransformer
- SimpleImputer
- StandardScaler
- OneHotEncoder

Pipeline outputs transformed feature matrices ready for machine learning.

---

### Data Leakage Prevention

Implemented proper preprocessing workflow:

```
Split Dataset
        │
        ▼
Fit Pipeline on Training Data
        │
        ▼
Transform Training Data
        │
        ▼
Transform Testing Data
```

No preprocessing step is fitted on the complete dataset.

---

# ⚙️ Source Modules

## preprocessing.py

Contains reusable preprocessing functions.

Features include

- Missing value analysis
- Missing value handling
- Duplicate detection
- Invalid value detection
- Category cleaning
- Outlier detection
- Outlier capping
- Feature engineering

---

## pipeline.py

Contains reusable Scikit-Learn preprocessing pipelines.

Includes

- Numeric pipeline
- Categorical pipeline
- ColumnTransformer
- Complete preprocessing pipeline

---

## utils.py

Utility functions used across notebooks.

---

# 📊 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Missingno
- Scikit-Learn

---

# 📈 Current Progress

| Phase | Status |
|---------|--------|
| Data Loading | ✅ |
| Data Exploration | ✅ |
| Data Quality Assessment | ✅ |
| Missing Value Treatment | ✅ |
| Outlier Detection | ✅ |
| Feature Engineering | ✅ |
| Categorical Encoding | ✅ |
| Numerical Scaling | ✅ |
| Data Splitting | ✅ |
| Data Leakage Prevention | ✅ |
| Scikit-Learn Pipeline | ✅ |
| Titanic Preprocessing | ⏳ |

---

# 🚀 Future Work

- Complete Titanic preprocessing notebook
- Create preprocessing reports
- Train baseline ML models
- Export preprocessing pipelines
- Compare preprocessing techniques across datasets

---

# 📚 References

House Prices Competition

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

Titanic Competition

https://www.kaggle.com/competitions/titanic

---

## 👨‍💻 Author

**Abubakr Kazmi**

BS Computer Science

Machine Learning Engineering Journey