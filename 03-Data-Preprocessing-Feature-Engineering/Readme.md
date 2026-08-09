# 🧹 Day 3 — Data Preprocessing & Feature Engineering

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikitlearn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)
![Git](https://img.shields.io/badge/Git-Version%20Control-F05032?logo=git)

A complete **Machine Learning Data Preprocessing & Feature Engineering** project built using industry-standard preprocessing techniques and reusable Scikit-Learn pipelines.

This project demonstrates how raw datasets are transformed into clean, machine-learning-ready data while following ML Engineering best practices such as modular code organization, reusable preprocessing pipelines, feature engineering, and data leakage prevention.

---

# 📖 Project Overview

Real-world datasets are rarely clean enough to train machine learning models directly. Missing values, inconsistent categories, incorrect data types, noisy data, and outliers can significantly impact model performance.

This project focuses on solving these challenges using two widely used datasets:

* 🏠 **House Prices (Regression)**
* 📱 **IBM Telco Customer Churn (Classification)**

Instead of placing all preprocessing logic inside Jupyter notebooks, reusable functions are organized into the **src/** directory to follow an engineering-oriented workflow.

---

# 📂 Datasets

| Dataset                                      | Problem Type   | Target Variable |
| -------------------------------------------- | -------------- | --------------- |
| House Prices: Advanced Regression Techniques | Regression     | SalePrice       |
| IBM Telco Customer Churn                     | Classification | Churn           |

## 🏠 House Prices

**Dataset**

House Prices: Advanced Regression Techniques (Ames Housing Dataset)

**Tasks Performed**

* Missing value analysis
* Missing value treatment
* Outlier detection
* Feature engineering
* Numerical scaling
* Pipeline creation

**Kaggle**

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

---

## 📱 IBM Telco Customer Churn

**Dataset**

IBM Telco Customer Churn

**Tasks Performed**

* Data type correction
* Category cleaning
* Feature engineering
* Categorical encoding
* Numerical scaling
* Pipeline creation
* Stratified train-test splitting

**Kaggle**

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

# 🎯 Project Objectives

This project demonstrates the complete preprocessing lifecycle including:

* Data exploration
* Data quality assessment
* Missing value handling
* Duplicate detection
* Invalid value detection
* Data type correction
* Category standardization
* Outlier detection
* Outlier treatment
* Feature engineering
* Categorical encoding
* Numerical scaling
* Train-test splitting
* Data leakage prevention
* Scikit-Learn preprocessing pipelines

---

# 🏗️ Repository Structure

```text
Day-3/
│
├── README.md
├── Requirement.txt
│
├── datasets/
│   ├── house-prices/
│   └── telco_customer_churn/
│
├── notebooks/
│   ├── 01_house_preprocessing.ipynb
│   └── 02_telco_preprocessing.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── pipeline.py
│
├── Reports/
│   ├── house_preprocessing_report.md
│   ├── house_missing_summary.csv
│   └── telco_preprocessing_report.md
│
└── images/
```

---

# 🏠 House Prices Preprocessing Workflow

## ✅ Data Exploration

* Loaded dataset
* Shape inspection
* Data type inspection
* Statistical summaries
* Missing value visualization

---

## ✅ Data Quality Assessment

Performed:

* Missing value analysis
* Missing value percentage calculation
* Duplicate detection
* Invalid value detection
* Category cleaning
* Data type verification

---

## ✅ Missing Value Treatment

| Feature           | Strategy          |
| ----------------- | ----------------- |
| PoolQC            | Dropped           |
| MiscFeature       | Dropped           |
| Alley             | Dropped           |
| Fence             | Dropped           |
| LotFrontage       | Median Imputation |
| Electrical        | Mode Imputation   |
| Garage Features   | NoGarage          |
| Basement Features | NoBasement        |
| FireplaceQu       | NoFireplace       |
| MasVnrType        | None              |
| MasVnrArea        | 0                 |
| GarageYrBlt       | 0                 |

---

## ✅ Imputation Comparison

Implemented and compared:

* Mean Imputation
* Median Imputation
* Mode Imputation
* KNN Imputation
* Iterative Imputation

---

## ✅ Outlier Detection

Techniques used:

* IQR Method
* Z-Score Method

Visualization:

* Boxplots

---

## ✅ Outlier Treatment

Compared:

* Outlier Removal
* Winsorization (Capping)

Applied IQR-based capping to the `SalePrice` feature.

---

## ✅ Feature Engineering

Created new features:

* HouseAge
* TotalBathrooms
* TotalArea
* TotalPorchArea

---

## ✅ Category Cleaning

Applied:

* Whitespace removal
* Lowercase conversion
* Category standardization

---

# 📱 Telco Customer Churn Preprocessing Workflow

## ✅ Data Exploration

* Dataset loading
* Shape inspection
* Statistical summaries
* Data type inspection

---

## ✅ Data Cleaning

Performed:

* Converted `TotalCharges` from string to numeric
* Verified missing values
* Duplicate detection
* Invalid value checks
* Category cleaning

---

## ✅ Feature Engineering

Created:

* TenureGroup
* AverageMonthlySpend
* IsSenior
* HasDependents

---

## ✅ Train-Test Split

Applied:

* Stratified train-test split

to preserve class distribution.

---

## ✅ Preprocessing Pipeline

Built reusable preprocessing pipelines using:

* SimpleImputer
* StandardScaler
* OneHotEncoder
* Pipeline
* ColumnTransformer

---

# ⚙️ Scikit-Learn Pipeline

The preprocessing workflow follows ML Engineering best practices.

```text
Raw Dataset
      │
      ▼
Train-Test Split
      │
      ▼
Fit Pipeline on Training Data
      │
      ▼
Transform Training Data
      │
      ▼
Transform Testing Data
      │
      ▼
Machine Learning Ready Dataset
```

---

# 🔒 Data Leakage Prevention

To avoid train-test contamination:

* Dataset was split before preprocessing.
* The preprocessing pipeline was fitted only on the training data.
* The fitted pipeline was reused to transform the testing data.
* No information from the testing set was used during preprocessing.

---

# 📦 Source Modules

## `preprocessing.py`

Reusable preprocessing functions including:

* Missing value analysis
* Missing value handling
* Duplicate detection
* Invalid value detection
* Category cleaning
* Data type conversion
* Outlier detection
* Outlier capping
* Feature engineering

---

## `pipeline.py`

Reusable Scikit-Learn preprocessing pipelines:

* Numerical Pipeline
* Categorical Pipeline
* StandardScaler
* OneHotEncoder
* ColumnTransformer
* Complete preprocessing pipeline

---

# 🛠️ Technologies Used

| Category         | Tools                          |
| ---------------- | ------------------------------ |
| Programming      | Python                         |
| Data Analysis    | Pandas, NumPy                  |
| Visualization    | Matplotlib, Seaborn, Missingno |
| Machine Learning | Scikit-Learn                   |
| Development      | Jupyter Notebook, VS Code      |
| Version Control  | Git & GitHub                   |

---

# 📊 Project Outcomes

✔ Cleaned two real-world datasets

✔ Implemented multiple missing value strategies

✔ Compared several imputation techniques

✔ Engineered meaningful features

✔ Detected and treated outliers

✔ Encoded categorical variables

✔ Scaled numerical features

✔ Built reusable preprocessing modules

✔ Created reusable Scikit-Learn pipelines

✔ Prevented data leakage using the correct preprocessing workflow

✔ Produced machine-learning-ready datasets

---

# 🚀 Future Improvements

* Train regression models for House Prices
* Train classification models for Telco Customer Churn
* Hyperparameter tuning
* Cross-validation
* Feature selection
* Pipeline serialization using Joblib
* Model deployment

---

# 📚 References

## House Prices

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

## IBM Telco Customer Churn

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

# 👨‍💻 Author

**Abubakr Kazmi**

BS Computer Science

Machine Learning Engineering Journey

GitHub: https://github.com/cyberAbubakr
