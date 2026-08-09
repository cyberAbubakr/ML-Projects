# Telco Customer Churn Data Preprocessing Report

## Project Information

**Project:** Day 3 – Data Preprocessing & Feature Engineering

**Dataset:** IBM Telco Customer Churn

**Problem Type:** Binary Classification

**Target Variable:** `Churn`

---

# 1. Introduction

This report documents the complete preprocessing workflow performed on the IBM Telco Customer Churn dataset. The objective was to clean, transform, and prepare the data for machine learning while following ML engineering best practices.

The preprocessing workflow was implemented using reusable modules located in the `src/` directory together with Scikit-Learn preprocessing pipelines.

---

# 2. Dataset Overview

- Total Records: **7,043**
- Original Features: **21**
- Problem Type: Binary Classification
- Target Variable: **Churn**

The dataset contains customer demographics, subscribed services, billing information, and customer churn status.

---

# 3. Data Quality Assessment

## Missing Values

A complete missing value assessment was performed before preprocessing.

### Observations

- No missing (`NaN`) values were detected in the dataset.
- All **21 features** contained complete information across all customer records.
- The dataset was considered suitable for further preprocessing without applying missing value imputation.

---

# 4. Duplicate Records

The dataset was examined for duplicate customer records.

### Observation

No duplicate records were found.

Each customer record is unique.

---

# 5. Invalid Values

Numerical columns were inspected for invalid values such as negative charges or negative tenure.

### Observation

No invalid numerical values were identified.

---

# 6. Data Type Verification

The data types of all features were verified before preprocessing.

### Conversion Performed

The `TotalCharges` column was converted from a text-based representation to a numerical (`float64`) data type to enable mathematical operations and compatibility with machine learning algorithms.

---

# 7. Category Cleaning

Categorical variables were standardized by:

- Removing leading and trailing whitespace
- Converting text to lowercase
- Standardizing category values

This ensured consistent categorical representations throughout the dataset.

---

# 8. Feature Engineering

Several additional features were created to improve model performance and provide more meaningful customer information.

### Engineered Features

- **TenureGroup** – Customer tenure grouped into time intervals.
- **AverageMonthlySpend** – TotalCharges divided by customer tenure.
- **IsSenior** – Simplified senior citizen indicator.
- **HasDependents** – Indicates whether the customer has a partner or dependents.

These engineered variables provide additional behavioral information that may improve churn prediction.

---

# 9. Train-Test Split

The dataset was divided into:

- Training Set
- Testing Set

using **Stratified Train-Test Splitting** to preserve the distribution of the target variable (`Churn`) across both datasets.

---

# 10. Scikit-Learn Pipeline

A reusable preprocessing pipeline was developed using:

- Pipeline
- ColumnTransformer
- SimpleImputer
- StandardScaler
- OneHotEncoder

The preprocessing pipeline automatically:

- Processes numerical features
- Encodes categorical variables
- Scales numerical variables
- Produces machine learning-ready feature matrices

---

# 11. Data Leakage Prevention

To ensure unbiased model evaluation:

- The dataset was split before preprocessing.
- The preprocessing pipeline was fitted only on the training dataset.
- The testing dataset was transformed using the fitted pipeline.

This prevents information from the testing dataset from influencing preprocessing or model training.

---

# 12. Processed Dataset Summary

## Original Dataset

- Training Features: **23**
- Testing Features: **23**

## Processed Dataset

- Training Shape: **(5634, 53)**
- Testing Shape: **(1409, 53)**

The increase in the number of features is expected because categorical variables were transformed using One-Hot Encoding.

The number of observations remained unchanged throughout preprocessing.

---

# 13. Pipeline Architecture

The preprocessing workflow follows the sequence below:

1. Data Loading
2. Data Quality Assessment
3. Data Type Verification
4. Category Cleaning
5. Feature Engineering
6. Train-Test Split
7. Numerical Preprocessing
8. Categorical Encoding
9. Feature Scaling
10. Machine Learning Ready Dataset

---

# 14. Conclusion

The IBM Telco Customer Churn dataset was successfully preprocessed using a modular and reusable ML engineering workflow.

The project demonstrated:

- Data quality assessment
- Duplicate detection
- Data type correction
- Category standardization
- Feature engineering
- Stratified train-test splitting
- Automated preprocessing with Scikit-Learn Pipelines
- Data leakage prevention

The resulting dataset is fully prepared for binary classification models and can be directly used for customer churn prediction.