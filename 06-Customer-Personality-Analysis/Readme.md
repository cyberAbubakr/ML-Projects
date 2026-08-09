# Customer Personality Analysis

# Data Understanding Report

**Project:** Customer Personality Analysis Dataset

**Dataset Source:** Kaggle Marketing Campaign Dataset

**Prepared By:** Abubakr Kazmi

**Tools Used:** Python, Pandas, NumPy, Matplotlib, Jupyter Notebook

**Report Type:** Initial Data Exploration and Data Quality Assessment

**Date:** July 2026

---

# Introduction

The first step of any machine learning project is to understand the dataset before performing preprocessing, feature engineering, or model building. A proper understanding of the data helps identify its structure, recognize quality issues, and determine what cleaning steps will be required in later stages.

In this project, I worked with the **Customer Personality Analysis** dataset obtained from Kaggle. The dataset contains customer demographic information, purchasing behavior, website activity, campaign responses, and complaint history. These features provide valuable information that can later be used for customer segmentation, behavior analysis, and predictive modeling.

The main objective of this phase was to perform an initial inspection of the dataset, identify data quality issues, document every variable, and generate supporting reports that will be used during the preprocessing stage.

---

# Dataset Overview

The dataset used in this project is the **Customer Personality Analysis** dataset available on Kaggle. It contains historical information about customers and their interactions with a company's marketing campaigns.

The original dataset is stored as a **tab-separated CSV file**, therefore the file was loaded using the appropriate separator (`sep='\t'`) in Pandas.

**Dataset File**

```
marketing_campaign.csv
```

The dataset contains information related to:

- Customer demographic details
- Household composition
- Product spending
- Purchase channels
- Marketing campaign responses
- Website engagement
- Customer complaints

---

# Basic Dataset Inspection

After loading the dataset into Pandas, I performed an initial inspection to understand its overall structure.

| Description | Value |
|-------------|------:|
| Total Records | 2,240 |
| Total Columns | 29 |
| File Format | CSV (Tab Delimited) |
| File Size | ~0.21 MB |

The dataset consists of **2,240 customer records** and **29 variables**, covering customer demographics, purchasing behavior, campaign responses, and customer engagement.

The following basic inspection steps were performed:

- Loaded the dataset successfully.
- Displayed the first few records.
- Checked the dataset dimensions.
- Examined column names.
- Verified data types.
- Generated descriptive statistics.

---

# Dataset Structure

To keep the project organized, all files were stored in a structured directory. Each folder contains a specific part of the data understanding phase.

```
Day-6/
│
├── 01_Raw_Data/
│   └── marketing_campaign.csv
│
├── 02_Inspection_Report/
│   ├── basic_dataset_inspection.xlsx
│   ├── variable_inspection.xlsx
│   ├── numerical_variables_summary.xlsx
│   └── categorical_variables_summary.xlsx
│
├── 03_Data_Quality/
│   ├── missing_values_report.xlsx
│   ├── duplicate_report.xlsx
│   ├── incorrect_values_report.xlsx
│   └── data_quality_report.xlsx
│
├── 04_Data_Dictionary/
│   └── data_dictionary.xlsx
│
├── 01_Data_Understanding.ipynb
│
└── README.md
```

This folder structure separates raw data, generated reports, notebooks, and documentation, making the project easier to understand and maintain.

---

# Variable Inspection

The next step was to inspect every column in the dataset. I examined the column names, identified their data types, and grouped the variables into numerical and categorical categories.

The dataset contains a mixture of demographic variables, spending variables, purchase history, campaign response indicators, and customer activity information.

---

## Numerical Variables

Most of the dataset consists of numerical variables representing customer characteristics and purchasing behavior.

The numerical variables include:

- ID
- Year_Birth
- Income
- Kidhome
- Teenhome
- Recency
- MntWines
- MntFruits
- MntMeatProducts
- MntFishProducts
- MntSweetProducts
- MntGoldProds
- NumDealsPurchases
- NumWebPurchases
- NumCatalogPurchases
- NumStorePurchases
- NumWebVisitsMonth
- AcceptedCmp1
- AcceptedCmp2
- AcceptedCmp3
- AcceptedCmp4
- AcceptedCmp5
- Complain
- Z_CostContact
- Z_Revenue
- Response

These variables represent customer demographics, product spending, purchase frequency, website activity, and marketing campaign responses.

---

## Categorical Variables

The dataset contains three non-numerical variables:

- **Education** – Customer education level.
- **Marital_Status** – Customer marital status.
- **Dt_Customer** – Customer enrollment date.

Although `Dt_Customer` is initially stored as an object (string), it represents date information and will be converted to the appropriate datetime format during the preprocessing stage.

The categorical variables provide useful information about customer background and will later be standardized and prepared for analysis.

# Missing Value Analysis

Missing value analysis was performed to identify incomplete records that could affect future analysis and machine learning models.

The dataset is generally complete, with missing values found in only one column.

| Column | Missing Values | Percentage |
|---------|---------------:|-----------:|
| Income | 24 | 1.07% |

The remaining **28 columns** contain no missing values.

Since **Income** is one of the most important variables for customer analysis and purchasing behavior, these missing values must be handled during the preprocessing stage instead of removing the affected records.

Because only a small percentage of the dataset is missing, imputing the missing values is more appropriate than deleting customer records.

---

# Duplicate Record Analysis

Duplicate records were checked to ensure that each customer appears only once in the dataset.

Two duplicate checks were performed:

- Complete row duplication
- Duplicate customer IDs

The results are summarized below.

| Check | Result |
|--------|-------:|
| Duplicate Rows | 0 |
| Duplicate IDs | 0 |

No duplicate customer records were found in the dataset.

This indicates that every customer record is unique and no duplicate removal is required during the data understanding phase.

---

# Incorrect Values and Data Quality Issues

Although the dataset is well structured, several data quality issues were identified during the inspection process. These issues will be addressed during the preprocessing stage.

## 1. Unrealistic Birth Years

The **Year_Birth** column contains three unrealistic values:

```
1893
1899
1900
```

These birth years correspond to customer ages greater than 110 years, which are highly unlikely for this dataset. These records should be reviewed and handled during preprocessing.

---

## 2. Missing Income Values

The **Income** column contains **24 missing values**.

Income is an important feature because it represents the purchasing capability of each customer. Leaving these missing values untreated may affect statistical analysis and future machine learning models.

These missing values will be handled during preprocessing using an appropriate imputation technique.

---

## 3. Income Outlier

The dataset contains one extremely large income value:

```
Income = 666,666
```

This value is significantly higher than the rest of the dataset and appears as an extreme statistical outlier.

The outlier will be evaluated during preprocessing before applying machine learning algorithms.

---

## 4. Non-Standard Marital Status Categories

The **Marital_Status** column contains several uncommon category values.

| Category | Count |
|----------|------:|
| Alone | 3 |
| Absurd | 2 |
| YOLO | 2 |

These values represent very few customers and are not standard relationship categories. They should be standardized during preprocessing to improve data consistency.

---

## 5. Constant Columns

Two columns contain the same value for every customer in the dataset.

- Z_CostContact
- Z_Revenue

Since these columns have no variation, they do not provide useful information for statistical analysis or machine learning models and can be removed during preprocessing.

---

# Business Understanding

The dataset provides valuable customer information that can be used for business intelligence and marketing analysis.

## Customer Demographics

Customer demographic variables such as age, education level, marital status, and household composition help understand the characteristics of different customer groups.

## Purchasing Behavior

Product spending variables provide insights into customer buying habits across different product categories, allowing businesses to identify popular products and high-value customers.

## Marketing Campaign Performance

Campaign acceptance variables indicate how customers responded to previous marketing campaigns. These variables can be used to measure campaign effectiveness and identify customers who are more likely to respond to future promotions.

## Website Engagement

The number of monthly website visits helps measure customer engagement with the company's online platform and can support digital marketing strategies.

## Customer Complaints

Complaint information helps identify dissatisfied customers and supports customer retention strategies by highlighting areas where service improvements may be required.

---

# Data Quality Summary

The overall data quality of the dataset is good, with only a few issues identified during the inspection phase.

| Issue | Column | Description | Recommended Action |
|-------|--------|-------------|--------------------|
| Missing Values | Income | 24 missing records | Handle during preprocessing |
| Unrealistic Values | Year_Birth | Three unrealistic birth years | Remove invalid records |
| Extreme Outlier | Income | One unusually high income value | Detect and treat outlier |
| Inconsistent Categories | Marital_Status | Alone, YOLO, Absurd | Standardize category names |
| Constant Features | Z_CostContact, Z_Revenue | No variation in values | Remove before modeling |

The identified issues are relatively minor and do not affect the overall usability of the dataset. After preprocessing, the dataset will be suitable for exploratory data analysis and machine learning tasks.

# Data Dictionary

To better understand the dataset, a data dictionary was prepared for all variables. The data dictionary provides the data type, description, and business purpose of each feature used in the dataset.

The dataset contains demographic information, household details, purchasing behavior, marketing campaign responses, website activity, and customer complaint records.

| Feature | Data Type | Description | Business Meaning |
|----------|-----------|-------------|------------------|
| ID | Integer | Unique customer identifier | Used to uniquely identify each customer |
| Year_Birth | Integer | Customer birth year | Used to calculate customer age |
| Education | Text | Customer education level | Represents educational background |
| Marital_Status | Text | Customer marital status | Indicates relationship status |
| Income | Float | Annual household income | Represents customer purchasing power |
| Kidhome | Integer | Number of children at home | Describes household composition |
| Teenhome | Integer | Number of teenagers at home | Describes household composition |
| Dt_Customer | Date | Customer enrollment date | Indicates when the customer joined the company |
| Recency | Integer | Days since last purchase | Measures customer activity |
| MntWines | Integer | Amount spent on wine | Customer spending on wine products |
| MntFruits | Integer | Amount spent on fruits | Customer spending on fruit products |
| MntMeatProducts | Integer | Amount spent on meat | Customer spending on meat products |
| MntFishProducts | Integer | Amount spent on fish | Customer spending on fish products |
| MntSweetProducts | Integer | Amount spent on sweets | Customer spending on sweet products |
| MntGoldProds | Integer | Amount spent on gold products | Customer spending on premium products |
| NumDealsPurchases | Integer | Purchases made using discounts | Indicates deal-seeking behavior |
| NumWebPurchases | Integer | Purchases made through the website | Measures online purchasing activity |
| NumCatalogPurchases | Integer | Purchases made using catalogs | Measures catalog purchases |
| NumStorePurchases | Integer | Purchases made in physical stores | Measures in-store shopping activity |
| NumWebVisitsMonth | Integer | Monthly website visits | Measures website engagement |
| AcceptedCmp1 | Binary | Accepted Campaign 1 | Indicates response to Campaign 1 |
| AcceptedCmp2 | Binary | Accepted Campaign 2 | Indicates response to Campaign 2 |
| AcceptedCmp3 | Binary | Accepted Campaign 3 | Indicates response to Campaign 3 |
| AcceptedCmp4 | Binary | Accepted Campaign 4 | Indicates response to Campaign 4 |
| AcceptedCmp5 | Binary | Accepted Campaign 5 | Indicates response to Campaign 5 |
| Complain | Binary | Customer complaint status | Indicates whether the customer submitted a complaint |
| Z_CostContact | Integer | Cost of customer contact | Constant metadata field |
| Z_Revenue | Integer | Revenue generated per customer | Constant metadata field |
| Response | Binary | Response to the latest marketing campaign | Target variable for campaign prediction |

---

# Generated Reports

During the data understanding phase, several Excel reports were generated automatically to summarize the dataset and document data quality findings.

## Inspection Reports

The **02_Inspection_Report** folder contains reports generated during the initial dataset inspection.

Generated files include:

- Basic dataset inspection
- Variable inspection
- Numerical variable summary
- Categorical variable summary

These reports provide a quick overview of the dataset structure and variable characteristics.

---

## Data Quality Reports

The **03_Data_Quality** folder contains reports generated during the data quality assessment.

Generated files include:

- Missing values report
- Duplicate records report
- Incorrect values report
- Complete data quality report

These reports summarize the quality issues identified during the inspection phase and provide a reference for the preprocessing stage.

---

## Data Dictionary

The **04_Data_Dictionary** folder contains the complete data dictionary generated for the dataset.

The data dictionary documents every variable, its data type, description, and business meaning, making the dataset easier to understand and use during future analysis.

---

# Final Summary

The Customer Personality Analysis dataset is well structured and contains valuable information about customer demographics, purchasing behavior, website activity, and marketing campaign responses.

During the data understanding phase, I examined the dataset structure, verified the data types, checked for missing values and duplicate records, identified incorrect values and outliers, and documented each variable using a comprehensive data dictionary.

The inspection revealed that the dataset has **2,240 customer records** and **29 variables**. Only the **Income** column contains missing values, no duplicate records were found, and a few data quality issues such as unrealistic birth years, extreme income values, non-standard marital status categories, and constant columns were identified.

Overall, the dataset has good data quality and provides a strong foundation for further preprocessing, exploratory data analysis, and machine learning tasks. The issues identified during this phase will be addressed in the preprocessing stage to produce a clean and reliable dataset for subsequent analysis.

---

## Prepared By

**Abubakr Kazmi**

Machine Learning Intern

AI Lab 99

**Project Phase:** Data Understanding and Quality Assessment

**Date:** July 2026
