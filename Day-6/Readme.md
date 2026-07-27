# Customer Personality Analysis

## Data Understanding Report

**Project:** Customer Personality Analysis Dataset  
**Dataset Source:** Kaggle Marketing Campaign Dataset  
**Prepared By:** Abubakr Kazmi  
**Tools Used:** Python, Pandas, Jupyter Notebook  
**Report Type:** Initial Data Exploration and Data Quality Analysis  
**Date:** July 2026  

---

# Introduction

This project focuses on understanding and analyzing the Customer Personality Analysis dataset before applying any preprocessing or machine learning techniques.

The main purpose of this phase was to inspect the dataset structure, understand different variables, check data quality problems, and document important findings that need to be considered during future analysis.

The dataset contains customer demographic information, purchasing behavior, marketing campaign responses, website activity, and complaint records. These attributes can help understand customer preferences and support future customer segmentation and prediction tasks.

---

# Dataset Overview

The original dataset was obtained from Kaggle and is stored in a tab-separated CSV format.

Dataset file:

```
marketing_campaign.csv
```

The dataset contains information about individual customers, including:

- Personal demographics
- Household information
- Product spending
- Purchase channels
- Campaign responses
- Customer complaints

---

# Basic Dataset Inspection

| Description | Value |
|-------------|-------|
| Total Records | 2,240 |
| Total Columns | 29 |
| File Format | CSV (Tab Delimited) |
| File Size | 0.21 MB |

The dataset contains 2,240 customer records with 29 different attributes.

---

# Dataset Structure

The project is organized as:

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
├── notebooks/
│   └── data_understanding.ipynb
│
└── README.md
```

---

# Variable Inspection

After checking the dataset, the variables were divided into numerical and categorical types.

## Numerical Variables

The dataset contains 25 numerical variables:

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

---

## Categorical Variables

The dataset contains three categorical/text variables:

- Education
- Marital_Status
- Dt_Customer

---

# Missing Value Analysis

Missing value checking was performed on all columns.

The only missing values were found in the Income column.

| Column | Missing Values | Percentage |
|--------|---------------|------------|
| Income | 24 | 1.07% |

The number of missing values is small compared to the total dataset size. However, Income is an important feature because it represents customer purchasing ability, so it needs proper handling during preprocessing.

---

# Duplicate Record Analysis

Duplicate records were checked using:

- Complete row comparison
- Customer ID comparison

Results:

| Check | Result |
|------|--------|
| Duplicate Rows | 0 |
| Duplicate IDs | 0 |

No duplicate customer records were found in the dataset.

---

# Incorrect Values and Data Quality Issues

During the inspection process, some data quality issues were identified.

## 1. Unrealistic Birth Years

Three records contain unusual birth years:

```
1893
1899
1899
```

These values indicate unrealistic customer ages and should be investigated before further analysis.

---

## 2. Income Outlier

One customer record has:

```
Income = 666,666
```

This value is significantly higher compared to other customers and may represent a data entry error.

---

## 3. Incorrect Marital Status Categories

Some unusual values were found in the Marital_Status column:

| Category | Count |
|----------|------:|
| Absurd | 2 |
| YOLO | 2 |
| Alone | 3 |

These categories should be reviewed and standardized.

---

# Business Understanding

The dataset can provide useful insights for marketing and customer behavior analysis.

## High Income Customers

Income information can help identify premium customers and create personalized marketing strategies.

## Customer Purchase Behavior

Product spending information helps understand customer interests and buying patterns.

## Campaign Response Analysis

Campaign acceptance data helps evaluate which customers are more likely to respond to marketing offers.

## Customer Complaints

Complaint records can help identify dissatisfaction and improve customer retention.

## Website Activity

Monthly website visits can show customer engagement with digital platforms.

---

# Data Quality Summary

| Issue | Column | Description | Action Required |
|------|--------|-------------|----------------|
| Missing Values | Income | 24 missing records | Handle during preprocessing |
| Invalid Values | Year_Birth | 3 unrealistic birth years | Review and correct |
| Outlier | Income | Extreme income value | Investigate |
| Category Issue | Marital_Status | Unusual labels | Standardize values |
| Constant Feature | Z_CostContact | No variation | Remove before modeling |

---

# Data Dictionary

| Feature | Type | Description | Business Meaning |
|---------|------|-------------|------------------|
| ID | Integer | Customer identifier | Unique customer reference |
| Year_Birth | Integer | Customer birth year | Customer age information |
| Education | Text | Education level | Customer background |
| Marital_Status | Text | Relationship status | Household information |
| Income | Float | Annual income | Purchasing capability |
| Kidhome | Integer | Number of children | Family information |
| Teenhome | Integer | Number of teenagers | Family information |
| Dt_Customer | Date/Text | Registration date | Customer relationship period |
| Recency | Integer | Days since last purchase | Customer activity |
| MntWines | Integer | Wine spending | Product preference |
| MntFruits | Integer | Fruit spending | Product preference |
| MntMeatProducts | Integer | Meat spending | Product preference |
| MntFishProducts | Integer | Fish spending | Product preference |
| MntSweetProducts | Integer | Sweet spending | Product preference |
| MntGoldProds | Integer | Gold spending | Premium product interest |
| NumDealsPurchases | Integer | Discount purchases | Deal preference |
| NumWebPurchases | Integer | Online purchases | Website usage |
| NumCatalogPurchases | Integer | Catalog purchases | Catalog engagement |
| NumStorePurchases | Integer | Store purchases | Physical store activity |
| NumWebVisitsMonth | Integer | Monthly visits | Digital engagement |
| AcceptedCmp1-5 | Binary | Campaign acceptance | Marketing response |
| Complain | Binary | Customer complaint | Satisfaction level |
| Response | Binary | Final campaign response | Campaign success |

---

# Generated Reports

## Dataset Inspection Reports

Location:

```
02_Inspection_Report/
```

Contains:

- Basic dataset inspection
- Variable inspection
- Numerical variable summary
- Categorical variable summary


## Data Quality Reports

Location:

```
03_Data_Quality/
```

Contains:

- Missing values report
- Duplicate report
- Incorrect values report
- Complete data quality report

---

# Final Summary

The Customer Personality Analysis dataset is well structured and contains useful information about customer demographics, spending behavior, and marketing responses.

The initial inspection showed that the dataset has:

- No duplicate records
- Very low missing values
- Good overall structure
- A few outliers and inconsistent values

The identified issues will need to be handled during the preprocessing stage before performing exploratory analysis or building machine learning models.

---

**Prepared By:**  
Abubakr Kazmi  

**Project Phase:**  
Data Understanding and Quality Assessment  

**Date:** July 2026
