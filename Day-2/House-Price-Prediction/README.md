# 📂 Dataset

This project uses the **House Prices: Advanced Regression Techniques (Ames Housing Dataset)** provided by Kaggle.

### 🏆 Kaggle Competition

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

### 📥 Download Dataset

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

### 📖 Dataset Description

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

The dataset contains the following files:

```text
house-prices-advanced-regression-techniques/
│
├── train.csv
├── test.csv
├── data_description.txt
└── sample_submission.csv
```

After downloading, place the dataset inside the project directory:

```text
House-Price-Prediction/
│
├── house-prices-advanced-regression-techniques/
│   ├── train.csv
│   ├── test.csv
│   ├── data_description.txt
│   └── sample_submission.csv
```

### 🎯 Target Variable

```text
SalePrice
```

The target variable was log-transformed using:

```python
SalePrice = np.log1p(SalePrice)
```

to reduce skewness and improve regression model performance.
