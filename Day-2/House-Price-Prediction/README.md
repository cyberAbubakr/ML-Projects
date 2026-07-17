# 📂 Dataset

This project uses the **House Prices: Advanced Regression Techniques (Ames Housing Dataset)** available on Kaggle.

**Kaggle Competition:**
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

**Download Dataset:**
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

Files used:

```text
house-prices-advanced-regression-techniques/
│
├── train.csv
├── test.csv
├── data_description.txt
└── sample_submission.csv
```

Place the downloaded dataset inside the project directory:

```text
House-Price-Prediction/
│
├── house-prices-advanced-regression-techniques/
│   ├── train.csv
│   ├── test.csv
│   ├── data_description.txt
│   └── sample_submission.csv
```

### Target Variable

```text
SalePrice
```

The target variable was log-transformed using:

```python
SalePrice = np.log1p(SalePrice)
```

to reduce skewness and improve model performance.