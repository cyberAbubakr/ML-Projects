"""
Data preprocessing module
House Prices - Advanced Regression Techniques

Contains reusable preprocessing functions:
- Data quality checks
- Missing value handling
- Outlier detection
- Feature engineering
"""



import pandas as pd
import numpy as np


# Missing Values

from sklearn.impute import (
    SimpleImputer,
    KNNImputer
)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer



# Scaling

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)



# ======================================================
# 1. Missing Value Analysis
# ======================================================


def missing_values_summary(df):

    """
    Returns missing values count and percentage
    """

    missing = pd.DataFrame(
        {
            "Missing_Count": df.isnull().sum(),
            "Missing_Percentage":
            (df.isnull().sum()/len(df))*100
        }
    )


    missing = (
        missing[missing["Missing_Count"] > 0]
        .sort_values(
            "Missing_Percentage",
            ascending=False
        )
    )


    return missing



# ======================================================
# 2. Compare Imputation Methods
# ======================================================


def compare_imputation_methods(df, column):

    """
    Compare:
    Mean
    Median
    Mode
    KNN
    """

    temp = df[[column]].copy()


    temp["Mean"] = (
        SimpleImputer(
            strategy="mean"
        )
        .fit_transform(temp)
    )


    temp["Median"] = (
        SimpleImputer(
            strategy="median"
        )
        .fit_transform(temp)
    )


    temp["Mode"] = (
        SimpleImputer(
            strategy="most_frequent"
        )
        .fit_transform(temp)
    )


    temp["KNN"] = (
        KNNImputer(
            n_neighbors=5
        )
        .fit_transform(temp)
    )


    return temp



# ======================================================
# 3. Advanced Imputation
# ======================================================


def advanced_imputation(df, columns):

    """
    KNN and Iterative Imputation comparison
    """

    temp = df[columns].copy()


    knn = KNNImputer(
        n_neighbors=5
    )


    iterative = IterativeImputer(
        random_state=42
    )


    knn_result = knn.fit_transform(temp)

    iterative_result = (
        iterative
        .fit_transform(temp)
    )


    result = pd.DataFrame(
        {
            "KNN":knn_result[:,0],
            "Iterative":iterative_result[:,0]
        }
    )


    return result



# ======================================================
# 4. Missing Value Treatment
# ======================================================


def handle_missing_values(df):

    """
    Domain based missing value handling
    """

    df = df.copy()



    # Remove high missing columns

    drop_columns = [

        "PoolQC",
        "MiscFeature",
        "Alley",
        "Fence"

    ]


    df.drop(
        columns=drop_columns,
        inplace=True,
        errors="ignore"
    )



    # Numerical


    if "LotFrontage" in df:

        df["LotFrontage"] = (
            df["LotFrontage"]
            .fillna(
                df["LotFrontage"].median()
            )
        )


    if "GarageYrBlt" in df:

        df["GarageYrBlt"] = (
            df["GarageYrBlt"]
            .fillna(0)
        )


    if "MasVnrArea" in df:

        df["MasVnrArea"] = (
            df["MasVnrArea"]
            .fillna(0)
        )



    # Mode


    if "Electrical" in df:

        df["Electrical"] = (
            df["Electrical"]
            .fillna(
                df["Electrical"].mode()[0]
            )
        )



    # Garage


    garage_columns = [

        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond"

    ]


    for col in garage_columns:

        if col in df:

            df[col] = (
                df[col]
                .fillna("NoGarage")
            )



    # Basement


    basement_columns = [

        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2"

    ]


    for col in basement_columns:

        if col in df:

            df[col] = (
                df[col]
                .fillna("NoBasement")
            )

 # Fireplace Features

    fireplace_columns = [
        "FireplaceQu"
    ]

    for col in fireplace_columns:

        if col in df:

            df[col] = (
                df[col]
                .fillna("NoFireplace")
            )


    # Masonry

    if "MasVnrType" in df:

        df["MasVnrType"] = (
            df["MasVnrType"]
            .fillna("None")
        )


    return df




# ======================================================
# 5. Duplicate Handling
# ======================================================


def check_duplicates(df):

    return df.duplicated().sum()



def remove_duplicates(df):

    return df.drop_duplicates()




# ======================================================
# 6. Invalid Values
# ======================================================


def check_invalid_values(df):

    invalid = {}


    numerical = df.select_dtypes(
        include=np.number
    ).columns



    for col in numerical:

        count = (
            df[col] < 0
        ).sum()


        if count > 0:

            invalid[col] = count



    return invalid




# ======================================================
# 7. Data Type Checking
# ======================================================


def check_data_types(df):

    return df.dtypes




# ======================================================
# 8. Category Cleaning
# ======================================================


def clean_categories(df):

    df = df.copy()


    categorical = df.select_dtypes(
        include="object"
    ).columns



    for col in categorical:

        df[col] = (
            df[col]
            .str.strip()
            .str.lower()
        )



    return df




# ======================================================
# 9. Outlier Detection IQR
# ======================================================


def detect_outliers_iqr(df,column):


    Q1 = df[column].quantile(0.25)

    Q3 = df[column].quantile(0.75)


    IQR = Q3-Q1


    lower = Q1 - 1.5*IQR

    upper = Q3 + 1.5*IQR



    outliers = df[
        (df[column] < lower) |
        (df[column] > upper)
    ]



    return outliers




# ======================================================
# 10. Z Score Outlier Detection
# ======================================================


def detect_outliers_zscore(df,column):


    mean = df[column].mean()

    std = df[column].std()



    z_score = (
        (df[column]-mean)/std
    )


    return df[
        abs(z_score)>3
    ]




# ======================================================
# 11. Outlier Capping
# ======================================================


def cap_outliers(df,column):


    df=df.copy()


    Q1=df[column].quantile(.25)

    Q3=df[column].quantile(.75)


    IQR=Q3-Q1



    lower=Q1-1.5*IQR

    upper=Q3+1.5*IQR



    df[column]=np.clip(
        df[column],
        lower,
        upper
    )


    return df




# ======================================================
# 12. House Price Feature Engineering
# ======================================================


def create_house_features(df):


    df=df.copy()



    # House Age


    if "YearBuilt" in df:

        df["HouseAge"] = (
            df["YrSold"]
            -
            df["YearBuilt"]
        )



    # Total Bathrooms


    if "FullBath" in df:


        df["TotalBathrooms"] = (

            df["FullBath"]

            +

            0.5*df["HalfBath"]

            +

            df["BsmtFullBath"]

            +

            0.5*df["BsmtHalfBath"]

        )



    # Total Area


    area_columns=[

        "TotalBsmtSF",
        "1stFlrSF",
        "2ndFlrSF"

    ]


    existing=[
        c for c in area_columns
        if c in df
    ]


    if existing:

        df["TotalArea"] = (
            df[existing]
            .sum(axis=1)
        )



    # Total Porch Area


    porch=[

        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch"

    ]


    existing=[
        c for c in porch
        if c in df
    ]


    if existing:

        df["TotalPorchArea"] = (
            df[existing]
            .sum(axis=1)
        )



    return df

# Telco - Data Type Conversion
# ======================================================

def convert_telco_data_types(df):
    """
    Convert Telco dataset columns to the correct data types.
    """

    df = df.copy()

    # Convert TotalCharges to numeric
    if "TotalCharges" in df.columns:

        df["TotalCharges"] = (
            df["TotalCharges"]
            .replace(" ", np.nan)
        )

        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

    return df

# Telco Feature Engineering

def create_telco_features(df):
    """
    Create new features for the Telco Customer Churn dataset.
    """

    df = df.copy()

    # Customer tenure groups
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 60, 72],
        labels=[
            "0-12 Months",
            "13-24 Months",
            "25-48 Months",
            "49-60 Months",
            "61-72 Months"
        ],
        include_lowest=True
    )

    # Average monthly spend
    df["AverageMonthlySpend"] = (
        df["TotalCharges"] /
        (df["tenure"].replace(0, 1))
    )

    # Senior citizen flag
    df["IsSenior"] = df["SeniorCitizen"]

    # Family support flag
    df["HasDependents"] = (
        (df["Partner"] == "yes") |
        (df["Dependents"] == "yes")
    ).astype(int)

    return df