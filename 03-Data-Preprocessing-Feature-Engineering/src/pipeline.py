"""
Machine Learning Preprocessing Pipeline

House Prices - Advanced Regression Techniques

Contains:
- Numerical preprocessing
- Categorical preprocessing
- Feature transformation
- ColumnTransformer pipeline
"""


import pandas as pd


from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer


from sklearn.impute import SimpleImputer


from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    RobustScaler
)




# ======================================================
# 1. Get Feature Types
# ======================================================


def get_feature_types(df):
    """
    Separate numerical and categorical columns.
    """

    df = df.copy()

    # Remove ID column if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    numerical_features = (
        df.select_dtypes(
            include=["int64", "float64"]
        )
        .columns
        .tolist()
    )

    categorical_features = (
        df.select_dtypes(
            include=["object", "category"]
        )
        .columns
        .tolist()
    )

    return numerical_features, categorical_features





# ======================================================
# 2. Numerical Pipeline
# ======================================================


def create_numeric_pipeline():

    """
    Numerical preprocessing:

    Missing values:
        Median

    Scaling:
        StandardScaler
    """


    numeric_pipeline = Pipeline(

        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),


            (
                "scaler",
                StandardScaler()
            )

        ]

    )


    return numeric_pipeline





# ======================================================
# 3. Categorical Pipeline
# ======================================================


def create_categorical_pipeline():

    """
    Categorical preprocessing:

    Missing values:
        Most frequent

    Encoding:
        One Hot Encoding
    """


    categorical_pipeline = Pipeline(

        steps=[


            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),


            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first"
                )
            )


        ]

    )


    return categorical_pipeline





# ======================================================
# 4. Complete Column Transformer
# ======================================================


def create_preprocessor(df):

    """
    Combine numerical and categorical pipelines
    """


    numerical_features, categorical_features = (
        get_feature_types(df)
    )



    numeric_pipeline = (
        create_numeric_pipeline()
    )


    categorical_pipeline = (
        create_categorical_pipeline()
    )



    preprocessor = ColumnTransformer(

        transformers=[


            (
                "numerical",
                numeric_pipeline,
                numerical_features
            ),



            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )


        ]

    )


    return preprocessor





# ======================================================
# 5. Fit Transform Helper
# ======================================================


def preprocess_data(
        train_df,
        test_df=None
):

    """
    Fit preprocessing pipeline
    """



    preprocessor = create_preprocessor(
        train_df
    )



    X_train = (
        preprocessor
        .fit_transform(train_df)
    )



    if test_df is not None:


        X_test = (
            preprocessor
            .transform(test_df)
        )


        return (
            X_train,
            X_test,
            preprocessor
        )



    return (
        X_train,
        preprocessor
    )