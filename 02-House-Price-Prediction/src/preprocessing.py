from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.compose import ColumnTransformer



def create_preprocessor(X):

    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns


    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns



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
                    handle_unknown="ignore"
                )
            )
        ]
    )



    preprocessor = ColumnTransformer(
        transformers=[

            (
                "num",
                numeric_pipeline,
                numeric_features
            ),

            (
                "cat",
                categorical_pipeline,
                categorical_features
            )

        ]
    )


    return preprocessor