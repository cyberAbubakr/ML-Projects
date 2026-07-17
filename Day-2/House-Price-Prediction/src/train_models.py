from sklearn.pipeline import Pipeline


from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)


from xgboost import XGBRegressor



def get_models():

    models = {


        "Linear Regression":
            LinearRegression(),


        "Ridge":
            Ridge(
                alpha=10
            ),


        "Lasso":
            Lasso(
                alpha=0.001
            ),


        "Elastic Net":
            ElasticNet(
                alpha=0.001,
                l1_ratio=0.5
            ),


        "Decision Tree":
            DecisionTreeRegressor(
                random_state=42
            ),


        "Random Forest":
            RandomForestRegressor(
                n_estimators=200,
                random_state=42
            ),


        "Gradient Boosting":
            GradientBoostingRegressor(
                random_state=42
            ),


        "XGBoost":
            XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )

    }


    return models




def train_models(
        models,
        preprocessor,
        X_train,
        y_train
):

    trained_models = {}



    for name, model in models.items():


        pipeline = Pipeline(
            steps=[

                (
                    "preprocessor",
                    preprocessor
                ),

                (
                    "model",
                    model
                )

            ]
        )


        print(
            f"Training {name}"
        )


        pipeline.fit(
            X_train,
            y_train
        )


        trained_models[name] = pipeline



    return trained_models