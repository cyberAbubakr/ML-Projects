import pandas as pd


def load_dataset(path):

    columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "income"
    ]

    df = pd.read_csv(
        path,
        header=None,
        names=columns,
        sep=",",
        skipinitialspace=True,
        na_values="?"
    )

    return df