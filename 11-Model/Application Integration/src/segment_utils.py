"""
Small helpers that connect the saved KMeans model to the business-facing
segment names established in Module 7/8. No model logic is redefined here -
this only reads what Module 7/8 already produced.
"""

import pandas as pd


def get_cluster_name_map(cluster_names_df: pd.DataFrame) -> dict:
    """{cluster_id (int): business segment name (str)} from Module 7's cluster_names.xlsx."""
    return dict(zip(cluster_names_df["Cluster"], cluster_names_df["Segment_Name"]))


def predict_segment_for_row(kmeans_model, features_row: pd.DataFrame):
    """
    Run inference only (no fitting) on a single already-engineered,
    already-scaled customer row, using the saved production model.

    NOTE: `customer_personality_features.csv` (the source of `features_row`)
    is already the OUTPUT of Module 6's saved scaler. It is passed straight
    to the model's .predict() and must NOT be re-scaled here, or the values
    would be scaled twice and the prediction would be wrong.
    """
    return int(kmeans_model.predict(features_row)[0])


def get_model_display_name(model) -> str:
    """
    Build a human-readable model label purely from the loaded model object
    (e.g. "KMeans (k=2)"), instead of hard-coding it, so it always reflects
    whatever model is actually saved in the production pipeline.
    """
    name = type(model).__name__
    if hasattr(model, "n_clusters"):
        return f"{name} (k={model.n_clusters})"
    if hasattr(model, "n_components"):
        return f"{name} (k={model.n_components})"
    return name


def get_segment_row(profiles_df: pd.DataFrame, cluster_id: int) -> pd.Series:
    """Look up a segment's full profile row by cluster id."""
    match = profiles_df[profiles_df["Cluster"] == cluster_id]
    if match.empty:
        return None
    return match.iloc[0]


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Encode a dataframe as CSV bytes for st.download_button. No data is altered."""
    return df.to_csv(index=False).encode("utf-8")
