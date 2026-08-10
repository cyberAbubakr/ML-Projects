"""
Loading utilities for the Day-9 dashboard.

Everything here READS existing, already-produced artifacts. Nothing is
retrained, re-clustered, or recomputed. Streamlit caching is used so the
model/data are loaded once per session, not on every interaction.
"""

import joblib
import pandas as pd
import streamlit as st

from src import config


@st.cache_resource(show_spinner=False)
def load_production_pipeline():
    """
    Load the single saved production pipeline bundle (Module 6):
    {'scaler': StandardScaler(), 'model': KMeans(...), 'features': [...]}.

    This is the PREFERRED way to get the model/scaler/feature-list for
    prediction, per the single-bundle artifact the project already saves.
    Returns None if pipeline.pkl is missing (caller should fall back to
    the individual files below).
    """
    if not config.PIPELINE_PATH.exists():
        return None
    return joblib.load(config.PIPELINE_PATH)


@st.cache_resource(show_spinner=False)
def load_kmeans_model():
    """Load the final saved KMeans clustering model (Module 6) individually.

    Kept for the About/Methodology page and as a fallback if pipeline.pkl
    is unavailable; the main prediction flow prefers load_production_pipeline().
    """
    return joblib.load(config.KMEANS_MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_scaler():
    """Load the fitted StandardScaler used in the final pipeline (Module 6).

    Kept as a fallback if pipeline.pkl is unavailable; not used to re-scale
    data that has already been scaled (see app.py Customer Segmentation page).
    """
    return joblib.load(config.STANDARD_SCALER_PATH)


@st.cache_resource(show_spinner=False)
def load_selected_features():
    """Load the exact ordered feature list used by the final model (Module 6)."""
    return joblib.load(config.SELECTED_FEATURES_PATH)


@st.cache_data(show_spinner=False)
def load_cleaned_customers():
    """
    Human-readable cleaned customer data (raw Income/Age/Education/etc.),
    Module 6's `customer_personality_cleaned.csv`. This is the PREFERRED
    source for the customer selector / customer-level display.

    Returns None (instead of raising) if the file is missing, so pages that
    don't need it can keep working; pages that do need it must check for
    None and show a clear message rather than fabricate data.
    """
    if not config.CLEANED_DATA_PATH.exists():
        return None
    return pd.read_csv(config.CLEANED_DATA_PATH)


@st.cache_data(show_spinner=False)
def load_engineered_features():
    """
    Module 6's `customer_personality_features.csv` — the final engineered
    AND already-scaled feature table used as direct model input. This is
    NOT raw customer data; it is only ever used as model input, never
    displayed as if it were the customer record.

    Row order matches load_cleaned_customers() 1:1 (verified against the
    same source pipeline). Returns None if the file is missing.
    """
    if not config.FEATURES_DATA_PATH.exists():
        return None
    return pd.read_csv(config.FEATURES_DATA_PATH)


@st.cache_data(show_spinner=False)
def load_algorithm_comparison() -> pd.DataFrame:
    """Model comparison table (KMeans/Hierarchical/GMM/DBSCAN) from Module 6."""
    return pd.read_excel(config.ALGORITHM_COMPARISON_PATH)


@st.cache_data(show_spinner=False)
def load_cluster_names() -> pd.DataFrame:
    """Segment names + justification from Module 7."""
    return pd.read_excel(config.CLUSTER_NAMES_PATH)


@st.cache_data(show_spinner=False)
def load_cluster_statistics() -> pd.DataFrame:
    """Per-cluster raw/standardized statistics from Module 7."""
    return pd.read_excel(config.CLUSTER_STATISTICS_PATH)


@st.cache_data(show_spinner=False)
def load_final_customer_profiles() -> pd.DataFrame:
    """Full per-segment profile table (Module 7)."""
    return pd.read_excel(config.FINAL_CUSTOMER_PROFILES_PATH)


@st.cache_data(show_spinner=False)
def load_customer_personas() -> pd.DataFrame:
    """Extended personas table (Module 8, builds on Module 7)."""
    return pd.read_excel(config.CUSTOMER_PERSONAS_M8_PATH)


@st.cache_data(show_spinner=False)
def load_segment_summary() -> pd.DataFrame:
    """Segment counts + percentages (Module 8)."""
    return pd.read_excel(config.CUSTOMER_SEGMENT_SUMMARY_PATH)


@st.cache_data(show_spinner=False)
def load_business_opportunity() -> pd.DataFrame:
    return pd.read_excel(config.BUSINESS_OPPORTUNITY_PATH)


@st.cache_data(show_spinner=False)
def load_marketing_strategy() -> pd.DataFrame:
    return pd.read_excel(config.MARKETING_STRATEGY_PATH)


@st.cache_data(show_spinner=False)
def load_marketing_strategy_matrix() -> pd.DataFrame:
    return pd.read_excel(config.MARKETING_STRATEGY_MATRIX_PATH)


@st.cache_data(show_spinner=False)
def load_product_recommendation() -> pd.DataFrame:
    return pd.read_excel(config.PRODUCT_RECOMMENDATION_PATH)


@st.cache_data(show_spinner=False)
def load_pricing_strategy() -> pd.DataFrame:
    return pd.read_excel(config.PRICING_STRATEGY_PATH)


@st.cache_data(show_spinner=False)
def load_retention_strategy() -> pd.DataFrame:
    return pd.read_excel(config.RETENTION_STRATEGY_PATH)


@st.cache_data(show_spinner=False)
def load_reactivation_strategy() -> pd.DataFrame:
    return pd.read_excel(config.REACTIVATION_STRATEGY_PATH)


@st.cache_data(show_spinner=False)
def load_campaign_action_plan() -> pd.DataFrame:
    return pd.read_excel(config.CAMPAIGN_ACTION_PLAN_PATH)


def check_artifacts_exist() -> dict:
    """
    Check expected project artifacts, split into:
      - "required": the app core cannot run without these (missing -> stop app)
      - "optional": display-only / supplementary; missing -> warn, keep running

    Returns {"required": [(label, path), ...], "optional": [(label, path), ...]}
    for any files that are actually missing (present files are omitted).
    """
    required = {
        "Production pipeline bundle (Module 6)": config.PIPELINE_PATH,
        "Selected feature list (Module 6)": config.SELECTED_FEATURES_PATH,
        "Cluster names (Module 7)": config.CLUSTER_NAMES_PATH,
        "Cluster statistics (Module 7)": config.CLUSTER_STATISTICS_PATH,
        "Segment summary (Module 8)": config.CUSTOMER_SEGMENT_SUMMARY_PATH,
        "Marketing strategy matrix (Module 8)": config.MARKETING_STRATEGY_MATRIX_PATH,
    }
    optional = {
        "Cleaned customer data (Module 6)": config.CLEANED_DATA_PATH,
        "Engineered feature data (Module 6)": config.FEATURES_DATA_PATH,
        "Individual KMeans model file (Module 6, also in pipeline.pkl)": config.KMEANS_MODEL_PATH,
        "Individual scaler file (Module 6, also in pipeline.pkl)": config.STANDARD_SCALER_PATH,
        "Algorithm comparison report (Module 6)": config.ALGORITHM_COMPARISON_PATH,
        "Final customer profiles (Module 7)": config.FINAL_CUSTOMER_PROFILES_PATH,
        "Customer personas, Module 7": config.CUSTOMER_PERSONAS_M7_PATH,
        "Customer personas, Module 8": config.CUSTOMER_PERSONAS_M8_PATH,
        "Business opportunity analysis (Module 8)": config.BUSINESS_OPPORTUNITY_PATH,
        "Marketing strategy (Module 8)": config.MARKETING_STRATEGY_PATH,
        "Product recommendation (Module 8)": config.PRODUCT_RECOMMENDATION_PATH,
        "Pricing strategy (Module 8)": config.PRICING_STRATEGY_PATH,
        "Retention strategy (Module 8)": config.RETENTION_STRATEGY_PATH,
        "Reactivation strategy (Module 8)": config.REACTIVATION_STRATEGY_PATH,
        "Campaign action plan (Module 8)": config.CAMPAIGN_ACTION_PLAN_PATH,
        "Dashboard summary image (Module 8)": config.DASHBOARD_SUMMARY_IMAGE_PATH,
    }
    return {
        "required": [(label, str(p)) for label, p in required.items() if not p.exists()],
        "optional": [(label, str(p)) for label, p in optional.items() if not p.exists()],
    }
