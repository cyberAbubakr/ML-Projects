"""
Central path configuration for the Day-9 Streamlit application.

Day-9 does NOT copy datasets, models, or reports from earlier modules.
It references the existing, already-submitted outputs of:
  - Module 6 (06-Customer-Personality-Analysis): cleaned data, engineered
    features, saved scaler + final KMeans model + selected feature list
  - Module 7 (07-Customer-Segmentation-Clustering): cluster naming and
    cluster/segment statistics reports
  - Module 8 (08-Marketing-Business-Insights): business insights and
    marketing recommendation reports

This file assumes the Day-9 folder sits alongside the module folders,
i.e. the project layout looks like:

    <project_root>/
        06-Customer-Personality-Analysis/
        07-Customer-Segmentation-Clustering/
        08-Marketing-Business-Insights/
        Day-9/                <- this app
"""

from pathlib import Path

# Day-9/src/config.py -> parents[1] = Day-9/, parents[2] = project root
APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent

MODULE6_DIR = PROJECT_ROOT / "06-Customer-Personality-Analysis"
MODULE7_DIR = PROJECT_ROOT / "07-Customer-Segmentation-Clustering"
MODULE8_DIR = PROJECT_ROOT / "08-Marketing-Business-Insights"

# ---- Module 6: data + saved production artifacts ----
CLEANED_DATA_PATH = MODULE6_DIR / "03_Cleaned_Data" / "customer_personality_cleaned.csv"
FEATURES_DATA_PATH = MODULE6_DIR / "03_Cleaned_Data" / "customer_personality_features.csv"

MODELS_DIR = MODULE6_DIR / "outputs" / "models"
KMEANS_MODEL_PATH = MODELS_DIR / "kmeans.pkl"
PIPELINE_PATH = MODELS_DIR / "pipeline.pkl"
STANDARD_SCALER_PATH = MODELS_DIR / "standard_scaler.pkl"
SELECTED_FEATURES_PATH = MODELS_DIR / "selected_features.pkl"

ALGORITHM_COMPARISON_PATH = MODULE6_DIR / "outputs" / "reports" / "algorithm_comparison.xlsx"

# ---- Module 7: cluster naming + statistics ----
MODULE7_REPORTS_DIR = MODULE7_DIR / "outputs" / "reports"
CLUSTER_NAMES_PATH = MODULE7_REPORTS_DIR / "cluster_names.xlsx"
CLUSTER_STATISTICS_PATH = MODULE7_REPORTS_DIR / "cluster_statistics.xlsx"
FINAL_CUSTOMER_PROFILES_PATH = MODULE7_REPORTS_DIR / "final_customer_profiles.xlsx"
CUSTOMER_PERSONAS_M7_PATH = MODULE7_REPORTS_DIR / "customer_personas.xlsx"

# ---- Module 8: business insights + marketing recommendations ----
MODULE8_REPORTS_DIR = MODULE8_DIR / "outputs" / "reports"
CUSTOMER_SEGMENT_SUMMARY_PATH = MODULE8_REPORTS_DIR / "Customer_Segment_Summary.xlsx"
CUSTOMER_PERSONAS_M8_PATH = MODULE8_REPORTS_DIR / "Customer_Personas.xlsx"
BUSINESS_OPPORTUNITY_PATH = MODULE8_REPORTS_DIR / "Business_Opportunity_Analysis.xlsx"
MARKETING_STRATEGY_PATH = MODULE8_REPORTS_DIR / "Marketing_Strategy.xlsx"
MARKETING_STRATEGY_MATRIX_PATH = MODULE8_REPORTS_DIR / "Marketing_Strategy_Matrix.xlsx"
PRODUCT_RECOMMENDATION_PATH = MODULE8_REPORTS_DIR / "Product_Recommendation.xlsx"
PRICING_STRATEGY_PATH = MODULE8_REPORTS_DIR / "Pricing_Strategy.xlsx"
RETENTION_STRATEGY_PATH = MODULE8_REPORTS_DIR / "Retention_Strategy.xlsx"
REACTIVATION_STRATEGY_PATH = MODULE8_REPORTS_DIR / "Reactivation_Strategy.xlsx"
CAMPAIGN_ACTION_PLAN_PATH = MODULE8_REPORTS_DIR / "Campaign_Action_Plan.xlsx"

# Module 8 dashboard figure (already generated, safe to display as-is)
DASHBOARD_SUMMARY_IMAGE_PATH = MODULE8_DIR / "outputs" / "dashboard" / "dashboard_summary.png"

PROJECT_TITLE = "Customer Segmentation & Marketing Insights Dashboard"
PROJECT_DESCRIPTION = (
    "An interactive view of the customer segmentation pipeline built in Modules "
    "6-8: engineered customer features, the final KMeans clustering model, and "
    "the resulting business insights and marketing recommendations."
)
