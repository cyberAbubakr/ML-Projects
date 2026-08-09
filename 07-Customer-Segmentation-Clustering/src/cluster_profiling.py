"""cluster_profiling.py — Module 7: Cluster Evaluation & Customer Profiling.

Reuses the Module 6 (Day-6) engineered dataset and saved production pipeline
(scaler + model + feature list) instead of retraining or recomputing anything.
Cluster labels are obtained by calling .predict() on the already-fitted
Day-6 KMeans model — no new model is trained here.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_DIR = Path.cwd()
DAY6_DIR = PROJECT_DIR.parent / "Day-6"
DAY6_MODELS = DAY6_DIR / "outputs" / "models"
DAY6_FIGURES = DAY6_DIR / "outputs" / "figures"
DAY6_FEATURES_PATH = DAY6_DIR / "03_Cleaned_Data" / "customer_personality_features.csv"

REPORTS_DIR = PROJECT_DIR / "outputs" / "reports"
FIGURES_DIR = PROJECT_DIR / "outputs" / "figures"

SPENDING_COLS = ["MntWines", "MntFruits", "MntMeatProducts",
                  "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
CHANNEL_COLS = ["NumWebPurchases", "NumStorePurchases", "NumCatalogPurchases"]
DEAL_COL = "NumDealsPurchases"
CAMPAIGN_COLS = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]
EDUCATION_COLS_PREFIX = "Education_"
MARITAL_COLS_PREFIX = "Marital_Status_"


# ------------------------------------------------------------
# 1. Load Project Outputs
# ------------------------------------------------------------
def load_day6_dataset(path: Path = DAY6_FEATURES_PATH) -> pd.DataFrame:
    """Load the Module 6 engineered dataset from Day-6 without modifying it."""
    return pd.read_csv(path)


def load_day6_pipeline(path: Path = DAY6_MODELS / "pipeline.pkl") -> dict:
    """Load the saved Day-6 production pipeline (scaler + model + feature list)."""
    return joblib.load(path)


def assign_cluster_labels(df: pd.DataFrame, pipeline: dict) -> np.ndarray:
    """Predict cluster labels with the already-fitted Day-6 model (no retraining)."""
    X = df[pipeline["features"]]
    X_scaled = pipeline["scaler"].transform(X)
    return pipeline["model"].predict(X_scaled)


# ------------------------------------------------------------
# 2. Dataset Validation
# ------------------------------------------------------------
def validate_dataset(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Validate the reloaded dataset and the assigned cluster labels."""
    checks = {
        "Shape": f"{df.shape[0]} rows x {df.shape[1]} columns",
        "Missing values": int(df.isnull().sum().sum()),
        "Duplicate rows": int(df.duplicated().sum()),
        "Cluster count": int(len(set(labels))),
        "Cluster label counts": dict(pd.Series(labels).value_counts().sort_index()),
    }
    return pd.DataFrame({"Check": list(checks.keys()), "Result": [str(v) for v in checks.values()]})


# ------------------------------------------------------------
# 3. Cluster Statistics
# ------------------------------------------------------------
def cluster_statistics(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Mean of every numeric feature by cluster, plus customer count per cluster."""
    numeric_df = df.select_dtypes(include="number").copy()
    numeric_df["Cluster"] = labels
    stats = numeric_df.groupby("Cluster").mean()
    stats.insert(0, "Customer_Count", numeric_df.groupby("Cluster").size())
    return stats


# ------------------------------------------------------------
# 4. Demographic Analysis
# ------------------------------------------------------------
def demographic_report(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Age, Income, Family_Size, Total_Children means plus Education/Marital_Status
    proportions (one-hot columns) by cluster."""
    work = df.copy()
    work["Cluster"] = labels
    numeric_cols = [c for c in ["Age", "Income", "Family_Size", "Total_Children"] if c in work.columns]
    cat_cols = [c for c in work.columns if c.startswith((EDUCATION_COLS_PREFIX, MARITAL_COLS_PREFIX))]
    report = work.groupby("Cluster")[numeric_cols + cat_cols].mean()
    report.insert(0, "Customer_Count", work.groupby("Cluster").size())
    return report


# ------------------------------------------------------------
# 5. Spending Behaviour
# ------------------------------------------------------------
def spending_report(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Spending means by cluster, with highest/lowest spending cluster flagged."""
    work = df.copy()
    work["Cluster"] = labels
    cols = [c for c in ["Total_Spending"] + SPENDING_COLS if c in work.columns]
    report = work.groupby("Cluster")[cols].mean()
    report.insert(0, "Customer_Count", work.groupby("Cluster").size())
    report["Segment_Label"] = np.where(
        report["Total_Spending"] == report["Total_Spending"].max(), "Premium (Highest Spending)",
        np.where(report["Total_Spending"] == report["Total_Spending"].min(),
                 "Budget-Conscious (Lowest Spending)", "Mid-Spending")
    )
    return report


# ------------------------------------------------------------
# 6. Shopping Behaviour & Engagement
# ------------------------------------------------------------
def shopping_behavior_report(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Channel, deal, engagement, and recency means by cluster, with behavior flags."""
    work = df.copy()
    work["Cluster"] = labels
    cols = [c for c in CHANNEL_COLS + [DEAL_COL, "Deal_Dependency", "NumWebVisitsMonth", "Recency"]
            if c in work.columns]
    report = work.groupby("Cluster")[cols].mean()
    report.insert(0, "Customer_Count", work.groupby("Cluster").size())

    report["Dominant_Channel"] = report[[c for c in CHANNEL_COLS if c in report.columns]].idxmax(axis=1)
    # Deal_Dependency (deals / total purchases, from Module 5) is the meaningful flag here,
    # not raw deal count, since raw counts scale up with overall purchase volume.
    report["Deal_Seeker"] = report["Deal_Dependency"] == report["Deal_Dependency"].max()
    report["Activity_Status"] = np.where(
        report["Recency"] == report["Recency"].min(), "Active (Lowest Recency)",
        np.where(report["Recency"] == report["Recency"].max(), "Inactive (Highest Recency)", "Moderate")
    )
    return report


# ------------------------------------------------------------
# 7. Marketing Campaign Analysis
# ------------------------------------------------------------
def campaign_report(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Campaign acceptance/complaint rates by cluster, with responsiveness flags."""
    work = df.copy()
    work["Cluster"] = labels
    cols = [c for c in CAMPAIGN_COLS + ["Response", "Complain"] if c in work.columns]
    report = work.groupby("Cluster")[cols].mean()
    report.insert(0, "Customer_Count", work.groupby("Cluster").size())

    response_col = "Response" if "Response" in report.columns else cols[0]
    report["Responsiveness"] = np.where(
        report[response_col] == report[response_col].max(), "Campaign-Responsive",
        "Marketing-Resistant (Needs Re-engagement)"
    )
    return report


# ------------------------------------------------------------
# 8. Business Segment Naming
# ------------------------------------------------------------
def name_segments(spending_df: pd.DataFrame, shopping_df: pd.DataFrame,
                   campaign_df: pd.DataFrame) -> pd.DataFrame:
    """Assign a business name to each cluster, justified by the actual computed stats."""
    records = []
    for cluster in spending_df.index:
        spend_label = spending_df.loc[cluster, "Segment_Label"]
        channel = shopping_df.loc[cluster, "Dominant_Channel"]
        deal_seeker = shopping_df.loc[cluster, "Deal_Seeker"]
        activity = shopping_df.loc[cluster, "Activity_Status"]
        responsiveness = campaign_df.loc[cluster, "Responsiveness"]

        if "Premium" in spend_label and responsiveness == "Campaign-Responsive":
            name = "Premium Loyal Customers"
        elif "Premium" in spend_label:
            name = "High-Value Low-Engagement"
        elif "Budget" in spend_label and deal_seeker:
            name = "Budget-Conscious Deal Seekers"
        elif "Budget" in spend_label:
            name = "Budget-Conscious Customers"
        else:
            name = "Moderate-Value Customers"

        justification = (
            f"Spending: {spend_label}; dominant channel: {channel}; "
            f"deal-seeking: {deal_seeker}; activity: {activity}; campaign response: {responsiveness}."
        )
        records.append({"Cluster": cluster, "Segment_Name": name, "Justification": justification})
    return pd.DataFrame(records).set_index("Cluster")


# ------------------------------------------------------------
# 9. Customer Personas
# ------------------------------------------------------------
def build_personas(df: pd.DataFrame, labels: np.ndarray, names_df: pd.DataFrame,
                    demo_df: pd.DataFrame, spending_df: pd.DataFrame,
                    shopping_df: pd.DataFrame, campaign_df: pd.DataFrame) -> pd.DataFrame:
    """One data-grounded persona per cluster. CLV is a simple average-historical-spend
    proxy (not a modeled forecast) and is labeled as such."""
    work = df.copy()
    work["Cluster"] = labels
    mnt_labels = {
        "MntWines": "Wine", "MntFruits": "Fruits", "MntMeatProducts": "Meat",
        "MntFishProducts": "Fish", "MntSweetProducts": "Sweets", "MntGoldProds": "Gold",
    }
    records = []
    for cluster in names_df.index:
        age_mean = work.loc[work["Cluster"] == cluster, "Age"].mean()
        income_mean = demo_df.loc[cluster, "Income"]
        top_product_col = spending_df.loc[cluster, SPENDING_COLS].idxmax()
        records.append({
            "Cluster": cluster,
            "Persona_Name": names_df.loc[cluster, "Segment_Name"],
            "Age_Profile": f"{'Above' if age_mean > 0 else 'Below'}-average age "
                            f"(standardized mean z={age_mean:.2f}; raw years not recoverable "
                            "since ID was dropped before scaling in Module 5)",
            "Income_Level": f"{income_mean:.2f} (standardized mean; higher = above dataset average)",
            "Family_Status": f"Avg family size {demo_df.loc[cluster, 'Family_Size']:.2f}, "
                              f"avg children {demo_df.loc[cluster, 'Total_Children']:.2f} (standardized means)",
            "Shopping_Habits": f"Dominant channel: {shopping_df.loc[cluster, 'Dominant_Channel']}, "
                                f"activity: {shopping_df.loc[cluster, 'Activity_Status']}",
            "Preferred_Product": mnt_labels[top_product_col],
            "Preferred_Channel": shopping_df.loc[cluster, "Dominant_Channel"],
            "Marketing_Responsiveness": campaign_df.loc[cluster, "Responsiveness"],
            "Customer_Challenges": (
                "Low campaign responsiveness" if "Resistant" in campaign_df.loc[cluster, "Responsiveness"]
                else "None flagged in available campaign data"
            ),
            "Recommended_Marketing_Strategy": (
                "Loyalty and retention offers on preferred product/channel"
                if "Premium" in names_df.loc[cluster, "Segment_Name"]
                else "Value-driven, deal-based re-engagement campaigns"
            ),
            "Estimated_CLV_Proxy": f"{spending_df.loc[cluster, 'Total_Spending']:.2f} "
                                    f"(avg. historical Total_Spending; not a forecasted CLV model)",
        })
    return pd.DataFrame(records).set_index("Cluster")


# ------------------------------------------------------------
# 10. Visualizations
# ------------------------------------------------------------
def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_distribution(labels: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots()
    sns.countplot(x=labels, ax=ax, hue=labels, palette="tab10", legend=False)
    ax.set_xlabel("Cluster")
    ax.set_title("Cluster Distribution")
    _save_fig(fig, save_path)


def plot_income_comparison(df: pd.DataFrame, labels: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots()
    sns.boxplot(x=labels, y=df["Income"], ax=ax, hue=labels, palette="tab10", legend=False)
    ax.set_xlabel("Cluster")
    ax.set_title("Income Comparison by Cluster")
    _save_fig(fig, save_path)


def plot_spending_comparison(spending_df: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots()
    spending_df["Total_Spending"].plot.bar(ax=ax, color="steelblue")
    ax.set_ylabel("Avg Total Spending")
    ax.set_title("Spending Comparison by Cluster")
    _save_fig(fig, save_path)


def plot_product_heatmap(spending_df: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(spending_df[SPENDING_COLS], annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
    ax.set_title("Product Preference Heatmap by Cluster")
    _save_fig(fig, save_path)


def plot_purchase_channels(shopping_df: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots()
    shopping_df[CHANNEL_COLS].plot.bar(ax=ax)
    ax.set_ylabel("Avg Purchases")
    ax.set_title("Purchase Channels by Cluster")
    _save_fig(fig, save_path)


def plot_campaign_response(campaign_df: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots()
    campaign_df[[c for c in CAMPAIGN_COLS + ["Response"] if c in campaign_df.columns]].plot.bar(ax=ax)
    ax.set_ylabel("Acceptance Rate")
    ax.set_title("Campaign Response by Cluster")
    _save_fig(fig, save_path)


def plot_recency_comparison(df: pd.DataFrame, labels: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots()
    sns.boxplot(x=labels, y=df["Recency"], ax=ax, hue=labels, palette="tab10", legend=False)
    ax.set_xlabel("Cluster")
    ax.set_title("Recency Comparison by Cluster")
    _save_fig(fig, save_path)


def plot_radar(stats_df: pd.DataFrame, features: list[str], save_path: Path) -> None:
    normalized = (stats_df[features] - stats_df[features].min()) / (
        stats_df[features].max() - stats_df[features].min())
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    for cluster, row in normalized.iterrows():
        values = row.tolist() + row.tolist()[:1]
        ax.plot(angles, values, label=f"Cluster {cluster}")
        ax.fill(angles, values, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_title("Cluster Radar Chart")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    _save_fig(fig, save_path)


def reuse_day6_figure(name_in_day6: str, name_in_day7: Path) -> Path:
    """Copy an already-generated Day-6 figure (e.g. PCA/t-SNE) instead of regenerating it."""
    src = DAY6_FIGURES / name_in_day6
    name_in_day7.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, name_in_day7)
    return name_in_day7


# ------------------------------------------------------------
# 11-12. Segment Comparison & Final Export
# ------------------------------------------------------------
def segment_comparison(names_df, demo_df, spending_df, shopping_df, campaign_df) -> pd.DataFrame:
    """Final side-by-side comparison table across all analyzed dimensions."""
    rows = []
    for cluster in names_df.index:
        rows.append({
            "Cluster": cluster,
            "Segment_Name": names_df.loc[cluster, "Segment_Name"],
            "Avg_Income": demo_df.loc[cluster, "Income"],
            "Avg_Total_Spending": spending_df.loc[cluster, "Total_Spending"],
            "Avg_Total_Purchases": shopping_df.loc[cluster, CHANNEL_COLS + [DEAL_COL]].sum(),
            "Engagement_Status": shopping_df.loc[cluster, "Activity_Status"],
            "Campaign_Response": campaign_df.loc[cluster, "Responsiveness"],
            "Preferred_Channel": shopping_df.loc[cluster, "Dominant_Channel"],
            "Business_Value": spending_df.loc[cluster, "Segment_Label"],
            "Marketing_Strategy": (
                "Loyalty and retention offers" if "Premium" in spending_df.loc[cluster, "Segment_Label"]
                else "Value-driven, deal-based campaigns"
            ),
        })
    return pd.DataFrame(rows).set_index("Cluster")


def save_report(df: pd.DataFrame, name: str, reports_dir: Path = REPORTS_DIR) -> Path:
    """Save a DataFrame to outputs/reports as an Excel file."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{name}.xlsx"
    df.to_excel(path, index=True)
    return path
