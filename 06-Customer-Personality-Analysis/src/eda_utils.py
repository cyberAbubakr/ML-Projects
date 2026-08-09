"""eda_utils.py — reusable helpers for Module 4 EDA."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path.cwd()
CLEANED_DATA_PATH = PROJECT_DIR / "03_Cleaned_Data" / "customer_personality_cleaned.csv"
REPORT_DIR = PROJECT_DIR / "05_EDA_Report"
CHART_DIR = REPORT_DIR / "Charts"

SPENDING_COLS = [
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds",
]
PURCHASE_COLS = [
    "NumWebPurchases", "NumStorePurchases",
    "NumCatalogPurchases", "NumDealsPurchases",
]
CAMPAIGN_COLS = [
    "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
    "AcceptedCmp4", "AcceptedCmp5",
]


def load_cleaned_data(path=CLEANED_DATA_PATH):
    """Load the Module 3 cleaned dataset."""
    df = pd.read_csv(path, parse_dates=["Dt_Customer"])
    return df


def save_fig(fig, name, show=True):
    """Save a matplotlib figure and optionally display it."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        CHART_DIR / f"{name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    if show:
        plt.show()

    plt.close(fig)

def save_excel(df, name):
    """Save a DataFrame to 05_EDA_Report as an Excel file."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_excel(REPORT_DIR / f"{name}.xlsx", index=False)


def top_correlations(corr, n=5):
    """Return top-n positive and top-n negative correlation pairs."""
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    pairs = corr.where(mask).stack().dropna().sort_values(ascending=False)
    top_pos = pairs.head(n)
    top_neg = pairs.tail(n).sort_values()
    return top_pos, top_neg


def add_engineered_columns(df):
    """Add Total_Spending and Total_Purchases for analysis (not persisted)."""
    df = df.copy()
    df["Total_Spending"] = df[SPENDING_COLS].sum(axis=1)
    df["Total_Purchases"] = df[PURCHASE_COLS].sum(axis=1)
    return df


def add_segments(df):
    """Return a DataFrame of boolean customer-segment flags."""
    seg = pd.DataFrame(index=df.index)
    seg["High_Value"] = df["Total_Spending"] > df["Total_Spending"].quantile(0.75)
    seg["Low_Value"] = df["Total_Spending"] < df["Total_Spending"].quantile(0.25)
    seg["Frequent_Buyer"] = df["Total_Purchases"] > df["Total_Purchases"].quantile(0.75)
    seg["Campaign_Responder"] = (df[CAMPAIGN_COLS].sum(axis=1) > 0) | (df["Response"] == 1)
    seg["Inactive"] = df["Recency"] > df["Recency"].quantile(0.75)
    seg["Discount_Seeker"] = df["NumDealsPurchases"] > df["NumDealsPurchases"].quantile(0.75)
    return seg
