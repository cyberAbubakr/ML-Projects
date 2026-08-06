"""business_insights.py — Module 8: Business Insights & Marketing Recommendations.

Builds entirely on the Day-7 cluster profiling outputs (segment stats, personas,
demographic/spending/shopping/campaign reports). No clustering, feature engineering,
or customer profiling is recomputed here — this module only reads Day-7's saved
Excel reports and translates them into business/marketing strategy.

All quantitative fields (counts, percentages, z-scored comparisons) are taken
directly from Day-7. Strategic fields (messaging, discount tiers, campaign
timing, etc.) are business recommendations derived from those real findings
and are labeled as recommendations, not measured outcomes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

PROJECT_DIR = Path.cwd()
DAY7_DIR = PROJECT_DIR.parent / "Day-7"
DAY7_REPORTS = DAY7_DIR / "outputs" / "reports"

REPORTS_DIR = PROJECT_DIR / "outputs" / "reports"
FIGURES_DIR = PROJECT_DIR / "outputs" / "figures"
DASHBOARD_DIR = PROJECT_DIR / "outputs" / "dashboard"

SPENDING_COLS = ["MntWines", "MntFruits", "MntMeatProducts",
                  "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
CAMPAIGN_COLS = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]


# ------------------------------------------------------------
# Load Day-7 outputs (source of truth — nothing recomputed)
# ------------------------------------------------------------
def load_day7_outputs() -> dict:
    """Load every Day-7 report needed for Module 8, keyed by report name."""
    names = ["cluster_names", "customer_personas", "spending_report", "shopping_behavior",
              "campaign_analysis", "segment_comparison", "demographic_report", "cluster_statistics"]
    return {name: pd.read_excel(DAY7_REPORTS / f"{name}.xlsx", index_col=0) for name in names}


# ------------------------------------------------------------
# Task 1: Customer Segment Summary
# ------------------------------------------------------------
def segment_summary(data: dict) -> pd.DataFrame:
    """Segment, business name, customer count, and percentage of the customer base."""
    names = data["cluster_names"]
    counts = data["spending_report"]["Customer_Count"]
    total = counts.sum()
    df = pd.DataFrame({
        "Segment": [f"Cluster {i}" for i in names.index],
        "Business_Name": names["Segment_Name"],
        "Customers": counts,
        "Percentage": (counts / total * 100).round(1).astype(str) + "%",
    })
    return df.reset_index(drop=True)


# ------------------------------------------------------------
# Task 2: Customer Personas (extended with Purchase Frequency, Risk, CLV)
# ------------------------------------------------------------
def build_personas(data: dict) -> pd.DataFrame:
    """Extend the Day-7 personas with purchase frequency, risk level, and a
    qualitative CLV tier — all derived from real Day-7 fields, not invented."""
    personas = data["customer_personas"].copy()
    shop = data["shopping_behavior"]
    camp = data["campaign_analysis"]
    spend = data["spending_report"]

    total_purchases = data["cluster_statistics"]["Total_Purchases"]
    personas["Purchase_Frequency"] = np.where(
        total_purchases == total_purchases.max(), "High (above-average Total_Purchases)",
        "Low (below-average Total_Purchases)"
    )
    personas["Deal_Dependency_Level"] = np.where(
        shop["Deal_Dependency"] == shop["Deal_Dependency"].max(),
        "High — deal-driven purchasing", "Low — price-insensitive purchasing"
    )
    personas["Activity_Level"] = shop["Activity_Status"]

    # Risk level: inactive + marketing-resistant = higher churn risk signal
    is_inactive = shop["Activity_Status"].str.contains("Inactive")
    is_resistant = camp["Responsiveness"].str.contains("Resistant")
    risk = pd.Series("Moderate", index=personas.index)
    risk[is_inactive & is_resistant] = "High (inactive and marketing-resistant)"
    risk[~is_inactive & ~is_resistant] = "Low (active and campaign-responsive)"
    personas["Risk_Level"] = risk

    personas["Customer_Lifetime_Value"] = np.where(
        spend["Total_Spending"] == spend["Total_Spending"].max(),
        "Qualitatively high — highest average historical spend of the two segments",
        "Qualitatively low — lowest average historical spend of the two segments"
    )
    return personas


# ------------------------------------------------------------
# Task 3: Business Opportunity Analysis
# ------------------------------------------------------------
def business_opportunity_analysis(data: dict, personas: pd.DataFrame) -> pd.DataFrame:
    """Business value, opportunity, risk, and marketing objective per segment."""
    names = data["cluster_names"]
    spend = data["spending_report"]
    camp = data["campaign_analysis"]
    records = []
    for c in names.index:
        is_premium = "Premium" in spend.loc[c, "Segment_Label"]
        records.append({
            "Segment": names.loc[c, "Segment_Name"],
            "Business_Value": spend.loc[c, "Segment_Label"],
            "Business_Opportunity": (
                "Upsell/cross-sell premium and bundled products; strengthen loyalty to protect "
                "high-margin revenue" if is_premium else
                "Convert deal-driven engagement into higher-margin repeat purchases via targeted offers"
            ),
            "Business_Risk": (
                "Higher Recency (less recent activity) despite high value — risk of quiet disengagement"
                if personas.loc[c, "Activity_Level"].startswith("Inactive") else
                "Low average spend and high deal-dependency — vulnerable to competitor discounting"
            ),
            "Marketing_Objective": (
                "Retain and deepen loyalty among high-value customers" if is_premium else
                "Increase purchase frequency and average order value"
            ),
        })
    return pd.DataFrame(records)


# ------------------------------------------------------------
# Task 4: Marketing Strategy
# ------------------------------------------------------------
def marketing_strategy(data: dict, personas: pd.DataFrame) -> pd.DataFrame:
    """Message, tone, channel, timing, frequency, and offer per segment."""
    names = data["cluster_names"]
    shop = data["shopping_behavior"]
    records = []
    for c in names.index:
        is_premium = "Premium" in data["spending_report"].loc[c, "Segment_Label"]
        channel = shop.loc[c, "Dominant_Channel"].replace("Num", "").replace("Purchases", "")
        records.append({
            "Segment": names.loc[c, "Segment_Name"],
            "Marketing_Message": (
                "An exclusive collection, curated for our most valued customers"
                if is_premium else "Real savings on the products you already love"
            ),
            "Tone": "Aspirational, appreciative" if is_premium else "Friendly, value-focused",
            "Communication_Channel": channel,
            "Campaign_Timing": (
                "Aligned to Recency pattern — send shortly after typical repurchase window "
                "given this segment's higher Recency" if personas.loc[c, "Activity_Level"].startswith("Inactive")
                else "Regular cadence, since this segment purchases most recently/actively"
            ),
            "Campaign_Frequency": "Bi-weekly" if is_premium else "Weekly",
            "Personalized_Offer": (
                "Early access to premium/new arrivals, loyalty-tier perks" if is_premium else
                "Stackable discount codes and deal bundles on top spending categories"
            ),
        })
    return pd.DataFrame(records)


# ------------------------------------------------------------
# Task 5: Product Recommendation Strategy
# ------------------------------------------------------------
def product_recommendation(data: dict) -> pd.DataFrame:
    """Primary/secondary/cross-sell/upsell/bundle recommendations from real spending ranks."""
    stats = data["cluster_statistics"]
    names = data["cluster_names"]
    label_map = {"MntWines": "Wine", "MntFruits": "Fruits", "MntMeatProducts": "Meat",
                 "MntFishProducts": "Fish", "MntSweetProducts": "Sweets", "MntGoldProds": "Gold"}
    records = []
    for c in names.index:
        ranked = stats.loc[c, SPENDING_COLS].sort_values(ascending=False)
        top2 = [label_map[x] for x in ranked.index[:2]]
        bottom2 = [label_map[x] for x in ranked.index[-2:]]
        records.append({
            "Segment": names.loc[c, "Segment_Name"],
            "Primary_Product": top2[0],
            "Secondary_Product": top2[1],
            "Cross_Selling": f"{top2[0]} buyers \u2192 introduce {ranked.index[2] and label_map[ranked.index[2]]}",
            "Upselling": f"Premium/larger-format {top2[0]} offerings",
            "Bundle_Recommendation": f"{top2[0]} + {top2[1]} bundle (this segment's two highest categories)",
            "Lowest_Engagement_Categories": ", ".join(bottom2),
        })
    return pd.DataFrame(records)


# ------------------------------------------------------------
# Task 6: Pricing and Discount Strategy
# ------------------------------------------------------------
def pricing_strategy(data: dict, personas: pd.DataFrame) -> pd.DataFrame:
    """Discount tiers driven directly by each segment's real Deal_Dependency finding."""
    names = data["cluster_names"]
    records = []
    for c in names.index:
        high_deal = "High" in personas.loc[c, "Deal_Dependency_Level"]
        records.append({
            "Segment": names.loc[c, "Segment_Name"],
            "Discount_Policy": "15-20% deal-driven discounts" if high_deal else "5-10% loyalty-only discounts",
            "Coupon_Strategy": "Stackable, frequent coupon codes" if high_deal else "Occasional, exclusive coupon codes",
            "Promotional_Offers": "Deal bundles, limited-time offers" if high_deal else "Early-access previews, VIP previews",
            "Seasonal_Offers": "Major seasonal sales (high price sensitivity)" if high_deal
                                else "Curated seasonal gift/premium collections",
            "Premium_Pricing_Opportunity": "Limited — segment is price-sensitive" if high_deal
                                            else "Viable — segment shows low deal-dependency and high spend",
        })
    return pd.DataFrame(records)


# ------------------------------------------------------------
# Task 7: Customer Retention Strategy
# ------------------------------------------------------------
def retention_strategy(data: dict, personas: pd.DataFrame) -> pd.DataFrame:
    """Loyalty/referral/support recommendations from real activity + responsiveness."""
    names = data["cluster_names"]
    records = []
    for c in names.index:
        is_premium = "Premium" in data["spending_report"].loc[c, "Segment_Label"]
        records.append({
            "Segment": names.loc[c, "Segment_Name"],
            "Loyalty_Program": "Tiered VIP loyalty program with premium perks" if is_premium
                                else "Points-based rewards on deal purchases",
            "Referral_Incentive": "High-value referral credit (protects high-CLV segment)" if is_premium
                                    else "Small discount-based referral incentive",
            "Membership_Plan": "Premium membership with early access" if is_premium else "Free tier with deal alerts",
            "Follow_Up_Frequency": "Bi-weekly relationship touchpoints" if is_premium else "Weekly value-offer touchpoints",
            "Customer_Support_Priority": "High priority / dedicated support" if is_premium else "Standard priority",
            "Retention_Campaign_Focus": (
                "Re-engagement given this segment's higher Recency despite high value"
                if personas.loc[c, "Activity_Level"].startswith("Inactive")
                else "Reinforce habitual engagement to sustain low Recency"
            ),
        })
    return pd.DataFrame(records)


# ------------------------------------------------------------
# Task 8: Customer Reactivation Strategy
# ------------------------------------------------------------
def reactivation_strategy(data: dict, personas: pd.DataFrame) -> pd.DataFrame:
    """Win-back plan, targeted specifically at the segment(s) flagged Inactive in Day-7."""
    names = data["cluster_names"]
    shop = data["shopping_behavior"]
    records = []
    for c in names.index:
        needs_reactivation = shop.loc[c, "Activity_Status"].startswith("Inactive")
        records.append({
            "Segment": names.loc[c, "Segment_Name"],
            "Needs_Reactivation": "Yes" if needs_reactivation else "No (currently the more active segment)",
            "Win_Back_Offer": "High-value win-back incentive (premium discount or gift with purchase)"
                                if needs_reactivation else "Not applicable",
            "Reminder_Campaign": "\"We miss you\" email/SMS referencing past premium purchases"
                                    if needs_reactivation else "Not applicable",
            "Personalized_Recommendation": "Recommend this segment's top historical category (see Task 5)"
                                             if needs_reactivation else "Not applicable",
            "Reactivation_Timeline": "Trigger campaign once Recency exceeds the segment's own average"
                                       if needs_reactivation else "Not applicable",
        })
    return pd.DataFrame(records)


# ------------------------------------------------------------
# Task 9: Campaign Action Plan
# ------------------------------------------------------------
def campaign_action_plan(data: dict, personas: pd.DataFrame) -> pd.DataFrame:
    """Concrete campaign table per segment, objective, channel, and KPI."""
    names = data["cluster_names"]
    shop = data["shopping_behavior"]
    records = []
    for c in names.index:
        is_premium = "Premium" in data["spending_report"].loc[c, "Segment_Label"]
        channel = shop.loc[c, "Dominant_Channel"].replace("Num", "").replace("Purchases", "")
        if is_premium:
            records.append({
                "Campaign": "VIP Loyalty Rewards", "Target_Segment": names.loc[c, "Segment_Name"],
                "Objective": "Increase loyalty and repeat purchases", "Channel": channel,
                "Budget_Priority": "High", "KPI": "Repeat Purchase Rate, Retention Rate",
                "Expected_Outcome": "Sustained high spend and reduced Recency drift",
            })
        else:
            records.append({
                "Campaign": "Value Discount Drive", "Target_Segment": names.loc[c, "Segment_Name"],
                "Objective": "Increase purchase frequency and conversion", "Channel": channel,
                "Budget_Priority": "Medium", "KPI": "Conversion Rate, Deal Redemption Rate",
                "Expected_Outcome": "Higher Total_Purchases and campaign acceptance",
            })
        if shop.loc[c, "Activity_Status"].startswith("Inactive"):
            records.append({
                "Campaign": "Win-Back Re-engagement", "Target_Segment": names.loc[c, "Segment_Name"],
                "Objective": "Reduce Recency / reactivate lapsed high-value customers", "Channel": "Email",
                "Budget_Priority": "High", "KPI": "Reactivation Rate, Recency Reduction",
                "Expected_Outcome": "Return to regular purchase cadence",
            })
    return pd.DataFrame(records)


# ------------------------------------------------------------
# Task 10: Marketing Strategy Matrix (consolidated)
# ------------------------------------------------------------
def marketing_strategy_matrix(names, mkt, prod, price, retention, reactivation) -> pd.DataFrame:
    """One consolidated matrix joining every prior strategy table by segment."""
    matrix = pd.DataFrame({"Segment": names["Segment_Name"].values})
    matrix["Marketing_Message"] = mkt["Marketing_Message"].values
    matrix["Communication_Channel"] = mkt["Communication_Channel"].values
    matrix["Product_Recommendation"] = prod["Primary_Product"].values + " + " + prod["Secondary_Product"].values
    matrix["Pricing_Strategy"] = price["Discount_Policy"].values
    matrix["Loyalty_Strategy"] = retention["Loyalty_Program"].values
    matrix["Cross_Selling"] = prod["Cross_Selling"].values
    matrix["Upselling"] = prod["Upselling"].values
    matrix["Campaign_Frequency"] = mkt["Campaign_Frequency"].values
    matrix["Reactivation_Strategy"] = reactivation["Win_Back_Offer"].values
    return matrix


# ------------------------------------------------------------
# Task 11: Business Dashboard
# ------------------------------------------------------------
def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_customers_per_segment(summary: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.bar(summary["Business_Name"], summary["Customers"], color="steelblue")
    ax.set_ylabel("Customers")
    ax.set_title("Customers per Segment")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _save_fig(fig, save_path)


def plot_value_contribution(data: dict, names: pd.DataFrame, save_path: Path) -> None:
    """Value contribution proxy: share of total historical spending (not modeled revenue)."""
    spend = data["spending_report"]
    counts = spend["Customer_Count"]
    # Un-standardize proxy: relative share using count-weighted spending rank position, labeled clearly.
    weight = spend["Total_Spending"].rank() * counts
    share = (weight / weight.sum() * 100).round(1)
    fig, ax = plt.subplots()
    ax.pie(share, labels=names["Segment_Name"], autopct="%1.1f%%", colors=sns.color_palette("Blues", len(share)))
    ax.set_title("Relative Value Contribution by Segment\n(rank-weighted proxy, not modeled revenue)")
    _save_fig(fig, save_path)


def plot_campaign_priority(action_plan: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots()
    order = {"High": 3, "Medium": 2, "Low": 1}
    action_plan = action_plan.copy()
    action_plan["Priority_Score"] = action_plan["Budget_Priority"].map(order)
    sns.barplot(data=action_plan, x="Campaign", y="Priority_Score", hue="Target_Segment", ax=ax)
    ax.set_ylabel("Budget Priority (1=Low, 3=High)")
    ax.set_title("Campaign Priority by Segment")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _save_fig(fig, save_path)


def plot_churn_risk(personas: pd.DataFrame, names: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots()
    risk_order = {"Low (active and campaign-responsive)": 1, "Moderate": 2,
                  "High (inactive and marketing-resistant)": 3}
    scores = personas["Risk_Level"].map(risk_order)
    ax.bar(names["Segment_Name"], scores, color=["seagreen" if s < 2 else "goldenrod" if s < 3 else "indianred" for s in scores])
    ax.set_ylabel("Risk Level (1=Low, 3=High)")
    ax.set_title("Churn Risk by Segment")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _save_fig(fig, save_path)


def plot_marketing_channels(mkt: pd.DataFrame, names: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.bar(names["Segment_Name"], [1] * len(mkt), color="slateblue")
    for i, ch in enumerate(mkt["Communication_Channel"]):
        ax.text(i, 0.5, ch, ha="center", va="center", color="white", fontweight="bold")
    ax.set_yticks([])
    ax.set_title("Recommended Marketing Channel by Segment")
    _save_fig(fig, save_path)


def plot_product_preferences(data: dict, save_path: Path) -> None:
    stats = data["cluster_statistics"]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(stats[SPENDING_COLS], annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    ax.set_title("Product Preference (standardized spending) by Segment")
    _save_fig(fig, save_path)


def plot_segment_comparison(data: dict, save_path: Path) -> None:
    comp = data["segment_comparison"]
    fig, ax = plt.subplots()
    comp[["Avg_Income", "Avg_Total_Spending", "Avg_Total_Purchases"]].plot.bar(ax=ax)
    ax.set_title("Segment Comparison (standardized values)")
    ax.set_xlabel("Cluster")
    plt.setp(ax.get_xticklabels(), rotation=0)
    _save_fig(fig, save_path)


def build_dashboard_summary(figure_paths: list[Path], save_path: Path) -> None:
    """Combine the individual dashboard figures into one summary sheet image."""
    n = len(figure_paths)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(18, 5 * rows))
    gs = gridspec.GridSpec(rows, cols, figure=fig)
    for i, p in enumerate(figure_paths):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        img = plt.imread(p)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(p.stem.replace("_", " ").title(), fontsize=11)
    fig.suptitle("Module 8 — Business Insights Dashboard Summary", fontsize=16, fontweight="bold")
    _save_fig(fig, save_path)


# ------------------------------------------------------------
# Export helper
# ------------------------------------------------------------
def save_report(df: pd.DataFrame, name: str, reports_dir: Path = REPORTS_DIR) -> Path:
    """Save a DataFrame to outputs/reports as an Excel file."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{name}.xlsx"
    df.to_excel(path, index=False)
    return path
