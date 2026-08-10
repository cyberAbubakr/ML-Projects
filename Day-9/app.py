"""
Day-9 — Customer Segmentation & Marketing Insights Dashboard

Connects the saved Module 6 clustering pipeline with the Module 7 segment
profiles and Module 8 business insights / marketing recommendations.

This app performs NO retraining and NO recomputation of clustering,
features, or business analysis. It only loads and displays existing,
already-submitted project artifacts.
"""

import pandas as pd
import streamlit as st

from src import config, data_loader, segment_utils

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
)

PAGES = [
    "Home / Overview",
    "Customer Segmentation",
    "Segment Profiles",
    "Business Insights",
    "Marketing Recommendations",
    "About / Methodology",
]


def sidebar_navigation() -> str:
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", PAGES, label_visibility="collapsed")
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Data, features, model, segments, and recommendations are loaded "
        "from the existing Module 6-8 outputs. Nothing is retrained."
    )
    return choice


def show_missing_artifacts_warning():
    status = data_loader.check_artifacts_exist()
    required_missing = status["required"]
    optional_missing = status["optional"]

    if optional_missing:
        with st.sidebar.expander("⚠️ Optional artifacts missing", expanded=False):
            for label, path in optional_missing:
                st.write(f"- **{label}** — expected at `{path}`")

    if required_missing:
        st.error(
            "Required Module 6/7/8 artifacts were not found. This app must "
            "sit alongside the `06-Customer-Personality-Analysis`, "
            "`07-Customer-Segmentation-Clustering`, and "
            "`08-Marketing-Business-Insights` folders."
        )
        for label, path in required_missing:
            st.write(f"- **{label}** — expected at `{path}`")
        st.stop()


# ----------------------------------------------------------------------
# PAGE: Home / Overview
# ----------------------------------------------------------------------
def page_home():
    st.title(config.PROJECT_TITLE)
    st.write(config.PROJECT_DESCRIPTION)

    cleaned_df = data_loader.load_cleaned_customers()
    segment_summary = data_loader.load_segment_summary()

    if cleaned_df is not None:
        total_customers = len(cleaned_df)
    else:
        # Fall back to Module 8's segment summary counts if the raw cleaned
        # CSV isn't available, rather than fabricating a number.
        total_customers = int(segment_summary["Customers"].sum())

    n_segments = segment_summary["Segment"].nunique()

    pipeline = data_loader.load_production_pipeline()
    model = pipeline["model"] if pipeline else data_loader.load_kmeans_model()
    model_label = segment_utils.get_model_display_name(model)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Customer Segments", n_segments)
    col3.metric("Final Model", model_label)

    st.markdown("### Segment Split")
    st.dataframe(segment_summary, use_container_width=True, hide_index=True)

    st.markdown("### Key Business Insight")
    opp_df = data_loader.load_business_opportunity()
    for _, row in opp_df.iterrows():
        st.markdown(f"**{row['Segment']}** — {row['Business_Value']}")
        st.caption(f"Opportunity: {row['Business_Opportunity']}")
        st.caption(f"Risk: {row['Business_Risk']}")
        st.markdown("")


# ----------------------------------------------------------------------
# PAGE: Customer Segmentation
# ----------------------------------------------------------------------
def page_segmentation():
    st.title("Customer Segmentation")
    st.write(
        "Select an existing customer to see their record and the segment "
        "already assigned by the saved production pipeline."
    )

    # Preferred source for customer-level display: the raw cleaned CSV.
    cleaned_df = data_loader.load_cleaned_customers()
    if cleaned_df is None:
        st.warning(
            "The raw cleaned customer dataset "
            "(`customer_personality_cleaned.csv`) is not available, so the "
            "customer-level segmentation view cannot be shown. No customer "
            "data is fabricated. Other pages that rely only on Module 7/8 "
            "reports remain available."
        )
        return

    # Model input: the engineered + already-scaled feature table. This is
    # NEVER treated as raw customer data — it is only used to feed the model.
    features_df = data_loader.load_engineered_features()
    if features_df is None:
        st.warning(
            "The engineered feature dataset "
            "(`customer_personality_features.csv`) is not available, so a "
            "segment cannot be predicted for a selected customer. Showing "
            "customer data only."
        )

    selected_features = data_loader.load_selected_features()
    cluster_names_df = data_loader.load_cluster_names()
    profiles_df = data_loader.load_final_customer_profiles()
    name_map = segment_utils.get_cluster_name_map(cluster_names_df)

    # Use the single saved production pipeline bundle whenever possible.
    pipeline = data_loader.load_production_pipeline()
    model = pipeline["model"] if pipeline else data_loader.load_kmeans_model()

    customer_ids = cleaned_df["ID"].tolist()
    selected_id = st.selectbox("Customer ID", customer_ids)

    row_index = cleaned_df.index[cleaned_df["ID"] == selected_id][0]
    raw_row = cleaned_df.loc[row_index]

    left, right = st.columns([1, 1])

    with left:
        st.markdown("#### Customer Data (from cleaned dataset)")
        display_cols = [
            "ID", "Age", "Education", "Marital_Status", "Income",
            "Kidhome", "Teenhome", "Recency", "Total_Spending",
            "Total_Purchases",
        ]
        display_cols = [c for c in display_cols if c in raw_row.index]
        st.table(raw_row[display_cols])

    if features_df is None:
        return

    feature_row = features_df.loc[[row_index], selected_features]

    with right:
        st.markdown("#### Model Input (engineered, already scaled)")
        st.caption(
            f"{len(selected_features)} engineered features from the saved "
            "production feature set — this is model input, not raw "
            "customer data, and is already scaled by Module 6's saved "
            "scaler (not re-scaled here)."
        )
        st.dataframe(feature_row.T.rename(columns={feature_row.index[0]: "value"}))

    st.markdown("#### Segmentation Model")
    st.write(f"**Algorithm:** {segment_utils.get_model_display_name(model)} "
             "(Module 6 final selection, loaded from the saved production pipeline)")

    # Already-scaled input -> saved production model -> cluster ID.
    predicted_cluster = segment_utils.predict_segment_for_row(model, feature_row)
    predicted_name = name_map.get(predicted_cluster, f"Cluster {predicted_cluster}")

    st.success(f"**Assigned Segment:** {predicted_name} (Cluster {predicted_cluster})")

    seg_row = segment_utils.get_segment_row(profiles_df, predicted_cluster)
    if seg_row is not None:
        st.markdown("#### Segment Description")
        st.write(seg_row["Justification"])


# ----------------------------------------------------------------------
# PAGE: Segment Profiles
# ----------------------------------------------------------------------
def page_segment_profiles():
    st.title("Segment Profiles")
    st.write("Profiles as established in Module 7's cluster profiling.")

    segment_summary = data_loader.load_segment_summary()
    profiles_df = data_loader.load_final_customer_profiles()
    personas_df = data_loader.load_customer_personas()

    for _, seg in segment_summary.iterrows():
        cluster_id = seg["Segment"]
        # "Segment" column holds values like "Cluster 0"
        cluster_num = int(str(cluster_id).split()[-1])
        profile = segment_utils.get_segment_row(profiles_df, cluster_num)
        persona = personas_df[personas_df["Persona_Name"] == seg["Business_Name"]]
        persona_row = persona.iloc[0] if not persona.empty else None

        with st.expander(f"{seg['Business_Name']} — {seg['Customers']:,} customers ({seg['Percentage']})", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Income & Spending**")
                if profile is not None:
                    st.write(f"Income (standardized mean): {profile['Income']:.2f}")
                    st.write(f"Total Spending (standardized mean): {profile['Total_Spending']:.2f}")
                    st.write(f"Preferred Product: {profile.get('Preferred_Product', 'N/A')}")
                st.markdown("**Purchase Behavior**")
                if persona_row is not None:
                    st.write(f"Purchase Frequency: {persona_row['Purchase_Frequency']}")
                    st.write(f"Deal Dependency: {persona_row['Deal_Dependency_Level']}")

            with col2:
                st.markdown("**Shopping Channel & Campaign Response**")
                if profile is not None:
                    st.write(f"Preferred Channel: {profile.get('Preferred_Channel', 'N/A')}")
                    st.write(f"Marketing Responsiveness: {profile.get('Marketing_Responsiveness', 'N/A')}")
                st.markdown("**Risk / Value**")
                if persona_row is not None:
                    st.write(f"Activity Level: {persona_row['Activity_Level']}")
                    st.write(f"Risk Level: {persona_row['Risk_Level']}")
                    st.write(f"Customer Lifetime Value: {persona_row['Customer_Lifetime_Value']}")

            if profile is not None:
                st.caption(profile["Justification"])


# ----------------------------------------------------------------------
# PAGE: Business Insights
# ----------------------------------------------------------------------
def page_business_insights():
    st.title("Business Insights")
    st.write("Findings established in Module 8's business insights analysis.")

    opp_df = data_loader.load_business_opportunity()

    st.markdown("### High-Value & At-Risk Segments")
    st.dataframe(opp_df, use_container_width=True, hide_index=True)

    st.markdown("### Segment Details")
    for _, row in opp_df.iterrows():
        st.markdown(f"**{row['Segment']}**")
        st.write(f"- Business value: {row['Business_Value']}")
        st.write(f"- Opportunity: {row['Business_Opportunity']}")
        st.write(f"- Risk: {row['Business_Risk']}")
        st.write(f"- Marketing objective: {row['Marketing_Objective']}")
        st.markdown("")

    if config.DASHBOARD_SUMMARY_IMAGE_PATH.exists():
        st.markdown("### Module 8 Business Dashboard")
        st.image(str(config.DASHBOARD_SUMMARY_IMAGE_PATH), use_container_width=True)


# ----------------------------------------------------------------------
# PAGE: Marketing Recommendations
# ----------------------------------------------------------------------
def page_marketing_recommendations():
    st.title("Marketing Recommendations")
    st.write("Recommendations already produced in Module 8. Nothing here is generated by this app.")

    strategy_matrix = data_loader.load_marketing_strategy_matrix()
    pricing_df = data_loader.load_pricing_strategy()
    retention_df = data_loader.load_retention_strategy()
    reactivation_df = data_loader.load_reactivation_strategy()
    product_df = data_loader.load_product_recommendation()
    campaign_df = data_loader.load_campaign_action_plan()

    st.markdown("### Marketing Message & Channel Strategy")
    st.dataframe(
        strategy_matrix[["Segment", "Marketing_Message", "Communication_Channel"]],
        use_container_width=True, hide_index=True,
    )

    st.markdown("### Product Recommendations")
    st.dataframe(product_df, use_container_width=True, hide_index=True)

    st.markdown("### Discount / Pricing Strategy")
    st.dataframe(pricing_df, use_container_width=True, hide_index=True)

    st.markdown("### Retention Strategy")
    st.dataframe(retention_df, use_container_width=True, hide_index=True)

    st.markdown("### Reactivation Strategy")
    st.dataframe(reactivation_df, use_container_width=True, hide_index=True)

    st.markdown("### Cross-Selling & Upselling")
    st.dataframe(
        strategy_matrix[["Segment", "Cross_Selling", "Upselling"]],
        use_container_width=True, hide_index=True,
    )

    st.markdown("### Campaign Action Plan")
    st.dataframe(campaign_df, use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------
# PAGE: About / Methodology
# ----------------------------------------------------------------------
def page_about():
    st.title("About / Methodology")

    selected_features = data_loader.load_selected_features()

    st.markdown(f"""
**Dataset:** Kaggle "Customer Personality Analysis" marketing campaign
dataset, cleaned in Module 6 (`06-Customer-Personality-Analysis`).

**Feature Engineering:** Module 6 engineered features such as `Age`,
`Total_Spending`, `Total_Purchases`, `Customer_Tenure`, `Family_Size`,
`Deal_Dependency`, `Digital_Engagement`, and one-hot encodings for
Education, Marital Status, Preferred Shopping Channel, and Product
Preference — {len(selected_features)} features in total, saved as the
project's `selected_features.pkl`.

**Scaling:** Numerical features were standardized with a fitted
`StandardScaler` (Module 6), saved as `standard_scaler.pkl`.
""")

    st.markdown("**Clustering & Model Evaluation (Module 6):**")
    comparison_df = data_loader.load_algorithm_comparison()
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    pipeline = data_loader.load_production_pipeline()
    model = pipeline["model"] if pipeline else data_loader.load_kmeans_model()
    model_label = segment_utils.get_model_display_name(model)
    st.caption(
        f"{model_label} was selected as the final model based on this "
        "algorithm comparison (silhouette / Davies-Bouldin scores)."
    )

    st.markdown(f"""
**Final Model Selection:** {model_label}, saved as `kmeans.pkl` and bundled
with its scaler and feature list in the single production pipeline file
`pipeline.pkl` (the preferred artifact this app uses for prediction).

**Business Interpretation:** Module 7 (`07-Customer-Segmentation-Clustering`)
assigned business-friendly names and profiles to each cluster. Module 8
(`08-Marketing-Business-Insights`) translated those profiles into business
opportunities, risks, and marketing recommendations — all of which this
app displays as-is.
""")


def main():
    show_missing_artifacts_warning()
    choice = sidebar_navigation()

    if choice == "Home / Overview":
        page_home()
    elif choice == "Customer Segmentation":
        page_segmentation()
    elif choice == "Segment Profiles":
        page_segment_profiles()
    elif choice == "Business Insights":
        page_business_insights()
    elif choice == "Marketing Recommendations":
        page_marketing_recommendations()
    elif choice == "About / Methodology":
        page_about()


if __name__ == "__main__":
    main()
