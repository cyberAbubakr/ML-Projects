# Dashboard Planning Document — Day 9

## 1. Data Source
- **Display data:** `06-Customer-Personality-Analysis/03_Cleaned_Data/customer_personality_cleaned.csv`
  (2,237 customers, human-readable raw values — Income, Age, Education, Marital Status, etc.)
- **Model input data:** `06-Customer-Personality-Analysis/03_Cleaned_Data/customer_personality_features.csv`
  (same 2,237 rows, same order, already engineered + scaled — verified 1:1 row alignment
  with the cleaned file, correlation ≈ 1.0 on shared raw fields such as Income/Age).
- Neither file is copied into Day-9; the app reads them directly from Module 6's folder.

## 2. Feature Source
- Exact final feature list loaded from `06-Customer-Personality-Analysis/outputs/models/selected_features.pkl`
  — 44 features (demographic, spending, purchase-channel, engineered ratios, and
  one-hot encoded categorical fields).
- The same list is also bundled inside `pipeline.pkl`; both are read as-is, never regenerated.

## 3. Model Source
- Final selected model: **KMeans, `n_clusters=2`, `random_state=42`**, saved at
  `06-Customer-Personality-Analysis/outputs/models/kmeans.pkl`.
- Selection is confirmed by Module 6's `algorithm_comparison.xlsx`, where KMeans (k=2)
  was compared against Hierarchical, GMM, and DBSCAN on silhouette / Davies-Bouldin scores.
- Verified: running the saved model's `.predict()` on `customer_personality_features.csv`
  reproduces the exact cluster sizes reported in Module 7/8 (Cluster 0 = 1,049, Cluster 1 = 1,188).
- The app only calls `.predict()` on existing rows — it never calls `.fit()`.

## 4. Segment Source
- Business segment names and justifications loaded from Module 7's
  `outputs/reports/cluster_names.xlsx`:
  - **Cluster 0 → "Budget-Conscious Deal Seekers"**
  - **Cluster 1 → "Premium Loyal Customers"**
- Full per-segment profiles (income, spending, channel, campaign response, etc.) loaded
  from Module 7's `final_customer_profiles.xlsx` and `customer_personas.xlsx`.

## 5. Business-Insight Source
- All business insights and marketing recommendations loaded from Module 8's
  `outputs/reports/*.xlsx`: `Business_Opportunity_Analysis`, `Customer_Segment_Summary`,
  `Marketing_Strategy`, `Marketing_Strategy_Matrix`, `Product_Recommendation`,
  `Pricing_Strategy`, `Retention_Strategy`, `Reactivation_Strategy`, `Campaign_Action_Plan`,
  `Customer_Personas`.
- Module 8's own notebook confirms these build entirely on Module 7's saved outputs with
  no re-clustering, and that the project has **2 real segments** (not 4), since Module 6
  selected k=2 via silhouette score.

## 6. Dashboard Pages
1. Home / Overview
2. Customer Segmentation
3. Segment Profiles
4. Business Insights
5. Marketing Recommendations
6. About / Methodology

## 7. Navigation Structure
A single-page Streamlit app with a **sidebar radio-button navigation** (`st.sidebar.radio`)
switching between the six pages above. No multi-page routing framework needed given the
small page count — kept lightweight per the resource constraint.

## 8. Application Technology
**Streamlit + Python**, using only libraries already present in the Module 6-8 project
(`pandas`, `scikit-learn`, `joblib`, `openpyxl`) plus `streamlit` itself.

## 9. Required Dependencies
```
streamlit>=1.61
pandas>=2.0
scikit-learn>=1.9
joblib>=1.5
openpyxl>=3.1
```

## Design Notes
- `st.cache_resource` is used for the model/scaler/feature-list (loaded once, reused).
- `st.cache_data` is used for all CSV/Excel reads (loaded once, reused).
- The app assumes it sits alongside the module folders:
  ```
  <project_root>/
      06-Customer-Personality-Analysis/
      07-Customer-Segmentation-Clustering/
      08-Marketing-Business-Insights/
      Day-9/          <- this app
  ```
- No datasets, models, or reports are duplicated into Day-9; all are referenced by
  relative path from `src/config.py`.
