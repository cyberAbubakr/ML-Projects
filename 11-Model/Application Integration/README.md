# Day 9 — Customer Segmentation & Marketing Insights Dashboard

## What this application does
A Streamlit dashboard that connects the customer-segmentation pipeline built in
Modules 6-8 with the business insights and marketing recommendations already
produced in Module 8. It **loads existing, saved artifacts** — it does not
retrain the clustering model, re-run feature engineering, or recompute any
business analysis.

Pages:
- **Home / Overview** — project summary, KPI cards (customers, segments, final model, top-value segment), a segment-size chart, and a segment-summary download
- **Customer Segmentation** — two modes:
  - *Existing Customer*: filter customers by segment/income/age, search by Customer ID, download the filtered list, pick a customer, view their features, see the segment assigned by the saved model, download that customer's record
  - *New Customer (manual input)*: enter a brand-new customer's raw information through a form; the app converts it into the exact 45-feature structure using Module 6's own feature-engineering code, scores it with the saved production KMeans model, and displays the predicted segment plus the matching Module 7 segment description and Module 8 recommendations
- **Segment Profiles** — a segment comparison chart, a segment selector ("All Segments" or one), full profile (income, spending, channel, campaign response, risk) per segment, and a profiles download
- **Business Insights** — a segment-size chart, high-value / at-risk segments, a segment selector, opportunities, risks (Module 8), and an insights download
- **Marketing Recommendations** — a segment filter applied across all tables (messaging, channel, product, discount, retention, reactivation, cross-/up-sell, campaign plan), each with its own CSV download (Module 8)
- **About / Methodology** — dataset, feature engineering, scaling, an interactive chart of the clustering algorithm comparison, evaluation, final model selection

## Day 2 additions
All Day 2 UI elements (filters, charts, KPI cards, search, segment selection, downloads)
are built on the same read-only artifacts as Day 1 — nothing is retrained or
recomputed. Per-customer segment labels shown in filters/search/charts are produced
by calling `.predict()` once on the saved production model against the existing,
already-scaled engineered feature table (the same inference already used for a
single customer), cached so it runs once per session.

## Day 3 additions — Model/Application Integration
The "New Customer (manual input)" mode on the Customer Segmentation page lets
someone who isn't in the historical dataset be scored by the real saved model:

1. A Streamlit form collects the customer's raw information (demographics,
   two-year spending, purchase behavior, campaign history).
2. `src/feature_engineering_bridge.py` imports (does not reimplement) Module
   6's actual functions from `06-Customer-Personality-Analysis/src/
   feature_engineering.py` and `eda_utils.py` to engineer the same 45
   features used to train the model, in the same order.
3. Because Module 6's saved `scaler` was fit *after* its features were
   already scaled (verified: mean ≈ 0, scale ≈ 1 for every continuous
   feature), it isn't a valid raw→scaled transformer for brand-new input.
   Module 6's real `scale_features()` step is the one that actually does
   this job, so the new customer's raw record is appended to the historical
   cleaned dataset and Module 6's unmodified pipeline is re-run once,
   placing the new customer in the exact same feature space as training.
   This was verified to reproduce `customer_personality_features.csv` to
   floating-point precision (~1e-15) for every existing customer.
4. The saved production KMeans model (`pipeline.pkl`) — never retrained —
   predicts the cluster from this feature vector.
5. The predicted cluster is mapped to its Module 7 segment name/description
   and its Module 8 business insight and marketing recommendations, all
   read from the existing report files.
6. Invalid or missing inputs (out-of-range age, negative amounts, unknown
   categories, missing enrollment date) are caught by
   `validate_customer_input()` before any prediction is attempted.

## Where the model artifacts come from
| Artifact | Source |
|---|---|
| Cleaned + engineered customer data | `06-Customer-Personality-Analysis/03_Cleaned_Data/` |
| Final KMeans model, scaler, feature list | `06-Customer-Personality-Analysis/outputs/models/` |
| Segment names & profiles | `07-Customer-Segmentation-Clustering/outputs/reports/` |
| Business insights & marketing recommendations | `08-Marketing-Business-Insights/outputs/reports/` |

This app must be placed **alongside** those three module folders:
```
<project_root>/
    06-Customer-Personality-Analysis/
    07-Customer-Segmentation-Clustering/
    08-Marketing-Business-Insights/
    Day-9/                <- this app
```

## Install dependencies
```bash
cd Day-9
pip install -r requirements.txt
```

## Run the application
```bash
streamlit run app.py
```
Then open the URL Streamlit prints (typically `http://localhost:8501`).

## Notes
- No datasets or model files are duplicated into `Day-9/` — everything is read
  by relative path from the existing Module 6-8 folders (see `src/config.py`).
- The `models/`, `data/`, and `outputs/` folders in this directory are placeholders
  for optional future artifacts and are not required for the app to run.
- Loading is cached (`st.cache_resource` / `st.cache_data`) so the model and
  reports are read once per session, not on every click.
