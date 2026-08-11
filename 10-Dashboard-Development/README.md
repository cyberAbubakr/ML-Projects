# Day 9 — Customer Segmentation & Marketing Insights Dashboard

## What this application does
A Streamlit dashboard that connects the customer-segmentation pipeline built in
Modules 6-8 with the business insights and marketing recommendations already
produced in Module 8. It **loads existing, saved artifacts** — it does not
retrain the clustering model, re-run feature engineering, or recompute any
business analysis.

Pages:
- **Home / Overview** — project summary, KPI cards (customers, segments, final model, top-value segment), a segment-size chart, and a segment-summary download
- **Customer Segmentation** — filter customers by segment/income/age, search by Customer ID, download the filtered list, pick a customer, view their features, see the segment assigned by the saved model, download that customer's record
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
