# StyleKart — Customer Segmentation for Organisation Growth

## SBR Session 5 | 20 Marks | Streamlit Deliverable

An **Integrated Customer Intelligence Framework** that combines RFM Segmentation, CLV Prediction, and Churn Prediction into a unified decision-making system for StyleKart — a mid-scale B2C fashion e-commerce platform.

---

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

Download the **Online Retail II** dataset from UCI:
- URL: https://archive.ics.uci.edu/ml/datasets/Online+Retail+II
- Place the `.xlsx` file at: `data/raw/online_retail_II.xlsx`

### 3. Run Full Analysis

```bash
python run_analysis.py
```

This generates all processed data, models, charts, and the master customer table in `data/processed/` and `models/`.

### 4. Launch Streamlit App

```bash
streamlit run app.py
```

---

## Project Architecture

```
stylekart_project/
├── app.py                    # Streamlit main entry point (YOU BUILD THIS)
├── run_analysis.py           # Full analysis pipeline (run first)
├── requirements.txt          # Dependencies
├── README.md
├── data/
│   ├── raw/                  # Place dataset here
│   └── processed/            # Generated: cleaned data, features, predictions
├── models/                   # Saved .pkl model files
├── utils/
│   ├── __init__.py
│   ├── data_cleaning.py      # Load, clean, prepare transactional data
│   ├── rfm.py                # RFM computation, scoring, segment mapping
│   ├── clv.py                # CLV via BG/NBD + Gamma-Gamma & ML
│   ├── churn.py              # Churn feature engineering & modelling
│   └── visualizations.py     # Reusable Plotly/Seaborn chart functions
└── notebooks/                # Jupyter notebooks for exploration
```

---

## Three Problems — One Framework

| Problem | Method | Key Output |
|---------|--------|------------|
| **1. RFM Segmentation** | Quartile-based R/F/M scoring → named segments | 6–10 actionable customer segments |
| **2. CLV Prediction** | BG/NBD + Gamma-Gamma (statistical) + XGBoost (ML) | Per-customer CLV & tier (Platinum/Gold/Silver/Bronze) |
| **3. Churn Prediction** | Logistic Regression + XGBoost with SMOTE | Per-customer churn probability & risk band |
| **Integrated** | Master table joining all three → priority matrix | Revenue at risk, priority action list, budget allocation |

---

## Key Outputs (in `data/processed/`)

| File | Description |
|------|-------------|
| `master_customer_table.csv` | **THE deliverable** — one row per customer with RFM segment, CLV tier, churn risk |
| `rfm_scored.csv` | Full RFM scores and segments |
| `clv_statistical.csv` | BG/NBD + Gamma-Gamma CLV predictions |
| `churn_predictions.csv` | Churn probabilities and risk bands |
| `priority_top_50.csv` | Top 50 customers needing immediate attention |
| `segment_summary.csv` | Per-segment summary statistics |
| `budget_analysis.csv` | CAC vs CLV profitability by tier |
| `hypothesis_results.json` | H1, H2, H3 validation results |
| `*.html` | Interactive Plotly charts (open in browser) |

---

## Streamlit App Pages (Your Implementation)

### Page 0 — Overview & EDA
- StyleKart company context card
- KPI cards (total customers, revenue, AOV)
- 6+ EDA charts from `visualizations.py`
- Cohort retention heatmap
- Data quality summary

### Page 1 — Customer Segmentation (RFM)
- Load `rfm_scored.csv` and `segment_summary.csv`
- Segment filter dropdown → updates all charts
- Charts: `plot_segment_distribution()`, `plot_segment_characteristics()`, `plot_rfm_scatter()`, `plot_segment_revenue_contribution()`
- Campaign recommendation cards from `rfm.get_recommendations()`

### Page 2 — Customer Lifetime Value
- Load `clv_statistical.csv` or `clv_features.csv`
- CLV tier slider, time horizon toggle
- Charts: `plot_clv_distribution()`, `plot_revenue_concentration_curve()`, `plot_clv_by_rfm_segment()`
- Budget calculator widget using `clv.budget_calculator()`

### Page 3 — Churn Prediction
- Load `churn_predictions.csv`, load model from `models/`
- Threshold slider (0.0–1.0) → updates confusion matrix + business impact
- Charts: `plot_roc_curve()`, `plot_feature_importance()`, `plot_confusion_matrix()`, `plot_churn_probability_dist()`
- Business impact section using `churn.business_impact_calculator()`

### Page 4 — Integrated Intelligence
- Load `master_customer_table.csv`
- Charts: `plot_integrated_heatmap()`, `plot_revenue_at_risk()`
- Filterable master table (by segment, CLV tier, churn band)
- Revenue at risk calculator
- Top 50 priority action list

---

## Hypotheses

| # | Hypothesis | Validated By |
|---|-----------|-------------|
| H1 | Top 20% by RFM → >60% revenue; At Risk ≥ 10% | `rfm.validate_h1()` |
| H2 | ≥30% unprofitable at CAC ₹380 | `clv.validate_h2()` |
| H3 | Single-purchase churn >70%; return_rate in top-5 features | `churn.validate_h3()` |

---

## Limitations (State in App)

1. Dataset is UK general retail (2009–2011), not Indian fashion
2. Currency in GBP (×105 for INR approximation)
3. No demographic data — purely behavioral segmentation
4. 25% null CustomerIDs dropped → possible survivorship bias
5. 90-day churn window is assumed; may need calibration for fashion
6. 15% profit margin assumed for CLV; actual varies by category

---

## Team Collaboration

| Role | Files |
|------|-------|
| Member 1 | `data_cleaning.py`, EDA section, Page 0 |
| Member 2 | `rfm.py`, Page 1 |
| Member 3 | `clv.py`, Page 2 |
| Member 4 | `churn.py`, Page 3 |
| All | `run_analysis.py`, Page 4, presentation |

---

## References

- Online Retail II Dataset: UCI ML Repository
- Blog Series Parts 1–4 (core), Parts 5–9 (supplementary)
- `lifetimes` library documentation: https://lifetimes.readthedocs.io
