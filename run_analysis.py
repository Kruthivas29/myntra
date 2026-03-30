"""
run_analysis.py — Full Analysis Pipeline
==========================================
Master script that orchestrates all 3 problems + integrated framework.

USAGE:
  1. Download Online Retail II from https://archive.ics.uci.edu/ml/datasets/Online+Retail+II
  2. Place the .xlsx file in data/raw/online_retail_II.xlsx
  3. Run: python run_analysis.py
  4. All outputs saved to data/processed/ and models/

NOTE: This script can also be run cell-by-cell in a Jupyter notebook.
      Each section is marked with # %% for VS Code / Jupytext compatibility.
"""

# %% [markdown]
# # StyleKart Customer Segmentation — Full Analysis
# **SBR Session 5 | 20 Marks**
#
# Three interconnected problems forming an Integrated Customer Intelligence Framework:
# 1. RFM-Based Customer Segmentation
# 2. Customer Lifetime Value (CLV) Prediction
# 3. Customer Churn Prediction

# %%
# ===========================================================================
# IMPORTS & SETUP
# ===========================================================================
import pandas as pd
import numpy as np
import os
import sys
import warnings
import json

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from utils.data_cleaning import (
    load_raw_data, clean_data, build_cancellation_table,
    build_customer_table, build_cohort_table, compute_eda_stats,
    save_processed,
)
from utils.rfm import (
    compute_rfm, score_rfm, map_segments, segment_summary,
    validate_h1, pareto_analysis, get_recommendations,
)
from utils.clv import (
    split_calibration_holdout, prepare_lifetimes_data,
    fit_bgnbd_model, fit_gamma_gamma_model, predict_clv_statistical,
    engineer_clv_features, train_clv_ml_models,
    assign_clv_tiers, validate_h2, budget_calculator,
)
from utils.churn import (
    split_feature_outcome, create_churn_labels, engineer_churn_features,
    train_churn_models, predict_churn_all_customers,
    business_impact_calculator, threshold_sweep, validate_h3,
    FEATURE_COLS, RETENTION_ACTIONS,
)
from utils.visualizations import *

print("=" * 70)
print("  StyleKart Customer Segmentation — Full Analysis Pipeline")
print("=" * 70)


# %%
# ===========================================================================
# SECTION 1: DATA LOADING & CLEANING
# ===========================================================================
print("\n" + "=" * 70)
print("  SECTION 1: DATA LOADING & CLEANING")
print("=" * 70)

# --- Load raw data ---
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "online_retail_II.xlsx")

# Check if file exists, if not try alternative names
if not os.path.exists(RAW_DATA_PATH):
    alt_paths = [
        os.path.join(PROJECT_ROOT, "data", "raw", "Online Retail.xlsx"),
        os.path.join(PROJECT_ROOT, "data", "raw", "online_retail.xlsx"),
        os.path.join(PROJECT_ROOT, "data", "raw", "Online_Retail_II.xlsx"),
    ]
    for p in alt_paths:
        if os.path.exists(p):
            RAW_DATA_PATH = p
            break

if not os.path.exists(RAW_DATA_PATH):
    print(f"\n[ERROR] Dataset not found at: {RAW_DATA_PATH}")
    print("Please download the Online Retail II dataset from:")
    print("  https://archive.ics.uci.edu/ml/datasets/Online+Retail+II")
    print("And place it in: data/raw/online_retail_II.xlsx")
    sys.exit(1)

df_raw = load_raw_data(RAW_DATA_PATH)
print(f"\nRaw data shape: {df_raw.shape}")
print(f"Columns: {list(df_raw.columns)}")
print(f"\nSample:")
print(df_raw.head())

# --- Build cancellation table (before filtering cancellations) ---
cancel_table = build_cancellation_table(df_raw, country_filter="United Kingdom")

# --- Clean data ---
df_clean, cleaning_report = clean_data(df_raw, country_filter="United Kingdom")

# --- Save cleaned data ---
save_processed(df_clean, os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_transactions.csv"))

# --- Save cleaning report ---
with open(os.path.join(PROJECT_ROOT, "data", "processed", "cleaning_report.json"), "w") as f:
    # Convert non-serializable items
    report_serializable = {k: str(v) if not isinstance(v, (int, float, str)) else v
                           for k, v in cleaning_report.items()}
    json.dump(report_serializable, f, indent=2)

print(f"\n[INFO] Cleaned data: {len(df_clean):,} transactions, "
      f"{df_clean['CustomerID'].nunique():,} customers")


# %%
# ===========================================================================
# SECTION 2: EXPLORATORY DATA ANALYSIS (EDA)
# ===========================================================================
print("\n" + "=" * 70)
print("  SECTION 2: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 70)

# --- Compute EDA statistics ---
eda_stats = compute_eda_stats(df_clean)

print(f"\n--- Key Metrics ---")
print(f"  Total Customers:           {eda_stats['total_customers']:,}")
print(f"  Total Transactions:        {eda_stats['total_transactions']:,}")
print(f"  Total Revenue:             £{eda_stats['total_revenue']:,.2f}")
print(f"  Avg Order Value (mean):    £{eda_stats['avg_order_value']:.2f}")
print(f"  Avg Order Value (median):  £{eda_stats['median_order_value']:.2f}")
print(f"  Avg Revenue/Customer:      £{eda_stats['avg_revenue_per_customer']:.2f}")
print(f"  Total Unique Products:     {eda_stats['total_products']:,}")
print(f"  Avg Items/Order:           {eda_stats['avg_items_per_order']:.1f}")
print(f"  Date Range:                {eda_stats['date_range_start'].date()} to {eda_stats['date_range_end'].date()}")

# --- Build customer table ---
customer_table = build_customer_table(df_clean)
save_processed(customer_table, os.path.join(PROJECT_ROOT, "data", "processed", "customer_table.csv"))

# --- Cohort analysis ---
retention_table = build_cohort_table(df_clean)
retention_table.to_csv(os.path.join(PROJECT_ROOT, "data", "processed", "cohort_retention.csv"))
print(f"\n[INFO] Cohort retention table: {retention_table.shape[0]} cohorts × {retention_table.shape[1]} periods")

# --- Generate EDA charts (save as HTML for review) ---
print("\n--- Generating EDA Charts ---")

fig_monthly = plot_monthly_revenue(eda_stats["monthly_revenue"])
fig_monthly.write_html(os.path.join(PROJECT_ROOT, "data", "processed", "eda_monthly_revenue.html"))

fig_freq = plot_order_frequency_dist(eda_stats["order_frequency_dist"])
fig_freq.write_html(os.path.join(PROJECT_ROOT, "data", "processed", "eda_order_frequency.html"))

fig_dow = plot_revenue_by_dow(eda_stats["revenue_by_dow"])
fig_dow.write_html(os.path.join(PROJECT_ROOT, "data", "processed", "eda_revenue_dow.html"))

fig_cohort = plot_cohort_heatmap(retention_table)
fig_cohort.write_html(os.path.join(PROJECT_ROOT, "data", "processed", "eda_cohort_heatmap.html"))

fig_nvr = plot_new_vs_returning(eda_stats["new_vs_returning"])
fig_nvr.write_html(os.path.join(PROJECT_ROOT, "data", "processed", "eda_new_vs_returning.html"))

fig_aov = plot_aov_distribution(df_clean)
fig_aov.write_html(os.path.join(PROJECT_ROOT, "data", "processed", "eda_aov_distribution.html"))

print("[INFO] EDA charts saved to data/processed/")


# %%
# ===========================================================================
# SECTION 3: PROBLEM 1 — RFM SEGMENTATION
# ===========================================================================
print("\n" + "=" * 70)
print("  SECTION 3: PROBLEM 1 — RFM-Based Customer Segmentation")
print("=" * 70)

# --- Hypotheses ---
print("""
HYPOTHESIS H1:
  - Top 20% of customers by RFM score contribute over 60% of total revenue.
  - The 'At Risk' segment represents at least 10% of the customer base.
""")

# --- Compute RFM ---
rfm = compute_rfm(df_clean)
print(f"\n--- RFM Summary ---")
print(rfm[["Recency", "Frequency", "Monetary"]].describe())

# --- Score RFM ---
rfm = score_rfm(rfm)
print(f"\n--- RFM Score Distribution ---")
print(f"  R_Score: {rfm['R_Score'].value_counts().sort_index().to_dict()}")
print(f"  F_Score: {rfm['F_Score'].value_counts().sort_index().to_dict()}")
print(f"  M_Score: {rfm['M_Score'].value_counts().sort_index().to_dict()}")

# --- Map to segments ---
rfm = map_segments(rfm)
print(f"\n--- Segment Distribution ---")
print(rfm["Segment"].value_counts())

# --- Segment summary ---
seg_summary = segment_summary(rfm)
print(f"\n--- Segment Summary ---")
print(seg_summary.to_string(index=False))

# --- Validate H1 ---
h1_result = validate_h1(rfm)

# --- Pareto analysis ---
pareto_data = pareto_analysis(rfm)

# --- Campaign recommendations ---
print(f"\n--- Campaign Recommendations ---")
for segment in rfm["Segment"].unique():
    rec = get_recommendations(segment)
    print(f"\n  [{segment}] Strategy: {rec['strategy']}")
    for action in rec["actions"]:
        print(f"    → {action}")

# --- Save RFM outputs ---
rfm.to_csv(os.path.join(PROJECT_ROOT, "data", "processed", "rfm_scored.csv"), index=False)
seg_summary.to_csv(os.path.join(PROJECT_ROOT, "data", "processed", "segment_summary.csv"), index=False)
pareto_data.to_csv(os.path.join(PROJECT_ROOT, "data", "processed", "pareto_analysis.csv"), index=False)

# --- Generate RFM charts ---
print("\n--- Generating RFM Charts ---")
plot_segment_distribution(rfm).write_html(os.path.join(PROJECT_ROOT, "data", "processed", "rfm_segment_dist.html"))
plot_segment_characteristics(seg_summary).write_html(os.path.join(PROJECT_ROOT, "data", "processed", "rfm_characteristics.html"))
plot_rfm_scatter(rfm).write_html(os.path.join(PROJECT_ROOT, "data", "processed", "rfm_scatter.html"))
plot_segment_revenue_contribution(seg_summary).write_html(os.path.join(PROJECT_ROOT, "data", "processed", "rfm_revenue_pie.html"))
plot_pareto(pareto_data).write_html(os.path.join(PROJECT_ROOT, "data", "processed", "pareto_chart.html"))
print("[INFO] RFM charts saved.")


# %%
# ===========================================================================
# SECTION 4: PROBLEM 2 — CLV PREDICTION
# ===========================================================================
print("\n" + "=" * 70)
print("  SECTION 4: PROBLEM 2 — Customer Lifetime Value Prediction")
print("=" * 70)

# --- Hypotheses ---
print("""
HYPOTHESIS H2:
  - At current CAC of ₹380 (~£3.62), at least 30% of customers are unprofitable
    over a 6-month horizon.
  - Sale-event customers have significantly lower CLV than organic customers.
""")

# --- 4A: STATISTICAL APPROACH (BG/NBD + Gamma-Gamma) ---
print("\n--- 4A: Statistical CLV (BG/NBD + Gamma-Gamma) ---")

# Split calibration / holdout
df_cal, df_holdout, split_date = split_calibration_holdout(df_clean, calibration_months=9, holdout_months=3)

# Prepare lifetimes data
lifetimes_data = prepare_lifetimes_data(df_cal, split_date)

try:
    # Fit BG/NBD
    bgf = fit_bgnbd_model(lifetimes_data)

    # Fit Gamma-Gamma
    ggf = fit_gamma_gamma_model(lifetimes_data)

    # Predict CLV (6-month horizon)
    clv_statistical = predict_clv_statistical(bgf, ggf, lifetimes_data, time_months=6)

    # Assign tiers
    clv_statistical, tier_boundaries = assign_clv_tiers(clv_statistical, clv_col="predicted_clv")

    # Save
    clv_statistical.to_csv(os.path.join(PROJECT_ROOT, "data", "processed", "clv_statistical.csv"), index=False)

    # Validate H2
    h2_result = validate_h2(clv_statistical, cac=380.0, clv_col="predicted_clv")

    statistical_clv_done = True
    print("[INFO] Statistical CLV complete.")

except Exception as e:
    print(f"[WARN] Statistical CLV failed: {e}")
    print("[INFO] Falling back to ML approach only.")
    statistical_clv_done = False

# --- 4B: ML APPROACH ---
print("\n--- 4B: ML-based CLV Prediction ---")

clv_features = engineer_clv_features(df_cal, df_holdout)
clv_ml_results, X_test_clv, y_test_clv = train_clv_ml_models(
    clv_features,
    save_dir=os.path.join(PROJECT_ROOT, "models"),
)

# Save features
clv_features.to_csv(os.path.join(PROJECT_ROOT, "data", "processed", "clv_features.csv"), index=False)

# --- Determine primary CLV source ---
if statistical_clv_done:
    primary_clv = clv_statistical[["CustomerID", "predicted_clv", "CLV_Tier", "prob_alive", "expected_purchases"]].copy()
    clv_col = "predicted_clv"
    print("[INFO] Using STATISTICAL CLV as primary.")
else:
    # Use best ML model to predict for all customers
    best_model_name = min(clv_ml_results, key=lambda k: clv_ml_results[k]["MAE"])
    best_model = clv_ml_results[best_model_name]["model"]
    feature_cols_ml = [
        "recency", "frequency", "monetary", "avg_order_value",
        "unique_products", "total_quantity", "tenure_days",
        "avg_days_between", "items_per_order",
    ]
    clv_features["predicted_clv"] = best_model.predict(
        clv_features[feature_cols_ml].fillna(0)
    ).clip(min=0)
    clv_features, tier_boundaries = assign_clv_tiers(clv_features, clv_col="predicted_clv")
    primary_clv = clv_features[["CustomerID", "predicted_clv", "CLV_Tier"]].copy()
    clv_col = "predicted_clv"
    h2_result = validate_h2(clv_features, cac=380.0, clv_col="predicted_clv")
    print(f"[INFO] Using ML CLV ({best_model_name}) as primary.")

# --- Budget calculator ---
budget_df = budget_calculator(primary_clv, cac=380.0, clv_col=clv_col)
print(f"\n--- Budget Analysis (CAC = ₹380 / ~£3.62) ---")
print(budget_df.to_string(index=False))
budget_df.to_csv(os.path.join(PROJECT_ROOT, "data", "processed", "budget_analysis.csv"), index=False)

# --- Save CLV charts ---
print("\n--- Generating CLV Charts ---")
try:
    plot_clv_distribution(primary_clv, clv_col).write_html(
        os.path.join(PROJECT_ROOT, "data", "processed", "clv_distribution.html"))
    plot_revenue_concentration_curve(primary_clv, clv_col).write_html(
        os.path.join(PROJECT_ROOT, "data", "processed", "clv_concentration.html"))
    print("[INFO] CLV charts saved.")
except Exception as e:
    print(f"[WARN] CLV chart error: {e}")


# %%
# ===========================================================================
# SECTION 5: PROBLEM 3 — CHURN PREDICTION
# ===========================================================================
print("\n" + "=" * 70)
print("  SECTION 5: PROBLEM 3 — Customer Churn Prediction")
print("=" * 70)

# --- Hypotheses ---
print("""
HYPOTHESIS H3:
  - Single-purchase customers in first 60 days have >70% churn probability.
  - Return rate and declining order value are top-5 predictive features.
""")

# --- Split feature / outcome ---
df_feature, df_outcome, feature_end, outcome_end = split_feature_outcome(
    df_clean, feature_months=9, outcome_days=90,
)

# --- Create labels ---
labels = create_churn_labels(df_feature, df_outcome)

# --- Engineer features ---
churn_features = engineer_churn_features(df_feature, cancel_table)
print(f"\n--- Churn Feature Summary ---")
print(churn_features.describe().round(2))

# --- Train models ---
churn_results = train_churn_models(
    churn_features, labels,
    save_dir=os.path.join(PROJECT_ROOT, "models"),
    use_smote=True,
)

# --- Predict churn for all customers ---
best_churn_model_name = "XGBoost" if "XGBoost" in churn_results else "Logistic Regression"
best_churn_res = churn_results[best_churn_model_name]

churn_predictions = predict_churn_all_customers(
    churn_features,
    best_churn_res["model"],
    scaler=best_churn_res.get("scaler"),
)

# --- Business impact analysis ---
test_data = churn_results["_test_data"]
impact_05 = business_impact_calculator(
    test_data["y_test"].values,
    best_churn_res["y_prob"],
    threshold=0.5,
    avg_customer_revenue=200.0,
    intervention_cost_per_customer=5.0,
    recovery_rate=0.30,
)

print(f"\n--- Business Impact (threshold=0.5) ---")
for k, v in impact_05.items():
    print(f"  {k}: {v}")

# --- Threshold sweep ---
sweep_df = threshold_sweep(
    test_data["y_test"].values,
    best_churn_res["y_prob"],
)
sweep_df.to_csv(os.path.join(PROJECT_ROOT, "data", "processed", "threshold_sweep.csv"), index=False)

# --- Validate H3 ---
h3_result = validate_h3(churn_features, labels, churn_results)

# --- Save churn outputs ---
churn_features.to_csv(os.path.join(PROJECT_ROOT, "data", "processed", "churn_features.csv"), index=False)
churn_predictions.to_csv(os.path.join(PROJECT_ROOT, "data", "processed", "churn_predictions.csv"), index=False)

# --- Generate churn charts ---
print("\n--- Generating Churn Charts ---")
try:
    plot_roc_curve(churn_results).write_html(
        os.path.join(PROJECT_ROOT, "data", "processed", "churn_roc_curve.html"))
    plot_feature_importance(best_churn_res["feature_importance"]).write_html(
        os.path.join(PROJECT_ROOT, "data", "processed", "churn_feature_importance.html"))
    plot_confusion_matrix(best_churn_res["confusion_matrix"]).write_html(
        os.path.join(PROJECT_ROOT, "data", "processed", "churn_confusion_matrix.html"))
    plot_churn_probability_dist(churn_predictions).write_html(
        os.path.join(PROJECT_ROOT, "data", "processed", "churn_prob_dist.html"))
    print("[INFO] Churn charts saved.")
except Exception as e:
    print(f"[WARN] Churn chart error: {e}")


# %%
# ===========================================================================
# SECTION 6: INTEGRATED CUSTOMER INTELLIGENCE FRAMEWORK
# ===========================================================================
print("\n" + "=" * 70)
print("  SECTION 6: INTEGRATED CUSTOMER INTELLIGENCE FRAMEWORK")
print("=" * 70)

# --- Build Master Customer Table ---
print("\n--- Building Master Customer Table ---")

master = rfm[["CustomerID", "Recency", "Frequency", "Monetary", "R_Score", "F_Score",
              "M_Score", "RFM_Score", "RFM_Agg_Score", "Segment"]].copy()
master = master.rename(columns={"Segment": "RFM_Segment"})

# Merge CLV
master = master.merge(
    primary_clv[["CustomerID", "predicted_clv", "CLV_Tier"]],
    on="CustomerID", how="left",
)

# Merge Churn
master = master.merge(
    churn_predictions[["CustomerID", "churn_probability", "churn_risk_band"]],
    on="CustomerID", how="left",
)
master = master.rename(columns={"churn_risk_band": "Churn_Risk_Band"})

# Fill NaN for customers not in all datasets (due to different time splits)
master["predicted_clv"] = master["predicted_clv"].fillna(0)
master["churn_probability"] = master["churn_probability"].fillna(0.5)
master["CLV_Tier"] = master["CLV_Tier"].fillna("Bronze")
master["Churn_Risk_Band"] = master["Churn_Risk_Band"].fillna("Medium")

print(f"[INFO] Master table: {len(master):,} customers, {len(master.columns)} columns")
print(f"\nColumns: {list(master.columns)}")
print(f"\n--- Sample Rows ---")
print(master.head(10).to_string(index=False))

# --- Cross-Problem Insights ---
print("\n--- Cross-Problem Insights ---")

# Insight 1: At Risk + High CLV + High Churn
high_priority = master[
    (master["RFM_Segment"] == "At Risk") &
    (master["CLV_Tier"].isin(["Platinum", "Gold"])) &
    (master["Churn_Risk_Band"].isin(["High", "Critical"]))
]
high_priority_rev = high_priority["predicted_clv"].sum()
print(f"\n  1. HIGHEST PRIORITY (At Risk + High CLV + High Churn):")
print(f"     Customers: {len(high_priority):,}")
print(f"     Total CLV at risk: £{high_priority_rev:,.2f}")
print(f"     → Every marketing dollar here has the HIGHEST ROI")

# Insight 2: Champions + Low Churn
stable_champs = master[
    (master["RFM_Segment"] == "Champions") &
    (master["Churn_Risk_Band"] == "Low")
]
print(f"\n  2. LEAVE ALONE (Champions + Low Churn):")
print(f"     Customers: {len(stable_champs):,}")
print(f"     → Don't over-market; focus on delight, not discounts")

# Insight 3: Lost + Low CLV + High Churn
let_go = master[
    (master["RFM_Segment"].isin(["Lost", "Hibernating"])) &
    (master["CLV_Tier"] == "Bronze") &
    (master["Churn_Risk_Band"].isin(["High", "Critical"]))
]
print(f"\n  3. LET GO (Lost/Hibernating + Low CLV + High Churn):")
print(f"     Customers: {len(let_go):,}")
print(f"     → Stop spending acquisition budget; CAC > CLV")

# Insight 4: Potential Loyalists + Medium CLV + Medium Churn
nurture = master[
    (master["RFM_Segment"] == "Potential Loyalists") &
    (master["Churn_Risk_Band"].isin(["Medium", "Low"]))
]
print(f"\n  4. NURTURE ZONE (Potential Loyalists + Medium Churn):")
print(f"     Customers: {len(nurture):,}")
print(f"     → Highest leverage for growth; invest in 2nd/3rd purchase")

# --- Priority action list (Top 50) ---
master["priority_score"] = (
    master["predicted_clv"] * master["churn_probability"] *
    master["RFM_Segment"].map({"At Risk": 3, "Potential Loyalists": 2,
                                "Need Attention": 2, "Loyal Customers": 1.5}).fillna(1)
)
top_50 = master.nlargest(50, "priority_score")
top_50.to_csv(os.path.join(PROJECT_ROOT, "data", "processed", "priority_top_50.csv"), index=False)
print(f"\n[INFO] Top 50 priority customers saved to priority_top_50.csv")

# --- Revenue at Risk ---
high_churn_customers = master[master["Churn_Risk_Band"].isin(["High", "Critical"])]
total_revenue_at_risk = high_churn_customers["predicted_clv"].sum()
print(f"\n--- Revenue at Risk ---")
print(f"  High/Critical churn customers: {len(high_churn_customers):,}")
print(f"  Total CLV at risk: £{total_revenue_at_risk:,.2f}")

# --- Save Master Table ---
master.to_csv(os.path.join(PROJECT_ROOT, "data", "processed", "master_customer_table.csv"), index=False)
print(f"\n[INFO] Master customer table saved: {len(master):,} rows")

# --- Generate integration charts ---
print("\n--- Generating Integration Charts ---")
try:
    plot_integrated_heatmap(master).write_html(
        os.path.join(PROJECT_ROOT, "data", "processed", "integrated_bubble.html"))
    plot_revenue_at_risk(master).write_html(
        os.path.join(PROJECT_ROOT, "data", "processed", "revenue_at_risk.html"))
    print("[INFO] Integration charts saved.")
except Exception as e:
    print(f"[WARN] Integration chart error: {e}")


# %%
# ===========================================================================
# SECTION 7: HYPOTHESIS VALIDATION SUMMARY
# ===========================================================================
print("\n" + "=" * 70)
print("  SECTION 7: HYPOTHESIS VALIDATION SUMMARY")
print("=" * 70)

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  H1: Top 20% by RFM → >60% revenue                                ║
║  Result: {h1_result['top_20_revenue_pct']}%  {'✓ VALIDATED' if h1_result['h1_revenue_validated'] else '✗ NOT VALIDATED':>25}  ║
║  At Risk ≥ 10%: {h1_result['at_risk_pct']}%  {'✓ VALIDATED' if h1_result['h1_at_risk_validated'] else '✗ NOT VALIDATED':>25}  ║
╠══════════════════════════════════════════════════════════════════════╣
║  H2: ≥30% unprofitable at CAC ₹380                                ║
║  Result: {h2_result['unprofitable_pct']}%  {'✓ VALIDATED' if h2_result['h2_validated'] else '✗ NOT VALIDATED':>25}  ║
╠══════════════════════════════════════════════════════════════════════╣
║  H3: Single-purchase churn >70%                                    ║
║  Result: {h3_result['single_purchase_churn_rate']}%  {'✓ VALIDATED' if h3_result['h3_churn_validated'] else '✗ NOT VALIDATED':>25}  ║
║  Best model AUC: {h3_result['best_auc']}                            ║
║  Top 5 features: {', '.join(h3_result['top_5_features'][:3])}...   ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# --- Save all hypothesis results ---
hypothesis_results = {"H1": h1_result, "H2": h2_result, "H3": h3_result}
with open(os.path.join(PROJECT_ROOT, "data", "processed", "hypothesis_results.json"), "w") as f:
    # Make JSON serializable
    def make_serializable(obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = {}
    for hyp, results in hypothesis_results.items():
        serializable[hyp] = {k: make_serializable(v) for k, v in results.items()}
    json.dump(serializable, f, indent=2)


# %%
# ===========================================================================
# SECTION 8: DATA LIMITATIONS & ASSUMPTIONS
# ===========================================================================
print("\n" + "=" * 70)
print("  SECTION 8: LIMITATIONS & ASSUMPTIONS (Required for Full Marks)")
print("=" * 70)

print("""
1. DATASET PROXY: The Online Retail II dataset is UK-based general retail (2009–2011),
   not Indian fashion e-commerce. In production, StyleKart's actual transaction data
   would replace it. Behavioral patterns may differ for fashion vs general retail.

2. CURRENCY: All values are in GBP. For StyleKart context, multiply by ~105 for INR
   equivalent. Budget calculations use conversion rate of 1 GBP ≈ ₹105.

3. TEMPORAL: Data is from 2009–2011. Analytical methods are timeless, but consumer
   behavior, digital adoption, and market dynamics have evolved significantly.

4. NO DEMOGRAPHICS: Segmentation is purely behavioral (transactional data only).
   Adding age, gender, location, device, and channel data would significantly improve
   model performance — especially for churn prediction.

5. SINGLE-MARKET FILTER: Analysis restricted to UK customers (~90% of data) for
   cleaner single-market analysis. Multi-country analysis would require additional
   normalization.

6. NULL CUSTOMER IDs: ~25% of records dropped due to null CustomerID. This could
   introduce survivorship bias — the dropped records may skew toward one-time buyers
   or B2B transactions without assigned customer accounts.

7. CHURN DEFINITION: 90-day no-purchase window is an assumption. Fashion e-commerce
   may warrant different windows (60 days for fast fashion, 120 days for premium).

8. CLV TIME HORIZON: 6-month CLV prediction assumes relatively stable customer behavior.
   External shocks (sales events, new competitors) are not modeled.

9. PROFIT MARGIN: Assumed 15% profit margin for CLV calculation. Actual StyleKart margins
   would vary by category and should be calibrated.

10. CLASS IMBALANCE: SMOTE was used for churn prediction. Alternative approaches
    (class_weight, undersampling, or cost-sensitive learning) could yield different results.
""")


# %%
# ===========================================================================
# FINAL OUTPUT SUMMARY
# ===========================================================================
print("\n" + "=" * 70)
print("  PIPELINE COMPLETE — OUTPUT FILES")
print("=" * 70)

output_dir = os.path.join(PROJECT_ROOT, "data", "processed")
for f in sorted(os.listdir(output_dir)):
    filepath = os.path.join(output_dir, f)
    size_kb = os.path.getsize(filepath) / 1024
    print(f"  {f:<45} {size_kb:>8.1f} KB")

model_dir = os.path.join(PROJECT_ROOT, "models")
if os.path.exists(model_dir):
    print(f"\n  --- Models ---")
    for f in sorted(os.listdir(model_dir)):
        filepath = os.path.join(model_dir, f)
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  {f:<45} {size_kb:>8.1f} KB")

print(f"\n{'=' * 70}")
print("  ✓ Analysis complete. Ready for Streamlit integration.")
print(f"{'=' * 70}")
