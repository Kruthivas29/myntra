"""
churn.py — Customer Churn Prediction
======================================
Problem 3: Predict which customers will churn (no purchase in next 90 days).

Pipeline:
  1. Define churn: no purchase in 90-day outcome window
  2. Split into feature period (months 1–9) and outcome period (months 10–12)
  3. Engineer 13+ features per customer from feature period
  4. Handle class imbalance (SMOTE / class_weight)
  5. Train Logistic Regression (baseline) + XGBoost (performance)
  6. Evaluate: Precision, Recall, F1, AUC-ROC, Confusion Matrix
  7. Business impact calculator
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import joblib
import os
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. FEATURE / OUTCOME PERIOD SPLIT
# ---------------------------------------------------------------------------
def split_feature_outcome(
    df_clean: pd.DataFrame,
    feature_months: int = 9,
    outcome_days: int = 90,
) -> tuple:
    """
    Split data into feature period and outcome period.
    
    Feature period: first `feature_months` months of data
    Outcome period: next `outcome_days` days after feature period
    
    Churn definition: customer made NO purchase in the outcome period.
    """
    min_date = df_clean["InvoiceDate"].min()
    max_date = df_clean["InvoiceDate"].max()

    total_days = (max_date - min_date).days
    feature_days = int(total_days * feature_months / 12)
    feature_end = min_date + timedelta(days=feature_days)
    outcome_end = feature_end + timedelta(days=outcome_days)

    # Cap outcome end to max date
    if outcome_end > max_date:
        outcome_end = max_date

    df_feature = df_clean[df_clean["InvoiceDate"] <= feature_end].copy()
    df_outcome = df_clean[
        (df_clean["InvoiceDate"] > feature_end) & (df_clean["InvoiceDate"] <= outcome_end)
    ].copy()

    print(f"[CHURN] Feature period:  {min_date.date()} to {feature_end.date()} ({len(df_feature):,} rows)")
    print(f"[CHURN] Outcome period:  {feature_end.date()} to {outcome_end.date()} ({len(df_outcome):,} rows)")

    return df_feature, df_outcome, feature_end, outcome_end


# ---------------------------------------------------------------------------
# 2. CHURN LABEL CREATION
# ---------------------------------------------------------------------------
def create_churn_labels(
    df_feature: pd.DataFrame,
    df_outcome: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create binary churn label:
      1 = customer did NOT purchase in outcome period (churned)
      0 = customer purchased in outcome period (retained)
    """
    # All customers in feature period
    all_customers = df_feature["CustomerID"].unique()

    # Customers who purchased in outcome period
    outcome_customers = df_outcome["CustomerID"].unique()

    labels = pd.DataFrame({"CustomerID": all_customers})
    labels["churned"] = (~labels["CustomerID"].isin(outcome_customers)).astype(int)

    churn_rate = labels["churned"].mean() * 100
    print(f"[CHURN] Churn rate: {churn_rate:.1f}% ({labels['churned'].sum():,} churned / {len(labels):,} total)")

    return labels


# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def engineer_churn_features(
    df_feature: pd.DataFrame,
    cancellation_table: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Engineer 13+ behavioral features per customer from the feature period.
    
    Features (as specified in project brief):
    - recency: days since last purchase
    - frequency: number of unique orders
    - monetary: total spend
    - avg_order_value: mean spend per order
    - avg_days_between_purchases: mean gap between consecutive orders
    - order_value_trend: slope of order values over time
    - unique_products_bought: count of distinct StockCodes
    - return_rate: % of orders cancelled/returned (from cancellation table)
    - days_since_first_purchase: customer tenure
    - purchases_last_30_days: recent activity
    - purchases_last_60_days: medium-term activity
    - purchase_frequency_change: frequency recent 3m vs prior 3m
    - is_single_purchase_customer: binary flag
    """
    reference_date = df_feature["InvoiceDate"].max() + pd.Timedelta(days=1)
    last_30_cutoff = reference_date - timedelta(days=30)
    last_60_cutoff = reference_date - timedelta(days=60)

    # Calculate midpoint for frequency change
    min_date = df_feature["InvoiceDate"].min()
    total_days = (reference_date - min_date).days
    midpoint = min_date + timedelta(days=total_days // 2)

    # ----- Core aggregations -----
    features = (
        df_feature.groupby("CustomerID")
        .agg(
            recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
            frequency=("InvoiceNo", "nunique"),
            monetary=("Revenue", "sum"),
            unique_products_bought=("StockCode", "nunique"),
            total_quantity=("Quantity", "sum"),
            first_purchase=("InvoiceDate", "min"),
            last_purchase=("InvoiceDate", "max"),
        )
        .reset_index()
    )

    features["avg_order_value"] = features["monetary"] / features["frequency"]
    features["days_since_first_purchase"] = (reference_date - features["first_purchase"]).dt.days
    features["is_single_purchase_customer"] = (features["frequency"] == 1).astype(int)

    # ----- Avg days between purchases -----
    features["avg_days_between_purchases"] = np.where(
        features["frequency"] > 1,
        (features["last_purchase"] - features["first_purchase"]).dt.days / (features["frequency"] - 1),
        0,
    )

    # ----- Purchases in last 30 / 60 days -----
    recent_30 = (
        df_feature[df_feature["InvoiceDate"] >= last_30_cutoff]
        .groupby("CustomerID")["InvoiceNo"]
        .nunique()
        .reset_index()
        .rename(columns={"InvoiceNo": "purchases_last_30_days"})
    )
    recent_60 = (
        df_feature[df_feature["InvoiceDate"] >= last_60_cutoff]
        .groupby("CustomerID")["InvoiceNo"]
        .nunique()
        .reset_index()
        .rename(columns={"InvoiceNo": "purchases_last_60_days"})
    )

    features = features.merge(recent_30, on="CustomerID", how="left")
    features = features.merge(recent_60, on="CustomerID", how="left")
    features["purchases_last_30_days"] = features["purchases_last_30_days"].fillna(0).astype(int)
    features["purchases_last_60_days"] = features["purchases_last_60_days"].fillna(0).astype(int)

    # ----- Purchase frequency change (recent half vs earlier half) -----
    freq_early = (
        df_feature[df_feature["InvoiceDate"] <= midpoint]
        .groupby("CustomerID")["InvoiceNo"]
        .nunique()
        .reset_index()
        .rename(columns={"InvoiceNo": "freq_early"})
    )
    freq_late = (
        df_feature[df_feature["InvoiceDate"] > midpoint]
        .groupby("CustomerID")["InvoiceNo"]
        .nunique()
        .reset_index()
        .rename(columns={"InvoiceNo": "freq_late"})
    )

    features = features.merge(freq_early, on="CustomerID", how="left")
    features = features.merge(freq_late, on="CustomerID", how="left")
    features["freq_early"] = features["freq_early"].fillna(0)
    features["freq_late"] = features["freq_late"].fillna(0)
    features["purchase_frequency_change"] = features["freq_late"] - features["freq_early"]

    # ----- Order value trend (slope) -----
    def _compute_order_value_trend(group):
        """Compute slope of order value over time using simple linear regression."""
        if len(group) < 2:
            return 0.0
        order_values = group.groupby("InvoiceNo").agg(
            date=("InvoiceDate", "first"),
            value=("Revenue", "sum"),
        ).sort_values("date")

        if len(order_values) < 2:
            return 0.0

        x = np.arange(len(order_values), dtype=float)
        y = order_values["value"].values.astype(float)

        # Simple slope: (y_n - y_0) / n
        slope = (y[-1] - y[0]) / len(y) if len(y) > 0 else 0.0
        return slope

    ovt = df_feature.groupby("CustomerID").apply(_compute_order_value_trend).reset_index()
    ovt.columns = ["CustomerID", "order_value_trend"]
    features = features.merge(ovt, on="CustomerID", how="left")
    features["order_value_trend"] = features["order_value_trend"].fillna(0)

    # ----- Return rate (from cancellation table) -----
    if cancellation_table is not None:
        features = features.merge(
            cancellation_table[["CustomerID", "return_rate"]],
            on="CustomerID",
            how="left",
        )
        features["return_rate"] = features["return_rate"].fillna(0)
    else:
        features["return_rate"] = 0.0

    # ----- Drop intermediate columns -----
    features = features.drop(columns=["first_purchase", "last_purchase", "freq_early", "freq_late"], errors="ignore")

    print(f"[CHURN] Features engineered for {len(features):,} customers, {len(features.columns) - 1} features")

    return features


# ---------------------------------------------------------------------------
# 4. MODEL TRAINING
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "recency", "frequency", "monetary", "avg_order_value",
    "avg_days_between_purchases", "order_value_trend",
    "unique_products_bought", "return_rate",
    "days_since_first_purchase", "purchases_last_30_days",
    "purchases_last_60_days", "purchase_frequency_change",
    "is_single_purchase_customer", "total_quantity",
]


def train_churn_models(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    save_dir: str = "models/",
    use_smote: bool = True,
) -> dict:
    """
    Train churn prediction models:
      1. Logistic Regression (baseline, interpretable)
      2. XGBoost (performance model)
    
    Handles class imbalance with SMOTE or class_weight.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        classification_report, confusion_matrix, roc_auc_score, roc_curve,
        precision_recall_curve, f1_score, precision_score, recall_score,
    )

    try:
        from xgboost import XGBClassifier
        has_xgboost = True
    except ImportError:
        has_xgboost = False

    try:
        from imblearn.over_sampling import SMOTE
        has_smote = True
    except ImportError:
        has_smote = False
        print("[WARN] imbalanced-learn not installed. Using class_weight='balanced' instead of SMOTE.")

    # Merge features + labels
    data = features.merge(labels, on="CustomerID")

    X = data[FEATURE_COLS].fillna(0)
    y = data["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    # Apply SMOTE on training set only
    if use_smote and has_smote:
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"[CHURN] SMOTE applied: {len(X_train)} → {len(X_train_res)} training samples")
        balance_method = "SMOTE"
    else:
        X_train_res, y_train_res = X_train, y_train
        balance_method = "class_weight='balanced'"

    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # ---- 1. Logistic Regression ----
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced" if not (use_smote and has_smote) else None,
        random_state=42,
    )
    lr.fit(X_train_scaled, y_train_res)
    y_pred_lr = lr.predict(X_test_scaled)
    y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

    lr_report = classification_report(y_test, y_pred_lr, output_dict=True)
    results["Logistic Regression"] = {
        "model": lr,
        "scaler": scaler,
        "y_pred": y_pred_lr,
        "y_prob": y_prob_lr,
        "auc_roc": roc_auc_score(y_test, y_prob_lr),
        "confusion_matrix": confusion_matrix(y_test, y_pred_lr),
        "classification_report": lr_report,
        "f1": f1_score(y_test, y_pred_lr, average="weighted"),
        "precision": precision_score(y_test, y_pred_lr, average="weighted"),
        "recall": recall_score(y_test, y_pred_lr, average="weighted"),
        "feature_importance": dict(zip(FEATURE_COLS, np.abs(lr.coef_[0]))),
        "roc_curve": roc_curve(y_test, y_prob_lr),
    }

    # ---- 2. XGBoost ----
    if has_xgboost:
        # Calculate scale_pos_weight for imbalanced classes
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight if not (use_smote and has_smote) else 1.0,
            random_state=42,
            eval_metric="logloss",
        )
        xgb.fit(X_train_res, y_train_res, eval_set=[(X_test, y_test)], verbose=False)
        y_pred_xgb = xgb.predict(X_test)
        y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

        xgb_report = classification_report(y_test, y_pred_xgb, output_dict=True)
        results["XGBoost"] = {
            "model": xgb,
            "scaler": None,  # XGBoost doesn't need scaling
            "y_pred": y_pred_xgb,
            "y_prob": y_prob_xgb,
            "auc_roc": roc_auc_score(y_test, y_prob_xgb),
            "confusion_matrix": confusion_matrix(y_test, y_pred_xgb),
            "classification_report": xgb_report,
            "f1": f1_score(y_test, y_pred_xgb, average="weighted"),
            "precision": precision_score(y_test, y_pred_xgb, average="weighted"),
            "recall": recall_score(y_test, y_pred_xgb, average="weighted"),
            "feature_importance": dict(zip(FEATURE_COLS, xgb.feature_importances_)),
            "roc_curve": roc_curve(y_test, y_prob_xgb),
        }

    # Print comparison
    print(f"\n[CHURN] Model Comparison (Imbalance handling: {balance_method}):")
    print(f"{'Model':<25} {'AUC-ROC':>10} {'F1 (wt)':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 65)
    for name, res in results.items():
        print(f"{name:<25} {res['auc_roc']:>10.4f} {res['f1']:>10.4f} "
              f"{res['precision']:>10.4f} {res['recall']:>10.4f}")

    # Save models
    os.makedirs(save_dir, exist_ok=True)
    for name, res in results.items():
        fname = name.lower().replace(" ", "_")
        joblib.dump(res["model"], os.path.join(save_dir, f"churn_{fname}.pkl"))
        if res.get("scaler"):
            joblib.dump(res["scaler"], os.path.join(save_dir, f"churn_{fname}_scaler.pkl"))

    # Store test data for later use
    results["_test_data"] = {"X_test": X_test, "y_test": y_test}
    results["_balance_method"] = balance_method

    return results


# ---------------------------------------------------------------------------
# 5. PREDICT CHURN PROBABILITIES (FULL DATASET)
# ---------------------------------------------------------------------------
def predict_churn_all_customers(
    features: pd.DataFrame,
    model,
    scaler=None,
    feature_cols: list = None,
) -> pd.DataFrame:
    """
    Predict churn probability for ALL customers using the trained model.
    Returns DataFrame with CustomerID and churn_probability.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    X_all = features[feature_cols].fillna(0)

    if scaler is not None:
        X_all_scaled = scaler.transform(X_all)
        probs = model.predict_proba(X_all_scaled)[:, 1]
    else:
        probs = model.predict_proba(X_all)[:, 1]

    result = features[["CustomerID"]].copy()
    result["churn_probability"] = probs

    # Assign risk bands
    result["churn_risk_band"] = pd.cut(
        result["churn_probability"],
        bins=[0, 0.3, 0.6, 0.8, 1.0],
        labels=["Low", "Medium", "High", "Critical"],
        include_lowest=True,
    )

    print("[CHURN] Risk band distribution:")
    for band in ["Low", "Medium", "High", "Critical"]:
        count = (result["churn_risk_band"] == band).sum()
        pct = count / len(result) * 100
        print(f"       {band}: {count:,} ({pct:.1f}%)")

    return result


# ---------------------------------------------------------------------------
# 6. BUSINESS IMPACT CALCULATOR
# ---------------------------------------------------------------------------
def business_impact_calculator(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    avg_customer_revenue: float = 200.0,
    intervention_cost_per_customer: float = 5.0,
    recovery_rate: float = 0.30,
) -> dict:
    """
    Calculate business impact of churn model at a given probability threshold.
    
    Parameters
    ----------
    threshold : probability cutoff for classifying as "churn"
    avg_customer_revenue : average annual revenue per at-risk customer (GBP)
    intervention_cost_per_customer : cost of retention action per customer (GBP)
    recovery_rate : % of flagged churners who are successfully retained
    """
    y_pred = (y_prob >= threshold).astype(int)

    tp = ((y_pred == 1) & (y_test == 1)).sum()  # True positives (caught churners)
    fp = ((y_pred == 1) & (y_test == 0)).sum()  # False positives (unnecessary interventions)
    fn = ((y_pred == 0) & (y_test == 1)).sum()  # False negatives (missed churners)
    tn = ((y_pred == 0) & (y_test == 0)).sum()  # True negatives

    total_interventions = tp + fp
    cost_of_interventions = total_interventions * intervention_cost_per_customer
    recovered_churners = int(tp * recovery_rate)
    revenue_saved = recovered_churners * avg_customer_revenue
    missed_revenue = fn * avg_customer_revenue
    net_roi = revenue_saved - cost_of_interventions

    impact = {
        "threshold": threshold,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "total_interventions": int(total_interventions),
        "cost_of_interventions": round(cost_of_interventions, 2),
        "recovered_churners": recovered_churners,
        "revenue_saved": round(revenue_saved, 2),
        "missed_revenue": round(missed_revenue, 2),
        "net_roi": round(net_roi, 2),
        "roi_multiple": round(revenue_saved / cost_of_interventions, 2) if cost_of_interventions > 0 else 0,
    }

    return impact


# ---------------------------------------------------------------------------
# 7. THRESHOLD SWEEP (for Streamlit slider)
# ---------------------------------------------------------------------------
def threshold_sweep(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    avg_customer_revenue: float = 200.0,
    intervention_cost: float = 5.0,
    recovery_rate: float = 0.30,
) -> pd.DataFrame:
    """
    Compute business impact metrics across a range of thresholds.
    Returns a DataFrame for plotting in Streamlit.
    """
    thresholds = np.arange(0.05, 1.0, 0.05)
    rows = []

    for t in thresholds:
        impact = business_impact_calculator(
            y_test, y_prob, t, avg_customer_revenue, intervention_cost, recovery_rate,
        )
        rows.append(impact)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8. HYPOTHESIS VALIDATION (H3)
# ---------------------------------------------------------------------------
def validate_h3(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    churn_results: dict,
) -> dict:
    """
    Validate H3: 
    - Single-purchase customers in first 60 days have >70% churn probability
    - Return rate and declining order value are top-5 features
    """
    # Merge
    data = features.merge(labels, on="CustomerID")

    # Single purchase customers churn rate
    single_purchase = data[data["is_single_purchase_customer"] == 1]
    single_churn_rate = single_purchase["churned"].mean() * 100

    # Top 5 features (from best model)
    best_model = "XGBoost" if "XGBoost" in churn_results else "Logistic Regression"
    fi = churn_results[best_model]["feature_importance"]
    top_5_features = sorted(fi, key=fi.get, reverse=True)[:5]

    result = {
        "hypothesis": "H3: Single-purchase customers have >70% churn, return_rate in top-5 features",
        "single_purchase_churn_rate": round(single_churn_rate, 2),
        "h3_churn_validated": single_churn_rate > 70,
        "top_5_features": top_5_features,
        "return_rate_in_top5": "return_rate" in top_5_features,
        "order_value_trend_in_top5": "order_value_trend" in top_5_features,
        "best_model": best_model,
        "best_auc": round(churn_results[best_model]["auc_roc"], 4),
    }

    print(f"[H3] Single-purchase churn rate: {result['single_purchase_churn_rate']}% "
          f"({'✓ VALIDATED' if result['h3_churn_validated'] else '✗ NOT validated'})")
    print(f"[H3] Top 5 features: {', '.join(top_5_features)}")
    print(f"[H3] return_rate in top 5: {'✓' if result['return_rate_in_top5'] else '✗'}")
    print(f"[H3] order_value_trend in top 5: {'✓' if result['order_value_trend_in_top5'] else '✗'}")

    return result


# ---------------------------------------------------------------------------
# 9. RETENTION ACTION MAPPING
# ---------------------------------------------------------------------------
RETENTION_ACTIONS = {
    "Low": {
        "risk_level": "Low",
        "probability_range": "0–30%",
        "action": "No intervention — standard experience",
        "estimated_cost_per_customer": 0,
    },
    "Medium": {
        "risk_level": "Medium",
        "probability_range": "30–60%",
        "action": "Soft nudge: personalized email + loyalty points reminder",
        "estimated_cost_per_customer": 2.0,
    },
    "High": {
        "risk_level": "High",
        "probability_range": "60–80%",
        "action": "Active retention: 15% discount + 'We miss you' campaign",
        "estimated_cost_per_customer": 8.0,
    },
    "Critical": {
        "risk_level": "Critical",
        "probability_range": "80–100%",
        "action": "Aggressive win-back: 25% discount + free shipping + call",
        "estimated_cost_per_customer": 15.0,
    },
}
