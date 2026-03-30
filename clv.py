"""
clv.py — Customer Lifetime Value Prediction
=============================================
Problem 2: Predict future customer value using two approaches:

Option A (Statistical — PRIMARY):
  - BG/NBD model for expected future purchases
  - Gamma-Gamma model for expected average profit per transaction
  - CLV = Expected Purchases × Expected Avg Profit × Profit Margin

Option B (Machine Learning — SECONDARY):
  - Feature engineering from calibration period
  - Target = actual spend in holdout period
  - Models: Linear Regression → Random Forest → XGBoost

Both approaches use a calibration/holdout split for validation.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
import joblib
import os

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. CALIBRATION / HOLDOUT SPLIT
# ---------------------------------------------------------------------------
def split_calibration_holdout(
    df_clean: pd.DataFrame,
    calibration_months: int = 9,
    holdout_months: int = 3,
) -> tuple:
    """
    Split cleaned transaction data into calibration and holdout periods.
    
    Default: first 9 months for calibration, last 3 months for holdout.
    """
    min_date = df_clean["InvoiceDate"].min()
    max_date = df_clean["InvoiceDate"].max()

    # Calculate split point
    total_days = (max_date - min_date).days
    cal_days = int(total_days * calibration_months / (calibration_months + holdout_months))
    split_date = min_date + timedelta(days=cal_days)

    df_cal = df_clean[df_clean["InvoiceDate"] <= split_date].copy()
    df_holdout = df_clean[df_clean["InvoiceDate"] > split_date].copy()

    print(f"[CLV] Calibration period: {min_date.date()} to {split_date.date()} ({len(df_cal):,} rows)")
    print(f"[CLV] Holdout period:     {split_date.date()} to {max_date.date()} ({len(df_holdout):,} rows)")

    return df_cal, df_holdout, split_date


# ---------------------------------------------------------------------------
# 2. PREPARE DATA FOR LIFETIMES LIBRARY
# ---------------------------------------------------------------------------
def prepare_lifetimes_data(df_cal: pd.DataFrame, split_date: pd.Timestamp) -> pd.DataFrame:
    """
    Prepare customer-level summary for BG/NBD and Gamma-Gamma models.
    
    Uses the `lifetimes` library format:
      - frequency: number of REPEAT purchases (total purchases - 1)
      - recency: time between first and last purchase (in days)
      - T: time between first purchase and end of calibration period
      - monetary_value: average order value (excluding first purchase)
    """
    customer_cal = (
        df_cal.groupby("CustomerID")
        .agg(
            first_purchase=("InvoiceDate", "min"),
            last_purchase=("InvoiceDate", "max"),
            total_orders=("InvoiceNo", "nunique"),
            total_revenue=("Revenue", "sum"),
        )
        .reset_index()
    )

    customer_cal["frequency"] = customer_cal["total_orders"] - 1  # repeat purchases only
    customer_cal["recency"] = (customer_cal["last_purchase"] - customer_cal["first_purchase"]).dt.days
    customer_cal["T"] = (split_date - customer_cal["first_purchase"]).dt.days

    # Monetary = avg order value for repeat customers (exclude first purchase)
    # For customers with frequency=0, monetary_value is undefined → set to 0
    repeat_customers = df_cal.copy()

    # Get first order date per customer
    first_orders = df_cal.groupby("CustomerID")["InvoiceDate"].min().reset_index()
    first_orders.columns = ["CustomerID", "first_order_date"]
    repeat_customers = repeat_customers.merge(first_orders, on="CustomerID")

    # Exclude first day's transactions for monetary calculation
    repeat_transactions = repeat_customers[
        repeat_customers["InvoiceDate"] > repeat_customers["first_order_date"]
    ]

    monetary_per_customer = (
        repeat_transactions.groupby("CustomerID")
        .agg(monetary_value=("Revenue", lambda x: x.sum() / repeat_transactions.loc[x.index, "InvoiceNo"].nunique()))
        .reset_index()
    )

    customer_cal = customer_cal.merge(monetary_per_customer, on="CustomerID", how="left")
    customer_cal["monetary_value"] = customer_cal["monetary_value"].fillna(0)

    # Keep only relevant columns
    lifetimes_data = customer_cal[["CustomerID", "frequency", "recency", "T", "monetary_value"]].copy()

    print(f"[CLV] Lifetimes data prepared: {len(lifetimes_data):,} customers")
    print(f"       Repeat customers (frequency > 0): {(lifetimes_data['frequency'] > 0).sum():,}")

    return lifetimes_data


# ---------------------------------------------------------------------------
# 3. BG/NBD + GAMMA-GAMMA MODEL (Option A)
# ---------------------------------------------------------------------------
def fit_bgnbd_model(lifetimes_data: pd.DataFrame):
    """
    Fit BG/NBD model for predicting expected future purchases.
    """
    from lifetimes import BetaGeoFitter

    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(
        lifetimes_data["frequency"],
        lifetimes_data["recency"],
        lifetimes_data["T"],
    )

    print(f"[CLV] BG/NBD model fitted.")
    print(f"       Parameters: {bgf.summary}")

    return bgf


def fit_gamma_gamma_model(lifetimes_data: pd.DataFrame):
    """
    Fit Gamma-Gamma model for predicting expected average profit.
    Only uses customers with frequency > 0 (repeat customers).
    """
    from lifetimes import GammaGammaFitter

    # Filter to repeat customers only
    repeat_data = lifetimes_data[lifetimes_data["frequency"] > 0].copy()

    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(
        repeat_data["frequency"],
        repeat_data["monetary_value"],
    )

    print(f"[CLV] Gamma-Gamma model fitted on {len(repeat_data):,} repeat customers.")

    return ggf


def predict_clv_statistical(
    bgf,
    ggf,
    lifetimes_data: pd.DataFrame,
    time_months: int = 6,
    discount_rate: float = 0.01,
    profit_margin: float = 0.15,
) -> pd.DataFrame:
    """
    Predict CLV using BG/NBD + Gamma-Gamma.
    
    CLV = Expected Purchases × Expected Avg Profit × Profit Margin
    
    Parameters
    ----------
    time_months : prediction horizon in months
    discount_rate : monthly discount rate for DCF adjustment
    profit_margin : assumed profit margin on revenue
    """
    from lifetimes import GammaGammaFitter

    data = lifetimes_data.copy()

    # Expected purchases in next `time_months` months (convert to days: ~30.44 days/month)
    t = time_months * 30.44
    data["expected_purchases"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        t,
        data["frequency"],
        data["recency"],
        data["T"],
    )

    # Probability alive
    data["prob_alive"] = bgf.conditional_probability_alive(
        data["frequency"],
        data["recency"],
        data["T"],
    )

    # Expected average profit (only for repeat customers)
    repeat_mask = data["frequency"] > 0
    data.loc[repeat_mask, "expected_avg_profit"] = ggf.conditional_expected_average_profit(
        data.loc[repeat_mask, "frequency"],
        data.loc[repeat_mask, "monetary_value"],
    )
    # For single-purchase customers, use overall mean monetary value as estimate
    data["expected_avg_profit"] = data["expected_avg_profit"].fillna(
        data.loc[repeat_mask, "monetary_value"].mean()
    )

    # Predicted CLV
    data["predicted_clv"] = (
        data["expected_purchases"] * data["expected_avg_profit"] * profit_margin
    )

    # Clip negative CLV to 0
    data["predicted_clv"] = data["predicted_clv"].clip(lower=0)

    print(f"[CLV] Statistical CLV predicted for {len(data):,} customers")
    print(f"       Mean CLV: £{data['predicted_clv'].mean():.2f}")
    print(f"       Median CLV: £{data['predicted_clv'].median():.2f}")
    print(f"       Max CLV: £{data['predicted_clv'].max():.2f}")

    return data


# ---------------------------------------------------------------------------
# 4. ML-BASED CLV PREDICTION (Option B)
# ---------------------------------------------------------------------------
def engineer_clv_features(df_cal: pd.DataFrame, df_holdout: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from calibration period for ML CLV prediction.
    Target = total spend in holdout period.
    """
    # Features from calibration period
    reference_date = df_cal["InvoiceDate"].max() + pd.Timedelta(days=1)

    features = (
        df_cal.groupby("CustomerID")
        .agg(
            recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
            frequency=("InvoiceNo", "nunique"),
            monetary=("Revenue", "sum"),
            avg_order_value=("Revenue", lambda x: x.sum()),
            unique_products=("StockCode", "nunique"),
            total_quantity=("Quantity", "sum"),
            first_purchase=("InvoiceDate", "min"),
            last_purchase=("InvoiceDate", "max"),
        )
        .reset_index()
    )

    features["avg_order_value"] = features["monetary"] / features["frequency"]
    features["tenure_days"] = (reference_date - features["first_purchase"]).dt.days
    features["avg_days_between"] = np.where(
        features["frequency"] > 1,
        (features["last_purchase"] - features["first_purchase"]).dt.days / (features["frequency"] - 1),
        0,
    )
    features["items_per_order"] = features["total_quantity"] / features["frequency"]

    # Drop date columns
    features = features.drop(columns=["first_purchase", "last_purchase"])

    # Target: actual spend in holdout
    holdout_spend = (
        df_holdout.groupby("CustomerID")["Revenue"]
        .sum()
        .reset_index()
        .rename(columns={"Revenue": "holdout_revenue"})
    )

    features = features.merge(holdout_spend, on="CustomerID", how="left")
    features["holdout_revenue"] = features["holdout_revenue"].fillna(0)

    return features


def train_clv_ml_models(features: pd.DataFrame, save_dir: str = "models/") -> dict:
    """
    Train ML models for CLV prediction.
    Models: Linear Regression → Random Forest → XGBoost
    """
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    try:
        from xgboost import XGBRegressor
        has_xgboost = True
    except ImportError:
        has_xgboost = False
        print("[WARN] XGBoost not installed. Skipping XGB model.")

    feature_cols = [
        "recency", "frequency", "monetary", "avg_order_value",
        "unique_products", "total_quantity", "tenure_days",
        "avg_days_between", "items_per_order",
    ]

    X = features[feature_cols].fillna(0)
    y = features["holdout_revenue"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}

    # --- Linear Regression ---
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test).clip(min=0)
    results["Linear Regression"] = {
        "model": lr,
        "MAE": mean_absolute_error(y_test, y_pred_lr),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        "R2": r2_score(y_test, y_pred_lr),
        "predictions": y_pred_lr,
    }

    # --- Random Forest ---
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test).clip(min=0)
    results["Random Forest"] = {
        "model": rf,
        "MAE": mean_absolute_error(y_test, y_pred_rf),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        "R2": r2_score(y_test, y_pred_rf),
        "predictions": y_pred_rf,
        "feature_importance": dict(zip(feature_cols, rf.feature_importances_)),
    }

    # --- XGBoost ---
    if has_xgboost:
        xgb = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        )
        xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        y_pred_xgb = xgb.predict(X_test).clip(min=0)
        results["XGBoost"] = {
            "model": xgb,
            "MAE": mean_absolute_error(y_test, y_pred_xgb),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
            "R2": r2_score(y_test, y_pred_xgb),
            "predictions": y_pred_xgb,
            "feature_importance": dict(zip(feature_cols, xgb.feature_importances_)),
        }

    # Print comparison
    print("\n[CLV ML] Model Comparison:")
    print(f"{'Model':<25} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
    print("-" * 55)
    for name, res in results.items():
        print(f"{name:<25} {res['MAE']:>10.2f} {res['RMSE']:>10.2f} {res['R2']:>10.4f}")

    # Save best model
    os.makedirs(save_dir, exist_ok=True)
    best_name = min(results, key=lambda k: results[k]["MAE"])
    joblib.dump(results[best_name]["model"], os.path.join(save_dir, "clv_best_model.pkl"))
    print(f"\n[CLV ML] Best model: {best_name} (saved to {save_dir})")

    return results, X_test, y_test


# ---------------------------------------------------------------------------
# 5. CLV TIER ASSIGNMENT
# ---------------------------------------------------------------------------
def assign_clv_tiers(clv_data: pd.DataFrame, clv_col: str = "predicted_clv") -> pd.DataFrame:
    """
    Assign CLV tiers: Platinum (top 10%), Gold (next 20%), Silver (next 30%), Bronze (bottom 40%).
    """
    data = clv_data.copy()

    percentiles = data[clv_col].quantile([0.40, 0.70, 0.90]).values

    def _tier(val):
        if val >= percentiles[2]:
            return "Platinum"
        elif val >= percentiles[1]:
            return "Gold"
        elif val >= percentiles[0]:
            return "Silver"
        else:
            return "Bronze"

    data["CLV_Tier"] = data[clv_col].apply(_tier)

    # Tier boundaries for display
    tier_boundaries = {
        "Bronze": f"< £{percentiles[0]:.2f}",
        "Silver": f"£{percentiles[0]:.2f} – £{percentiles[1]:.2f}",
        "Gold": f"£{percentiles[1]:.2f} – £{percentiles[2]:.2f}",
        "Platinum": f"> £{percentiles[2]:.2f}",
    }

    print("[CLV] Tier distribution:")
    for tier, boundary in tier_boundaries.items():
        count = (data["CLV_Tier"] == tier).sum()
        print(f"       {tier}: {count:,} customers ({boundary})")

    return data, tier_boundaries


# ---------------------------------------------------------------------------
# 6. HYPOTHESIS VALIDATION (H2)
# ---------------------------------------------------------------------------
def validate_h2(clv_data: pd.DataFrame, cac: float = 380.0, clv_col: str = "predicted_clv") -> dict:
    """
    Validate H2: At current CAC, at least 30% of customers are unprofitable.
    """
    # Convert CAC to GBP (1 GBP ≈ ₹105)
    cac_gbp = cac / 105.0

    total = len(clv_data)
    unprofitable = (clv_data[clv_col] < cac_gbp).sum()
    unprofitable_pct = unprofitable / total * 100

    result = {
        "hypothesis": "H2: At current CAC, ≥30% of customers are unprofitable over 6-month horizon",
        "cac_inr": cac,
        "cac_gbp": round(cac_gbp, 2),
        "unprofitable_count": unprofitable,
        "unprofitable_pct": round(unprofitable_pct, 2),
        "h2_validated": unprofitable_pct >= 30,
        "mean_clv": round(clv_data[clv_col].mean(), 2),
        "median_clv": round(clv_data[clv_col].median(), 2),
    }

    print(f"[H2] Unprofitable customers (CLV < CAC): {result['unprofitable_pct']}% "
          f"({'✓ VALIDATED' if result['h2_validated'] else '✗ NOT validated'})")

    return result


# ---------------------------------------------------------------------------
# 7. BUDGET CALCULATOR LOGIC
# ---------------------------------------------------------------------------
def budget_calculator(clv_data: pd.DataFrame, cac: float, clv_col: str = "predicted_clv") -> pd.DataFrame:
    """
    For each CLV tier, compute: avg CLV, CAC, CAC as % of CLV, profit/loss per customer.
    Used for the Streamlit budget calculator widget.
    """
    cac_gbp = cac / 105.0

    tier_analysis = (
        clv_data.groupby("CLV_Tier")
        .agg(
            customer_count=("CustomerID", "count"),
            avg_clv=(clv_col, "mean"),
            median_clv=(clv_col, "median"),
            total_clv=(clv_col, "sum"),
        )
        .reset_index()
    )

    tier_analysis["cac_gbp"] = cac_gbp
    tier_analysis["profit_per_customer"] = tier_analysis["avg_clv"] - cac_gbp
    tier_analysis["cac_as_pct_of_clv"] = (cac_gbp / tier_analysis["avg_clv"] * 100).round(2)
    tier_analysis["is_profitable"] = tier_analysis["profit_per_customer"] > 0

    # Order tiers
    tier_order = ["Platinum", "Gold", "Silver", "Bronze"]
    tier_analysis["CLV_Tier"] = pd.Categorical(tier_analysis["CLV_Tier"], categories=tier_order, ordered=True)
    tier_analysis = tier_analysis.sort_values("CLV_Tier").reset_index(drop=True)

    return tier_analysis
