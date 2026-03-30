"""
data_cleaning.py — Data Loading, Cleaning & Feature Preparation
================================================================
Handles the full cleaning pipeline for Online Retail II dataset:
  1. Load raw Excel sheets
  2. Drop null CustomerIDs
  3. Handle cancellations (InvoiceNo starting with 'C')
  4. Remove negative/zero UnitPrice
  5. Filter to UK-only transactions
  6. Cap/remove outliers
  7. Create Revenue column
  8. Parse dates properly
  9. Export cleaned data
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. LOAD RAW DATA
# ---------------------------------------------------------------------------
def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load Online Retail II Excel file (both sheets) and concatenate.
    The UCI dataset ships as two sheets: Year 2009-2010 and Year 2010-2011.
    """
    print("[INFO] Loading raw data from:", filepath)

    if filepath.endswith(".xlsx") or filepath.endswith(".xls"):
        # Read both sheets and combine
        sheet1 = pd.read_excel(filepath, sheet_name="Year 2009-2010", engine="openpyxl")
        sheet2 = pd.read_excel(filepath, sheet_name="Year 2010-2011", engine="openpyxl")
        df = pd.concat([sheet1, sheet2], ignore_index=True)
    elif filepath.endswith(".csv"):
        df = pd.read_csv(filepath, encoding="ISO-8859-1")
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

    # Standardize column names (UCI dataset sometimes has 'Invoice' vs 'InvoiceNo')
    col_map = {
        "Invoice": "InvoiceNo",
        "Customer ID": "CustomerID",
        "Price": "UnitPrice",
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    print(f"[INFO] Raw data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# 2. CLEANING PIPELINE
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame, country_filter: str = "United Kingdom") -> pd.DataFrame:
    """
    Full cleaning pipeline. Returns cleaned DataFrame + a cleaning report dict.
    """
    report = {"raw_rows": len(df)}

    # --- a. Parse InvoiceDate ---
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    # --- b. Drop null CustomerIDs ---
    null_cust = df["CustomerID"].isna().sum()
    df = df.dropna(subset=["CustomerID"]).copy()
    df["CustomerID"] = df["CustomerID"].astype(int)
    report["null_customer_ids_dropped"] = null_cust

    # --- c. Separate cancellations ---
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["is_cancellation"] = df["InvoiceNo"].str.startswith("C")
    cancellation_count = df["is_cancellation"].sum()
    report["cancellation_rows"] = cancellation_count

    # Keep cancellation flag but filter them out for core analysis
    df_clean = df[~df["is_cancellation"]].copy()

    # --- d. Remove negative / zero Quantity ---
    neg_qty = (df_clean["Quantity"] <= 0).sum()
    df_clean = df_clean[df_clean["Quantity"] > 0].copy()
    report["negative_quantity_removed"] = neg_qty

    # --- e. Remove negative / zero UnitPrice ---
    neg_price = (df_clean["UnitPrice"] <= 0).sum()
    df_clean = df_clean[df_clean["UnitPrice"] > 0].copy()
    report["negative_price_removed"] = neg_price

    # --- f. Filter to single country ---
    if country_filter:
        other_countries = df_clean[df_clean["Country"] != country_filter].shape[0]
        df_clean = df_clean[df_clean["Country"] == country_filter].copy()
        report["non_uk_rows_removed"] = other_countries

    # --- g. Create Revenue ---
    df_clean["Revenue"] = df_clean["Quantity"] * df_clean["UnitPrice"]

    # --- h. Outlier handling ---
    # Cap extreme bulk orders (Quantity > 5000) and high unit prices (> £500)
    qty_outliers = (df_clean["Quantity"] > 5000).sum()
    price_outliers = (df_clean["UnitPrice"] > 500).sum()
    df_clean = df_clean[df_clean["Quantity"] <= 5000].copy()
    df_clean = df_clean[df_clean["UnitPrice"] <= 500].copy()
    report["quantity_outliers_removed"] = qty_outliers
    report["price_outliers_removed"] = price_outliers

    # --- i. Remove duplicate rows ---
    dup_count = df_clean.duplicated().sum()
    df_clean = df_clean.drop_duplicates().copy()
    report["duplicates_removed"] = dup_count

    report["clean_rows"] = len(df_clean)
    report["unique_customers"] = df_clean["CustomerID"].nunique()
    report["date_range"] = (
        df_clean["InvoiceDate"].min().strftime("%Y-%m-%d"),
        df_clean["InvoiceDate"].max().strftime("%Y-%m-%d"),
    )

    print("[INFO] Cleaning complete:")
    for k, v in report.items():
        print(f"       {k}: {v}")

    return df_clean, report


# ---------------------------------------------------------------------------
# 3. CANCELLATION TABLE (for churn features — return rate)
# ---------------------------------------------------------------------------
def build_cancellation_table(df_raw: pd.DataFrame, country_filter="United Kingdom") -> pd.DataFrame:
    """
    Build a per-customer cancellation summary from the raw data.
    Used as a feature in churn prediction (return_rate).
    """
    df = df_raw.copy()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int)

    if country_filter:
        df = df[df["Country"] == country_filter]

    df["is_cancellation"] = df["InvoiceNo"].str.startswith("C")

    cancel_summary = (
        df.groupby("CustomerID")
        .agg(
            total_orders=("InvoiceNo", "nunique"),
            cancelled_orders=("is_cancellation", "sum"),
        )
        .reset_index()
    )
    cancel_summary["return_rate"] = (
        cancel_summary["cancelled_orders"] / cancel_summary["total_orders"]
    ).fillna(0)

    return cancel_summary[["CustomerID", "return_rate", "cancelled_orders"]]


# ---------------------------------------------------------------------------
# 4. AGGREGATE CUSTOMER-LEVEL DATA
# ---------------------------------------------------------------------------
def build_customer_table(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction-level data to customer level.
    Returns a table with one row per customer.
    """
    reference_date = df_clean["InvoiceDate"].max() + pd.Timedelta(days=1)

    customer = (
        df_clean.groupby("CustomerID")
        .agg(
            first_purchase=("InvoiceDate", "min"),
            last_purchase=("InvoiceDate", "max"),
            frequency=("InvoiceNo", "nunique"),
            total_revenue=("Revenue", "sum"),
            total_quantity=("Quantity", "sum"),
            unique_products=("StockCode", "nunique"),
            avg_order_value=("Revenue", lambda x: x.sum() / df_clean.loc[x.index, "InvoiceNo"].nunique()),
        )
        .reset_index()
    )

    customer["recency_days"] = (reference_date - customer["last_purchase"]).dt.days
    customer["tenure_days"] = (reference_date - customer["first_purchase"]).dt.days
    customer["avg_days_between_purchases"] = np.where(
        customer["frequency"] > 1,
        (customer["last_purchase"] - customer["first_purchase"]).dt.days / (customer["frequency"] - 1),
        0,
    )
    customer["reference_date"] = reference_date

    return customer


# ---------------------------------------------------------------------------
# 5. COHORT TABLE (for EDA)
# ---------------------------------------------------------------------------
def build_cohort_table(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Build a monthly cohort retention table.
    Returns a pivot table: rows = cohort month, columns = period index, values = retention %.
    """
    df = df_clean.copy()
    df["order_month"] = df["InvoiceDate"].dt.to_period("M")

    # Assign cohort = month of first purchase per customer
    cohort = df.groupby("CustomerID")["order_month"].min().reset_index()
    cohort.columns = ["CustomerID", "cohort_month"]

    df = df.merge(cohort, on="CustomerID")
    df["period_index"] = (df["order_month"] - df["cohort_month"]).apply(lambda x: x.n)

    # Count unique customers per cohort + period
    cohort_data = (
        df.groupby(["cohort_month", "period_index"])["CustomerID"]
        .nunique()
        .reset_index()
    )
    cohort_data.columns = ["cohort_month", "period_index", "customers"]

    # Pivot
    cohort_pivot = cohort_data.pivot(index="cohort_month", columns="period_index", values="customers")

    # Convert to retention rate (% of cohort still active)
    cohort_sizes = cohort_pivot.iloc[:, 0]
    retention_table = cohort_pivot.divide(cohort_sizes, axis=0) * 100

    return retention_table


# ---------------------------------------------------------------------------
# 6. EDA SUMMARY STATISTICS
# ---------------------------------------------------------------------------
def compute_eda_stats(df_clean: pd.DataFrame) -> dict:
    """
    Compute key EDA statistics for the overview dashboard.
    """
    stats = {
        "total_customers": df_clean["CustomerID"].nunique(),
        "total_transactions": df_clean["InvoiceNo"].nunique(),
        "total_revenue": df_clean["Revenue"].sum(),
        "avg_order_value": df_clean.groupby("InvoiceNo")["Revenue"].sum().mean(),
        "median_order_value": df_clean.groupby("InvoiceNo")["Revenue"].sum().median(),
        "avg_revenue_per_customer": df_clean.groupby("CustomerID")["Revenue"].sum().mean(),
        "date_range_start": df_clean["InvoiceDate"].min(),
        "date_range_end": df_clean["InvoiceDate"].max(),
        "total_products": df_clean["StockCode"].nunique(),
        "avg_items_per_order": df_clean.groupby("InvoiceNo")["Quantity"].sum().mean(),
    }

    # Monthly revenue
    monthly_rev = (
        df_clean.set_index("InvoiceDate")
        .resample("M")["Revenue"]
        .sum()
        .reset_index()
    )
    monthly_rev.columns = ["month", "revenue"]
    stats["monthly_revenue"] = monthly_rev

    # Order frequency distribution
    order_freq = df_clean.groupby("CustomerID")["InvoiceNo"].nunique().reset_index()
    order_freq.columns = ["CustomerID", "order_count"]
    stats["order_frequency_dist"] = order_freq

    # Revenue by day of week
    dow_rev = df_clean.copy()
    dow_rev["day_of_week"] = dow_rev["InvoiceDate"].dt.day_name()
    dow_rev = dow_rev.groupby("day_of_week")["Revenue"].sum().reset_index()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_rev["day_of_week"] = pd.Categorical(dow_rev["day_of_week"], categories=day_order, ordered=True)
    dow_rev = dow_rev.sort_values("day_of_week")
    stats["revenue_by_dow"] = dow_rev

    # New vs returning by month
    first_purchase = df_clean.groupby("CustomerID")["InvoiceDate"].min().reset_index()
    first_purchase.columns = ["CustomerID", "first_purchase_date"]
    first_purchase["first_month"] = first_purchase["first_purchase_date"].dt.to_period("M")

    df_temp = df_clean.merge(first_purchase, on="CustomerID")
    df_temp["order_month"] = df_temp["InvoiceDate"].dt.to_period("M")
    df_temp["is_new"] = df_temp["order_month"] == df_temp["first_month"]

    new_vs_ret = (
        df_temp.groupby(["order_month", "is_new"])["CustomerID"]
        .nunique()
        .unstack(fill_value=0)
        .reset_index()
    )
    new_vs_ret.columns = ["month", "returning", "new"]
    stats["new_vs_returning"] = new_vs_ret

    return stats


# ---------------------------------------------------------------------------
# 7. SAVE / LOAD HELPERS
# ---------------------------------------------------------------------------
def save_processed(df: pd.DataFrame, path: str):
    """Save processed DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[INFO] Saved: {path} ({len(df):,} rows)")


def load_processed(path: str) -> pd.DataFrame:
    """Load processed DataFrame from CSV."""
    df = pd.read_csv(path)
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df
