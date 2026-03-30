"""
rfm.py — RFM Segmentation Engine
==================================
Problem 1: Customer Segmentation using Recency, Frequency, Monetary analysis.

Pipeline:
  1. Compute R, F, M per customer
  2. Score each on 1–4 scale (quartile-based)
  3. Map combined RFM scores to named segments
  4. Generate segment-level summaries and recommendations
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# 1. COMPUTE RFM VALUES
# ---------------------------------------------------------------------------
def compute_rfm(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Recency, Frequency, Monetary values per customer.
    
    Parameters
    ----------
    df_clean : Cleaned transaction DataFrame with columns:
               CustomerID, InvoiceNo, InvoiceDate, Revenue
    
    Returns
    -------
    rfm_table : DataFrame with CustomerID, Recency, Frequency, Monetary
    """
    reference_date = df_clean["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df_clean.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("Revenue", "sum"),
        )
        .reset_index()
    )

    rfm["reference_date"] = reference_date
    return rfm


# ---------------------------------------------------------------------------
# 2. SCORE RFM (Quartile-Based, 1-4)
# ---------------------------------------------------------------------------
def score_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Assign 1–4 scores to R, F, M using quartile-based binning.
    For Recency, lower is better → 4 = most recent.
    For Frequency and Monetary, higher is better → 4 = highest.
    """
    rfm = rfm.copy()

    # Recency: lower = better → reverse scoring
    rfm["R_Score"] = pd.qcut(rfm["Recency"], q=4, labels=[4, 3, 2, 1], duplicates="drop")
    rfm["R_Score"] = rfm["R_Score"].astype(int)

    # Frequency: higher = better
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=4, labels=[1, 2, 3, 4], duplicates="drop")
    rfm["F_Score"] = rfm["F_Score"].astype(int)

    # Monetary: higher = better
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], q=4, labels=[1, 2, 3, 4], duplicates="drop")
    rfm["M_Score"] = rfm["M_Score"].astype(int)

    # Combined RFM score string
    rfm["RFM_Score"] = rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)

    # Numeric aggregate score (for sorting)
    rfm["RFM_Agg_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    return rfm


# ---------------------------------------------------------------------------
# 3. SEGMENT MAPPING
# ---------------------------------------------------------------------------
SEGMENT_RULES = {
    # (R range, F range, M range) → segment name
    # Using a function-based approach for flexibility
}


def map_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Map RFM scores to named customer segments using rule-based logic.
    
    Segment Definitions (from project brief):
    - Champions:          R=4, F=4, M=4
    - Loyal Customers:    R∈{3,4}, F∈{3,4}, M∈{3,4} (excl Champions)
    - Potential Loyalists: R∈{3,4}, F∈{1,2}, M∈{1,2,3,4}
    - At Risk:            R∈{1,2}, F∈{3,4}, M∈{3,4}
    - Hibernating:        R∈{1,2}, F∈{1,2}, M∈{1,2}
    - Lost:               R=1, F=1, M=1
    
    Additional segments for richer analysis:
    - Big Spenders:       R∈{2,3,4}, F∈{1,2}, M∈{3,4}
    - About to Sleep:     R=2, F∈{2,3}, M∈{2,3}
    """
    rfm = rfm.copy()

    def _assign_segment(row):
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]

        # Most specific first
        if r == 1 and f == 1 and m == 1:
            return "Lost"
        elif r >= 3 and f >= 3 and m >= 3:
            if r == 4 and f == 4 and m == 4:
                return "Champions"
            return "Loyal Customers"
        elif r >= 3 and f <= 2:
            return "Potential Loyalists"
        elif r <= 2 and f >= 3 and m >= 3:
            return "At Risk"
        elif r <= 2 and f <= 2 and m <= 2:
            return "Hibernating"
        elif r <= 2 and f >= 3 and m <= 2:
            return "Need Attention"
        elif r >= 3 and f >= 3 and m <= 2:
            return "Promising"
        elif r <= 2 and f <= 2 and m >= 3:
            return "Big Spenders (Lapsed)"
        else:
            return "Others"

    rfm["Segment"] = rfm.apply(_assign_segment, axis=1)
    return rfm


# ---------------------------------------------------------------------------
# 4. SEGMENT SUMMARY
# ---------------------------------------------------------------------------
def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Generate per-segment summary statistics.
    """
    summary = (
        rfm.groupby("Segment")
        .agg(
            customer_count=("CustomerID", "count"),
            avg_recency=("Recency", "mean"),
            avg_frequency=("Frequency", "mean"),
            avg_monetary=("Monetary", "mean"),
            total_revenue=("Monetary", "sum"),
            median_monetary=("Monetary", "median"),
        )
        .reset_index()
    )

    total_customers = summary["customer_count"].sum()
    total_rev = summary["total_revenue"].sum()

    summary["pct_customers"] = (summary["customer_count"] / total_customers * 100).round(2)
    summary["pct_revenue"] = (summary["total_revenue"] / total_rev * 100).round(2)

    summary = summary.sort_values("total_revenue", ascending=False).reset_index(drop=True)
    return summary


# ---------------------------------------------------------------------------
# 5. CAMPAIGN RECOMMENDATIONS
# ---------------------------------------------------------------------------
CAMPAIGN_RECOMMENDATIONS = {
    "Champions": {
        "strategy": "Reward & Advocate",
        "actions": [
            "Early access to new collections and flash sales",
            "Referral program with premium incentives (₹500 credit per referral)",
            "VIP events and exclusive StyleKart Insider Gold perks",
        ],
        "channel": "In-app + Personalized email",
        "discount": "None needed — value lies in exclusivity",
    },
    "Loyal Customers": {
        "strategy": "Upsell & Retain",
        "actions": [
            "Loyalty tier upgrade nudge (Silver → Gold benefits preview)",
            "Cross-sell complementary categories (accessories with apparel)",
            "Birthday/anniversary surprise discount (10% off)",
        ],
        "channel": "Email + Push notification",
        "discount": "10% targeted on cross-sell categories",
    },
    "Potential Loyalists": {
        "strategy": "Nurture & Convert",
        "actions": [
            "Second-purchase incentive: 15% off within 14 days",
            "Personalized product recommendations based on first purchase",
            "Welcome series drip campaign (3-email sequence)",
        ],
        "channel": "Email automation + Retargeting ads",
        "discount": "15% off next purchase",
    },
    "At Risk": {
        "strategy": "Urgent Win-Back",
        "actions": [
            "'We miss you' email campaign with 20% off, 7-day validity",
            "Show 'customers like you bought...' personalized picks",
            "SMS reminder with limited-time free shipping",
        ],
        "channel": "Email + SMS + Retargeting",
        "discount": "20% off with urgency timer",
    },
    "Hibernating": {
        "strategy": "Re-Engage or Release",
        "actions": [
            "Deep discount (30% off) with re-engagement survey",
            "New arrivals showcase email (show what they've missed)",
            "If no response in 30 days, move to 'Lost' and reduce spend",
        ],
        "channel": "Email + one-time push",
        "discount": "30% off",
    },
    "Lost": {
        "strategy": "Paid Retargeting or Exclude",
        "actions": [
            "Low-cost retargeting ads (social media) with steep discount",
            "If CLV < CAC, exclude from paid campaigns entirely",
            "Survey: 'What made you leave?' for product/service insights",
        ],
        "channel": "Paid social (low budget) or suppress",
        "discount": "35% off or exclude to save budget",
    },
    "Need Attention": {
        "strategy": "Re-activate with Value",
        "actions": [
            "Highlight new arrivals in their preferred category",
            "Limited time offer to drive a quick purchase",
            "Feedback request: ask what would bring them back",
        ],
        "channel": "Email + Push notification",
        "discount": "15% off",
    },
    "Promising": {
        "strategy": "Encourage Higher Spend",
        "actions": [
            "Bundle offers to increase basket size",
            "Free shipping on orders above a threshold",
            "Product recommendations in higher price ranges",
        ],
        "channel": "Email + In-app banners",
        "discount": "Free shipping above ₹1,500",
    },
    "Big Spenders (Lapsed)": {
        "strategy": "Priority Win-Back",
        "actions": [
            "Personal outreach: curated collection email",
            "Exclusive comeback offer with premium positioning",
            "Loyalty points reinstatement if previously active",
        ],
        "channel": "Personalized email + Phone (if high CLV)",
        "discount": "20% off + bonus loyalty points",
    },
    "Others": {
        "strategy": "Monitor & Nudge",
        "actions": [
            "Standard email campaigns with personalized content",
            "Track behavior for 30 days before assigning strategy",
            "A/B test different offer types to find best response",
        ],
        "channel": "Email",
        "discount": "Standard promotional offers",
    },
}


def get_recommendations(segment: str) -> dict:
    """Return campaign recommendations for a given segment."""
    return CAMPAIGN_RECOMMENDATIONS.get(segment, CAMPAIGN_RECOMMENDATIONS["Others"])


# ---------------------------------------------------------------------------
# 6. HYPOTHESIS VALIDATION (H1)
# ---------------------------------------------------------------------------
def validate_h1(rfm: pd.DataFrame) -> dict:
    """
    Validate H1: Top 20% by RFM score contribute >60% of total revenue.
    Also check 'At Risk' segment size.
    """
    total_revenue = rfm["Monetary"].sum()
    total_customers = len(rfm)

    # Top 20% by aggregate RFM score
    top_20_threshold = rfm["RFM_Agg_Score"].quantile(0.80)
    top_20 = rfm[rfm["RFM_Agg_Score"] >= top_20_threshold]
    top_20_rev_pct = top_20["Monetary"].sum() / total_revenue * 100

    # At Risk segment size
    at_risk_count = len(rfm[rfm["Segment"] == "At Risk"])
    at_risk_pct = at_risk_count / total_customers * 100

    result = {
        "hypothesis": "H1: Top 20% by RFM score contribute >60% of total revenue",
        "top_20_revenue_pct": round(top_20_rev_pct, 2),
        "top_20_customer_count": len(top_20),
        "h1_revenue_validated": top_20_rev_pct > 60,
        "at_risk_count": at_risk_count,
        "at_risk_pct": round(at_risk_pct, 2),
        "h1_at_risk_validated": at_risk_pct >= 10,
    }

    print(f"[H1] Top 20% by RFM score → {result['top_20_revenue_pct']}% revenue "
          f"({'✓ VALIDATED' if result['h1_revenue_validated'] else '✗ NOT validated'})")
    print(f"[H1] At Risk segment → {result['at_risk_pct']}% of customers "
          f"({'✓ VALIDATED' if result['h1_at_risk_validated'] else '✗ NOT validated'})")

    return result


# ---------------------------------------------------------------------------
# 7. PARETO ANALYSIS (Customer Concentration)
# ---------------------------------------------------------------------------
def pareto_analysis(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pareto (customer concentration) curve data.
    Returns DataFrame with cumulative % customers vs cumulative % revenue.
    """
    sorted_rfm = rfm.sort_values("Monetary", ascending=False).reset_index(drop=True)
    sorted_rfm["cum_revenue"] = sorted_rfm["Monetary"].cumsum()
    sorted_rfm["cum_revenue_pct"] = sorted_rfm["cum_revenue"] / sorted_rfm["Monetary"].sum() * 100
    sorted_rfm["cum_customer_pct"] = (sorted_rfm.index + 1) / len(sorted_rfm) * 100

    return sorted_rfm[["CustomerID", "Monetary", "cum_revenue_pct", "cum_customer_pct"]]
