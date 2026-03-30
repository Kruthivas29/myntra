"""
visualizations.py — Reusable Chart Functions
==============================================
Provides Plotly-based (primary) and Matplotlib/Seaborn (secondary) charts
for all pages of the Streamlit app.

Color palette: consistent across all pages.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# GLOBAL STYLE CONFIG
# ---------------------------------------------------------------------------
BRAND_PALETTE = {
    "primary": "#6366F1",      # Indigo
    "secondary": "#EC4899",    # Pink
    "success": "#10B981",      # Green
    "warning": "#F59E0B",      # Amber
    "danger": "#EF4444",       # Red
    "info": "#3B82F6",         # Blue
    "dark": "#1F2937",
    "light": "#F9FAFB",
}

SEGMENT_COLORS = {
    "Champions": "#10B981",
    "Loyal Customers": "#3B82F6",
    "Potential Loyalists": "#6366F1",
    "At Risk": "#F59E0B",
    "Hibernating": "#9CA3AF",
    "Lost": "#EF4444",
    "Need Attention": "#F97316",
    "Promising": "#8B5CF6",
    "Big Spenders (Lapsed)": "#EC4899",
    "Others": "#6B7280",
}

CLV_TIER_COLORS = {
    "Platinum": "#6366F1",
    "Gold": "#F59E0B",
    "Silver": "#9CA3AF",
    "Bronze": "#92400E",
}

CHURN_RISK_COLORS = {
    "Low": "#10B981",
    "Medium": "#F59E0B",
    "High": "#F97316",
    "Critical": "#EF4444",
}

PLOTLY_TEMPLATE = "plotly_white"


# ====================================================================
# PAGE 0 — EDA VISUALIZATIONS
# ====================================================================

def plot_monthly_revenue(monthly_rev: pd.DataFrame) -> go.Figure:
    """Line chart: Revenue over time (monthly)."""
    fig = px.line(
        monthly_rev, x="month", y="revenue",
        title="Monthly Revenue Trend",
        labels={"month": "Month", "revenue": "Revenue (£)"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(line_color=BRAND_PALETTE["primary"], line_width=2.5)
    fig.update_layout(hovermode="x unified")
    return fig


def plot_order_frequency_dist(order_freq: pd.DataFrame) -> go.Figure:
    """Histogram: Order frequency distribution (how many buy 1x, 2x, 3x…)."""
    # Cap at 20 for readability
    capped = order_freq.copy()
    capped["order_count"] = capped["order_count"].clip(upper=20)

    fig = px.histogram(
        capped, x="order_count", nbins=20,
        title="Order Frequency Distribution",
        labels={"order_count": "Number of Orders", "count": "Customer Count"},
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=[BRAND_PALETTE["primary"]],
    )
    fig.update_layout(bargap=0.1)
    return fig


def plot_pareto(pareto_data: pd.DataFrame) -> go.Figure:
    """Pareto chart: Customer concentration (top X% → Y% revenue)."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=pareto_data["cum_customer_pct"],
            y=pareto_data["Monetary"],
            name="Revenue",
            marker_color=BRAND_PALETTE["primary"],
            opacity=0.4,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=pareto_data["cum_customer_pct"],
            y=pareto_data["cum_revenue_pct"],
            name="Cumulative Revenue %",
            line=dict(color=BRAND_PALETTE["danger"], width=2.5),
        ),
        secondary_y=True,
    )

    # Add reference lines
    fig.add_hline(y=80, line_dash="dash", line_color="gray", secondary_y=True)
    fig.add_vline(x=20, line_dash="dash", line_color="gray")

    fig.update_layout(
        title="Pareto Analysis — Customer Revenue Concentration",
        xaxis_title="Cumulative % of Customers",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_yaxes(title_text="Revenue (£)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Revenue %", secondary_y=True)

    return fig


def plot_aov_distribution(df_clean: pd.DataFrame) -> go.Figure:
    """Histogram: Average order value distribution."""
    order_values = df_clean.groupby("InvoiceNo")["Revenue"].sum().reset_index()

    fig = px.histogram(
        order_values, x="Revenue", nbins=50,
        title="Order Value Distribution",
        labels={"Revenue": "Order Value (£)"},
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=[BRAND_PALETTE["info"]],
    )
    # Cap x-axis for readability
    fig.update_xaxes(range=[0, order_values["Revenue"].quantile(0.95)])
    return fig


def plot_new_vs_returning(new_vs_ret: pd.DataFrame) -> go.Figure:
    """Stacked bar: New vs returning customers by month."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=new_vs_ret["month"].astype(str), y=new_vs_ret["new"],
        name="New Customers", marker_color=BRAND_PALETTE["primary"],
    ))
    fig.add_trace(go.Bar(
        x=new_vs_ret["month"].astype(str), y=new_vs_ret["returning"],
        name="Returning Customers", marker_color=BRAND_PALETTE["success"],
    ))
    fig.update_layout(
        barmode="stack",
        title="New vs Returning Customers by Month",
        xaxis_title="Month", yaxis_title="Unique Customers",
        template=PLOTLY_TEMPLATE,
    )
    return fig


def plot_revenue_by_dow(dow_rev: pd.DataFrame) -> go.Figure:
    """Bar chart: Revenue by day of week."""
    fig = px.bar(
        dow_rev, x="day_of_week", y="Revenue",
        title="Revenue by Day of Week",
        labels={"day_of_week": "Day", "Revenue": "Total Revenue (£)"},
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=[BRAND_PALETTE["secondary"]],
    )
    return fig


def plot_cohort_heatmap(retention_table: pd.DataFrame) -> go.Figure:
    """Heatmap: Cohort retention rates."""
    fig = go.Figure(data=go.Heatmap(
        z=retention_table.values,
        x=[f"Month {i}" for i in retention_table.columns],
        y=[str(idx) for idx in retention_table.index],
        colorscale="RdYlGn",
        text=np.round(retention_table.values, 1),
        texttemplate="%{text}%",
        textfont={"size": 9},
        hovertemplate="Cohort: %{y}<br>Period: %{x}<br>Retention: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Cohort Retention Heatmap",
        xaxis_title="Months Since First Purchase",
        yaxis_title="Cohort Month",
        template=PLOTLY_TEMPLATE,
        height=500,
    )
    return fig


# ====================================================================
# PAGE 1 — RFM VISUALIZATIONS
# ====================================================================

def plot_segment_distribution(rfm: pd.DataFrame) -> go.Figure:
    """Bar chart / treemap: Segment distribution."""
    seg_counts = rfm["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]

    fig = px.treemap(
        seg_counts, path=["Segment"], values="Count",
        title="Customer Segment Distribution",
        color="Segment",
        color_discrete_map=SEGMENT_COLORS,
    )
    return fig


def plot_segment_characteristics(summary: pd.DataFrame) -> go.Figure:
    """Grouped bar: avg R, F, M per segment."""
    fig = go.Figure()

    for metric, color in [("avg_recency", BRAND_PALETTE["primary"]),
                           ("avg_frequency", BRAND_PALETTE["success"]),
                           ("avg_monetary", BRAND_PALETTE["secondary"])]:
        fig.add_trace(go.Bar(
            x=summary["Segment"], y=summary[metric],
            name=metric.replace("avg_", "Avg ").title(),
            marker_color=color,
        ))

    fig.update_layout(
        barmode="group",
        title="Segment Characteristics — Avg Recency, Frequency, Monetary",
        template=PLOTLY_TEMPLATE,
    )
    return fig


def plot_rfm_scatter(rfm: pd.DataFrame) -> go.Figure:
    """2D scatter: Frequency vs Monetary colored by segment."""
    fig = px.scatter(
        rfm, x="Frequency", y="Monetary",
        color="Segment",
        color_discrete_map=SEGMENT_COLORS,
        title="Customer Segments — Frequency vs Monetary Value",
        labels={"Frequency": "Purchase Frequency", "Monetary": "Total Revenue (£)"},
        template=PLOTLY_TEMPLATE,
        opacity=0.6,
        hover_data=["CustomerID", "Recency", "RFM_Score"],
    )
    fig.update_layout(height=600)
    return fig


def plot_segment_revenue_contribution(summary: pd.DataFrame) -> go.Figure:
    """Pie/stacked bar: Revenue contribution by segment."""
    fig = px.pie(
        summary, values="total_revenue", names="Segment",
        title="Revenue Contribution by Segment",
        color="Segment",
        color_discrete_map=SEGMENT_COLORS,
        template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


# ====================================================================
# PAGE 2 — CLV VISUALIZATIONS
# ====================================================================

def plot_clv_distribution(clv_data: pd.DataFrame, clv_col: str = "predicted_clv") -> go.Figure:
    """Histogram: CLV distribution with tier boundaries."""
    fig = px.histogram(
        clv_data, x=clv_col, nbins=50, color="CLV_Tier",
        color_discrete_map=CLV_TIER_COLORS,
        title="Customer Lifetime Value Distribution",
        labels={clv_col: "Predicted CLV (£)"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_xaxes(range=[0, clv_data[clv_col].quantile(0.95)])
    return fig


def plot_revenue_concentration_curve(clv_data: pd.DataFrame, clv_col="predicted_clv") -> go.Figure:
    """Revenue concentration curve: top N% customers vs % total revenue."""
    sorted_data = clv_data.sort_values(clv_col, ascending=False).reset_index(drop=True)
    sorted_data["cum_pct_customers"] = (sorted_data.index + 1) / len(sorted_data) * 100
    sorted_data["cum_revenue"] = sorted_data[clv_col].cumsum()
    sorted_data["cum_pct_revenue"] = sorted_data["cum_revenue"] / sorted_data[clv_col].sum() * 100

    fig = px.line(
        sorted_data, x="cum_pct_customers", y="cum_pct_revenue",
        title="Revenue Concentration Curve",
        labels={"cum_pct_customers": "Top N% of Customers", "cum_pct_revenue": "% of Total CLV"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(line_color=BRAND_PALETTE["primary"], line_width=2.5)
    fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100,
                  line=dict(dash="dash", color="gray"))
    return fig


def plot_clv_by_rfm_segment(master_table: pd.DataFrame, clv_col="predicted_clv") -> go.Figure:
    """Box plot: CLV distribution by RFM segment (connects Problem 1 → Problem 2)."""
    fig = px.box(
        master_table, x="RFM_Segment", y=clv_col,
        color="RFM_Segment",
        color_discrete_map=SEGMENT_COLORS,
        title="CLV Distribution by RFM Segment",
        labels={clv_col: "Predicted CLV (£)", "RFM_Segment": "Segment"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_yaxes(range=[0, master_table[clv_col].quantile(0.95)])
    return fig


def plot_actual_vs_predicted_clv(y_actual, y_predicted) -> go.Figure:
    """Scatter: Actual vs Predicted CLV for holdout validation."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_actual, y=y_predicted,
        mode="markers", marker=dict(color=BRAND_PALETTE["primary"], opacity=0.5, size=4),
        name="Customers",
    ))
    max_val = max(max(y_actual), max(y_predicted))
    fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                  line=dict(dash="dash", color="red"))
    fig.update_layout(
        title="Actual vs Predicted CLV (Holdout Validation)",
        xaxis_title="Actual Revenue (£)",
        yaxis_title="Predicted CLV (£)",
        template=PLOTLY_TEMPLATE,
    )
    return fig


# ====================================================================
# PAGE 3 — CHURN VISUALIZATIONS
# ====================================================================

def plot_roc_curve(results: dict) -> go.Figure:
    """ROC curve with AUC scores for all models."""
    fig = go.Figure()
    colors = [BRAND_PALETTE["primary"], BRAND_PALETTE["secondary"]]

    for i, (name, res) in enumerate(results.items()):
        if name.startswith("_"):
            continue
        fpr, tpr, _ = res["roc_curve"]
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f"{name} (AUC={res['auc_roc']:.3f})",
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(dash="dash", color="gray"))
    fig.update_layout(
        title="ROC Curve — Churn Prediction Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template=PLOTLY_TEMPLATE,
    )
    return fig


def plot_feature_importance(feature_importance: dict, top_n: int = 10) -> go.Figure:
    """Horizontal bar: Top N feature importance."""
    sorted_fi = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features = [f[0] for f in sorted_fi][::-1]
    values = [f[1] for f in sorted_fi][::-1]

    fig = go.Figure(go.Bar(
        x=values, y=features,
        orientation="h",
        marker_color=BRAND_PALETTE["primary"],
    ))
    fig.update_layout(
        title=f"Top {top_n} Feature Importance — Churn Prediction",
        xaxis_title="Importance",
        template=PLOTLY_TEMPLATE,
    )
    return fig


def plot_confusion_matrix(cm: np.ndarray, labels=None) -> go.Figure:
    """Heatmap: Confusion matrix."""
    if labels is None:
        labels = ["Retained", "Churned"]

    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale="Blues",
        text=cm, texttemplate="%{text}",
        textfont={"size": 18},
    ))
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template=PLOTLY_TEMPLATE,
    )
    return fig


def plot_churn_probability_dist(churn_probs: pd.DataFrame) -> go.Figure:
    """Histogram: Churn probability distribution with risk band coloring."""
    fig = px.histogram(
        churn_probs, x="churn_probability", nbins=50,
        color="churn_risk_band",
        color_discrete_map=CHURN_RISK_COLORS,
        title="Churn Probability Distribution",
        labels={"churn_probability": "Churn Probability", "churn_risk_band": "Risk Band"},
        template=PLOTLY_TEMPLATE,
    )
    return fig


# ====================================================================
# PAGE 4 — INTEGRATED DASHBOARD VISUALIZATIONS
# ====================================================================

def plot_integrated_heatmap(master_table: pd.DataFrame, clv_col="predicted_clv") -> go.Figure:
    """
    Bubble chart: X=CLV Tier, Y=Churn Risk Band, Size=Customer Count, Color=RFM Segment.
    THE key integration visualization.
    """
    grouped = (
        master_table.groupby(["CLV_Tier", "Churn_Risk_Band", "RFM_Segment"])
        .agg(customer_count=("CustomerID", "count"))
        .reset_index()
    )

    fig = px.scatter(
        grouped,
        x="CLV_Tier", y="Churn_Risk_Band",
        size="customer_count", color="RFM_Segment",
        color_discrete_map=SEGMENT_COLORS,
        title="Integrated Customer Intelligence — CLV × Churn Risk × Segment",
        labels={"CLV_Tier": "CLV Tier", "Churn_Risk_Band": "Churn Risk Band"},
        template=PLOTLY_TEMPLATE,
        size_max=50,
        hover_data=["customer_count"],
    )

    # Order axes
    fig.update_xaxes(categoryorder="array", categoryarray=["Platinum", "Gold", "Silver", "Bronze"])
    fig.update_yaxes(categoryorder="array", categoryarray=["Critical", "High", "Medium", "Low"])
    fig.update_layout(height=600)
    return fig


def plot_revenue_at_risk(master_table: pd.DataFrame, clv_col="predicted_clv") -> go.Figure:
    """
    Bar chart showing revenue at risk by segment + churn risk combination.
    """
    at_risk = master_table[master_table["Churn_Risk_Band"].isin(["High", "Critical"])].copy()
    rev_at_risk = (
        at_risk.groupby("RFM_Segment")
        .agg(
            customer_count=("CustomerID", "count"),
            total_clv_at_risk=(clv_col, "sum"),
        )
        .reset_index()
        .sort_values("total_clv_at_risk", ascending=False)
    )

    fig = px.bar(
        rev_at_risk, x="RFM_Segment", y="total_clv_at_risk",
        color="RFM_Segment",
        color_discrete_map=SEGMENT_COLORS,
        title="Revenue at Risk — High/Critical Churn Customers",
        labels={"total_clv_at_risk": "Total CLV at Risk (£)", "RFM_Segment": "Segment"},
        template=PLOTLY_TEMPLATE,
        text="customer_count",
    )
    fig.update_traces(texttemplate="%{text} customers", textposition="outside")
    return fig
