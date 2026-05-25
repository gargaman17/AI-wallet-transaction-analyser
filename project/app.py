from __future__ import annotations

from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from models.fraud_detection import detect_fraud
from utils.ai_insights import generate_ai_insights
from utils.analytics import build_summary_metrics, category_spending, spending_trend, top_merchants
from utils.data_processing import clean_transactions, filter_transactions
from utils.sample_data import ensure_sample_csv


BASE_DIR = Path(__file__).parent
SAMPLE_CSV = BASE_DIR / "sample_data" / "wallet_transactions_sample.csv"


st.set_page_config(
    page_title="AI Wallet Transaction Analyzer",
    page_icon="AI",
    layout="wide",
)


def apply_theme(theme: str) -> None:
    is_dark = theme == "Dark"
    background = "#0f172a" if is_dark else "#f6f8fb"
    surface = "#111827" if is_dark else "#ffffff"
    text = "#e5e7eb" if is_dark else "#172033"
    muted = "#94a3b8" if is_dark else "#5f6b7a"
    border = "#263244" if is_dark else "#dce3ee"
    accent = "#12b886"
    danger = "#ef4444"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {background};
            color: {text};
        }}
        h1, h2, h3, p, label, span {{
            color: {text};
        }}
        [data-testid="stMetric"] {{
            background: {surface};
            border: 1px solid {border};
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }}
        [data-testid="stMetricLabel"] p {{
            color: {muted};
        }}
        .dashboard-card {{
            background: {surface};
            border: 1px solid {border};
            border-radius: 8px;
            padding: 18px;
            margin-bottom: 16px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }}
        .risk-badge {{
            display: inline-block;
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 12px;
            font-weight: 700;
        }}
        .risk-low {{ background: rgba(18, 184, 134, 0.14); color: {accent}; }}
        .risk-medium {{ background: rgba(245, 158, 11, 0.18); color: #f59e0b; }}
        .risk-high {{ background: rgba(239, 68, 68, 0.16); color: {danger}; }}
        .stDataFrame {{
            border: 1px solid {border};
            border-radius: 8px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_sample_data() -> pd.DataFrame:
    path = ensure_sample_csv(SAMPLE_CSV, record_count=2000)
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def prepare_transactions(raw_df: pd.DataFrame, use_ai_categories: bool) -> pd.DataFrame:
    cleaned = clean_transactions(raw_df, use_ai_categories=use_ai_categories)
    return detect_fraud(cleaned)


def money(value: float) -> str:
    return f"${value:,.2f}"


def risk_badge(level: str) -> str:
    css = level.lower()
    return f'<span class="risk-badge risk-{css}">{level}</span>'


def build_pie_chart(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    category_data = category_spending(df)
    if category_data.empty:
        ax.text(0.5, 0.5, "No category data", ha="center", va="center")
        ax.axis("off")
        return fig
    colors = ["#12b886", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#14b8a6", "#64748b", "#f97316"]
    ax.pie(
        category_data["amount"],
        labels=category_data["category"],
        autopct="%1.1f%%",
        startangle=120,
        colors=colors[: len(category_data)],
        textprops={"fontsize": 8},
    )
    ax.set_title("Category-wise Spending")
    return fig


def build_bar_chart(df: pd.DataFrame):
    merchant_data = top_merchants(df, limit=10).sort_values("amount")
    fig, ax = plt.subplots(figsize=(7, 4))
    if merchant_data.empty:
        ax.text(0.5, 0.5, "No merchant data", ha="center", va="center")
        ax.axis("off")
        return fig
    ax.barh(merchant_data["merchant"], merchant_data["amount"], color="#12b886")
    ax.set_xlabel("Spending")
    ax.set_title("Top Merchants")
    ax.grid(axis="x", alpha=0.2)
    return fig


def build_line_chart(df: pd.DataFrame, frequency: str):
    trend = spending_trend(df, frequency=frequency)
    fig, ax = plt.subplots(figsize=(9, 4))
    if trend.empty:
        ax.text(0.5, 0.5, "No trend data", ha="center", va="center")
        ax.axis("off")
        return fig
    ax.plot(trend["period"], trend["amount"], color="#3b82f6", linewidth=2)
    ax.fill_between(trend["period"], trend["amount"], color="#3b82f6", alpha=0.12)
    ax.set_ylabel("Spending")
    ax.set_title("Spending Trend")
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    return fig


def style_risk_rows(row: pd.Series) -> list[str]:
    if row.get("risk_level") == "High":
        return ["background-color: rgba(239, 68, 68, 0.14)"] * len(row)
    if row.get("risk_level") == "Medium":
        return ["background-color: rgba(245, 158, 11, 0.10)"] * len(row)
    return [""] * len(row)


def render_header() -> None:
    st.markdown(
        """
        <div class="dashboard-card">
            <h1 style="margin:0;">AI Wallet Transaction Analyzer</h1>
            <p style="margin:8px 0 0 0;">
                Upload wallet transactions, monitor spending behavior, and surface suspicious activity with machine learning and AI insights.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    with st.sidebar:
        st.header("Controls")
        theme = st.radio("Theme", ["Light", "Dark"], horizontal=True)
        apply_theme(theme)

        use_ai_categories = st.toggle(
            "AI merchant categorization fallback",
            value=False,
            help="Uses OPENAI_API_KEY when rule-based categorization cannot classify a merchant.",
        )
        trend_frequency = st.selectbox("Trend granularity", ["Daily", "Monthly"], index=0)
        uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])

    render_header()

    try:
        if uploaded_file is None:
            raw_df = load_sample_data()
            st.info("Using generated sample data. Upload a CSV to analyze your own transactions.")
        else:
            raw_df = pd.read_csv(uploaded_file)

        analyzed_df = prepare_transactions(raw_df, use_ai_categories)
    except Exception as exc:
        st.error(f"Could not load or analyze transactions: {exc}")
        st.stop()

    with st.sidebar:
        st.divider()
        st.header("Filters")
        categories = sorted(analyzed_df["category"].dropna().unique().tolist())
        selected_categories = st.multiselect("Categories", categories, default=categories)
        selected_risks = st.multiselect(
            "Risk levels", ["Low", "Medium", "High"], default=["Low", "Medium", "High"]
        )
        search_text = st.text_input("Search transactions")

    filtered_df = filter_transactions(
        analyzed_df,
        categories=selected_categories,
        risk_levels=selected_risks,
        search_text=search_text,
    )

    metrics = build_summary_metrics(filtered_df)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Total Spending", money(metrics["total_spending"]))
    metric_cols[1].metric("Average Transaction", money(metrics["average_transaction"]))
    metric_cols[2].metric("Transactions", f"{metrics['transaction_count']:,}")
    metric_cols[3].metric("High Risk", f"{metrics['high_risk_count']:,}")

    st.subheader("Analytics Overview")
    chart_cols = st.columns([1, 1])
    with chart_cols[0]:
        st.pyplot(build_pie_chart(filtered_df), use_container_width=True)
    with chart_cols[1]:
        st.pyplot(build_bar_chart(filtered_df), use_container_width=True)

    frequency_code = "D" if trend_frequency == "Daily" else "M"
    st.pyplot(build_line_chart(filtered_df, frequency_code), use_container_width=True)

    st.subheader("Fraud Detection Table")
    table_columns = [
        "transaction_id",
        "user_id",
        "amount",
        "merchant",
        "category",
        "timestamp",
        "location",
        "payment_status",
        "fraud_score",
        "risk_level",
        "suspicious_reasons",
    ]
    table_df = filtered_df[table_columns].copy()
    table_df["timestamp"] = table_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(
        table_df.style.apply(style_risk_rows, axis=1).format({"amount": "${:,.2f}", "fraud_score": "{:.1f}"}),
        use_container_width=True,
        height=460,
    )

    risk_counts = filtered_df["risk_level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
    st.markdown(
        " ".join(
            risk_badge(level) + f" {int(count):,}"
            for level, count in risk_counts.items()
        ),
        unsafe_allow_html=True,
    )

    csv_buffer = StringIO()
    download_df = filtered_df.copy()
    download_df["timestamp"] = download_df["timestamp"].astype(str)
    download_df.to_csv(csv_buffer, index=False)
    st.download_button(
        "Download analyzed CSV",
        data=csv_buffer.getvalue(),
        file_name="analyzed_wallet_transactions.csv",
        mime="text/csv",
    )

    st.subheader("AI Insights Panel")
    if st.button("Generate AI insights", type="primary"):
        with st.spinner("Analyzing wallet behavior and fraud signals..."):
            st.markdown(generate_ai_insights(filtered_df))
    else:
        st.caption("Set OPENAI_API_KEY for live OpenAI insights, or click to use the local fallback summary.")

    with st.expander("Data quality snapshot"):
        unique_after_cleaning = analyzed_df["transaction_id"].nunique()
        st.write(
            {
                "rows_after_cleaning": len(analyzed_df),
                "uploaded_or_sample_rows": len(raw_df),
                "duplicate_transaction_ids_removed": max(len(raw_df) - unique_after_cleaning, 0),
                "sample_csv_path": str(SAMPLE_CSV),
            }
        )


if __name__ == "__main__":
    main()
