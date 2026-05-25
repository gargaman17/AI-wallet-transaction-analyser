from __future__ import annotations

import os

import pandas as pd

from utils.analytics import build_summary_metrics, category_spending, top_merchants


def _offline_insights(df: pd.DataFrame) -> str:
    metrics = build_summary_metrics(df)
    category_table = category_spending(df)
    top_category = category_table.iloc[0]["category"] if not category_table.empty else "N/A"
    high_risk = df[df.get("risk_level", "") == "High"] if "risk_level" in df else pd.DataFrame()
    monthly = df.groupby("month")["amount"].sum().sort_index()
    trend_text = "stable"
    if len(monthly) >= 2:
        trend_text = "up" if monthly.iloc[-1] > monthly.iloc[-2] else "down"

    return (
        f"Spending totals ${metrics['total_spending']:,.2f} across "
        f"{metrics['transaction_count']:,} transactions, with an average transaction of "
        f"${metrics['average_transaction']:,.2f}. The leading category is {top_category}. "
        f"Recent monthly spending is trending {trend_text}. "
        f"{len(high_risk):,} transactions are marked high risk; review large, late-night, "
        "international, or rapid repeated payments before approving them."
    )


def generate_ai_insights(df: pd.DataFrame) -> str:
    """Generate financial and fraud insights through OpenAI, with local fallback."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _offline_insights(df)

    try:
        from openai import OpenAI

        metrics = build_summary_metrics(df)
        categories = category_spending(df).head(8).to_dict(orient="records")
        merchants = top_merchants(df, limit=8).to_dict(orient="records")
        risky = (
            df[df["risk_level"].isin(["High", "Medium"])]
            .head(12)[["amount", "merchant", "category", "location", "risk_level", "suspicious_reasons"]]
            .to_dict(orient="records")
            if "risk_level" in df
            else []
        )
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a fintech wallet risk analyst. Write concise, practical insights "
                        "for operations and fraud teams. Avoid overclaiming; call suspicious items "
                        "signals, not proven fraud."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Summary metrics: {metrics}\n"
                        f"Category spending: {categories}\n"
                        f"Top merchants: {merchants}\n"
                        f"Risk examples: {risky}\n"
                        "Return 4 short bullets: spending behavior, fraud warnings, financial insights, monthly trend."
                    ),
                },
            ],
            temperature=0.25,
            max_tokens=320,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"{_offline_insights(df)}\n\nAI service note: {exc}"
