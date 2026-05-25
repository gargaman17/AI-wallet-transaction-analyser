from __future__ import annotations

import pandas as pd


def build_summary_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    return {
        "total_spending": float(df["amount"].sum()),
        "average_transaction": float(df["amount"].mean()) if not df.empty else 0,
        "transaction_count": int(len(df)),
        "high_risk_count": int((df.get("risk_level") == "High").sum()) if "risk_level" in df else 0,
    }


def category_spending(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("category", as_index=False)["amount"]
        .sum()
        .sort_values("amount", ascending=False)
    )


def top_merchants(df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    return (
        df.groupby("merchant", as_index=False)
        .agg(amount=("amount", "sum"), transactions=("transaction_id", "count"))
        .sort_values("amount", ascending=False)
        .head(limit)
    )


def spending_trend(df: pd.DataFrame, frequency: str = "D") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["period", "amount"])
    trend = (
        df.set_index("timestamp")
        .resample(frequency)["amount"]
        .sum()
        .reset_index()
        .rename(columns={"timestamp": "period"})
    )
    return trend
