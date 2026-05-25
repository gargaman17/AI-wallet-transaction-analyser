from __future__ import annotations

import pandas as pd

from utils.categorization import categorize_merchant, normalize_category


REQUIRED_COLUMNS = [
    "transaction_id",
    "user_id",
    "amount",
    "merchant",
    "category",
    "timestamp",
    "location",
    "payment_status",
]


def validate_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in REQUIRED_COLUMNS if column not in df.columns]


def clean_transactions(df: pd.DataFrame, use_ai_categories: bool = False) -> pd.DataFrame:
    """Clean uploaded wallet transactions and enrich them with time/category fields."""
    cleaned = df.copy()
    missing_columns = validate_columns(cleaned)
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    cleaned = cleaned.drop_duplicates(subset=["transaction_id"]).copy()
    cleaned["amount"] = pd.to_numeric(cleaned["amount"], errors="coerce")
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce")

    cleaned["transaction_id"] = cleaned["transaction_id"].fillna("UNKNOWN").astype(str)
    cleaned["user_id"] = cleaned["user_id"].fillna("UNKNOWN").astype(str)
    cleaned["merchant"] = cleaned["merchant"].fillna("Unknown Merchant").astype(str).str.strip()
    cleaned["location"] = cleaned["location"].fillna("Unknown").astype(str).str.strip()
    cleaned["payment_status"] = (
        cleaned["payment_status"].fillna("Unknown").astype(str).str.strip().str.title()
    )
    cleaned["category"] = cleaned["category"].apply(normalize_category)
    cleaned["amount"] = cleaned["amount"].fillna(cleaned["amount"].median()).clip(lower=0)
    cleaned["timestamp"] = cleaned["timestamp"].fillna(cleaned["timestamp"].median())

    inferred_categories = cleaned["merchant"].apply(
        lambda merchant: categorize_merchant(merchant, use_ai_fallback=use_ai_categories)
    )
    cleaned["category"] = cleaned["category"].where(
        cleaned["category"] != "Uncategorized", inferred_categories
    )
    cleaned["category"] = cleaned["category"].apply(normalize_category)

    cleaned["date"] = cleaned["timestamp"].dt.date
    cleaned["month"] = cleaned["timestamp"].dt.to_period("M").astype(str)
    cleaned["hour"] = cleaned["timestamp"].dt.hour
    cleaned["day_name"] = cleaned["timestamp"].dt.day_name()
    return cleaned


def filter_transactions(
    df: pd.DataFrame,
    categories: list[str] | None = None,
    risk_levels: list[str] | None = None,
    search_text: str = "",
) -> pd.DataFrame:
    filtered = df.copy()
    if categories:
        filtered = filtered[filtered["category"].isin(categories)]
    if risk_levels and "risk_level" in filtered:
        filtered = filtered[filtered["risk_level"].isin(risk_levels)]
    if search_text:
        needle = search_text.lower()
        haystack = (
            filtered["transaction_id"].astype(str)
            + " "
            + filtered["user_id"].astype(str)
            + " "
            + filtered["merchant"].astype(str)
            + " "
            + filtered["location"].astype(str)
        ).str.lower()
        filtered = filtered[haystack.str.contains(needle, na=False)]
    return filtered
