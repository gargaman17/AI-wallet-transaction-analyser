from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    features["amount"] = df["amount"].astype(float)
    features["hour"] = df["hour"].astype(int)
    features["is_unusual_hour"] = df["hour"].between(0, 5).astype(int)
    features["is_international"] = (~df["location"].str.contains("USA", case=False, na=False)).astype(int)
    features["is_failed_or_pending"] = df["payment_status"].isin(["Failed", "Pending"]).astype(int)
    features["user_txn_count"] = df.groupby("user_id")["transaction_id"].transform("count")
    features["merchant_txn_count"] = df.groupby("merchant")["transaction_id"].transform("count")
    features["same_user_merchant_count"] = df.groupby(["user_id", "merchant"])["transaction_id"].transform("count")

    sorted_df = df.sort_values(["user_id", "timestamp"])
    minute_gaps = sorted_df.groupby("user_id")["timestamp"].diff().dt.total_seconds().div(60)
    features.loc[sorted_df.index, "minutes_since_previous"] = minute_gaps.fillna(9999)
    features["rapid_repeat"] = (features["minutes_since_previous"] <= 20).astype(int)
    features["daily_user_spend"] = df.groupby(["user_id", "date"])["amount"].transform("sum")
    features["category_user_spend"] = df.groupby(["user_id", "category"])["amount"].transform("sum")
    return features.fillna(0)


def _risk_level(score: float) -> str:
    if score >= 75:
        return "High"
    if score >= 45:
        return "Medium"
    return "Low"


def detect_fraud(df: pd.DataFrame, contamination: float = 0.06) -> pd.DataFrame:
    """Detect suspicious wallet transactions with Isolation Forest plus domain signals."""
    if df.empty:
        return df.assign(fraud_score=[], risk_level=[])

    analyzed = df.copy()
    features = _build_features(analyzed)
    scaled = StandardScaler().fit_transform(features)
    model = IsolationForest(
        n_estimators=180,
        contamination=min(max(contamination, 0.01), 0.2),
        random_state=42,
    )
    model.fit(scaled)

    anomaly_strength = -model.score_samples(scaled)
    normalized_anomaly = (
        (anomaly_strength - anomaly_strength.min())
        / (anomaly_strength.max() - anomaly_strength.min() + 1e-9)
    )

    amount_cutoff = analyzed["amount"].quantile(0.95)
    daily_spend_cutoff = features["daily_user_spend"].quantile(0.95)
    rule_score = (
        (analyzed["amount"] >= amount_cutoff).astype(int) * 20
        + features["is_unusual_hour"] * 16
        + features["is_international"] * 14
        + features["rapid_repeat"] * 16
        + (features["daily_user_spend"] >= daily_spend_cutoff).astype(int) * 14
        + features["is_failed_or_pending"] * 8
    )
    analyzed["fraud_score"] = np.clip((normalized_anomaly * 62) + rule_score, 0, 100).round(1)
    analyzed["risk_level"] = analyzed["fraud_score"].apply(_risk_level)
    analyzed["suspicious_reasons"] = _build_reasons(analyzed, features, amount_cutoff, daily_spend_cutoff)
    return analyzed.sort_values("fraud_score", ascending=False).reset_index(drop=True)


def _build_reasons(
    df: pd.DataFrame,
    features: pd.DataFrame,
    amount_cutoff: float,
    daily_spend_cutoff: float,
) -> list[str]:
    reasons: list[str] = []
    for idx, row in df.iterrows():
        flags = []
        if row["amount"] >= amount_cutoff:
            flags.append("large amount")
        if features.loc[idx, "is_unusual_hour"]:
            flags.append("unusual hour")
        if features.loc[idx, "is_international"]:
            flags.append("international location")
        if features.loc[idx, "rapid_repeat"]:
            flags.append("rapid repeat")
        if features.loc[idx, "daily_user_spend"] >= daily_spend_cutoff:
            flags.append("daily spending spike")
        if features.loc[idx, "is_failed_or_pending"]:
            flags.append("non-completed payment")
        reasons.append(", ".join(flags) if flags else "normal pattern")
    return reasons
