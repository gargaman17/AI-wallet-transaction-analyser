from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


SAMPLE_COLUMNS = [
    "transaction_id",
    "user_id",
    "amount",
    "merchant",
    "category",
    "timestamp",
    "location",
    "payment_status",
]


NORMAL_MERCHANTS = {
    "Food": ["Starbucks", "Chipotle", "DoorDash", "Subway", "Local Deli"],
    "Shopping": ["Amazon", "Walmart", "Target", "Best Buy", "Apple Store"],
    "Travel": ["Uber", "Lyft", "Delta Airlines", "Airbnb", "Marriott"],
    "Bills": ["Verizon", "Comcast", "PG&E", "Netflix", "Spotify"],
    "Groceries": ["Whole Foods", "Trader Joe's", "Costco", "Safeway", "Kroger"],
    "ATM": ["Chase ATM", "Bank of America ATM", "Wells Fargo ATM"],
    "Crypto": ["Coinbase", "Binance", "Kraken", "Crypto.com"],
}

INTERNATIONAL_LOCATIONS = [
    "London, UK",
    "Singapore",
    "Dubai, UAE",
    "Tokyo, Japan",
    "Berlin, Germany",
    "Toronto, Canada",
]

LOCAL_LOCATIONS = [
    "New York, USA",
    "San Francisco, USA",
    "Austin, USA",
    "Chicago, USA",
    "Seattle, USA",
    "Los Angeles, USA",
]


def generate_sample_transactions(record_count: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic wallet transactions with injected suspicious behavior."""
    random.seed(seed)
    np.random.seed(seed)

    start_date = datetime.now() - timedelta(days=365)
    users = [f"U{str(i).zfill(4)}" for i in range(1, 151)]
    records: list[dict[str, object]] = []

    for i in range(record_count):
        category = random.choices(
            list(NORMAL_MERCHANTS.keys()),
            weights=[24, 22, 12, 14, 18, 5, 5],
            k=1,
        )[0]
        merchant = random.choice(NORMAL_MERCHANTS[category])
        timestamp = start_date + timedelta(
            days=random.randint(0, 364),
            hours=random.randint(7, 22),
            minutes=random.randint(0, 59),
        )

        if category == "Food":
            amount = np.random.gamma(2.2, 9)
        elif category == "Shopping":
            amount = np.random.gamma(2.5, 35)
        elif category == "Travel":
            amount = np.random.gamma(2.0, 75)
        elif category == "Bills":
            amount = np.random.normal(95, 35)
        elif category == "Groceries":
            amount = np.random.normal(75, 25)
        elif category == "ATM":
            amount = random.choice([40, 60, 80, 100, 120, 200])
        else:
            amount = np.random.gamma(1.7, 90)

        records.append(
            {
                "transaction_id": f"TXN-{i + 1:06d}",
                "user_id": random.choice(users),
                "amount": round(max(float(amount), 2.5), 2),
                "merchant": merchant,
                "category": category,
                "timestamp": timestamp.isoformat(timespec="minutes"),
                "location": random.choices(
                    LOCAL_LOCATIONS + INTERNATIONAL_LOCATIONS,
                    weights=[18, 18, 18, 18, 18, 18, 2, 2, 2, 2, 2, 2],
                    k=1,
                )[0],
                "payment_status": random.choices(
                    ["Completed", "Pending", "Failed"],
                    weights=[92, 5, 3],
                    k=1,
                )[0],
            }
        )

    suspicious_merchants = [
        "Unknown Crypto Wallet",
        "Offshore Exchange",
        "Luxury Wire Transfer",
        "Night ATM",
        "Unverified Marketplace",
    ]
    suspicious_users = random.sample(users, 12)

    for j in range(max(80, record_count // 20)):
        user = random.choice(suspicious_users)
        burst_start = datetime.now() - timedelta(days=random.randint(0, 120))
        burst_time = burst_start.replace(
            hour=random.choice([0, 1, 2, 3, 4]), minute=random.randint(0, 59)
        )
        records.append(
            {
                "transaction_id": f"TXN-SUSP-{j + 1:05d}",
                "user_id": user,
                "amount": round(float(np.random.uniform(900, 6500)), 2),
                "merchant": random.choice(suspicious_merchants),
                "category": random.choice(["Crypto", "ATM", "Travel", "Shopping"]),
                "timestamp": (
                    burst_time + timedelta(minutes=random.randint(0, 18))
                ).isoformat(timespec="minutes"),
                "location": random.choice(INTERNATIONAL_LOCATIONS),
                "payment_status": random.choices(
                    ["Completed", "Pending", "Failed"],
                    weights=[78, 10, 12],
                    k=1,
                )[0],
            }
        )

    df = pd.DataFrame(records, columns=SAMPLE_COLUMNS)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def ensure_sample_csv(path: str | Path, record_count: int = 2000) -> Path:
    """Create sample CSV data when the file does not already exist."""
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        generate_sample_transactions(record_count=record_count).to_csv(csv_path, index=False)
    return csv_path
