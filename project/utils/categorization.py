from __future__ import annotations

import os
from functools import lru_cache


MERCHANT_CATEGORY_RULES = {
    "Food": ["starbucks", "chipotle", "doordash", "subway", "deli", "restaurant", "cafe"],
    "Shopping": ["amazon", "walmart", "target", "best buy", "apple", "marketplace"],
    "Travel": ["uber", "lyft", "delta", "airbnb", "marriott", "hotel", "airlines"],
    "Bills": ["verizon", "comcast", "pg&e", "netflix", "spotify", "utility"],
    "Groceries": ["whole foods", "trader joe", "costco", "safeway", "kroger"],
    "ATM": ["atm", "cash withdrawal"],
    "Crypto": ["coinbase", "binance", "kraken", "crypto", "wallet", "exchange"],
    "Transfers": ["wire", "transfer", "zelle", "venmo", "cash app"],
}


def normalize_category(category: object) -> str:
    if category is None:
        return "Uncategorized"
    text = str(category).strip()
    if not text or text.lower() in {"nan", "none", "unknown"}:
        return "Uncategorized"
    return text.replace("_", " ").replace("-", " ").title()


def categorize_merchant_rule_based(merchant: object) -> str:
    merchant_text = str(merchant or "").strip().lower()
    for category, keywords in MERCHANT_CATEGORY_RULES.items():
        if any(keyword in merchant_text for keyword in keywords):
            return category
    return "Uncategorized"


@lru_cache(maxsize=256)
def categorize_merchant_ai(merchant: str) -> str:
    """AI fallback categorization. Returns Uncategorized if no API key/client is available."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Uncategorized"

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify a payment merchant into exactly one category: Food, "
                        "Shopping, Travel, Bills, Groceries, ATM, Crypto, Transfers, or Uncategorized."
                    ),
                },
                {"role": "user", "content": f"Merchant: {merchant}"},
            ],
            temperature=0,
            max_tokens=8,
        )
        category = response.choices[0].message.content.strip()
        return category if category in set(MERCHANT_CATEGORY_RULES) | {"Uncategorized"} else "Uncategorized"
    except Exception:
        return "Uncategorized"


def categorize_merchant(merchant: object, use_ai_fallback: bool = False) -> str:
    category = categorize_merchant_rule_based(merchant)
    if category == "Uncategorized" and use_ai_fallback:
        return categorize_merchant_ai(str(merchant or ""))
    return category
