# AI Wallet Transaction Analyzer

A Streamlit fintech dashboard for analyzing wallet transaction CSV files, detecting suspicious activity with Isolation Forest, and generating AI-assisted financial insights.

## Features

- CSV upload for wallet transactions
- Auto-generated sample dataset with 2,000+ records
- Data cleaning for missing values, timestamps, duplicates, and normalized categories
- Rule-based merchant categorization with optional OpenAI fallback
- Fraud scoring with Scikit-learn Isolation Forest
- Risk levels: Low, Medium, High
- Category pie chart, merchant bar chart, and spending trend line chart
- Filters, search, risk highlighting, risk badges, and analyzed CSV download
- Light and dark dashboard modes
- OpenAI-powered insights when `OPENAI_API_KEY` is available

## Expected CSV Columns

```text
transaction_id,user_id,amount,merchant,category,timestamp,location,payment_status
```

## Local Setup

```bash
cd project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## OpenAI Setup

The app works without an API key by using a local fallback summary. For live AI categorization and insights:

```bash
export OPENAI_API_KEY="your_api_key_here"
export OPENAI_MODEL="gpt-4o-mini"
```

## Replit Deployment

1. Upload or create this `project` folder in Replit.
2. Install dependencies from `requirements.txt`.
3. Add `OPENAI_API_KEY` in Replit Secrets if you want AI insights.
4. Use this run command:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## Project Structure

```text
project/
├── app.py
├── utils/
│   ├── ai_insights.py
│   ├── analytics.py
│   ├── categorization.py
│   ├── data_processing.py
│   └── sample_data.py
├── models/
│   └── fraud_detection.py
├── data/
├── sample_data/
│   └── wallet_transactions_sample.csv
├── requirements.txt
└── README.md
```

## Notes

Fraud scores are risk signals, not proof of fraud. Review flagged transactions before taking account action.
