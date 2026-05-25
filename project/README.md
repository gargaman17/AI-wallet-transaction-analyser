AI Wallet Transaction Analyzer
A Streamlit fintech dashboard for analyzing wallet transaction CSV files, detecting suspicious activity with Isolation Forest, and generating AI-assisted financial insights.

Features
CSV upload for wallet transactions
Auto-generated sample dataset with 2,000+ records
Data cleaning for missing values, timestamps, duplicates, and normalized categories
Rule-based merchant categorization with optional OpenAI fallback
Fraud scoring with Scikit-learn Isolation Forest
Risk levels: Low, Medium, High
Category pie chart, merchant bar chart, and spending trend line chart
Filters, search, risk highlighting, risk badges, and analyzed CSV download
Light and dark dashboard modes
SQLite backend for saved analysis runs and fraud review cases
OpenAI-powered insights when OPENAI_API_KEY is available
Expected CSV Columns
transaction_id,user_id,amount,merchant,category,timestamp,location,payment_status
Local Setup
cd project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
OpenAI Setup
The app works without an API key by using a local fallback summary. For live AI categorization and insights:

export OPENAI_API_KEY="your_api_key_here"
export OPENAI_MODEL="gpt-4o-mini"
Replit Deployment
Upload or create this project folder in Replit.
Install dependencies from requirements.txt.
Add OPENAI_API_KEY in Replit Secrets if you want AI insights.
Use this run command:
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
Project Structure
project/
├── app.py
├── utils/
│   ├── ai_insights.py
│   ├── analytics.py
│   ├── backend.py
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
Use Cases
Wallet fraud monitoring for fintech or mobile money transaction exports
Payment operations dashboards for reviewing high-risk transactions
Merchant and category spend analytics for personal finance products
AML/KYT-style triage signals for suspicious behavior patterns
Customer support review workflows for failed, pending, or unusual payments
Synthetic fraud analytics demos for portfolios, hackathons, and ML projects
Monthly spend summaries for banking, wallet, and expense management apps
Notes
Fraud scores are risk signals, not proof of fraud. Review flagged transactions before taking account action.
