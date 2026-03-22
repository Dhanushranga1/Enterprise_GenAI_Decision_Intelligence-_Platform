import sqlite3
import pandas as pd
import json
import os
from datetime import datetime
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load environment variables
load_dotenv()

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
LOCAL_DB_PATH = os.path.join(DATA_DIR, 'enterprise_data.db')
INSIGHTS_PATH = os.path.join(DATA_DIR, 'ml_insights.json')

def get_db_engine():
    """Returns the database engine based on environment configuration."""
    db_url = os.getenv('DATABASE_URL')
    if db_url and db_url.startswith('postgres'):
        if db_url.startswith('postgres://'):
            db_url = db_url.replace('postgres://', 'postgresql://', 1)
        print("Connecting to External Database (Supabase)...")
        return create_engine(db_url)
    else:
        print(f"Connecting to Local SQLite: {LOCAL_DB_PATH}")
        return create_engine(f'sqlite:///{LOCAL_DB_PATH}')

def analyze_churn():
    engine = get_db_engine()
    
    try:
        df = pd.read_sql('SELECT * FROM customer_metrics', engine)
    except Exception as e:
        print(f"Error reading from database: {e}")
        return

    print(f"Loaded {len(df)} rows for training.")

    # Preprocessing
    # Drop customerID
    X = df.drop(['customerID', 'Churn'], axis=1)
    y = df['Churn']

    # Handle Categoricals
    cat_cols = X.select_dtypes(include=['object']).columns
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluation
    accuracy = clf.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Feature Importance
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)

    top_features = feature_importances.head(5).to_dict('records')
    
    # Generate Insights
    high_risk_prob = clf.predict_proba(X)[:, 1]
    df['churn_probability'] = high_risk_prob
    high_risk_customers = df[df['churn_probability'] > 0.7] # High threshold
    
    # Segment Analysis: Who is high risk?
    # Simple aggregation on original df to find patterns in high risk
    # We use the original DF (before encoding) for readable insights, but we need to re-attach categorical values
    # Actually, let's just use the original df we loaded.
    
    df_high_risk = df.loc[high_risk_customers.index]
    
    # Find most common attribute in high risk group
    # Handle empty DataFrames gracefully
    if len(df_high_risk) == 0:
        top_contract = "Unknown"
        top_internet = "Unknown"
        avg_monthly = 0.0
    else:
        contract_mode = df_high_risk['Contract'].mode()
        internet_mode = df_high_risk['InternetService'].mode()
        top_contract = contract_mode[0] if len(contract_mode) > 0 else "Unknown"
        top_internet = internet_mode[0] if len(internet_mode) > 0 else "Unknown"
        avg_monthly = df_high_risk['MonthlyCharges'].mean()

    # ── Drift baseline ─────────────────────────────────────────────────────────
    # Captured from training data so future runs can compare live DB against it.
    contract_dist = (
        df['Contract'].value_counts(normalize=True).round(4).to_dict()
    )
    drift_baseline = {
        "churn_rate":           round(float(y.mean()), 4),
        "avg_tenure":           round(float(df['tenure'].mean()), 2),
        "avg_monthly_charges":  round(float(df['MonthlyCharges'].mean()), 2),
        "contract_distribution": contract_dist,
    }

    insights = {
        "generated_at": datetime.now().isoformat(),
        "drift_baseline": drift_baseline,
        "summary": {
            "model_accuracy": f"{accuracy:.1%}",
            "total_customers": len(df),
            "overall_churn_rate": f"{y.mean():.1%}",
            "high_risk_customers_count": len(high_risk_customers)
        },
        "top_risk_factors": [
            {
                "feature": f.get("feature"),
                "importance": f"{f.get('importance'):.2f}"
            } for f in top_features
        ],
        "critical_segments": [
            {
                "segment_name": f"{top_contract} Users with {top_internet}",
                "risk_score": "Critical",
                "reasoning": f"This segment shows disproportionately high churn probability (>70%). Average monthly spend is ${avg_monthly:.2f}."
            }
        ],
        "recommendations_foundation": [
            f"Offer long-term contract incentives to {top_contract} users.",
            f"Review service quality for {top_internet} customers.",
            "Implement 'save-teams' for customers with Monthly Charges > $80."
        ]
    }

    with open(INSIGHTS_PATH, 'w') as f:
        json.dump(insights, f, indent=4)
    
    print(f"Real ML insights generated and saved to {INSIGHTS_PATH}")

if __name__ == "__main__":
    analyze_churn()
