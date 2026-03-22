# Enterprise GenAI Decision Intelligence Platform

A platform that combines a trained ML churn model with a multi-agent GenAI reasoning system to help business teams make faster, data-grounded decisions. Instead of a generic chatbot, every answer is backed by real customer data, pre-computed ML insights, and company policy documents.

---

## What it does

Executives and analysts can ask natural language questions — like "Why is churn high among fiber optic customers?" or "Which segments should we prioritize for retention this month?" — and get structured, actionable responses that cite actual numbers from the database.

Three specialized agents handle different workflows:

- **QA Agent** — answers specific business questions by reasoning across the SQL database, ML insights, and policy documents in a single chain
- **Briefing Agent** — runs autonomously each morning without a user prompt, pulls from all data sources, and produces a structured daily executive report
- **Drift Monitor Agent** — checks whether the trained model is still reliable by comparing current customer distributions to the training baseline and recommends whether retraining is needed

---

## Architecture

```
Frontend (React + Vite)
    |
    | HTTP
    v
Backend (FastAPI)
    |-- /ask           -> QA Agent
    |-- /briefing      -> Briefing Agent
    |-- /drift         -> Raw drift metrics (no agent)
    |-- /drift/analyze -> Drift Monitor Agent
    |
    v
LangChain Tool Layer
    |-- query_db()          -> SQLite (7,043 customer records)
    |-- get_ml_insights()   -> Pre-trained RandomForest insights JSON
    |-- get_company_policy()-> ChromaDB vector store (policy RAG)
    |-- check_data_drift()  -> Compares live DB to training baseline
    |
    v
Data Layer
    |-- enterprise_data.db  -> Customer metrics (Telco churn dataset)
    |-- ml_insights.json    -> Model accuracy, feature importance, risk segments
    |-- data/vector_db/     -> Embedded company policy documents
```

The ML pipeline runs separately and feeds the agent layer — keeping inference fast and the LLM grounded in pre-computed facts rather than raw data.

---

## Project structure

```
backend/
    agents.py            # QA, Briefing, and Drift Monitor agent definitions
    main.py              # FastAPI app with all endpoints
    tools.py             # LangChain tools (SQL, ML insights, RAG, drift)
    ingest_knowledge.py  # One-time script to build the ChromaDB vector store

frontend/
    src/App.jsx          # React dashboard with chat interface and side panel
    src/App.css          # Dark theme, viewport-locked layout

ml/
    data_generator.py    # Loads the CSV into SQLite
    churn_reasoning.py   # Trains RandomForest, saves insights + drift baseline

airflow/
    refresh_insights_dag.py  # Daily DAG: re-ingest data and re-run ML pipeline

docs/
    pricing_policy.md        # Ingested into ChromaDB for policy RAG
    retention_policy.md
    retention_strategy.md

data/
    WA_Fn-UseC_-Telco-Customer-Churn.csv  # Source dataset (7,043 rows)
    enterprise_data.db                    # SQLite database
    ml_insights.json                      # Generated ML output + drift baseline
    vector_db/                            # Persisted ChromaDB embeddings
```

---

## Setup

### Requirements

- Python 3.12
- Node.js 18+
- A Groq API key (free tier works fine — uses `llama-3.3-70b-versatile`)

### Backend

```bash
# Create and activate a virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add your Groq API key
cp .env.example .env
# Edit .env and set GROQ_API_KEY=your_key_here

# Run the data pipeline (only needed once, or when refreshing data)
python -m ml.data_generator
python -m ml.churn_reasoning

# Build the policy knowledge base (only needed once)
python -m backend.ingest_knowledge

# Start the API server
uvicorn backend.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The frontend proxies `/api` requests to the backend on port 8000.

---

## How drift detection works

When the ML model is trained, `churn_reasoning.py` saves a baseline snapshot of the training data distribution — overall churn rate, average tenure, average monthly charges, and contract mix.

At any point, the `/drift` endpoint (or the Drift Monitor Agent) queries the live database and compares its current distributions to that baseline. It reports drift percentages for:

- The full customer population (expected to be near-zero on static data)
- The recent customer cohort — customers with tenure of 6 months or less

New customers on month-to-month contracts churn at roughly twice the rate of the overall base, so the recent cohort consistently flags a real distribution shift. In a live system where new customers are added over time, overall drift would also accumulate and trigger retraining.

---

## Agents and tools

| Agent | Tools used | Triggered by |
|---|---|---|
| QA Agent | query_db, get_ml_insights, get_company_policy, check_data_drift | POST /ask |
| Briefing Agent | query_db, get_ml_insights, get_company_policy, check_data_drift | POST /briefing |
| Drift Monitor | check_data_drift, get_ml_insights | POST /drift/analyze |

Each agent response also triggers a lightweight follow-up call to suggest the next three questions an executive might want to ask, which appear as clickable chips below the response.

---

## Data

The platform uses the IBM Telco Customer Churn dataset (7,043 records). This is publicly available, real-world data — not synthetic. The ML model is a RandomForest classifier trained on the full dataset, achieving around 79.5% accuracy. Feature importances and high-risk segments are computed from the actual model output.

Company policy documents in `/docs` are realistic sample policies covering retention tiers, discount authority, and pricing. These are embedded into ChromaDB using `all-MiniLM-L6-v2` from HuggingFace and retrieved via similarity search at query time.

---

## Airflow integration

Drop `airflow/refresh_insights_dag.py` into your Airflow DAGs folder. The DAG runs two tasks in sequence daily:

1. `ingest_data` — re-loads the CSV into the database
2. `run_ml_analytics` — retrains the model and refreshes `ml_insights.json` with an updated drift baseline

This keeps the agent grounded in current data without any manual steps.
