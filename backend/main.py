"""
main.py — FastAPI application for the Enterprise GenAI Decision Intelligence Platform.

Endpoints:
  GET  /               Health check
  GET  /metrics        Live metrics from the latest ML insights JSON
  GET  /drift          Raw drift metrics vs training baseline (fast, no agent)
  POST /ask            Q&A agent — answers a specific business question
  POST /briefing       Briefing agent — autonomously generates a daily executive report
  POST /drift/analyze  Drift Monitor agent — interprets drift and recommends action
"""

import json
import os
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from .agents import build_llm, build_qa_agent, build_briefing_agent, build_drift_agent
from .tools import _run_drift_check

load_dotenv()

# ── App setup ───────────────────────────────────────────────────────────────────

app = FastAPI(title="Enterprise GenAI Decision Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)
INSIGHTS_PATH = os.path.join(BASE_DIR, "..", "data", "ml_insights.json")

# ── Build agents on startup ─────────────────────────────────────────────────────

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("WARNING: GROQ_API_KEY not set — agent endpoints will return an error.")

llm               = build_llm(groq_api_key)      if groq_api_key else None
qa_executor       = build_qa_agent(llm)          if llm else None
briefing_executor = build_briefing_agent(llm)    if llm else None
drift_executor    = build_drift_agent(llm)        if llm else None

# ── Pydantic models ─────────────────────────────────────────────────────────────

class Query(BaseModel):
    message: str

class Response(BaseModel):
    answer: str
    suggested_questions: List[str] = []

class BriefingResponse(BaseModel):
    briefing: str
    suggested_questions: List[str] = []

class DriftAnalysisResponse(BaseModel):
    analysis: str
    suggested_questions: List[str] = []

# ── Helper: follow-up question suggester ───────────────────────────────────────

def get_suggested_questions(user_query: str, agent_answer: str) -> List[str]:
    """
    One lightweight LLM call (no agent loop) that returns 3 follow-up questions
    an executive might want to ask next based on the Q&A just completed.
    """
    if not llm:
        return []
    try:
        response = llm.invoke([
            SystemMessage(content=(
                "You are a business intelligence assistant. "
                "Based on the Q&A below, suggest exactly 3 short follow-up questions "
                "an executive might ask next. "
                "Return ONLY a valid JSON array of 3 strings — no markdown, no explanation."
            )),
            HumanMessage(content=f"Q: {user_query}\nA: {agent_answer}"),
        ])
        raw = response.content.strip()
        # Strip markdown fences if the model wraps the JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        questions = json.loads(raw)
        if isinstance(questions, list):
            return [q for q in questions if isinstance(q, str)][:3]
    except Exception as e:
        print(f"[Suggestions] Failed to generate follow-up questions: {e}")
    return []

# ── Endpoints ───────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "online", "message": "Enterprise GenAI Platform is running."}


@app.get("/metrics")
async def get_metrics():
    """Return live metrics pulled directly from the latest ML insights JSON."""
    try:
        if not os.path.exists(INSIGHTS_PATH):
            return {"churn_rate": "N/A", "high_risk_count": "N/A",
                    "last_updated": "Pipeline not run yet", "model_accuracy": "N/A"}
        with open(INSIGHTS_PATH) as f:
            data = json.load(f)
        s = data.get("summary", {})
        return {
            "churn_rate":       s.get("overall_churn_rate", "N/A"),
            "high_risk_count":  s.get("high_risk_customers_count", "N/A"),
            "model_accuracy":   s.get("model_accuracy", "N/A"),
            "total_customers":  s.get("total_customers", "N/A"),
            "last_updated":     data.get("generated_at", "Unknown"),
        }
    except Exception as e:
        print(f"[Metrics] Error: {e}")
        return {"churn_rate": "N/A", "high_risk_count": "N/A",
                "last_updated": "Error reading insights", "model_accuracy": "N/A"}


@app.post("/ask", response_model=Response)
async def ask_question(query: Query):
    """Q&A Agent — reasons across SQL, ML insights, and policy docs to answer a question."""
    if not qa_executor:
        return Response(answer=(
            "### Error: GROQ_API_KEY not configured.\n\n"
            "Please set `GROQ_API_KEY` in your `.env` file to enable the reasoning engine."
        ))
    try:
        print(f"[QA Agent] {query.message}")
        result = qa_executor.invoke({"input": query.message})
        answer = result["output"]
        suggestions = get_suggested_questions(query.message, answer)
        return Response(answer=answer, suggested_questions=suggestions)
    except Exception as e:
        print(f"[QA Agent] Error: {e}")
        if "BadRequestError" in str(e) or "400" in str(e):
            return Response(answer=(
                "### Reasoning Error\n\n"
                "The model had trouble processing this request. Please try rephrasing."
            ))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/briefing", response_model=BriefingResponse)
async def generate_briefing():
    """
    Briefing Agent — autonomously calls all data sources and returns a
    Daily Executive Briefing without needing a user question.
    """
    if not briefing_executor:
        return BriefingResponse(briefing=(
            "### Error: GROQ_API_KEY not configured."
        ))
    try:
        print("[Briefing Agent] Starting autonomous investigation...")
        result = briefing_executor.invoke({"input": "Generate today's executive briefing."})
        briefing = result["output"]
        suggestions = get_suggested_questions("Daily executive briefing", briefing)
        return BriefingResponse(briefing=briefing, suggested_questions=suggestions)
    except Exception as e:
        print(f"[Briefing Agent] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/drift")
async def get_drift():
    """
    Raw drift metrics — fast, no agent loop.
    Compares the live DB against the training baseline and returns
    severity levels + per-metric percentages for the frontend card.
    """
    try:
        result = _run_drift_check()
        return json.loads(result)
    except Exception as e:
        print(f"[Drift] Error: {e}")
        return {"error": str(e)}


@app.post("/drift/analyze", response_model=DriftAnalysisResponse)
async def analyze_drift():
    """
    Drift Monitor Agent — calls check_data_drift() and get_ml_insights(),
    then returns an interpreted Model Health Report with a plain-language
    explanation and a clear retraining recommendation.
    """
    if not drift_executor:
        return DriftAnalysisResponse(analysis="### Error: GROQ_API_KEY not configured.")
    try:
        print("[Drift Agent] Analyzing model health...")
        result = drift_executor.invoke({"input": "Analyze the current model drift and produce a health report."})
        analysis = result["output"]
        suggestions = get_suggested_questions("Model drift analysis", analysis)
        return DriftAnalysisResponse(analysis=analysis, suggested_questions=suggestions)
    except Exception as e:
        print(f"[Drift Agent] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
