"""
agents.py — Agent definitions for the Enterprise GenAI Decision Intelligence Platform.

Three agents defined using LangChain's tool-calling agent framework:

  QAAgent           Answers specific business questions by reasoning across SQL
                    data, ML insights, policy docs, and drift status.

  BriefingAgent     Proactively investigates all data sources without a user
                    question and produces a structured Daily Executive Briefing.

  DriftMonitorAgent Specialized agent that interprets drift metrics, explains
                    what they mean for model reliability, and recommends action.

Usage:
    from .agents import build_llm, build_qa_agent, build_briefing_agent, build_drift_agent

    llm      = build_llm(api_key)
    qa       = build_qa_agent(llm)
    briefing = build_briefing_agent(llm)
    drift    = build_drift_agent(llm)

    answer   = qa.invoke({"input": "Why is churn high?"})["output"]
    report   = briefing.invoke({"input": "Generate briefing."})["output"]
    health   = drift.invoke({"input": "Analyze model drift."})["output"]
"""

from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .tools import query_db, get_ml_insights, get_company_policy, check_data_drift

# ── Tool lists ──────────────────────────────────────────────────────────────────

# Full tool set — used by Q&A and Briefing agents
TOOLS = [query_db, get_ml_insights, get_company_policy, check_data_drift]

# Drift-only tool set — focused, prevents the drift agent from going off-topic
DRIFT_TOOLS = [check_data_drift, get_ml_insights]

# ── System prompts ──────────────────────────────────────────────────────────────

QA_SYSTEM_PROMPT = """\
You are the Enterprise Decision Intelligence Agent.

Your job is to help executives understand business performance and churn risk \
by combining four data sources:
  1. SQL database     — live customer metrics (tenure, contract type, charges, churn)
  2. ML insights      — pre-computed churn risk, feature importances, high-risk segments
  3. Policy docs      — approved retention offers, pricing tiers, discount authority
  4. Drift detection  — whether the model is still reliable on current customer data

HOW TO REASON:
- Start with get_ml_insights() for overall context and risk factors.
- Use query_db() when you need specific numbers, counts, or segment breakdowns.
- Use get_company_policy() when the answer requires knowing approved actions or rules.
- Use check_data_drift() if asked about model health, reliability, or retraining needs.
- You may call tools more than once or in any order the question demands.

FORMAT your final answer exactly as:
### Analysis
<what the data shows>

### Root Cause
<why this is happening, grounded in the data>

### Recommended Actions
<specific, policy-aligned steps — cite offer names and discount limits where relevant>
"""

BRIEFING_SYSTEM_PROMPT = """\
You are a Proactive Enterprise Intelligence Analyst.

Your job is to autonomously investigate the current state of the business \
and produce a Daily Executive Briefing — no user question required.

You MUST call all three tools before writing the briefing:

  1. get_ml_insights()
     → Retrieve model accuracy, overall churn rate, top 5 risk factors, \
high-risk segment description.

  2. query_db() with this SQL:
     SELECT Contract,
            COUNT(*)                        AS total_customers,
            SUM(Churn)                      AS churned,
            ROUND(AVG(MonthlyCharges), 2)   AS avg_monthly_charge
     FROM customer_metrics
     GROUP BY Contract
     ORDER BY churned DESC

  3. get_company_policy("retention discount offer")
     → Find what retention actions are currently policy-approved.

Once you have the tool results, write the briefing with these four sections:

### Key Metrics Summary
### Top Risk Areas
### Priority Actions for Today
### Policy Reminders

Rules:
- Cite actual numbers from your tool calls (not placeholders).
- Keep each section concise — bullet points preferred.
- This will be read by the C-suite; be direct and actionable.
"""

# ── LLM factory ────────────────────────────────────────────────────────────────

def build_llm(api_key: str, temperature: float = 0.0) -> ChatGroq:
    """Return a ChatGroq LLM instance using Llama-3.3-70b-versatile."""
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=temperature,
    )

# ── Agent factories ─────────────────────────────────────────────────────────────

def _make_executor(llm: ChatGroq, system_prompt: str) -> AgentExecutor:
    """Build an AgentExecutor from a system prompt — shared by both agents."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    return AgentExecutor(agent=agent, tools=TOOLS, verbose=True)


def build_qa_agent(llm: ChatGroq) -> AgentExecutor:
    """
    Q&A Agent — answers specific executive questions using SQL, ML insights,
    and policy docs. Returns structured Analysis / Root Cause / Recommended Actions.
    """
    return _make_executor(llm, QA_SYSTEM_PROMPT)


def build_briefing_agent(llm: ChatGroq) -> AgentExecutor:
    """
    Briefing Agent — proactively calls all three tools without a user question
    and produces a structured Daily Executive Briefing for the C-suite.
    """
    return _make_executor(llm, BRIEFING_SYSTEM_PROMPT)


# ── Drift Monitor Agent ─────────────────────────────────────────────────────────

DRIFT_SYSTEM_PROMPT = """\
You are the Model Health Monitor Agent.

Your sole job is to assess whether the ML churn model is still reliable \
and whether retraining is needed.

You MUST call both tools before writing your report:

  1. check_data_drift()
     → Get current drift metrics: severity levels, per-metric percentage deviations,
       and the recent customer cohort analysis (tenure ≤ 6 months).

  2. get_ml_insights()
     → Get the model's current accuracy, overall churn rate, and top risk factors
       for context.

Once you have the results, write a concise Model Health Report:

### Model Health Status
<overall verdict: HEALTHY / WATCH / RETRAIN NEEDED — include the severity label>

### Drift Analysis
<what is drifting and by how much — be specific with the percentages>
<explain what this means in plain business terms, not just numbers>

### Recommendation
<clear action: no action / monitor weekly / schedule retraining — and why>

Keep the report short and direct. This is read by the Head of Data Science.
"""


def build_drift_agent(llm: ChatGroq) -> AgentExecutor:
    """
    Drift Monitor Agent — calls check_data_drift() and get_ml_insights(),
    then produces an interpreted Model Health Report with a retraining recommendation.
    Uses a focused tool set (no SQL, no policy) to stay on-topic.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", DRIFT_SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, DRIFT_TOOLS, prompt)
    return AgentExecutor(agent=agent, tools=DRIFT_TOOLS, verbose=True)
