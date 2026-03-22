import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Send, BarChart3, ShieldAlert, TrendingUp, Activity, Cpu, RefreshCw, BookOpen } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// ── Markdown-lite renderer ────────────────────────────────────────────────────
function formatText(text) {
    const lines = text.split('\n');
    const elements = [];
    let listItems = [];

    const flushList = (key) => {
        if (listItems.length > 0) {
            elements.push(<ul key={`ul-${key}`}>{listItems}</ul>);
            listItems = [];
        }
    };

    lines.forEach((line, i) => {
        if (line.startsWith('### ')) {
            flushList(i);
            elements.push(<h3 key={i}>{line.slice(4)}</h3>);
        } else if (line.startsWith('## ')) {
            flushList(i);
            elements.push(<h2 key={i}>{line.slice(3)}</h2>);
        } else if (line.startsWith('- ') || line.startsWith('* ')) {
            listItems.push(<li key={i}>{line.slice(2)}</li>);
        } else if (line.trim() === '') {
            flushList(i);
        } else {
            flushList(i);
            elements.push(<p key={i}>{line}</p>);
        }
    });
    flushList('end');
    return elements;
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function driftColor(sev) {
    return { HIGH: '#f87171', MEDIUM: '#fbbf24', LOW: '#4ade80' }[sev] ?? '#64748b';
}

function formatTs(ts) {
    if (!ts || ts === 'Unknown' || ts === 'Loading...' || ts === 'Pipeline not run yet') return ts;
    try { return new Date(ts).toLocaleString(); } catch { return ts; }
}

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
    const [input, setInput]         = useState('');
    const [loading, setLoading]     = useState(false);
    const [briefing, setBriefing]   = useState(false);
    const [drift, setDrift]         = useState(null);
    const [metrics, setMetrics]     = useState({ churnRate: '—', highRiskCount: '—', modelAccuracy: '—', lastUpdated: '' });
    const [messages, setMessages]   = useState([{
        type: 'ai',
        text: "### Welcome to Decision Intelligence\nConnected to enterprise data, ML pipeline, and policy knowledge base. Ask a strategic question or run the Daily Briefing.",
        suggestions: [
            "What are the top churn risk factors?",
            "Which customer segment has the highest churn rate?",
            "What retention actions should we take this month?",
        ],
    }]);

    const endRef = useRef(null);

    useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

    useEffect(() => {
        axios.get('/api/metrics').then(r => setMetrics({
            churnRate:      r.data.churn_rate,
            highRiskCount:  String(r.data.high_risk_count),
            modelAccuracy:  r.data.model_accuracy,
            lastUpdated:    r.data.last_updated,
        })).catch(() => {});

        axios.get('/api/drift').then(r => setDrift(r.data)).catch(() => {});
    }, []);

    const busy = loading || briefing;

    const push = (type, text, suggestions = []) =>
        setMessages(prev => [...prev, { type, text, suggestions }]);

    const sendMessage = async (text) => {
        if (!text.trim() || busy) return;
        push('user', text);
        setInput('');
        setLoading(true);
        try {
            const r = await axios.post('/api/ask', { message: text });
            push('ai', r.data.answer, r.data.suggested_questions ?? []);
        } catch {
            push('ai', '### Error\nCould not reach the reasoning engine. Check the backend is running and GROQ_API_KEY is set.');
        } finally {
            setLoading(false);
        }
    };

    const runBriefing = async () => {
        if (busy) return;
        push('user', 'Generate Daily Executive Briefing');
        setBriefing(true);
        try {
            const r = await axios.post('/api/briefing');
            push('ai', r.data.briefing, r.data.suggested_questions ?? []);
        } catch {
            push('ai', '### Error\nCould not generate briefing. Please try again.');
        } finally {
            setBriefing(false);
        }
    };

    const runDriftAnalysis = async () => {
        if (busy) return;
        push('user', 'Run Model Health & Drift Analysis');
        setLoading(true);
        try {
            const r = await axios.post('/api/drift/analyze');
            push('ai', r.data.analysis, r.data.suggested_questions ?? []);
        } catch {
            push('ai', '### Error\nCould not run drift analysis. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    const recentSev = drift?.recent_cohort?.severity;

    return (
        <div className="app">

            {/* ── Header ── */}
            <header className="header">
                <div className="brand">
                    <span className="brand-name">Enterprise GenAI</span>
                    <span className="brand-divider" />
                    <span className="brand-sub">Decision Intelligence</span>
                </div>

                <div className="header-right">
                    <button className="briefing-btn" onClick={runBriefing} disabled={busy}>
                        <BookOpen size={12} />
                        {briefing ? 'Analyzing…' : 'Daily Briefing'}
                    </button>
                    <div className="live-badge">
                        <span className="live-dot" />
                        Live
                    </div>
                </div>
            </header>

            {/* ── Workspace ── */}
            <div className="workspace">

                {/* ── Chat panel ── */}
                <div className="chat-panel">
                    <div className="messages">
                        <AnimatePresence initial={false}>
                            {messages.map((msg, i) => (
                                <motion.div
                                    key={i}
                                    className={`msg ${msg.type}`}
                                    initial={{ opacity: 0, y: 8 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ duration: 0.18 }}
                                >
                                    <div className="bubble">
                                        {formatText(msg.text)}
                                        {msg.suggestions?.length > 0 && (
                                            <div className="chips">
                                                {msg.suggestions.map((q, j) => (
                                                    <button
                                                        key={j}
                                                        className="chip"
                                                        onClick={() => sendMessage(q)}
                                                        disabled={busy}
                                                    >
                                                        {q}
                                                    </button>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                </motion.div>
                            ))}
                        </AnimatePresence>

                        {busy && (
                            <div className="thinking">
                                <div className="dots">
                                    <span /><span /><span />
                                </div>
                                {briefing
                                    ? 'Briefing agent investigating all data sources…'
                                    : 'Reasoning across data sources…'}
                            </div>
                        )}

                        <div ref={endRef} />
                    </div>

                    <div className="input-bar">
                        <input
                            className="input-field"
                            value={input}
                            onChange={e => setInput(e.target.value)}
                            onKeyDown={e => e.key === 'Enter' && sendMessage(input)}
                            placeholder="Ask a strategic question — e.g. 'Why is churn high among fiber optic customers?'"
                            disabled={busy}
                        />
                        <button className="send-btn" onClick={() => sendMessage(input)} disabled={busy}>
                            <Send size={15} />
                        </button>
                    </div>
                </div>

                {/* ── Side panel ── */}
                <aside className="side">

                    {/* Churn rate */}
                    <div className="card card-red">
                        <div className="card-label">
                            <BarChart3 size={12} /> Current Churn Rate
                        </div>
                        <div className="card-value" style={{ color: '#f87171' }}>{metrics.churnRate}</div>
                        <div className="card-sub">Live from ML pipeline</div>
                        <div className="progress">
                            <div
                                className="progress-fill"
                                style={{
                                    width: metrics.churnRate !== '—' ? metrics.churnRate : '0%',
                                    background: 'linear-gradient(90deg, #ef4444, #f87171)',
                                }}
                            />
                        </div>
                    </div>

                    {/* High risk */}
                    <div className="card card-orange">
                        <div className="card-label">
                            <ShieldAlert size={12} /> High-Risk Customers
                        </div>
                        <div className="card-value" style={{ color: '#fb923c' }}>{metrics.highRiskCount}</div>
                        <div className="card-sub">Churn probability &gt; 70%</div>
                    </div>

                    {/* Model accuracy */}
                    <div className="card card-purple">
                        <div className="card-label">
                            <TrendingUp size={12} /> Model Accuracy
                        </div>
                        <div className="card-value" style={{ color: '#a78bfa' }}>{metrics.modelAccuracy}</div>
                        <div className="card-sub">Random Forest Classifier</div>
                        <div className="progress">
                            <div
                                className="progress-fill"
                                style={{
                                    width: metrics.modelAccuracy !== '—' ? metrics.modelAccuracy : '0%',
                                    background: 'linear-gradient(90deg, #7c3aed, #a78bfa)',
                                }}
                            />
                        </div>
                    </div>

                    {/* Drift monitor */}
                    <div className="card">
                        <div className="drift-title-row">
                            <div className="card-label" style={{ margin: 0 }}>
                                <Activity size={12} /> Model Drift
                            </div>
                            {recentSev && (
                                <span className={`sev sev-${recentSev}`}>{recentSev}</span>
                            )}
                        </div>

                        {drift && !drift.error ? (
                            <>
                                <div className="drift-row">
                                    <span className="drift-key">Population</span>
                                    <span style={{ color: driftColor(drift.overall?.severity) }}>
                                        {drift.overall?.max_drift_pct ?? '—'}%
                                    </span>
                                </div>
                                <div className="drift-row">
                                    <span className="drift-key">Recent cohort</span>
                                    <span style={{ color: driftColor(recentSev) }}>
                                        {drift.recent_cohort?.max_drift_pct ?? '—'}%
                                    </span>
                                </div>
                                <p className="drift-note">{drift.recommendation}</p>
                                <button
                                    className="analyze-btn"
                                    onClick={runDriftAnalysis}
                                    disabled={busy}
                                >
                                    Deep Analysis
                                </button>
                            </>
                        ) : (
                            <p className="drift-note">{drift?.error ?? 'Loading…'}</p>
                        )}
                    </div>

                    {/* Pipeline */}
                    <div className="card">
                        <div className="card-label">
                            <Cpu size={12} /> Active Pipeline
                        </div>
                        <div className="pipeline-row">
                            <RefreshCw size={11} className="spin" />
                            refresh_insights_dag
                        </div>
                        {metrics.lastUpdated && (
                            <p className="last-run">Updated {formatTs(metrics.lastUpdated)}</p>
                        )}
                    </div>

                </aside>
            </div>
        </div>
    );
}
