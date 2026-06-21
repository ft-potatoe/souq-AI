import { useEffect, useRef, useState } from 'react';
import AnomalyFeedback from './AnomalyFeedback';
import AnomalyIndicator from './AnomalyIndicator';
import SimilarityCard from './SimilarityCard';
import SimilarityChart from './SimilarityChart';
import AnalyticsPanel from './AnalyticsPanel';
import './ChatWindow.css';

// ── Categorised welcome chips ─────────────────────────────────────────────────
const CHIP_CATEGORIES = [
  {
    label: 'Market',
    chips: ["Was today's volume unusual?", "How does today's return rank?"],
  },
  {
    label: 'Flows',
    chips: ['Foreign investor flows today?', 'Who is driving the market?'],
  },
  {
    label: 'Regime',
    chips: ['What is the current market regime?', 'Similar historical sessions?'],
  },
  {
    label: 'GCC',
    chips: ['How does QSE compare to peers?', 'Decoupled from GCC today?'],
  },
];

// ── Metrics row ───────────────────────────────────────────────────────────────
function MetricCell({ label, value, sub, valueColor, subColor, noBorder }) {
  return (
    <div className={`metric-cell${noBorder ? ' metric-cell--last' : ''}`}>
      <span className="metric-label">{label}</span>
      <span className="metric-value" style={{ color: valueColor }}>{value ?? '—'}</span>
      {sub && <span className="metric-sub" style={{ color: subColor }}>{sub}</span>}
    </div>
  );
}

function TodayMetrics({ metrics, regime }) {
  if (!metrics && !regime) return null;

  const vol = metrics?.volume != null
    ? (metrics.volume / 1e6).toFixed(1) + 'M'
    : null;
  const volPct = metrics?.volume_zscore != null
    ? `z-score ${metrics.volume_zscore.toFixed(1)}`
    : null;

  const anomScore = metrics?.anomaly_score != null
    ? Math.round(metrics.anomaly_score * 100) + '%'
    : null;

  const fnet = metrics?.foreign_net != null
    ? (metrics.foreign_net >= 0 ? '+' : '') + (metrics.foreign_net / 1e6).toFixed(1) + 'M'
    : null;

  const regimeLabel = regime?.current_regime ?? metrics?.regime;
  const regimeProb = regime?.regime_probability != null
    ? Math.round(regime.regime_probability * 100) + '%'
    : null;
  const regimeSessions = regime?.sessions_in_current_regime;

  const REGIME_COLORS = { bull: 'var(--green)', bear: 'var(--red)', sideways: 'var(--amber)' };
  const regColor = REGIME_COLORS[regimeLabel] ?? 'var(--text-muted)';

  return (
    <div className="today-metrics-wrap">
      <div className="today-metrics-title">Today's Session Overview</div>
      <div className="today-metrics-row">
        {vol && (
          <MetricCell
            label="Volume"
            value={vol}
            sub={volPct}
            valueColor="var(--text)"
            subColor="var(--green)"
          />
        )}
        {regimeLabel && (
          <MetricCell
            label="Regime"
            value={regimeLabel.charAt(0).toUpperCase() + regimeLabel.slice(1)}
            sub={[regimeProb, regimeSessions ? `${regimeSessions} sessions` : null].filter(Boolean).join(' · ')}
            valueColor={regColor}
            subColor="var(--text-muted)"
          />
        )}
        {anomScore && (
          <MetricCell
            label="Anomaly Score"
            value={anomScore}
            sub="Check anomaly details"
            valueColor="var(--red)"
            subColor="var(--qse-maroon)"
          />
        )}
        {fnet && (
          <MetricCell
            label="Foreign Net"
            value={fnet}
            sub={metrics?.foreign_net >= 0 ? 'Net buying' : 'Net selling'}
            valueColor={metrics?.foreign_net >= 0 ? 'var(--green)' : 'var(--red)'}
            subColor="var(--text-muted)"
            noBorder
          />
        )}
      </div>
    </div>
  );
}

// ── Markdown-like text renderer ───────────────────────────────────────────────
function InlineText({ text }) {
  const parts = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) parts.push({ type: 'text', content: text.slice(last, m.index) });
    const raw = m[0];
    if (raw.startsWith('**')) parts.push({ type: 'bold', content: raw.slice(2, -2) });
    else if (raw.startsWith('*')) parts.push({ type: 'italic', content: raw.slice(1, -1) });
    else parts.push({ type: 'code', content: raw.slice(1, -1) });
    last = m.index + raw.length;
  }
  if (last < text.length) parts.push({ type: 'text', content: text.slice(last) });

  return (
    <>
      {parts.map((p, i) => {
        if (p.type === 'bold') return <strong key={i}>{p.content}</strong>;
        if (p.type === 'italic') return <em key={i}>{p.content}</em>;
        if (p.type === 'code') return <code key={i} className="msg-inline-code">{p.content}</code>;
        return p.content;
      })}
    </>
  );
}

function MsgText({ text }) {
  if (!text) return null;
  const lines = text.split('\n');
  const elements = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (/^\d+\.\s/.test(line)) {
      const items = [];
      while (i < lines.length && /^\d+\.\s/.test(lines[i])) {
        items.push(lines[i].replace(/^\d+\.\s/, ''));
        i++;
      }
      elements.push(
        <ol key={`ol-${i}`} className="msg-list msg-list--ol">
          {items.map((it, j) => <li key={j}><InlineText text={it} /></li>)}
        </ol>
      );
      continue;
    }

    if (/^[-*]\s/.test(line)) {
      const items = [];
      while (i < lines.length && /^[-*]\s/.test(lines[i])) {
        items.push(lines[i].replace(/^[-*]\s/, ''));
        i++;
      }
      elements.push(
        <ul key={`ul-${i}`} className="msg-list msg-list--ul">
          {items.map((it, j) => <li key={j}><InlineText text={it} /></li>)}
        </ul>
      );
      continue;
    }

    if (/^#{1,3}\s/.test(line)) {
      const level = line.match(/^(#+)/)[1].length;
      const content = line.replace(/^#+\s/, '');
      const Tag = `h${Math.min(level + 3, 6)}`;
      elements.push(
        <Tag key={`h-${i}`} className={`msg-heading msg-heading--${level}`}>
          <InlineText text={content} />
        </Tag>
      );
      i++;
      continue;
    }

    if (!line.trim()) {
      elements.push(<div key={`sp-${i}`} className="msg-spacer" />);
      i++;
      continue;
    }

    elements.push(
      <p key={`p-${i}`} className="msg-para">
        <InlineText text={line} />
      </p>
    );
    i++;
  }

  return <div className="msg-text">{elements}</div>;
}

// ── Regime inline strip ───────────────────────────────────────────────────────
function RegimeInline({ regime }) {
  if (!regime) return null;
  const COLORS = { bull: 'var(--green)', bear: 'var(--red)', sideways: 'var(--amber)' };
  const color = COLORS[regime.current_regime] ?? 'var(--text-muted)';
  const prob = regime.regime_probability != null ? Math.round(regime.regime_probability * 100) : null;

  return (
    <div className="regime-inline">
      <span className="regime-inline-dot" style={{ background: color }} />
      <span className="regime-inline-label" style={{ color }}>
        {regime.current_regime?.charAt(0).toUpperCase() + regime.current_regime?.slice(1)} regime
      </span>
      {prob != null && <span className="regime-inline-prob">{prob}%</span>}
      {regime.sessions_in_current_regime != null && (
        <span className="regime-inline-meta">{regime.sessions_in_current_regime} sessions</span>
      )}
      {regime.regime_start_date && (
        <span className="regime-inline-meta">since {regime.regime_start_date}</span>
      )}
    </div>
  );
}

// ── Anomaly score bar ─────────────────────────────────────────────────────────
function AnomalyBar({ assessment }) {
  if (!assessment) return null;
  const pct = Math.round(assessment.anomaly_score * 100);
  const isAnom = assessment.anomaly_score > 0.65;
  return (
    <div className="anomaly-bar-wrap">
      <span className="anomaly-bar-label">Anomaly Score</span>
      <div className="anomaly-bar-track">
        <div
          className="anomaly-bar-fill"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="anomaly-bar-value" style={{ color: isAnom ? 'var(--red)' : 'var(--amber)' }}>
        {pct}%
      </span>
      {isAnom && (
        <span className="anomaly-bar-badge">Anomalous</span>
      )}
    </div>
  );
}

// ── Thumbs ────────────────────────────────────────────────────────────────────
function ThumbButtons({ onUp, onDown, voted }) {
  return (
    <div className="thumb-buttons">
      <button
        className={`thumb-btn ${voted === 'up' ? 'thumb-btn--active-up' : ''}`}
        onClick={onUp} disabled={voted != null} title="Helpful"
      >&#128077;</button>
      <button
        className={`thumb-btn ${voted === 'down' ? 'thumb-btn--active-down' : ''}`}
        onClick={onDown} disabled={voted != null} title="Not helpful"
      >&#128078;</button>
    </div>
  );
}

// ── Message ───────────────────────────────────────────────────────────────────
function Message({ msg, date }) {
  const [voted, setVoted] = useState(null);

  async function vote(type) {
    setVoted(type);
    try {
      await fetch('http://localhost:8000/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query_date: date,
          feedback_type: type === 'up' ? 'thumbs_up' : 'thumbs_down',
          question: msg.question,
        }),
      });
    } catch { /* silent */ }
  }

  if (msg.role === 'user') {
    return (
      <div className="msg msg--user" style={{ animation: 'fadeUp 0.3s ease both' }}>
        <div className="msg-bubble msg-bubble--user">{msg.text}</div>
      </div>
    );
  }

  const similarity        = msg.payload?.similarity ?? [];
  const anomalyAssessment = msg.payload?.anomaly ?? null;
  const regime            = msg.payload?.regime ?? null;

  return (
    <div className="msg msg--assistant" style={{ animation: 'fadeUp 0.4s 0.14s ease both' }}>
      <div className="msg-bubble msg-bubble--assistant">
        {msg.loading ? (
          <span className="msg-loading"><span /><span /><span /></span>
        ) : (
          <>
            {regime && <RegimeInline regime={regime} />}
            <MsgText text={msg.text} />
            <ThumbButtons voted={voted} onUp={() => vote('up')} onDown={() => vote('down')} />

            {anomalyAssessment && <AnomalyBar assessment={anomalyAssessment} />}
            {anomalyAssessment && <AnomalyIndicator assessment={anomalyAssessment} />}
            {anomalyAssessment?.anomaly_score > 0.65 && (
              <AnomalyFeedback date={date} score={anomalyAssessment.anomaly_score} />
            )}

            {similarity.length > 0 && (
              <div className="sim-section">
                <div className="sim-section-title">Similar sessions</div>
                <div className="sim-cards">
                  {similarity.slice(0, 5).map((m, i) => (
                    <SimilarityCard key={i} match={m} queryDate={date} />
                  ))}
                </div>
                <SimilarityChart matches={similarity} />
              </div>
            )}

            {msg.payload && <AnalyticsPanel payload={msg.payload} />}

            {msg.payload?.response_time_ms != null && (
              <div className="msg-meta">
                {msg.payload.data_date && (
                  <span className="msg-meta-date">data: {msg.payload.data_date}</span>
                )}
                <span>{msg.payload.response_time_ms.toFixed(0)} ms</span>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// ── ChatWindow ────────────────────────────────────────────────────────────────
export default function ChatWindow({
  date, disabled, onNewMessage,
  activeHistoryItem, todayMetrics,
  pendingSend, onPendingSendConsumed,
}) {
  const [messages, setMessages] = useState([]);
  const [input, setInput]       = useState('');
  const [loading, setLoading]   = useState(false);
  const [regime, setRegime]     = useState(null);
  const bottomRef = useRef(null);
  const inputRef  = useRef(null);

  // Fetch regime for welcome metrics
  useEffect(() => {
    fetch('http://localhost:8000/regime/current')
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(setRegime)
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (activeHistoryItem) setMessages(activeHistoryItem.messages);
  }, [activeHistoryItem]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Consume pendingSend from sidebar Quick Ask
  useEffect(() => {
    if (pendingSend && !loading && date && !disabled) {
      send(pendingSend);
      onPendingSendConsumed?.();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingSend]);

  async function send(question) {
    const q = (question ?? input).trim();
    if (!q || loading || !date || disabled) return;
    setInput('');
    setLoading(true);

    const userMsg   = { role: 'user', text: q };
    const placeholder = { role: 'assistant', text: '', loading: true, payload: null };
    setMessages(prev => [...prev, userMsg, placeholder]);

    try {
      const res = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, date }),
      });
      const data   = res.ok ? await res.json() : null;
      const answer = data?.answer ?? 'No response received.';
      const payload = data ? {
        similarity:     data.similarity_results ?? [],
        anomaly:        data.anomaly_assessment ?? null,
        regime:         data.regime_context ?? null,
        analytics_used: data.analytics_used ?? [],
        data_date:      data.data_date,
        response_time_ms: data.response_time_ms,
      } : null;

      setMessages(prev => {
        const next = [...prev];
        const idx  = next.findLastIndex(m => m.loading);
        if (idx !== -1) next[idx] = { role: 'assistant', text: answer, loading: false, payload };
        onNewMessage?.({
          question: q, answer, date,
          messages: next,
        });
        return next;
      });
    } catch {
      setMessages(prev => {
        const next = [...prev];
        const idx  = next.findLastIndex(m => m.loading);
        if (idx !== -1) {
          next[idx] = { role: 'assistant', text: 'Error: could not reach the API.', loading: false, payload: null };
          onNewMessage?.({ question: q, answer: next[idx].text, date, messages: next });
        }
        return next;
      });
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }

  function onKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  }

  const isEmpty = messages.length === 0;

  return (
    <div className="chat-window">
      <div className="chat-messages">
        {isEmpty ? (
          <div className="chat-empty">
            {/* Logo block */}
            <div className="welcome-logo-block">
              <img src="/qse_logo.png" className="welcome-logo" alt="QSE" />
              <div className="welcome-exchange">Qatar Stock Exchange</div>
              <h1 className="welcome-title">Market Copilot</h1>
              <p className="welcome-sub">
                Natural language market intelligence. Every answer is analytics-backed — no hallucinated figures.
              </p>
            </div>

            {/* Metrics row */}
            {!disabled && (
              <TodayMetrics metrics={todayMetrics} regime={regime} />
            )}

            {/* Categorised chips */}
            {!disabled && (
              <div className="chip-categories">
                {CHIP_CATEGORIES.map(cat => (
                  <div key={cat.label} className="chip-row">
                    <span className="chip-category-label">{cat.label}</span>
                    <div className="chip-list">
                      {cat.chips.map(q => (
                        <button
                          key={q}
                          className="welcome-chip"
                          onClick={() => send(q)}
                          disabled={loading}
                        >
                          {q}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : (
          messages.map((msg, i) => (
            <Message key={i} msg={msg} date={date} />
          ))
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div className="chat-input-area">
        <div className="chat-input-row">
          <input
            ref={inputRef}
            type="text"
            className="chat-input"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={disabled ? 'Loading session date…' : 'Ask about market activity, trends, anomalies…'}
            disabled={loading || disabled}
          />
          <button
            className="chat-send"
            onClick={() => send()}
            disabled={loading || disabled || !input.trim()}
          >
            {loading ? (
              <span className="send-loading"><span /><span /><span /></span>
            ) : 'Send →'}
          </button>
        </div>
      </div>
    </div>
  );
}
