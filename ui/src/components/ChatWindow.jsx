import { useEffect, useRef, useState } from 'react';
import AnomalyFeedback from './AnomalyFeedback';
import AnomalyIndicator from './AnomalyIndicator';
import SimilarityCard from './SimilarityCard';
import SimilarityChart from './SimilarityChart';
import AnalyticsPanel from './AnalyticsPanel';
import './ChatWindow.css';

const SUGGESTED_QUESTIONS = [
  'Was today unusual or anomalous?',
  'What similar sessions have occurred historically?',
  'What is the current market regime?',
  'How does today compare to GCC peers?',
  'What are the foreign investor flows today?',
  'How does today\'s return rank historically?',
];

function ThumbButtons({ onUp, onDown, voted }) {
  return (
    <div className="thumb-buttons">
      <button
        className={`thumb-btn ${voted === 'up' ? 'thumb-btn--active-up' : ''}`}
        onClick={onUp}
        disabled={voted != null}
        title="Helpful"
      >
        &#128077;
      </button>
      <button
        className={`thumb-btn ${voted === 'down' ? 'thumb-btn--active-down' : ''}`}
        onClick={onDown}
        disabled={voted != null}
        title="Not helpful"
      >
        &#128078;
      </button>
    </div>
  );
}

// Minimal markdown-like renderer: **bold**, *italic*, `code`, bullet lists, numbered lists
function MsgText({ text }) {
  if (!text) return null;

  const lines = text.split('\n');
  const elements = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Numbered list
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

    // Bullet list
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

    // Heading (###)
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

    // Empty line
    if (!line.trim()) {
      elements.push(<div key={`sp-${i}`} className="msg-spacer" />);
      i++;
      continue;
    }

    // Normal paragraph
    elements.push(
      <p key={`p-${i}`} className="msg-para">
        <InlineText text={line} />
      </p>
    );
    i++;
  }

  return <div className="msg-text">{elements}</div>;
}

function InlineText({ text }) {
  // split on **bold**, *italic*, `code`
  const parts = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0;
  let m;
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
    } catch {
      // silent
    }
  }

  if (msg.role === 'user') {
    return (
      <div className="msg msg--user">
        <div className="msg-bubble msg-bubble--user">{msg.text}</div>
      </div>
    );
  }

  const similarity = msg.payload?.similarity ?? [];
  const anomalyAssessment = msg.payload?.anomaly ?? null;
  const regime = msg.payload?.regime ?? null;

  return (
    <div className="msg msg--assistant">
      <div className="msg-bubble msg-bubble--assistant">
        {msg.loading ? (
          <span className="msg-loading"><span /><span /><span /></span>
        ) : (
          <>
            {regime && <RegimeInline regime={regime} />}
            <MsgText text={msg.text} />
            <ThumbButtons voted={voted} onUp={() => vote('up')} onDown={() => vote('down')} />

            {anomalyAssessment && (
              <AnomalyIndicator assessment={anomalyAssessment} />
            )}
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
              <div className="msg-meta">{msg.payload.response_time_ms.toFixed(0)} ms</div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default function ChatWindow({ date, disabled, onNewMessage, activeHistoryItem }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    if (activeHistoryItem) setMessages(activeHistoryItem.messages);
  }, [activeHistoryItem]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  async function send(question) {
    const q = (question ?? input).trim();
    if (!q || loading || !date || disabled) return;
    setInput('');
    setLoading(true);

    const userMsg = { role: 'user', text: q };
    const placeholder = { role: 'assistant', text: '', loading: true, payload: null };
    setMessages(prev => [...prev, userMsg, placeholder]);

    try {
      const res = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, date }),
      });
      const data = res.ok ? await res.json() : null;
      const answer = data?.answer ?? 'No response received.';
      const payload = data ? {
        similarity: data.similarity_results ?? [],
        anomaly: data.anomaly_assessment ?? null,
        regime: data.regime_context ?? null,
        analytics_used: data.analytics_used ?? [],
        data_date: data.data_date,
        response_time_ms: data.response_time_ms,
        // pass structured analytics buckets through for AnalyticsPanel
        ...buildAnalyticsBuckets(data),
      } : null;

      setMessages(prev => {
        const next = [...prev];
        const idx = next.findLastIndex(m => m.loading);
        if (idx !== -1) next[idx] = { role: 'assistant', text: answer, loading: false, payload };
        return next;
      });

      onNewMessage?.({
        question: q,
        answer,
        date,
        messages: [...messages, userMsg, { role: 'assistant', text: answer, loading: false, payload }],
      });
    } catch {
      setMessages(prev => {
        const next = [...prev];
        const idx = next.findLastIndex(m => m.loading);
        if (idx !== -1) next[idx] = { role: 'assistant', text: 'Error: could not reach the API.', loading: false, payload: null };
        return next;
      });
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }

  function onKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  const isEmpty = messages.length === 0;

  return (
    <div className="chat-window">
      <div className="chat-messages">
        {isEmpty && (
          <div className="chat-empty">
            <div className="chat-empty-icon">&#128202;</div>
            <div className="chat-empty-title">QSE Market Copilot</div>
            <div className="chat-empty-sub">Ask a natural language question about QSE market activity.</div>
            {!disabled && (
              <div className="chat-suggestions">
                {SUGGESTED_QUESTIONS.map((q, i) => (
                  <button
                    key={i}
                    className="chat-suggestion-chip"
                    onClick={() => send(q)}
                    disabled={loading}
                  >
                    {q}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
        {messages.map((msg, i) => (
          <Message key={i} msg={msg} date={date} />
        ))}
        <div ref={bottomRef} />
      </div>
      <div className="chat-input-row">
        <textarea
          ref={inputRef}
          className="chat-input"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={disabled ? 'Loading session date...' : 'Ask about market activity, trends, anomalies... (Enter to send)'}
          rows={2}
          disabled={loading || disabled}
        />
        <button
          className="chat-send"
          onClick={() => send()}
          disabled={loading || disabled || !input.trim()}
        >
          {loading ? (
            <span className="send-loading"><span /><span /><span /></span>
          ) : 'Send'}
        </button>
      </div>
    </div>
  );
}

// Pull analytics bucket results out of the API response for AnalyticsPanel
function buildAnalyticsBuckets(data) {
  // The API doesn't return raw bucket results directly — they're inside the LLM payload
  // We pass through what we have from the structured fields
  // If the API is extended to return raw buckets, they'd be threaded here
  return {};
}
