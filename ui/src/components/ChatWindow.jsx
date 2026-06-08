import { useEffect, useRef, useState } from 'react';
import AnomalyFeedback from './AnomalyFeedback';
import SimilarityCard from './SimilarityCard';
import AnalyticsPanel from './AnalyticsPanel';
import './ChatWindow.css';

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
  const anomalyScore = msg.payload?.anomaly?.anomaly_score;

  return (
    <div className="msg msg--assistant">
      <div className="msg-bubble msg-bubble--assistant">
        {msg.loading ? (
          <span className="msg-loading"><span /><span /><span /></span>
        ) : (
          <>
            <div className="msg-text">{msg.text}</div>
            <ThumbButtons
              voted={voted}
              onUp={() => vote('up')}
              onDown={() => vote('down')}
            />
            {anomalyScore != null && anomalyScore > 0.65 && (
              <AnomalyFeedback date={date} score={anomalyScore} />
            )}
            {similarity.length > 0 && (
              <div className="sim-section">
                <div className="sim-section-title">Similar sessions</div>
                <div className="sim-cards">
                  {similarity.slice(0, 5).map((m, i) => (
                    <SimilarityCard key={i} match={m} queryDate={date} />
                  ))}
                </div>
              </div>
            )}
            {msg.payload && <AnalyticsPanel payload={msg.payload} />}
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
    if (activeHistoryItem) {
      setMessages(activeHistoryItem.messages);
    }
  }, [activeHistoryItem]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  async function send() {
    const q = input.trim();
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
      // Map top-level API fields into the payload the Message component consumes
      const payload = data ? {
        similarity: data.similarity_results ?? [],
        anomaly: data.anomaly_assessment ?? null,
        regime: data.regime_context ?? null,
        analytics_used: data.analytics_used ?? [],
        data_date: data.data_date,
      } : null;

      setMessages(prev => {
        const next = [...prev];
        const idx = next.findLastIndex(m => m.loading);
        if (idx !== -1) next[idx] = { role: 'assistant', text: answer, loading: false, payload };
        return next;
      });

      onNewMessage?.({ question: q, answer, date, messages: [...messages, userMsg, { role: 'assistant', text: answer, loading: false, payload }] });
    } catch (err) {
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

  return (
    <div className="chat-window">
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <div className="chat-empty-icon">&#128202;</div>
            <div className="chat-empty-title">QSE Market Copilot</div>
            <div className="chat-empty-sub">Ask a natural language question about QSE market activity.</div>
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
          placeholder={disabled ? 'Loading session date...' : 'Ask about market activity, trends, anomalies...'}
          rows={2}
          disabled={loading || disabled}
        />
        <button
          className="chat-send"
          onClick={send}
          disabled={loading || disabled || !input.trim()}
        >
          {loading ? '...' : 'Send'}
        </button>
      </div>
    </div>
  );
}
