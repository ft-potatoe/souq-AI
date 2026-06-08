import { useState } from 'react';
import './AnomalyFeedback.css';

export default function AnomalyFeedback({ date, score }) {
  const [sent, setSent] = useState(null);
  const [loading, setLoading] = useState(false);

  if (score == null || score <= 0.65) return null;

  async function submit(type) {
    setLoading(true);
    try {
      await fetch('http://localhost:8000/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query_date: date, feedback_type: type }),
      });
      setSent(type);
    } catch {
      setSent('error');
    } finally {
      setLoading(false);
    }
  }

  if (sent === 'error') {
    return <div className="anomaly-feedback anomaly-feedback--error">Failed to submit feedback.</div>;
  }

  if (sent) {
    return (
      <div className="anomaly-feedback anomaly-feedback--sent">
        Feedback recorded — thank you.
      </div>
    );
  }

  return (
    <div className="anomaly-feedback">
      <span className="anomaly-label">
        Anomaly score {Math.round(score * 100)}% — was this session genuinely unusual?
      </span>
      <div className="anomaly-buttons">
        <button
          className="anomaly-btn anomaly-btn--yes"
          onClick={() => submit('anomaly_confirm')}
          disabled={loading}
        >
          Yes
        </button>
        <button
          className="anomaly-btn anomaly-btn--no"
          onClick={() => submit('anomaly_reject')}
          disabled={loading}
        >
          No
        </button>
      </div>
    </div>
  );
}
