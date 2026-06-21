import { useEffect, useState } from 'react';
import './ModelStatus.css';

const FEEDBACK_ICONS = {
  thumbs_up: '👍',
  thumbs_down: '👎',
  anomaly_confirm: '✓ confirm',
  anomaly_reject: '✗ reject',
  similarity_rating: '★ sim',
};

export default function ModelStatus() {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetch('http://localhost:8000/models/status')
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(setData)
      .catch(() => {});
  }, []);

  if (!data) return null;

  const feedback = data.feedback_counts ?? {};
  const feedbackEntries = Object.entries(feedback).filter(([k]) => k !== 'since');
  const lastRetrain = data.last_retrain
    ? new Date(data.last_retrain).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: '2-digit' })
    : null;

  return (
    <div className="model-status">
      <div className="ms-left">
        {data.models && Object.entries(data.models).map(([name, info]) => (
          <span key={name} className="ms-model-chip">
            <span className="ms-name">{name.replace(/_/g, ' ')}</span>
            {info.version && <span className="ms-version">{info.version}</span>}
            {info.deployed_at && (
              <span className="ms-deployed">
                {new Date(info.deployed_at).toLocaleDateString('en-GB', { day: 'numeric', month: 'short' })}
              </span>
            )}
          </span>
        ))}
      </div>

      <div className="ms-right">
        {feedbackEntries.map(([type, count]) => (
          <span key={type} className="ms-feedback-chip">
            {FEEDBACK_ICONS[type] ?? type} {count}
          </span>
        ))}
        {lastRetrain && (
          <span className="ms-retrain">retrained {lastRetrain}</span>
        )}
      </div>
    </div>
  );
}
