import { useEffect, useState } from 'react';
import './ModelStatus.css';

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
  const since = feedback.since
    ? new Date(feedback.since).toLocaleDateString()
    : null;

  return (
    <div className="model-status">
      {data.models && Object.entries(data.models).map(([name, info]) => (
        <span key={name} className="ms-model">
          <span className="ms-name">{name}</span>
          {info.version && <span className="ms-version">v{info.version}</span>}
        </span>
      ))}
      {feedbackEntries.length > 0 && (
        <span className="ms-sep" />
      )}
      {feedbackEntries.map(([type, count]) => (
        <span key={type} className="ms-feedback">
          {type}: {count}
        </span>
      ))}
      {data.last_retrain && (
        <>
          <span className="ms-sep" />
          <span className="ms-retrain">
            retrained {new Date(data.last_retrain).toLocaleDateString()}
            {since && ` · feedback since ${since}`}
          </span>
        </>
      )}
    </div>
  );
}
