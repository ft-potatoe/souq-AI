import { useEffect, useState } from 'react';
import './RegimeBadge.css';

const REGIME_CONFIG = {
  bull: { label: 'Bull', color: 'var(--green)', bg: 'rgba(46,204,113,0.12)' },
  bear: { label: 'Bear', color: 'var(--red)', bg: 'rgba(231,76,60,0.12)' },
  sideways: { label: 'Sideways', color: 'var(--amber)', bg: 'rgba(243,156,18,0.12)' },
};

export default function RegimeBadge() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    fetch('http://localhost:8000/regime/current')
      .then(r => r.ok ? r.json() : Promise.reject(r.status))
      .then(setData)
      .catch(() => setError(true));
  }, []);

  if (error) return <div className="regime-badge regime-badge--error">Regime unavailable</div>;
  if (!data) return <div className="regime-badge regime-badge--loading">Loading regime...</div>;

  const cfg = REGIME_CONFIG[data.regime] ?? REGIME_CONFIG.sideways;
  const conf = data.confidence != null ? Math.round(data.confidence * 100) : null;

  return (
    <div className="regime-badge" style={{ borderColor: cfg.color, background: cfg.bg }}>
      <span className="regime-dot" style={{ background: cfg.color }} />
      <span className="regime-label" style={{ color: cfg.color }}>{cfg.label}</span>
      {conf != null && <span className="regime-conf">{conf}%</span>}
      {data.session_count != null && (
        <span className="regime-meta">{data.session_count} sessions</span>
      )}
      {data.start_date && (
        <span className="regime-meta">since {data.start_date}</span>
      )}
      {data.model_version && (
        <span className="regime-version">v{data.model_version}</span>
      )}
    </div>
  );
}
