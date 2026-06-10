import { useEffect, useState } from 'react';
import './RegimeBadge.css';

const TREND_CONFIG = {
  bull:     { label: 'Bull',     color: 'var(--green)', bg: 'rgba(46,204,113,0.12)' },
  bear:     { label: 'Bear',     color: 'var(--red)',   bg: 'rgba(231,76,60,0.12)'  },
  sideways: { label: 'Sideways', color: 'var(--amber)', bg: 'rgba(243,156,18,0.12)' },
};

const VOL_CONFIG = {
  low_vol:  { label: 'Low Vol',  color: 'var(--teal,#1abc9c)',  bg: 'rgba(26,188,156,0.10)' },
  high_vol: { label: 'High Vol', color: 'var(--purple,#9b59b6)', bg: 'rgba(155,89,182,0.10)' },
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

  const trendKey = data.current_regime ?? data.regime;
  const tcfg = TREND_CONFIG[trendKey] ?? { label: trendKey ?? '?', color: 'var(--text-muted)', bg: 'rgba(255,255,255,0.05)' };
  const prob = data.regime_probability != null ? Math.round(data.regime_probability * 100) : null;

  const vcfg = VOL_CONFIG[data.vol_regime] ?? null;
  const volPct = data.volatility_20d_percentile != null ? Math.round(data.volatility_20d_percentile) : null;

  return (
    <div className="regime-badge-group">
      {/* Trend regime pill */}
      <div className="regime-badge" style={{ borderColor: tcfg.color, background: tcfg.bg }}>
        <span className="regime-dot" style={{ background: tcfg.color }} />
        <span className="regime-label" style={{ color: tcfg.color }}>{tcfg.label}</span>
        {prob != null && <span className="regime-conf">{prob}%</span>}
        {data.sessions_in_current_regime != null && (
          <span className="regime-meta">{data.sessions_in_current_regime} sessions</span>
        )}
        {data.regime_start_date && (
          <span className="regime-meta">since {data.regime_start_date}</span>
        )}
      </div>

      {/* Vol regime pill */}
      {vcfg && (
        <div className="regime-badge regime-badge--vol" style={{ borderColor: vcfg.color, background: vcfg.bg }}>
          <span className="regime-dot" style={{ background: vcfg.color }} />
          <span className="regime-label" style={{ color: vcfg.color }}>{vcfg.label}</span>
          {volPct != null && (
            <span className="regime-meta" style={{ color: vcfg.color, opacity: 0.85 }}>
              {volPct}th pct
            </span>
          )}
        </div>
      )}
    </div>
  );
}
