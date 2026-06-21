import { useEffect, useState } from 'react';
import './RegimeBadge.css';

const TREND_CONFIG = {
  bull:     { label: 'Bull',     color: 'var(--green)' },
  bear:     { label: 'Bear',     color: 'var(--red)'   },
  sideways: { label: 'Sideways', color: 'var(--amber)'  },
};

const VOL_CONFIG = {
  low_vol:  { label: 'Low Vol',  color: 'var(--teal,#1abc9c)'   },
  high_vol: { label: 'High Vol', color: 'var(--purple,#9b59b6)' },
};

export default function RegimeBadge() {
  const [data, setData]   = useState(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    fetch('http://localhost:8000/regime/current')
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(setData)
      .catch(() => setError(true));
  }, []);

  if (error || !data) return null;

  const trendKey = data.current_regime ?? data.regime;
  const tcfg = TREND_CONFIG[trendKey] ?? { label: trendKey ?? '?', color: 'var(--text-muted)' };
  const prob = data.regime_probability != null ? Math.round(data.regime_probability * 100) : null;
  const sessions = data.sessions_in_current_regime;
  const since = data.regime_start_date
    ? new Date(data.regime_start_date).toLocaleDateString('en-GB', { day: 'numeric', month: 'short' })
    : null;

  const vcfg = VOL_CONFIG[data.vol_regime] ?? null;
  const volPct = data.volatility_20d_percentile != null ? Math.round(data.volatility_20d_percentile) : null;

  return (
    <div className="regime-inline-group">
      <span className="regime-dot" style={{ background: tcfg.color }} />
      <span className="regime-name" style={{ color: tcfg.color }}>{tcfg.label}</span>
      <span className="regime-detail">
        {[
          prob != null ? `${prob}%` : null,
          sessions != null ? `${sessions} sessions` : null,
          since ? `since ${since}` : null,
        ].filter(Boolean).join(' · ')}
      </span>
      {vcfg && (
        <>
          <span className="regime-pipe">|</span>
          <span className="regime-vol" style={{ color: vcfg.color }}>{vcfg.label}</span>
          {volPct != null && <span className="regime-detail">{volPct}th pct</span>}
        </>
      )}
    </div>
  );
}
