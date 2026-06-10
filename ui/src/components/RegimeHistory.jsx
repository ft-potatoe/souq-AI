import { useEffect, useState } from 'react';
import './RegimeHistory.css';

const REGIME_CONFIG = {
  bull:     { color: 'var(--green)',  bg: 'rgba(46,204,113,0.15)', label: 'Bull' },
  bear:     { color: 'var(--red)',    bg: 'rgba(231,76,60,0.15)',  label: 'Bear' },
  sideways: { color: 'var(--amber)',  bg: 'rgba(243,156,18,0.15)', label: 'Sideways' },
};

function cfg(regime) {
  return REGIME_CONFIG[regime] ?? { color: 'var(--text-dim)', bg: 'rgba(255,255,255,0.05)', label: regime };
}

export default function RegimeHistory() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!open || data) return;
    fetch('http://localhost:8000/regime/history')
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(setData)
      .catch(() => setError(true));
  }, [open, data]);

  return (
    <div className="regime-history">
      <button className="rh-toggle" onClick={() => setOpen(o => !o)}>
        <span className="rh-chevron">{open ? '▾' : '▸'}</span>
        Regime history
      </button>

      {open && (
        <div className="rh-body">
          {error && <div className="rh-error">Could not load regime history.</div>}
          {!error && !data && <div className="rh-loading">Loading...</div>}
          {data && <HistoryContent data={data} />}
        </div>
      )}
    </div>
  );
}

function HistoryContent({ data }) {
  // data can be an array of {date, regime, probability} rows or {history: [...], transitions: [...]}
  const rows = Array.isArray(data) ? data : (data.history ?? []);
  const transitions = Array.isArray(data) ? [] : (data.transitions ?? []);

  if (!rows.length) return <div className="rh-empty">No history available.</div>;

  // Build a compact segment view: consecutive same-regime runs
  const segments = [];
  let cur = null;
  for (const row of rows) {
    if (!cur || cur.regime !== row.regime) {
      cur = { regime: row.regime, start: row.date, end: row.date, count: 1 };
      segments.push(cur);
    } else {
      cur.end = row.date;
      cur.count++;
    }
  }

  const total = rows.length;

  return (
    <>
      {/* Timeline bar */}
      <div className="rh-timeline">
        {segments.map((seg, i) => {
          const c = cfg(seg.regime);
          const width = (seg.count / total) * 100;
          return (
            <div
              key={i}
              className="rh-seg"
              style={{ width: `${width}%`, background: c.bg, borderColor: c.color }}
              title={`${c.label}: ${seg.start} → ${seg.end} (${seg.count} sessions)`}
            >
              {width > 8 && (
                <span className="rh-seg-label" style={{ color: c.color }}>{c.label}</span>
              )}
            </div>
          );
        })}
      </div>

      {/* Transitions table */}
      {transitions.length > 0 ? (
        <div className="rh-transitions">
          <div className="rh-section-title">Recent transitions</div>
          {transitions.slice(-8).reverse().map((t, i) => {
            const from = cfg(t.from_regime ?? t.from);
            const to = cfg(t.to_regime ?? t.to);
            return (
              <div key={i} className="rh-transition-row">
                <span className="rh-trans-date">{t.date}</span>
                <span className="rh-trans-from" style={{ color: from.color }}>{from.label}</span>
                <span className="rh-trans-arrow">&#8594;</span>
                <span className="rh-trans-to" style={{ color: to.color }}>{to.label}</span>
              </div>
            );
          })}
        </div>
      ) : (
        /* segment list fallback */
        <div className="rh-segments">
          <div className="rh-section-title">Regime spans</div>
          {segments.slice(-8).reverse().map((seg, i) => {
            const c = cfg(seg.regime);
            return (
              <div key={i} className="rh-segment-row">
                <span className="rh-seg-dot" style={{ background: c.color }} />
                <span className="rh-seg-label-sm" style={{ color: c.color }}>{c.label}</span>
                <span className="rh-seg-dates">{seg.start} – {seg.end}</span>
                <span className="rh-seg-count">{seg.count}d</span>
              </div>
            );
          })}
        </div>
      )}
    </>
  );
}
