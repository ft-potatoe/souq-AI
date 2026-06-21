import { useEffect, useState } from 'react';
import './RegimeHistory.css';

const TREND_CONFIG = {
  bull:     { color: 'var(--green)',          bg: 'rgba(46,204,113,0.15)',  label: 'Bull' },
  bear:     { color: 'var(--red)',            bg: 'rgba(231,76,60,0.15)',   label: 'Bear' },
  sideways: { color: 'var(--amber)',          bg: 'rgba(243,156,18,0.15)',  label: 'Sideways' },
};

const VOL_CONFIG = {
  low_vol:  { color: 'var(--teal,#1abc9c)',   bg: 'rgba(26,188,156,0.15)', label: 'Low Vol' },
  high_vol: { color: 'var(--purple,#9b59b6)', bg: 'rgba(155,89,182,0.15)', label: 'High Vol' },
};

function cfg(regime) {
  return TREND_CONFIG[regime] ?? { color: 'var(--text-dim)', bg: 'rgba(255,255,255,0.05)', label: regime ?? '?' };
}

function vcfg(vol) {
  return VOL_CONFIG[vol] ?? { color: 'var(--text-dim)', bg: 'rgba(255,255,255,0.05)', label: vol ?? '?' };
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

function buildSegments(rows, key) {
  const segs = [];
  let cur = null;
  for (const row of rows) {
    const val = row[key];
    if (!cur || cur.val !== val) {
      cur = { val, start: row.date, end: row.date, count: 1 };
      segs.push(cur);
    } else {
      cur.end = row.date;
      cur.count++;
    }
  }
  return segs;
}

function TimelineBar({ segments, total, cfgFn, label }) {
  return (
    <div className="rh-timeline-wrap">
      {label && <div className="rh-timeline-label">{label}</div>}
      <div className="rh-timeline">
        {segments.map((seg, i) => {
          const c = cfgFn(seg.val);
          const width = (seg.count / total) * 100;
          return (
            <div
              key={i}
              className="rh-seg"
              style={{ width: `${width}%`, background: c.bg, borderColor: c.color }}
              title={`${c.label}: ${seg.start} - ${seg.end} (${seg.count} sessions)`}
            >
              {width > 8 && (
                <span className="rh-seg-label" style={{ color: c.color }}>{c.label}</span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function HistoryContent({ data }) {
  const rows = Array.isArray(data) ? data : (data.history ?? []);
  const transitions = Array.isArray(data) ? [] : (data.transitions ?? []);

  if (!rows.length) return <div className="rh-empty">No history available.</div>;

  const trendSegs = buildSegments(rows, 'regime');
  const volRows = rows.filter(r => r.vol_regime != null);
  const volSegs = volRows.length > 0 ? buildSegments(volRows, 'vol_regime') : [];
  const total = rows.length;

  return (
    <>
      <TimelineBar segments={trendSegs} total={total} cfgFn={cfg} label="Trend" />
      {volSegs.length > 0 && (
        <TimelineBar segments={volSegs} total={volRows.length} cfgFn={vcfg} label="Vol" />
      )}

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
        <div className="rh-segments">
          <div className="rh-section-title">Regime spans</div>
          {trendSegs.slice(-8).reverse().map((seg, i) => {
            const c = cfg(seg.val);
            // Dominant vol regime during this trend span
            const spanRows = rows.filter(r => r.date >= seg.start && r.date <= seg.end && r.vol_regime);
            const volCounts = {};
            spanRows.forEach(r => { volCounts[r.vol_regime] = (volCounts[r.vol_regime] ?? 0) + 1; });
            const domVol = Object.entries(volCounts).sort((a, b) => b[1] - a[1])[0]?.[0];
            const vc = domVol ? vcfg(domVol) : null;
            return (
              <div key={i} className="rh-segment-row">
                <span className="rh-seg-dot" style={{ background: c.color }} />
                <span className="rh-seg-label-sm" style={{ color: c.color }}>{c.label}</span>
                {vc && <span className="rh-seg-vol" style={{ color: vc.color }}>{vc.label}</span>}
                <span className="rh-seg-dates">{seg.start} - {seg.end}</span>
                <span className="rh-seg-count">{seg.count}d</span>
              </div>
            );
          })}
        </div>
      )}
    </>
  );
}
