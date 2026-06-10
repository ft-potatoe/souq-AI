import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Cell,
} from 'recharts';
import './SimilarityChart.css';

function pct(v) {
  return v != null ? `${(v * 100).toFixed(1)}%` : null;
}

function ReturnTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="sim-chart-tooltip">
      <div className="sct-date">{label}</div>
      {payload.map(p => (
        <div key={p.dataKey} className="sct-row">
          <span className="sct-name">{p.name}</span>
          <span
            className="sct-val"
            style={{ color: p.value >= 0 ? 'var(--green)' : 'var(--red)' }}
          >
            {pct(p.value)}
          </span>
        </div>
      ))}
    </div>
  );
}

export default function SimilarityChart({ matches }) {
  if (!matches?.length) return null;

  const data = matches
    .filter(m => m.forward_return_5d != null || m.forward_return_10d != null)
    .map(m => ({
      date: m.date,
      '5d': m.forward_return_5d ?? null,
      '10d': m.forward_return_10d ?? null,
      regime: m.regime ?? 'unknown',
    }));

  if (!data.length) return null;

  return (
    <div className="sim-chart-wrap">
      <div className="sim-chart-title">Post-session forward returns (similar sessions)</div>
      <ResponsiveContainer width="100%" height={160}>
        <BarChart data={data} barCategoryGap="30%" barGap={3} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
          <XAxis
            dataKey="date"
            tick={{ fill: 'var(--text-dim)', fontSize: 10 }}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            tickFormatter={v => `${(v * 100).toFixed(0)}%`}
            tick={{ fill: 'var(--text-dim)', fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            width={38}
          />
          <Tooltip content={<ReturnTooltip />} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
          <ReferenceLine y={0} stroke="var(--border)" strokeWidth={1} />
          <Bar dataKey="5d" name="5-day return" radius={[3, 3, 0, 0]}>
            {data.map((d, i) => (
              <Cell
                key={i}
                fill={d['5d'] >= 0 ? 'var(--green)' : 'var(--red)'}
                fillOpacity={0.75}
              />
            ))}
          </Bar>
          <Bar dataKey="10d" name="10-day return" radius={[3, 3, 0, 0]}>
            {data.map((d, i) => (
              <Cell
                key={i}
                fill={d['10d'] >= 0 ? 'rgba(46,204,113,0.45)' : 'rgba(231,76,60,0.45)'}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="sim-chart-legend">
        <span className="scl-swatch" style={{ background: 'var(--green)', opacity: 0.75 }} />
        <span className="scl-label">5-day</span>
        <span className="scl-swatch" style={{ background: 'var(--green)', opacity: 0.45 }} />
        <span className="scl-label">10-day</span>
        <span className="scl-note">(none = within 10 trading days of today)</span>
      </div>
    </div>
  );
}
