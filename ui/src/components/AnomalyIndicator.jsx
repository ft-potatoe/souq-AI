import { RadialBarChart, RadialBar, PolarAngleAxis, ResponsiveContainer } from 'recharts';
import './AnomalyIndicator.css';

const THRESHOLDS = [
  { max: 0.45, label: 'Normal', color: 'var(--green)' },
  { max: 0.65, label: 'Elevated', color: 'var(--amber)' },
  { max: 1.0,  label: 'Anomalous', color: 'var(--red)' },
];

function classify(score) {
  return THRESHOLDS.find(t => score <= t.max) ?? THRESHOLDS[THRESHOLDS.length - 1];
}

export default function AnomalyIndicator({ assessment }) {
  if (!assessment) return null;

  const { anomaly_score, confidence, top_contributing_features = [] } = assessment;
  const cls = classify(anomaly_score);
  const pct = Math.round(anomaly_score * 100);
  const confPct = Math.round(confidence * 100);

  const chartData = [{ value: anomaly_score * 100, fill: cls.color }];

  return (
    <div className="anomaly-indicator">
      <div className="ai-header">
        <span className="ai-title">Anomaly Assessment</span>
        <span className="ai-label" style={{ color: cls.color, borderColor: cls.color }}>
          {cls.label}
        </span>
      </div>

      <div className="ai-body">
        <div className="ai-gauge">
          <ResponsiveContainer width={110} height={110}>
            <RadialBarChart
              cx="50%"
              cy="50%"
              innerRadius={32}
              outerRadius={50}
              startAngle={210}
              endAngle={-30}
              data={chartData}
            >
              <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
              <RadialBar
                dataKey="value"
                background={{ fill: 'var(--surface-2)' }}
                cornerRadius={4}
              />
            </RadialBarChart>
          </ResponsiveContainer>
          <div className="ai-gauge-center">
            <span className="ai-pct" style={{ color: cls.color }}>{pct}%</span>
            <span className="ai-conf">{confPct}% conf</span>
          </div>
        </div>

        {top_contributing_features.length > 0 && (
          <div className="ai-features">
            <div className="ai-feat-title">Top drivers</div>
            {top_contributing_features.slice(0, 5).map((f, i) => (
              <div key={i} className="ai-feat-row">
                <span className="ai-feat-rank">{i + 1}</span>
                <span className="ai-feat-name">{f.replace(/_/g, ' ')}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
