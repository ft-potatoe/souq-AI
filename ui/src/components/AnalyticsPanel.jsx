import { useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from 'recharts';
import './AnalyticsPanel.css';

// ── helpers ──────────────────────────────────────────────────────────────────

function fmt(v, decimals = 2) {
  if (v == null) return '—';
  if (typeof v === 'boolean') return v ? 'Yes' : 'No';
  if (typeof v === 'number') return Number.isInteger(v) ? v.toLocaleString() : v.toFixed(decimals);
  return String(v);
}

function pctFmt(v) {
  if (v == null) return '—';
  const p = v * 100;
  return `${p >= 0 ? '+' : ''}${p.toFixed(2)}%`;
}

function StatRow({ label, value, highlight }) {
  return (
    <div className={`ap-stat-row ${highlight ? 'ap-stat-row--hl' : ''}`}>
      <span className="ap-stat-label">{label}</span>
      <span className="ap-stat-value">{value}</span>
    </div>
  );
}

// ── Trend panel ───────────────────────────────────────────────────────────────

function TrendView({ data }) {
  const rsi = data.rsi_14;
  const macd = data.macd;
  const bb = data.bollinger_bands;

  return (
    <div className="ap-section">
      <div className="ap-section-title">Trend &amp; Momentum</div>
      <div className="ap-stat-grid">
        {data.return_1d != null && (
          <StatRow label="1-day return" value={pctFmt(data.return_1d)} highlight />
        )}
        {data.return_5d != null && (
          <StatRow label="5-day return" value={pctFmt(data.return_5d)} />
        )}
        {data.sma_20 != null && (
          <StatRow label="SMA-20" value={fmt(data.sma_20)} />
        )}
        {data.sma_200 != null && (
          <StatRow label="SMA-200" value={fmt(data.sma_200)} />
        )}
        {data.above_sma_20 != null && (
          <StatRow label="Above SMA-20" value={fmt(data.above_sma_20)} />
        )}
        {data.above_sma_200 != null && (
          <StatRow label="Above SMA-200" value={fmt(data.above_sma_200)} />
        )}
        {rsi != null && (
          <StatRow
            label="RSI-14"
            value={
              <span style={{ color: rsi > 70 ? 'var(--red)' : rsi < 30 ? 'var(--green)' : 'var(--text)' }}>
                {fmt(rsi)}
                {rsi > 70 ? ' (overbought)' : rsi < 30 ? ' (oversold)' : ''}
              </span>
            }
            highlight
          />
        )}
        {macd?.macd_line != null && (
          <StatRow label="MACD" value={fmt(macd.macd_line, 4)} />
        )}
        {macd?.signal_line != null && (
          <StatRow label="Signal" value={fmt(macd.signal_line, 4)} />
        )}
        {bb?.upper != null && (
          <StatRow label="BB Upper" value={fmt(bb.upper)} />
        )}
        {bb?.lower != null && (
          <StatRow label="BB Lower" value={fmt(bb.lower)} />
        )}
        {data.atr_14 != null && (
          <StatRow label="ATR-14" value={fmt(data.atr_14)} />
        )}
        {data.slope_10d != null && (
          <StatRow label="Slope 10d" value={fmt(data.slope_10d, 4)} />
        )}
      </div>
    </div>
  );
}

// ── Distribution panel ────────────────────────────────────────────────────────

function DistributionView({ data }) {
  return (
    <div className="ap-section">
      <div className="ap-section-title">Distribution</div>
      <div className="ap-stat-grid">
        {data.return_1d_zscore != null && (
          <StatRow label="Return z-score" value={fmt(data.return_1d_zscore)} highlight />
        )}
        {data.return_1d_percentile != null && (
          <StatRow label="Percentile" value={`${Math.round(data.return_1d_percentile * 100)}th`} />
        )}
        {data.volume_zscore != null && (
          <StatRow label="Volume z-score" value={fmt(data.volume_zscore)} />
        )}
        {data.value_traded_zscore != null && (
          <StatRow label="Value z-score" value={fmt(data.value_traded_zscore)} />
        )}
        {data.skewness != null && (
          <StatRow label="Skewness" value={fmt(data.skewness)} />
        )}
        {data.kurtosis != null && (
          <StatRow label="Kurtosis" value={fmt(data.kurtosis)} />
        )}
        {data.rolling_mean_20d != null && (
          <StatRow label="Rolling mean 20d" value={pctFmt(data.rolling_mean_20d)} />
        )}
        {data.rolling_std_20d != null && (
          <StatRow label="Rolling std 20d" value={pctFmt(data.rolling_std_20d)} />
        )}
      </div>
    </div>
  );
}

// ── Flows panel ───────────────────────────────────────────────────────────────

function FlowsView({ data }) {
  const netForeign = data.foreign_net ?? data.net_foreign;
  const netDomestic = data.domestic_net ?? data.net_domestic;

  const chartData = [];
  if (data.foreign_buy != null) chartData.push({ name: 'For. Buy', value: data.foreign_buy });
  if (data.foreign_sell != null) chartData.push({ name: 'For. Sell', value: -data.foreign_sell });
  if (data.domestic_buy != null) chartData.push({ name: 'Dom. Buy', value: data.domestic_buy });
  if (data.domestic_sell != null) chartData.push({ name: 'Dom. Sell', value: -data.domestic_sell });

  return (
    <div className="ap-section">
      <div className="ap-section-title">Investor Flows</div>
      <div className="ap-stat-grid">
        {data.foreign_buy != null && (
          <StatRow label="Foreign buy" value={fmt(data.foreign_buy, 0)} />
        )}
        {data.foreign_sell != null && (
          <StatRow label="Foreign sell" value={fmt(data.foreign_sell, 0)} />
        )}
        {netForeign != null && (
          <StatRow
            label="Foreign net"
            value={<span style={{ color: netForeign >= 0 ? 'var(--green)' : 'var(--red)' }}>{fmt(netForeign, 0)}</span>}
            highlight
          />
        )}
        {data.domestic_buy != null && (
          <StatRow label="Domestic buy" value={fmt(data.domestic_buy, 0)} />
        )}
        {data.domestic_sell != null && (
          <StatRow label="Domestic sell" value={fmt(data.domestic_sell, 0)} />
        )}
        {netDomestic != null && (
          <StatRow
            label="Domestic net"
            value={<span style={{ color: netDomestic >= 0 ? 'var(--green)' : 'var(--red)' }}>{fmt(netDomestic, 0)}</span>}
            highlight
          />
        )}
        {data.foreign_buy_zscore != null && (
          <StatRow label="Foreign buy z" value={fmt(data.foreign_buy_zscore)} />
        )}
        {data.foreign_net_zscore != null && (
          <StatRow label="Foreign net z" value={fmt(data.foreign_net_zscore)} />
        )}
      </div>
    </div>
  );
}

// ── GCC panel ─────────────────────────────────────────────────────────────────

function GccView({ data }) {
  const peers = data.peers ?? data.gcc_peers ?? [];
  return (
    <div className="ap-section">
      <div className="ap-section-title">GCC Regional</div>
      {data.qse_return != null && (
        <div className="ap-stat-grid" style={{ marginBottom: 8 }}>
          <StatRow label="QSE return" value={pctFmt(data.qse_return)} highlight />
          {data.gcc_avg_return != null && (
            <StatRow label="GCC avg" value={pctFmt(data.gcc_avg_return)} />
          )}
          {data.qse_rank_in_gcc != null && (
            <StatRow label="QSE rank" value={`#${data.qse_rank_in_gcc} of ${data.gcc_market_count ?? '?'}`} />
          )}
        </div>
      )}
      {peers.length > 0 && (
        <div className="ap-gcc-peers">
          {peers.map((p, i) => {
            const ret = p.daily_change_pct ?? p.return;
            return (
              <div key={i} className="ap-gcc-peer">
                <span className="ap-gcc-name">{p.market_name ?? p.market ?? p.name}</span>
                {ret != null && (
                  <span
                    className="ap-gcc-ret"
                    style={{ color: ret >= 0 ? 'var(--green)' : 'var(--red)' }}
                  >
                    {ret >= 0 ? '+' : ''}{ret.toFixed ? ret.toFixed(2) : ret}%
                  </span>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ── Seasonality panel ─────────────────────────────────────────────────────────

const DOW_NAMES = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu'];

function SeasonalityView({ data }) {
  const dayStats = data.day_of_week_stats ?? [];
  const chartData = dayStats.map(d => ({
    name: DOW_NAMES[d.day_of_week] ?? `D${d.day_of_week}`,
    avg_return: d.avg_return,
  }));

  return (
    <div className="ap-section">
      <div className="ap-section-title">Seasonality</div>
      <div className="ap-stat-grid">
        {data.today_day_of_week != null && (
          <StatRow
            label="Day of week"
            value={DOW_NAMES[data.today_day_of_week] ?? data.today_day_of_week}
          />
        )}
        {data.day_of_week_rank != null && (
          <StatRow label="DoW rank" value={data.day_of_week_rank} />
        )}
        {data.is_ramadan != null && (
          <StatRow
            label="Ramadan"
            value={<span style={{ color: data.is_ramadan ? 'var(--gold)' : 'var(--text-dim)' }}>
              {data.is_ramadan ? 'Yes' : 'No'}
            </span>}
          />
        )}
      </div>
      {chartData.length > 0 && (
        <div className="ap-chart-wrap">
          <ResponsiveContainer width="100%" height={90}>
            <LineChart data={chartData} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="2 2" stroke="var(--border)" vertical={false} />
              <XAxis dataKey="name" tick={{ fill: 'var(--text-dim)', fontSize: 10 }} tickLine={false} axisLine={false} />
              <YAxis
                tickFormatter={v => `${(v * 100).toFixed(1)}%`}
                tick={{ fill: 'var(--text-dim)', fontSize: 9 }}
                tickLine={false}
                axisLine={false}
                width={36}
              />
              <Tooltip
                formatter={(v) => [`${(v * 100).toFixed(2)}%`, 'Avg return']}
                contentStyle={{ background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: 6, fontSize: 11 }}
                labelStyle={{ color: 'var(--text-muted)' }}
              />
              <ReferenceLine y={0} stroke="var(--border)" />
              <Line
                type="monotone"
                dataKey="avg_return"
                stroke="var(--gold)"
                strokeWidth={2}
                dot={{ fill: 'var(--gold)', r: 3 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

// ── Correlation panel ─────────────────────────────────────────────────────────

function CorrelationView({ data }) {
  return (
    <div className="ap-section">
      <div className="ap-section-title">Correlation</div>
      <div className="ap-stat-grid">
        {data.rolling_corr_20d != null && (
          <StatRow label="Rolling corr 20d" value={fmt(data.rolling_corr_20d)} highlight />
        )}
        {data.percentile_of_current_corr != null && (
          <StatRow label="Corr percentile" value={`${Math.round(data.percentile_of_current_corr * 100)}th`} />
        )}
        {data.corr_vol_return != null && (
          <StatRow label="Vol-return corr" value={fmt(data.corr_vol_return)} />
        )}
        {data.corr_foreign_domestic != null && (
          <StatRow label="For-Dom corr" value={fmt(data.corr_foreign_domestic)} />
        )}
      </div>
    </div>
  );
}

// ── Summary panel ─────────────────────────────────────────────────────────────

function SummaryView({ data }) {
  const records = data.records ?? {};
  return (
    <div className="ap-section">
      <div className="ap-section-title">Historical Summary</div>
      <div className="ap-stat-grid">
        {data.avg_daily_return_52w != null && (
          <StatRow label="Avg return 52w" value={pctFmt(data.avg_daily_return_52w)} />
        )}
        {data.up_days_ytd != null && data.down_days_ytd != null && (
          <StatRow
            label="YTD up/down"
            value={<><span style={{ color: 'var(--green)' }}>{data.up_days_ytd} up</span> / <span style={{ color: 'var(--red)' }}>{data.down_days_ytd} down</span></>}
          />
        )}
      </div>
      {Object.entries(records).map(([metric, windows]) => (
        <div key={metric} className="ap-summary-metric">
          <div className="ap-summary-metric-name">{metric.replace(/_/g, ' ')}</div>
          <div className="ap-stat-grid">
            {windows && typeof windows === 'object' && Object.entries(windows).map(([window, rec]) => (
              rec && (
                <StatRow
                  key={window}
                  label={window.replace(/_/g, ' ')}
                  value={rec.value != null ? `${fmt(rec.value)} (${rec.date ?? '?'})` : '—'}
                />
              )
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Fallback raw JSON ─────────────────────────────────────────────────────────

function JsonNode({ data, depth = 0 }) {
  const [collapsed, setCollapsed] = useState(depth >= 2);

  if (data === null) return <span className="json-null">null</span>;
  if (typeof data === 'boolean') return <span className="json-bool">{String(data)}</span>;
  if (typeof data === 'number') return <span className="json-num">{data}</span>;
  if (typeof data === 'string') return <span className="json-str">"{data}"</span>;

  if (Array.isArray(data)) {
    if (data.length === 0) return <span className="json-bracket">[]</span>;
    return (
      <span>
        <button className="json-toggle" onClick={() => setCollapsed(c => !c)}>
          {collapsed ? `[${data.length}]` : '['}
        </button>
        {!collapsed && (
          <span className="json-children">
            {data.map((v, i) => (
              <div key={i} className="json-row" style={{ paddingLeft: 16 }}>
                <JsonNode data={v} depth={depth + 1} />
                {i < data.length - 1 && <span className="json-comma">,</span>}
              </div>
            ))}
            <div>]</div>
          </span>
        )}
      </span>
    );
  }

  if (typeof data === 'object') {
    const keys = Object.keys(data);
    if (keys.length === 0) return <span className="json-bracket">{'{}'}</span>;
    return (
      <span>
        <button className="json-toggle" onClick={() => setCollapsed(c => !c)}>
          {collapsed ? `{${keys.length}}` : '{'}
        </button>
        {!collapsed && (
          <span className="json-children">
            {keys.map((k, i) => (
              <div key={k} className="json-row" style={{ paddingLeft: 16 }}>
                <span className="json-key">"{k}"</span>
                <span className="json-colon">: </span>
                <JsonNode data={data[k]} depth={depth + 1} />
                {i < keys.length - 1 && <span className="json-comma">,</span>}
              </div>
            ))}
            <div>{'}'}</div>
          </span>
        )}
      </span>
    );
  }
  return <span>{String(data)}</span>;
}

// ── Main component ────────────────────────────────────────────────────────────

const BUCKET_VIEWS = {
  trend: TrendView,
  distribution: DistributionView,
  flows: FlowsView,
  gcc: GccView,
  seasonality: SeasonalityView,
  correlation: CorrelationView,
  summary: SummaryView,
};

const BUCKET_LABELS = {
  trend: 'Trend',
  distribution: 'Distribution',
  flows: 'Flows',
  gcc: 'GCC',
  seasonality: 'Seasonality',
  correlation: 'Correlation',
  summary: 'Summary',
  regime: 'Regime',
  anomaly: 'Anomaly',
  similarity: 'Similarity',
};

export default function AnalyticsPanel({ payload }) {
  const [open, setOpen] = useState(false);
  const [rawOpen, setRawOpen] = useState(false);
  if (!payload) return null;

  const structuredBuckets = Object.keys(payload).filter(
    k => !['similarity', 'anomaly', 'regime', 'analytics_used', 'data_date'].includes(k)
  );

  if (structuredBuckets.length === 0) return null;

  return (
    <div className="analytics-panel">
      <button className="analytics-toggle" onClick={() => setOpen(o => !o)}>
        <span className="analytics-chevron">{open ? '▾' : '▸'}</span>
        Analytics detail
        <span className="analytics-buckets">
          {structuredBuckets.map(b => (
            <span key={b} className="ap-bucket-chip">{BUCKET_LABELS[b] ?? b}</span>
          ))}
        </span>
      </button>
      {open && (
        <div className="analytics-body">
          {structuredBuckets.map(bucket => {
            const View = BUCKET_VIEWS[bucket];
            if (View) return <View key={bucket} data={payload[bucket]} />;
            return (
              <div key={bucket} className="ap-section">
                <div className="ap-section-title">{BUCKET_LABELS[bucket] ?? bucket}</div>
                <div className="ap-raw-fallback">
                  <JsonNode data={payload[bucket]} depth={0} />
                </div>
              </div>
            );
          })}
          <button
            className="ap-raw-toggle"
            onClick={() => setRawOpen(o => !o)}
          >
            {rawOpen ? '▾' : '▸'} Raw JSON
          </button>
          {rawOpen && (
            <div className="ap-raw-body">
              <JsonNode data={payload} depth={0} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
