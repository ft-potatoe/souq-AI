import { useState } from 'react';
import './AnalyticsPanel.css';

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

export default function AnalyticsPanel({ payload }) {
  const [open, setOpen] = useState(false);
  if (!payload) return null;

  return (
    <div className="analytics-panel">
      <button className="analytics-toggle" onClick={() => setOpen(o => !o)}>
        <span className="analytics-chevron">{open ? '▾' : '▸'}</span>
        Raw analytics
      </button>
      {open && (
        <div className="analytics-body">
          <JsonNode data={payload} depth={0} />
        </div>
      )}
    </div>
  );
}
