import './ClusteringCard.css';

// Compact one-line profile of a cluster's most distinctive features.
function profileSummary(characteristics) {
  if (!characteristics) return null;
  const keys = ['return_1d', 'volatility_20d', 'breadth_ratio'];
  const parts = keys
    .filter(k => characteristics[k] != null)
    .map(k => `${k.replace(/_/g, ' ')} ${Number(characteristics[k]).toFixed(3)}`);
  return parts.length ? parts.join('  |  ') : null;
}

export default function ClusteringCard({ clustering }) {
  if (!clustering) return null;

  const {
    current_cluster_label,
    is_outlier,
    all_clusters = [],
    noise_fraction,
    note,
  } = clustering;

  return (
    <div className="clustering-card">
      <div className="clu-header">
        <span className="clu-title">Market-state clusters</span>
        <span className={`clu-current ${is_outlier ? 'clu-current--outlier' : ''}`}>
          {current_cluster_label ?? 'unknown'}
        </span>
      </div>

      <div className="clu-body">
        {all_clusters.map(c => {
          const summary = profileSummary(c.characteristics);
          const active = c.label === current_cluster_label && !is_outlier;
          return (
            <div key={c.cluster_id} className={`clu-item ${active ? 'clu-item--active' : ''}`}>
              <div className="clu-item-head">
                <span className="clu-item-label">{c.label}</span>
                <span className="clu-item-size">{c.size} sessions</span>
              </div>
              {summary && <div className="clu-item-profile">{summary}</div>}
            </div>
          );
        })}
      </div>

      <div className="clu-footer">
        {noise_fraction != null && (
          <span className="clu-noise">{(noise_fraction * 100).toFixed(0)}% atypical days</span>
        )}
        {note && <span className="clu-note">{note}</span>}
      </div>
    </div>
  );
}
