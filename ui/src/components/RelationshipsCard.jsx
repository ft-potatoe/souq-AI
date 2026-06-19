import './RelationshipsCard.css';

function humanise(name) {
  return name ? name.replace(/_/g, ' ') : '';
}

export default function RelationshipsCard({ relationships }) {
  if (!relationships) return null;

  const {
    primary_relationships = [],
    conditional,
    note,
  } = relationships;

  return (
    <div className="relationships-card">
      <div className="rel-header">
        <span className="rel-title">Discovered relationships</span>
        <span className="rel-disclaimer">associations, not causation</span>
      </div>

      <div className="rel-body">
        {primary_relationships.slice(0, 5).map((r, i) => (
          <div key={i} className="rel-item">
            <span className="rel-pair">
              {humanise(r.feature_a)}
              <span className={`rel-dir rel-dir--${r.direction}`}>{r.direction}</span>
              {humanise(r.feature_b)}
            </span>
            <span className="rel-strength">{r.spearman?.toFixed(2)}</span>
          </div>
        ))}
      </div>

      {conditional && (
        <div className="rel-conditional">{conditional.plain}</div>
      )}

      {note && <div className="rel-note">{note}</div>}
    </div>
  );
}
