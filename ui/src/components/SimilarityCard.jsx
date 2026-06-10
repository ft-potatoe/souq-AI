import { useState } from 'react';
import './SimilarityCard.css';

const REGIME_COLORS = {
  bull: 'var(--green)',
  bear: 'var(--red)',
  sideways: 'var(--amber)',
};

function StarRating({ value, onChange, disabled }) {
  const [hovered, setHovered] = useState(null);
  const display = hovered ?? value ?? 0;

  return (
    <div className="star-rating" onMouseLeave={() => setHovered(null)}>
      {[1, 2, 3, 4, 5].map(n => (
        <button
          key={n}
          className={`star ${display >= n ? 'star--on' : ''}`}
          onMouseEnter={() => !disabled && setHovered(n)}
          onClick={() => !disabled && onChange(n)}
          disabled={disabled}
          aria-label={`${n} star`}
        >
          &#9733;
        </button>
      ))}
    </div>
  );
}

function ReturnBar({ value, label }) {
  if (value == null) return (
    <div className="sim-return-row">
      <span className="sim-return-label">{label}</span>
      <span className="sim-return-na">n/a</span>
    </div>
  );
  const pct = value * 100;
  const positive = pct >= 0;
  const width = Math.min(Math.abs(pct) * 6, 100);
  return (
    <div className="sim-return-row">
      <span className="sim-return-label">{label}</span>
      <div className="sim-return-bar-wrap">
        <div
          className={`sim-return-bar ${positive ? 'sim-return-bar--pos' : 'sim-return-bar--neg'}`}
          style={{ width: `${width}%` }}
        />
      </div>
      <span
        className="sim-return-val"
        style={{ color: positive ? 'var(--green)' : 'var(--red)' }}
      >
        {positive ? '+' : ''}{pct.toFixed(1)}%
      </span>
    </div>
  );
}

export default function SimilarityCard({ match, queryDate }) {
  const [rating, setRating] = useState(null);
  const [sent, setSent] = useState(false);

  const regime = match.regime ?? 'unknown';
  const regimeColor = REGIME_COLORS[regime] ?? 'var(--text-muted)';
  const score = match.similarity_score ?? match.ranker_score ?? match.score;
  const scoreWidth = score != null ? Math.round(score * 100) : null;

  async function submitRating(stars) {
    setRating(stars);
    try {
      await fetch('http://localhost:8000/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query_date: queryDate,
          feedback_type: 'similarity_rating',
          rating: stars,
          target_date: match.date,
        }),
      });
      setSent(true);
    } catch {
      // optimistic
    }
  }

  return (
    <div className="similarity-card">
      <div className="sim-header">
        <span className="sim-rank">#{match.rank ?? '?'}</span>
        <span className="sim-date">{match.date}</span>
        <span className="sim-regime" style={{ color: regimeColor }}>
          {regime.charAt(0).toUpperCase() + regime.slice(1)}
        </span>
      </div>

      {scoreWidth != null && (
        <div className="sim-score-row">
          <span className="sim-score-label">Match</span>
          <div className="sim-score-track">
            <div className="sim-score-fill" style={{ width: `${scoreWidth}%` }} />
          </div>
          <span className="sim-score-num">{score.toFixed(3)}</span>
        </div>
      )}

      <div className="sim-returns">
        <ReturnBar value={match.forward_return_5d} label="5d" />
        <ReturnBar value={match.forward_return_10d} label="10d" />
      </div>

      <div className="sim-footer">
        <StarRating value={rating} onChange={submitRating} disabled={sent} />
        {sent && <span className="sim-rated">Rated</span>}
      </div>
    </div>
  );
}
