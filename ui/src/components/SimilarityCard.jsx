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
          ★
        </button>
      ))}
    </div>
  );
}

export default function SimilarityCard({ match, queryDate }) {
  const [rating, setRating] = useState(null);
  const [sent, setSent] = useState(false);

  const regime = match.regime ?? 'unknown';
  const regimeColor = REGIME_COLORS[regime] ?? 'var(--text-muted)';
  const score = match.score ?? match.ranker_score ?? match.similarity_score;

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
      // rating optimistically shown, silent failure
    }
  }

  return (
    <div className="similarity-card">
      <div className="sim-header">
        <span className="sim-date">{match.date}</span>
        <span className="sim-regime" style={{ color: regimeColor }}>
          {regime.charAt(0).toUpperCase() + regime.slice(1)}
        </span>
        {score != null && (
          <span className="sim-score">score {score.toFixed ? score.toFixed(3) : score}</span>
        )}
      </div>
      <div className="sim-footer">
        <StarRating value={rating} onChange={submitRating} disabled={sent} />
        {sent && <span className="sim-rated">Rated</span>}
      </div>
    </div>
  );
}
