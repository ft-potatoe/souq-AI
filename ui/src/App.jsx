import { useEffect, useState } from 'react';
import RegimeBadge from './components/RegimeBadge';
import ChatWindow from './components/ChatWindow';
import ModelStatus from './components/ModelStatus';
import RegimeHistory from './components/RegimeHistory';
import './App.css';

export default function App() {
  const [date, setDate] = useState(null);
  const [history, setHistory] = useState([]);
  const [activeIdx, setActiveIdx] = useState(null);

  useEffect(() => {
    fetch('http://localhost:8000/features/today')
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(data => { if (data?.date) setDate(String(data.date).slice(0, 10)); })
      .catch(() => {});
  }, []);

  function handleNewMessage({ question, date: msgDate, messages }) {
    setHistory(prev => {
      const next = [
        { question, date: msgDate, messages, ts: Date.now() },
        ...prev.filter(h => h.question !== question || h.date !== msgDate),
      ].slice(0, 20);
      return next;
    });
    setActiveIdx(0);
  }

  return (
    <div className="app">
      <header className="app-header">
        <span className="app-logo">QSE Market Copilot</span>
        <RegimeBadge />
        <div className="header-spacer" />
        <div className="date-selector-wrap">
          <label htmlFor="date-pick">Session date</label>
          <input
            id="date-pick"
            type="date"
            className="date-input"
            value={date ?? ''}
            onChange={e => setDate(e.target.value)}
            disabled={date === null}
          />
        </div>
      </header>

      <div className="app-body">
        <aside className="sidebar">
          <div className="sidebar-title">Recent queries</div>
          <div className="sidebar-list">
            {history.length === 0 && (
              <div className="sidebar-empty">No queries yet</div>
            )}
            {history.map((item, i) => (
              <div
                key={i}
                className={`sidebar-item ${activeIdx === i ? 'active' : ''}`}
                onClick={() => setActiveIdx(i)}
              >
                <div className="sidebar-item-date">{item.date}</div>
                <div className="sidebar-item-question">{item.question}</div>
              </div>
            ))}
          </div>
        </aside>

        <main className="main-area">
          <ChatWindow
            date={date}
            disabled={date === null}
            onNewMessage={handleNewMessage}
            activeHistoryItem={activeIdx != null ? history[activeIdx] : null}
          />
        </main>
      </div>

      <footer className="app-footer">
        <RegimeHistory />
        <ModelStatus />
      </footer>
    </div>
  );
}
