import { useEffect, useState } from 'react';
import RegimeBadge from './components/RegimeBadge';
import ChatWindow from './components/ChatWindow';
import ModelStatus from './components/ModelStatus';
import RegimeHistory from './components/RegimeHistory';
import './App.css';

const QUICK_ASK = ['Anomaly?', 'vs GCC?', 'Flows?', 'Regime?'];
const QUICK_ASK_FULL = {
  'Anomaly?': 'Was today anomalous or unusual?',
  'vs GCC?':  'How does QSE compare to GCC peers today?',
  'Flows?':   'What are the foreign investor flows today?',
  'Regime?':  'What is the current market regime?',
};

export default function App() {
  const [date, setDate]             = useState(null);
  const [apiDown, setApiDown]       = useState(false);
  const [history, setHistory]       = useState([]);
  const [activeIdx, setActiveIdx]   = useState(null);
  const [todayMetrics, setTodayMetrics] = useState(null);
  const [theme, setTheme]           = useState(
    () => document.documentElement.dataset.theme || 'dark'
  );
  // ref for ChatWindow's send function
  const [pendingSend, setPendingSend] = useState(null);

  useEffect(() => {
    fetch('http://localhost:8000/features/today')
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(data => {
        if (data?.date) setDate(String(data.date).slice(0, 10));
        setTodayMetrics(data ?? null);
      })
      .catch(() => setApiDown(true));
  }, []);

  function toggleTheme() {
    const next = theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    localStorage.setItem('qse-theme', next);
    setTheme(next);
  }

  function handleNewMessage({ question, date: msgDate, messages }) {
    setHistory(prev => {
      const next = [
        { question, date: msgDate, messages, ts: Date.now() },
        ...prev.filter(h => h.question !== question || h.date !== msgDate),
      ].slice(0, 20);
      // new/updated item always lands at index 0
      setActiveIdx(0);
      return next;
    });
  }

  // Separate today vs earlier queries
  const todayStr = date;
  const todayItems = history.filter(h => h.date === todayStr);
  const earlierItems = history.filter(h => h.date !== todayStr);

  return (
    <div className="app">
      <header className="app-header">
        <img src="/qse_logo.png" className="header-logo" alt="QSE" />
        <div className="header-divider" />
        <span className="header-title">Market Copilot</span>
        <RegimeBadge />
        <div className="header-spacer" />
        <button className="theme-toggle" onClick={toggleTheme}>
          {theme === 'dark' ? '☀ Light' : '🌙 Dark'}
        </button>
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

      {apiDown && (
        <div className="api-down-banner">
          API unavailable — start the backend: <code>uvicorn api.main:app --port 8000</code>
        </div>
      )}

      <div className="app-body">
        <aside className="sidebar">
          <div className="sidebar-section-label">Today</div>
          <div className="sidebar-list">
            {todayItems.length === 0 && (
              <div className="sidebar-empty">No queries today</div>
            )}
            {todayItems.map((item, i) => {
              const realIdx = history.indexOf(item);
              return (
                <div
                  key={i}
                  className={`sidebar-item ${activeIdx === realIdx ? 'active' : ''}`}
                  onClick={() => setActiveIdx(realIdx)}
                >
                  <div className="sidebar-item-question">{item.question}</div>
                  <div className="sidebar-item-date">
                    {new Date(item.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    {' · '}{item.date}
                  </div>
                </div>
              );
            })}
          </div>

          {earlierItems.length > 0 && (
            <>
              <div className="sidebar-section-label sidebar-section-label--sep">Earlier</div>
              <div className="sidebar-list sidebar-list--earlier">
                {earlierItems.map((item, i) => {
                  const realIdx = history.indexOf(item);
                  return (
                    <div
                      key={i}
                      className={`sidebar-item ${activeIdx === realIdx ? 'active' : ''}`}
                      onClick={() => setActiveIdx(realIdx)}
                    >
                      <div className="sidebar-item-question">{item.question}</div>
                      <div className="sidebar-item-date">{item.date}</div>
                    </div>
                  );
                })}
              </div>
            </>
          )}

          <div className="sidebar-quickask">
            <div className="sidebar-quickask-label">Quick Ask</div>
            <div className="sidebar-quickask-chips">
              {QUICK_ASK.map(q => (
                <button
                  key={q}
                  className="sidebar-chip"
                  onClick={() => setPendingSend(QUICK_ASK_FULL[q])}
                  disabled={!date}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        </aside>

        <main className="main-area">
          <ChatWindow
            date={date}
            disabled={date === null}
            onNewMessage={handleNewMessage}
            activeHistoryItem={activeIdx != null ? history[activeIdx] : null}
            todayMetrics={todayMetrics}
            pendingSend={pendingSend}
            onPendingSendConsumed={() => setPendingSend(null)}
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
