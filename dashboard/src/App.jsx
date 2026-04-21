import React, { useEffect, useMemo, useState } from 'react';
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import DecisionGraph from './DecisionGraph';

const API_URL = 'http://localhost:5000/api/state';

function normalizeRecentActions(actions = []) {
  return actions
    .map((action, index) => {
      if (typeof action === 'string') {
        return {
          id: `${index}-${action}`,
          label: action,
          level: 'R2',
          step: index + 1,
        };
      }

      return {
        id: `${index}-${action.action || action.action_id || 'action'}`,
        label: action.action || action.action_id || 'unknown_action',
        level: action.reversibility || action.level || `R${action.r_level ?? action.actual_r_level ?? 2}`,
        step: action.step ?? index + 1,
      };
    })
    .reverse();
}

function normalizeCatastropheSeries(raw = []) {
  if (!Array.isArray(raw)) {
    return [];
  }
  return raw.map((point, index) => {
    if (typeof point === 'number') {
      return { step: index + 1, catastrophe_rate: point };
    }
    if (typeof point === 'object' && point !== null) {
      return {
        step: point.step ?? index + 1,
        catastrophe_rate: point.catastrophe_rate ?? point.value ?? 0,
      };
    }
    return { step: index + 1, catastrophe_rate: 0 };
  });
}

function normalizeLockedActions(rawLockedActions = {}) {
  if (Array.isArray(rawLockedActions)) {
    return Object.fromEntries(rawLockedActions.map((actionId) => [actionId, 'Locked by prior irreversible action']));
  }

  if (rawLockedActions && typeof rawLockedActions === 'object') {
    return rawLockedActions;
  }

  return {};
}

function normalizeThinking(rawThinking) {
  if (Array.isArray(rawThinking)) {
    return rawThinking.map((entry) => String(entry)).filter(Boolean);
  }

  if (typeof rawThinking === 'string') {
    return rawThinking
      .split(/\r?\n+/)
      .map((line) => line.trim())
      .filter(Boolean);
  }

  if (rawThinking && typeof rawThinking === 'object') {
    const values = Object.values(rawThinking)
      .flatMap((value) => (Array.isArray(value) ? value : [value]))
      .map((value) => String(value).trim())
      .filter(Boolean);
    return values;
  }

  return [];
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function TrustGauge({ catastropheSeries, lockedCount, recentThinking }) {
  const latestCatastrophe = catastropheSeries.length ? catastropheSeries[catastropheSeries.length - 1].catastrophe_rate : 0;
  const trustValue = clamp(Math.round(100 - latestCatastrophe * 72 - lockedCount * 1.7), 0, 100);
  const flash = latestCatastrophe > 0.35 || lockedCount > 6;
  const warning = trustValue < 55;

  return (
    <section className={`panel trust-panel ${flash ? 'trust-flash' : ''}`}>
      <div className="card-header trust-header">
        <div>
          <h2>Board Trust</h2>
          <p>Live reputation pressure from catastrophe spikes and action lockout.</p>
        </div>
        <div className={`trust-readout ${warning ? 'warning' : 'stable'}`}>
          <span>{trustValue}</span>
          <small>/ 100</small>
        </div>
      </div>

      <div className="gauge-shell" aria-label="Board Trust gauge">
        <div className="gauge-track">
          <div className="gauge-fill" style={{ width: `${trustValue}%` }} />
        </div>
        <div className="gauge-meta">
          <span>Confidence</span>
          <strong>{flash ? 'ALERT' : warning ? 'UNDER PRESSURE' : 'STABLE'}</strong>
        </div>
      </div>

      <div className="ticker-note">
        <span className="ticker-label">Reasoning signal</span>
        <p>{recentThinking.length ? recentThinking[0] : 'Awaiting raw_thinking from the training loop...'}</p>
      </div>
    </section>
  );
}

function ReasoningTicker({ rawThinkingLines }) {
  return (
    <section className="panel ticker-panel">
      <div className="card-header ticker-header">
        <div>
          <h2>Reasoning Ticker</h2>
          <p>Streaming raw_thinking text from the live training process.</p>
        </div>
        <div className="pulse-chip terminal-chip">LIVE</div>
      </div>

      <div className="terminal-window" role="log" aria-live="polite" aria-label="Reasoning ticker window">
        <div className="terminal-scanline" />
        {rawThinkingLines.length ? (
          rawThinkingLines.map((line, index) => (
            <div className="terminal-line" key={`${index}-${line}`}>
              <span className="terminal-prompt">&gt;</span>
              <span>{line}</span>
            </div>
          ))
        ) : (
          <div className="terminal-line muted">
            <span className="terminal-prompt">&gt;</span>
            <span>Waiting for raw_thinking telemetry...</span>
          </div>
        )}
      </div>
    </section>
  );
}

function FlashRow({ item }) {
  const danger = item.level === 'R4' || item.level === 'R5';
  const className = danger ? 'flash-row danger' : 'flash-row safe';

  return (
    <div className={className}>
      <div className="flash-row-top">
        <span className="flash-step">Step {item.step}</span>
        <span className="flash-level">{item.level}</span>
      </div>
      <div className="flash-label">{item.label}</div>
    </div>
  );
}

export default function App() {
  const [state, setState] = useState({
    recent_actions: [],
    locked_actions: {},
    critical_options: {},
    catastrophe_rate: [],
    raw_thinking: [],
  });
  const [connected, setConnected] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);

  useEffect(() => {
    let mounted = true;

    const fetchState = async () => {
      try {
        const response = await fetch(API_URL, { cache: 'no-store' });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        if (mounted) {
          setState(data);
          setConnected(true);
          setLastUpdated(new Date());
        }
      } catch (error) {
        if (mounted) {
          setConnected(false);
        }
      }
    };

    fetchState();
    const interval = window.setInterval(fetchState, 1000);
    return () => {
      mounted = false;
      window.clearInterval(interval);
    };
  }, []);

  const lockedActions = useMemo(() => normalizeLockedActions(state.locked_actions || {}), [state.locked_actions]);
  const recentActions = useMemo(() => normalizeRecentActions(state.recent_actions || []), [state.recent_actions]);
  const catastropheSeries = useMemo(() => normalizeCatastropheSeries(state.catastrophe_rate || []), [state.catastrophe_rate]);
  const rawThinkingLines = useMemo(() => normalizeThinking(state.raw_thinking || state.thinking || state.reasoning || []), [state.raw_thinking, state.thinking, state.reasoning]);

  const lockedCount = Object.keys(lockedActions).length;
  const criticalCount = Object.values(state.critical_options || {}).filter(Boolean).length;

  return (
    <div className="app-shell">
      <div className="background-orb orb-one" />
      <div className="background-orb orb-two" />

      <header className="hero-bar">
        <div>
          <p className="eyebrow">PermanenceEnv Command Center</p>
          <h1>Live Decision Physics</h1>
          <p className="hero-copy">
            Tracking irreversible choices, option lockout, and catastrophe decay in real time.
          </p>
        </div>
        <div className={`status-pill ${connected ? 'online' : 'offline'}`}>
          <span className="status-dot" />
          {connected ? 'Connected' : 'Offline'}
        </div>
      </header>

      <main className="mission-grid">
        <aside className="left-rail">
          <ReasoningTicker rawThinkingLines={rawThinkingLines} />
          <TrustGauge catastropheSeries={catastropheSeries} lockedCount={lockedCount} recentThinking={rawThinkingLines} />
        </aside>

        <section className="center-rail">
          <DecisionGraph lockedActions={lockedActions} recentActions={recentActions} />

          <section className="panel chart-panel">
            <div className="card-header">
              <div>
                <h2>Catastrophe Rate</h2>
                <p>Desired slope: downward as the policy learns permanence.</p>
              </div>
              <div className="metric-group">
                <div className="metric">
                  <span className="metric-label">Locked</span>
                  <strong>{lockedCount}</strong>
                </div>
                <div className="metric">
                  <span className="metric-label">Critical</span>
                  <strong>{criticalCount}</strong>
                </div>
              </div>
            </div>

            <div className="chart-frame">
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={catastropheSeries}>
                  <defs>
                    <linearGradient id="catastropheStroke" x1="0" y1="0" x2="1" y2="0">
                      <stop offset="0%" stopColor="#ff4d6d" />
                      <stop offset="100%" stopColor="#ffd166" />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="rgba(148, 163, 184, 0.12)" strokeDasharray="4 6" />
                  <XAxis dataKey="step" stroke="#8b97b4" tick={{ fill: '#8b97b4', fontSize: 12 }} />
                  <YAxis stroke="#8b97b4" tick={{ fill: '#8b97b4', fontSize: 12 }} domain={[0, 1]} />
                  <Tooltip
                    contentStyle={{
                      background: 'rgba(8, 12, 22, 0.92)',
                      border: '1px solid rgba(148, 163, 184, 0.2)',
                      borderRadius: '14px',
                      color: '#ecf2ff',
                      boxShadow: '0 20px 40px rgba(0,0,0,0.35)',
                    }}
                    labelStyle={{ color: '#f8fafc' }}
                  />
                  <Line
                    type="monotone"
                    dataKey="catastrophe_rate"
                    stroke="url(#catastropheStroke)"
                    strokeWidth={3}
                    dot={false}
                    activeDot={{ r: 5, stroke: '#ffffff', strokeWidth: 2 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>
        </section>

        <aside className="right-rail">
          <section className="panel feed-panel">
            <div className="card-header">
              <div>
                <h2>Recent Actions</h2>
                <p>Color-coded by predicted reversibility.</p>
              </div>
              <div className="pulse-chip">{recentActions.length} events</div>
            </div>

            <div className="feed-list">
              {recentActions.length ? (
                recentActions.map((item) => <FlashRow item={item} key={item.id} />)
              ) : (
                <div className="empty-state">Waiting for training telemetry...</div>
              )}
            </div>
          </section>

          <section className="panel feed-panel compact">
            <div className="card-header">
              <div>
                <h2>Critical Options</h2>
                <p>Live availability from the current state.</p>
              </div>
            </div>
            <div className="option-list">
              {Object.entries(state.critical_options || {}).map(([name, enabled]) => (
                <div key={name} className={`option-row ${enabled ? 'enabled' : 'disabled'}`}>
                  <span>{name}</span>
                  <strong>{enabled ? 'OPEN' : 'LOCKED'}</strong>
                </div>
              ))}
            </div>
          </section>
        </aside>
      </main>

      <footer className="footer-bar">
        <span>Last update: {lastUpdated ? lastUpdated.toLocaleTimeString() : 'never'}</span>
        <span>API: {API_URL}</span>
      </footer>
    </div>
  );
}
