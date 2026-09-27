import React, { useState, useRef, useEffect, useCallback } from 'react'

const API_BASE = 'http://localhost:8000'

const LANGUAGES = [
  { code: 'en', label: 'English' },
  { code: 'hi', label: 'हिंदी' },
  { code: 'bn', label: 'বাংলা' },
  { code: 'ur', label: 'اردو' },
  { code: 'pa', label: 'ਪੰਜਾਬੀ' },
  { code: 'mr', label: 'मराठी' },
]

/* ─── Scroll-reveal hook ─── */
function useReveal() {
  const ref = useRef(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const observer = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) { el.classList.add('visible'); observer.unobserve(el) } },
      { threshold: 0.12 }
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [])
  return ref
}

function Reveal({ children, className = '', delay = 0 }) {
  const ref = useReveal()
  const delayClass = delay ? `reveal-delay-${delay}` : ''
  return <div ref={ref} className={`reveal ${delayClass} ${className}`}>{children}</div>
}

/* ─── RiskSeal (logic unchanged) ─── */
function RiskSeal({ riskScore, label }) {
  const isDanger = label === 'spam'
  const color = isDanger ? '#e11d48' : '#059669'
  const stamp = isDanger ? 'FLAGGED' : 'VERIFIED SAFE'

  return (
    <div className="seal-mini" style={{ '--seal-color': color }}>
      <svg viewBox="0 0 200 200" className="seal-svg">
        <circle cx="100" cy="100" r="92" className="seal-ring-outer" />
        <circle cx="100" cy="100" r="78" className="seal-ring-inner" />
        <path
          id="sealTextPath"
          d="M 100,100 m -78,0 a 78,78 0 1,1 156,0 a 78,78 0 1,1 -156,0"
          fill="none"
        />
        <text className="seal-stamp-text">
          <textPath href="#sealTextPath" startOffset="2%">
            {stamp} • {stamp} •
          </textPath>
        </text>
      </svg>
      <div className="seal-center">
        <span className="seal-score">{riskScore}</span>
        <span className="seal-score-label">risk / 100</span>
      </div>
    </div>
  )
}

/* ─── SignalChips (logic unchanged) ─── */
function SignalChips({ signals }) {
  if (!signals || signals.length === 0) {
    return <p className="no-signals">No fraud indicators detected in this message.</p>
  }
  return (
    <div className="chip-row">
      {signals.map((s, i) => (
        <span className="chip" key={i}>
          <span className="chip-icon">{s.icon}</span> {s.tag}
        </span>
      ))}
    </div>
  )
}

/* ─── ScanButton (logic unchanged) ─── */
function ScanButton({ loading, disabled, onClick, label }) {
  return (
    <button
      className={`scan-btn ${loading ? 'scanning' : ''}`}
      onClick={onClick}
      disabled={disabled}
    >
      {loading ? (
        <span className="scan-dots">
          Scanning<span>.</span><span>.</span><span>.</span>
        </span>
      ) : (
        <>
          {label} <span className="btn-arrow">→</span>
        </>
      )}
    </button>
  )
}

/* ─── Main App (all state, handlers, API calls UNCHANGED) ─── */
export default function App() {
  const [mode, setMode] = useState('single')
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [history, setHistory] = useState([])

  const [batchText, setBatchText] = useState('')
  const [batchResults, setBatchResults] = useState([])

  const scanCardRef = useRef(null)

  async function scanMessage() {
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })
      if (!res.ok) throw new Error('Scan failed — check that the backend is running.')
      const data = await res.json()
      setResult(data)
      setHistory((h) => [data, ...h].slice(0, 8))
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function scanBatch() {
    const lines = batchText.split('\n').map((l) => l.trim()).filter(Boolean)
    if (lines.length === 0) return
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/predict_batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: lines }),
      })
      if (!res.ok) throw new Error('Batch scan failed — check that the backend is running.')
      const data = await res.json()
      setBatchResults(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const flaggedCount = batchResults.filter((r) => r.label === 'spam').length
  const safeCount = batchResults.length - flaggedCount

  const scrollToScan = useCallback(() => {
    setMode('single')
    setTimeout(() => {
      scanCardRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }, 50)
  }, [])

  function switchToBatch() {
    setMode('batch')
  }

  function getVerdictClass(r) {
    if (r.label !== 'spam') return 'safe'
    return r.risk_score >= 70 ? 'danger' : 'warning'
  }

  function getVerdictText(r) {
    if (r.label !== 'spam') return 'SAFE'
    return r.risk_score >= 70 ? 'HIGH' : 'MEDIUM'
  }

  return (
    <div className="app">

      {/* ═══════ NAVBAR ═══════ */}
      <nav className="navbar">
        <div className="navbar-inner">
          <div className="nav-brand">
            <div className="nav-logo">🛡️</div>
            <span className="nav-wordmark">SMS Suraksha</span>
          </div>

          <ul className="nav-links">
            <li>
              <button className={`nav-link ${mode === 'single' ? 'active' : ''}`} onClick={() => { setMode('single'); window.scrollTo({ top: 0, behavior: 'smooth' }) }}>
                Home
              </button>
            </li>
            <li>
              <button className="nav-link" onClick={scrollToScan}>
                Live Demo
              </button>
            </li>
            <li>
              <button className="nav-link" onClick={() => document.getElementById('how-section')?.scrollIntoView({ behavior: 'smooth' })}>
                How it Works
              </button>
            </li>
            <li>
              <button className="nav-link" onClick={() => document.getElementById('languages-section')?.scrollIntoView({ behavior: 'smooth' })}>
                Languages
              </button>
            </li>
          </ul>

          <div className="nav-actions">
            <button className="btn-nav-ghost" onClick={scrollToScan}>
              Documentation
            </button>
            <button className="btn-nav-solid" onClick={scrollToScan}>
              Scan a Message
            </button>
          </div>
        </div>
      </nav>

      {/* ═══════ HERO ═══════ */}
      <section className="hero">
        <div className="ambient-glow glow-hero" />
        <div className="hero-content">
          <Reveal>
            <div className="hero-badge">
              <span className="hero-badge-index">01</span>
              <span className="hero-badge-dot">◆</span>
              sms suraksha · multilingual fraud detection
            </div>
          </Reveal>

          <Reveal>
            <h1>Catch the fraud message<br />before it catches someone.</h1>
          </Reveal>

          <Reveal>
            <p className="hero-sub">
              AI-powered SMS scanning across English, Hindi, Bengali, Urdu, Punjabi, and Marathi — instant, explainable verdicts powered by MuRIL and TF-IDF.
            </p>
          </Reveal>

          <Reveal>
            <div className="hero-ctas">
              <button className="btn-primary" onClick={scrollToScan}>
                Scan a Message <span className="btn-arrow">→</span>
              </button>
              <button className="btn-secondary" onClick={switchToBatch}>
                Try Batch Demo
              </button>
            </div>
          </Reveal>

          <Reveal>
            <div className="marquee-section">
              <span className="marquee-label">Built with</span>
              <div className="marquee-strip">
                <span className="marquee-pill">FastAPI</span>
                <span className="marquee-pill">React</span>
                <span className="marquee-pill">MuRIL</span>
                <span className="marquee-pill">TF-IDF</span>
                <span className="marquee-pill">LIME</span>
                <span className="marquee-pill">Vite</span>
              </div>
            </div>
          </Reveal>
        </div>
      </section>

      {/* Hidden mode tabs for a11y */}
      <nav className="mode-tabs" aria-hidden="true">
        <button id="tab-single" className={mode === 'single' ? 'tab active' : 'tab'} onClick={() => setMode('single')}>Scan Message</button>
        <button id="tab-batch" className={mode === 'batch' ? 'tab active' : 'tab'} onClick={() => setMode('batch')}>Batch Scan</button>
      </nav>

      {/* ═══════ TAB SELECTOR RAIL ═══════ */}
      <div className="tab-rail-section">
        <div className="tab-rail">
          <button className={`tab-pill ${mode === 'single' ? 'active' : ''}`} onClick={() => setMode('single')}>
            <span className="tab-pip" /> Single Scan
          </button>
          <button className={`tab-pill ${mode === 'batch' ? 'active' : ''}`} onClick={() => setMode('batch')}>
            <span className="tab-pip" /> Batch Scan
          </button>
        </div>
      </div>

      {/* ═══════ SINGLE SCAN MODE ═══════ */}
      {mode === 'single' && (
        <>
          {/* ── How it Works ── */}
          <section className="how-section" id="how-section">
            <div className="how-content">
              <Reveal>
                <div className="section-header">
                  <div className="section-header-left">
                    <div className="section-badge">
                      <span className="section-badge-num">02</span>
                      How it works
                    </div>
                    <h2>Three-step<br />fraud detection</h2>
                  </div>
                  <div className="section-header-right">
                    <p>Paste any SMS, let our AI ensemble analyze it, and get an instant verdict with explainable fraud signals.</p>
                  </div>
                </div>
              </Reveal>
              <hr className="section-divider" />
              <Reveal>
                <div className="how-grid">
                  <div className="how-card">
                    <div className="how-card-step">Step 01</div>
                    <span className="how-card-icon">📱</span>
                    <h3>Paste a Message</h3>
                    <p>Copy any suspicious SMS you've received and paste it into the scanner below.</p>
                  </div>
                  <div className="how-card">
                    <div className="how-card-step">Step 02</div>
                    <span className="how-card-icon">🧠</span>
                    <h3>AI Analysis</h3>
                    <p>Our MuRIL + TF-IDF ensemble model analyzes the text for fraud patterns across 6 languages.</p>
                  </div>
                  <div className="how-card">
                    <div className="how-card-step">Step 03</div>
                    <span className="how-card-icon">✅</span>
                    <h3>Get a Verdict</h3>
                    <p>Receive an instant, explainable verdict with confidence scores and fraud signal indicators.</p>
                  </div>
                </div>
              </Reveal>
            </div>
          </section>

          {/* ── Languages ── */}
          <section className="languages-section" id="languages-section">
            <div className="lang-content">
              <Reveal>
                <div className="section-header">
                  <div className="section-header-left">
                    <div className="section-badge">
                      <span className="section-badge-num">03</span>
                      Language support
                    </div>
                    <h2>Six languages,<br />one scanner</h2>
                  </div>
                  <div className="section-header-right">
                    <p>Native script detection for India's most spoken languages — no translation step required.</p>
                  </div>
                </div>
              </Reveal>
              <hr className="section-divider" />
              <Reveal>
                <div className="lang-grid">
                  {LANGUAGES.map((l) => (
                    <div className="lang-card" key={l.code}>
                      <span className="lang-card-code">{l.code}</span>
                      <span>{l.label}</span>
                    </div>
                  ))}
                </div>
              </Reveal>
            </div>
          </section>

          {/* ── LIVE SCAN CARD ── */}
          <section className="scan-card-section" ref={scanCardRef}>
            <div className="ambient-glow glow-scan" />
            <div className="scan-card">
              <div className="scan-card-beacon"><div className="beacon" /></div>
              <div className="scan-card-badge">
                <span className="scan-card-badge-dot">◆</span>
                sms suraksha · live scan
              </div>
              <h2>Scan a Message</h2>
              <p className="scan-card-sub">Paste any SMS and get an instant, explainable verdict.</p>
              <label htmlFor="sms-input" style={{ position: 'absolute', width: 1, height: 1, overflow: 'hidden', clip: 'rect(0,0,0,0)' }}>
                Paste an SMS message
              </label>
              <textarea
                id="sms-input"
                className="scan-textarea"
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="e.g. Dear customer, your electricity connection will be disconnected today. Update KYC immediately: bit.ly/xyz123"
                rows={5}
              />
              <ScanButton
                loading={loading}
                disabled={loading || !text.trim()}
                onClick={scanMessage}
                label="Scan message"
              />
              <div className="trust-row">
                <span className="trust-item">⏱ Instant verdict</span>
                <span className="trust-item">🔒 Nothing stored</span>
              </div>
              {error && <p className="error-text">{error}</p>}
            </div>
          </section>

          {/* ── RESULT CARD ── */}
          {result && (
            <section className="result-section">
              <div className="result-card">
                <div className="result-header">
                  <div className="result-avatar">
                    <RiskSeal riskScore={result.risk_score} label={result.label} />
                  </div>
                  <div className="result-info">
                    <p className="result-name">
                      "{result.text ? (result.text.slice(0, 55) + (result.text.length > 55 ? '…' : '')) : 'Scanned message'}"
                    </p>
                    <p className="result-role">
                      {LANGUAGES.find((l) => l.code === result.language_detected)?.label || result.language_detected} · Detected Language
                    </p>
                  </div>
                  <div className="result-metrics">
                    <div className="metric-badge">
                      <span className="metric-value">{(result.confidence * 100).toFixed(0)}%</span>
                      <span className="metric-label">Confidence</span>
                    </div>
                    <div className="metric-badge">
                      <span className="metric-value">{result.risk_score}<small style={{ fontSize: '0.6em', color: '#c0c0c0' }}>/100</small></span>
                      <span className="metric-label">Risk</span>
                    </div>
                  </div>
                </div>

                <div className="result-body">
                  <div className="result-col">
                    <div className="result-col-header">Fraud Signals</div>
                    <SignalChips signals={result.signals} />
                  </div>
                  <div className="result-col">
                    <div className="result-col-header">Model Verdict</div>
                    <div className={`verdict-pill ${getVerdictClass(result)}`}>
                      {result.label === 'spam' ? '⚠' : '✓'} {getVerdictText(result)}
                    </div>
                    <p className="verdict-sentence">
                      {result.label === 'spam'
                        ? 'This message has been flagged as likely fraud or spam. Exercise caution and do not click any links.'
                        : 'This message appears to be legitimate. No fraud indicators were found.'}
                    </p>
                    <span className="model-tag">{result.model_backend}</span>
                  </div>
                </div>
              </div>
            </section>
          )}

          {/* ── HISTORY ── */}
          {history.length > 0 && (
            <section className="history-section">
              <div className="history-card">
                <div className="history-header">Recent Scans</div>
                <ul className="history-list">
                  {history.map((h, i) => (
                    <li key={i} className={h.label === 'spam' ? 'history-item danger' : 'history-item safe'}>
                      <span className="history-badge">{h.label === 'spam' ? '⚠' : '✓'}</span>
                      <span className="history-text">{h.text.slice(0, 70)}{h.text.length > 70 ? '…' : ''}</span>
                      <span className="history-score">{h.risk_score}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </section>
          )}

          {/* ── CTA / Conversion ── */}
          <section className="cta-section">
            <div className="ambient-glow glow-cta" />
            <Reveal>
              <div className="cta-card">
                <div className="section-badge" style={{ justifyContent: 'center' }}>
                  <span className="section-badge-num">→</span>
                  Get started
                </div>
                <h2>Ready to scan<br />your inbox?</h2>
                <p className="cta-card-sub">
                  Try the batch scanner to analyze multiple messages at once, or scan a single suspicious SMS right now.
                </p>
                <div className="cta-actions">
                  <button className="btn-primary" onClick={scrollToScan}>
                    Scan a Message <span className="btn-arrow">→</span>
                  </button>
                  <button className="btn-secondary" onClick={switchToBatch}>
                    Try Batch Demo
                  </button>
                </div>
              </div>
            </Reveal>
          </section>
        </>
      )}

      {/* ═══════ BATCH SCAN MODE ═══════ */}
      {mode === 'batch' && (
        <section className="batch-section" key="batch">
          <div className="batch-input-card">
            <div className="scan-card-beacon"><div className="beacon" /></div>
            <div className="scan-card-badge">
              <span className="scan-card-badge-dot">◆</span>
              sms suraksha · batch scan
            </div>
            <h2>Batch Scan</h2>
            <p className="batch-input-card-sub">
              Paste multiple messages (one per line) — simulates scanning an inbox.
            </p>
            <label htmlFor="batch-input" style={{ position: 'absolute', width: 1, height: 1, overflow: 'hidden', clip: 'rect(0,0,0,0)' }}>
              Paste multiple messages
            </label>
            <textarea
              id="batch-input"
              className="scan-textarea"
              value={batchText}
              onChange={(e) => setBatchText(e.target.value)}
              placeholder={'Hey, are we still meeting at 6?\nCongratulations! You have won ₹50,000. Claim now: bit.ly/win123'}
              rows={6}
            />
            <ScanButton
              loading={loading}
              disabled={loading || !batchText.trim()}
              onClick={scanBatch}
              label="Scan inbox"
            />
            {error && <p className="error-text">{error}</p>}
          </div>

          {batchResults.length > 0 && (
            <div className="batch-results-card">
              <div className="batch-stats-row">
                <div className="summary-stat">
                  <span className="summary-number danger">{flaggedCount}</span>
                  <span className="summary-label">Flagged as fraud</span>
                </div>
                <div className="summary-stat">
                  <span className="summary-number safe-count">{safeCount}</span>
                  <span className="summary-label">Safe messages</span>
                </div>
                <div className="summary-stat">
                  <span className="summary-number">{batchResults.length}</span>
                  <span className="summary-label">Total scanned</span>
                </div>
              </div>
              <table className="batch-table">
                <thead>
                  <tr>
                    <th>Message</th>
                    <th>Language</th>
                    <th>Risk</th>
                    <th>Verdict</th>
                  </tr>
                </thead>
                <tbody>
                  {batchResults.map((r, i) => (
                    <tr key={i} className={r.label === 'spam' ? 'row-danger' : ''}>
                      <td>{r.text.slice(0, 60)}{r.text.length > 60 ? '…' : ''}</td>
                      <td>{r.language_detected}</td>
                      <td>{r.risk_score}</td>
                      <td>
                        <span className={`batch-verdict-pill ${r.label === 'spam' ? 'danger' : 'safe'}`}>
                          {r.label === 'spam' ? '⚠ Fraud' : '✓ Safe'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}

      {/* ═══════ FOOTER ═══════ */}
      <footer className="footer">
        <div className="footer-top">
          <div className="footer-brand-col">
            <div className="footer-logo">
              <div className="footer-logo-mark">🛡️</div>
              <span className="footer-logo-text">SMS Suraksha</span>
            </div>
            <p className="footer-desc">
              AI-powered multilingual fraud SMS detector for Indian languages. Scan messages across English, Hindi, Bengali, Urdu, Punjabi, and Marathi.
            </p>
          </div>
          <div className="footer-links-area">
            <div className="footer-col">
              <h4>Product</h4>
              <ul>
                <li onClick={scrollToScan}>Single Scan</li>
                <li onClick={switchToBatch}>Batch Scan</li>
                <li>API Docs</li>
              </ul>
            </div>
            <div className="footer-col">
              <h4>Stack</h4>
              <ul>
                <li>FastAPI</li>
                <li>MuRIL</li>
                <li>TF-IDF / LIME</li>
              </ul>
            </div>
            <div className="footer-col">
              <h4>Languages</h4>
              <ul>
                <li>English</li>
                <li>Hindi · Bengali</li>
                <li>Urdu · Punjabi · Marathi</li>
              </ul>
            </div>
          </div>
        </div>
        <div className="footer-bottom">
          <span className="footer-legal">
            © 2026 SMS Suraksha · Multi-lingual Fraud SMS Detector Project
          </span>
          <span className="footer-status">
            <span className="status-dot" />
            All systems operational
          </span>
        </div>
      </footer>
    </div>
  )
}
