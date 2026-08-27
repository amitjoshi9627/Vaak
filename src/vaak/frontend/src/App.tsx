import React, { useState } from 'react';
import './App.css';

interface PredictionResult {
  is_spoof: boolean;
  spoof_probability: number;
  chunk_probabilities: number[];
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      setFile(event.target.files[0]);
      setResult(null);
      setError(null);
    }
  };

  const handleReset = () => {
    setFile(null);
    setResult(null);
    setError(null);
  };

  const analyzeAudio = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.statusText}`);
      }

      const data: PredictionResult = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="layout">
      {/* Top Navigation */}
      <nav className="topbar">
        <div className="brand">VAAK [वाक्]</div>
        <div className="nav-links">
          <a href="#" className="nav-link">INSPECT</a>
          <a href="#" className="nav-link active">MIRROR TEST</a>
          <a href="#" className="nav-link">ARCHIVE</a>
        </div>
        <div className="status-indicator">
          <div className="dot"></div>
          CORE 02 / ONLINE
        </div>
      </nav>

      {/* Status Bar for Errors/Loading */}
      {error && <div className="status-bar error">SYSTEM ERROR: {error}</div>}
      {loading && <div className="status-bar">PROCESSING: EXECUTING ARTIFACT ANALYSIS...</div>}

      {/* Hero Section */}
      <header className="hero">
        <div className="hero-text-block">
          <div className="eyebrow">
            INSTRUMENT 02 <span>/</span> PARALLEL LISTENING
          </div>
          <h1>The voice,<br />against itself.</h1>
        </div>
        <div className="hero-action-block">
          <p className="hero-description">
            Compare a questioned recording with its acoustic counterfactual.
            Listen where the room ends and synthesis begins.
          </p>

          {/* File Upload (Hidden when results are shown) */}
          {!result && (
            <div className="file-upload-wrapper">
              <input
                type="file"
                accept=".wav,.flac"
                onChange={handleFileChange}
                className="file-input"
                disabled={loading}
              />
              <button className="load-btn" disabled={loading}>
                {file ? `STAGED: ${file.name.toUpperCase()}` : 'LOAD NEW RECORD ↗'}
              </button>
            </div>
          )}

          {/* Action Buttons */}
          <div className="action-group">
            {file && !result && (
              <button
                className="brutalist-btn"
                onClick={analyzeAudio}
                disabled={loading}
              >
                {loading ? 'EXECUTING...' : 'EXECUTE ANALYSIS →'}
              </button>
            )}

            {result && (
              <button
                className="brutalist-btn outline"
                onClick={handleReset}
              >
                ↺ INITIALIZE NEW SESSION
              </button>
            )}
          </div>

        </div>
      </header>

      {/* Analysis Results Grid */}
      {result && (
        <main className="analysis-grid">

          {/* Left: Finding Statement */}
          <div className="finding-block">
            <div className="finding-number">
              <span>FINDING</span>
              01
            </div>
            <div className="finding-label">ACOUSTIC DISTANCE</div>
            <h2 className="finding-statement">
              {result.is_spoof
                ? "The speaker is human. The passage is not."
                : "Acoustic continuity preserved. Synthesis undetected."}
            </h2>
            <p className="finding-desc">
              {result.is_spoof
                ? "Room tone and breath cadence diverge significantly from natural human vocal projection models."
                : "Vocal tract modeling, room tone, and breath cadence remain continuous throughout the duration."}
            </p>
          </div>

          {/* Center: Global Score */}
          <div className="score-block">
            <div className="score-value">
              {(result.spoof_probability * 100).toFixed(0)}<span>%</span>
            </div>
            <div className="score-label">Synthetic Probability</div>
          </div>

          {/* Right: Temporal Breakdown */}
          <div className="temporal-block">
            {result.chunk_probabilities.map((prob, idx) => {
              const start = (idx * 2).toFixed(2).padStart(5, '0');
              const end = (idx * 2 + 4).toFixed(2).padStart(5, '0');
              const isHighRisk = prob > 0.5;

              return (
                <div key={idx} className={`temporal-row ${isHighRisk ? 'danger' : ''}`}>
                  <div>
                    <div className="temporal-label">
                      {isHighRisk ? 'ALTERED' : 'CONSISTENT'}
                    </div>
                    <div>{start}s &mdash; {end}s</div>
                  </div>
                  <div className="temporal-value">
                    {prob.toFixed(2)}
                  </div>
                </div>
              );
            })}
          </div>

        </main>
      )}
    </div>
  );
}