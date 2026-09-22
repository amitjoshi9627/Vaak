import { useState } from 'react';
import { Icon } from '../ui/Icon';

interface EvidenceArtifact {
  id: string;
  timeRange: string;
  title: string;
  type: 'high' | 'moderate';
  category: string;
  description: string;
  spectrogramHint: string;
}

export function EvidenceSection() {
  const [activeSpecimen, setActiveSpecimen] = useState<'synthetic' | 'bonafide'>('synthetic');
  const [selectedArtifactId, setSelectedArtifactId] = useState<string>('art-1');

  const artifacts: EvidenceArtifact[] = [
    {
      id: 'art-1',
      timeRange: '01.40s – 02.15s',
      title: 'Elevated Synthetic Likelihood',
      type: 'high',
      category: 'Segment Score: 0.94',
      description: 'The model assigned a higher synthetic score to this segment than to surrounding audio.',
      spectrogramHint: 'Strongest localized model activation within the test sample.'
    },
    {
      id: 'art-2',
      timeRange: '02.90s – 03.65s',
      title: 'Secondary Signal Cluster',
      type: 'high',
      category: 'Segment Score: 0.88',
      description: 'A recurring cluster of elevated scores observed across the transition into voiced speech.',
      spectrogramHint: 'Consistent synthetic signal across overlapping evaluation windows.'
    },
    {
      id: 'art-3',
      timeRange: '04.10s – 04.85s',
      title: 'Moderate Anomaly Signal',
      type: 'moderate',
      category: 'Segment Score: 0.72',
      description: 'Model response shows moderate deviation from typical human baseline patterns.',
      spectrogramHint: 'Intermediate likelihood value indicating acoustic uncertainty.'
    }
  ];

  const selectedArtifact = artifacts.find(a => a.id === selectedArtifactId) || artifacts[0];

  return (
    <section className="evidence-section" id="evidence">
      {/* Visual Narrative Connector: Stream from Listen Room */}
      <div className="section-narrative-cue evidence-cue" aria-hidden="true">
        <span className="narrative-line" />
        <span className="narrative-node">02 & 03 / EVIDENCE LOCALIZATION</span>
        <span className="narrative-line" />
      </div>

      <div className="section-intro evidence-intro centered-intro">
        <h2>Synthetic speech can reveal<br />patterns across time.</h2>
        <p className="evidence-lead-copy">
          Rather than collapsing a recording into a single score, Vaak evaluates overlapping segments across time to show where synthetic signals are strongest.
        </p>
      </div>

      {/* Forensic Specimen Visualizer */}
      <div className="forensic-specimen-deck">
        <div className="specimen-deck-header">
          <div className="specimen-selector">
            <span className="specimen-label">AUDIO INSPECTOR:</span>
            <button
              className={`specimen-toggle ${activeSpecimen === 'synthetic' ? 'active' : ''}`}
              onClick={() => setActiveSpecimen('synthetic')}
            >
              <i className="status-dot alert" />
              <span>Synthetic Voice (Neural TTS)</span>
            </button>
            <button
              className={`specimen-toggle ${activeSpecimen === 'bonafide' ? 'active' : ''}`}
              onClick={() => setActiveSpecimen('bonafide')}
            >
              <i className="status-dot natural" />
              <span>Human Voice</span>
            </button>
          </div>
          <div className="specimen-meta">
            <span>WINDOW: 40ms / 20ms HOP</span>
            <span className="sep">/</span>
            <span>MODEL: VAAK-CORE 16kHz</span>
          </div>
        </div>

        {/* Timeline Visualization */}
        <div className="specimen-timeline-container">
          <div className="timeline-ruler">
            <span>00:00.0</span>
            <span>00:01.0</span>
            <span>00:02.0</span>
            <span>00:03.0</span>
            <span>00:04.0</span>
            <span>00:05.0</span>
            <span>00:06.0</span>
          </div>

          {/* Acoustic Waveform & Detection Track */}
          <div className="specimen-wave-track">
            {/* Visualized Waveform Slices */}
            <div className="specimen-wave-bars" aria-hidden="true">
              {Array.from({ length: 64 }).map((_, i) => {
                const isHighRisk1 = i >= 14 && i <= 22;
                const isHighRisk2 = i >= 30 && i <= 38;
                const isModRisk = i >= 43 && i <= 51;
                const isSyntheticAnomaly = activeSpecimen === 'synthetic' && (isHighRisk1 || isHighRisk2 || isModRisk);

                const baseHeight = 20 + Math.sin(i * 0.35) * 15 + Math.cos(i * 0.7) * 10;
                const height = Math.max(12, Math.min(84, baseHeight + (isSyntheticAnomaly ? 18 : 0)));

                return (
                  <span
                    key={i}
                    className={`specimen-bar ${isSyntheticAnomaly ? 'anomaly' : ''}`}
                    style={{ height: `${height}%` }}
                  />
                );
              })}
            </div>

            {/* Overlay Evidence Highlight Bands (Visible only on synthetic specimen) */}
            {activeSpecimen === 'synthetic' && (
              <div className="evidence-overlay-bands">
                <button
                  className={`evidence-band band-1 ${selectedArtifactId === 'art-1' ? 'active' : ''}`}
                  onClick={() => setSelectedArtifactId('art-1')}
                  title="01.40s – 02.15s: Elevated Synthetic Likelihood"
                >
                  <span className="band-tag">01.4s · Elevated Synthetic Likelihood</span>
                </button>

                <button
                  className={`evidence-band band-2 ${selectedArtifactId === 'art-2' ? 'active' : ''}`}
                  onClick={() => setSelectedArtifactId('art-2')}
                  title="02.90s – 03.65s: Secondary Signal Cluster"
                >
                  <span className="band-tag">02.9s · Secondary Signal Cluster</span>
                </button>

                <button
                  className={`evidence-band band-3 ${selectedArtifactId === 'art-3' ? 'active' : ''}`}
                  onClick={() => setSelectedArtifactId('art-3')}
                  title="04.10s – 04.85s: Moderate Anomaly Signal"
                >
                  <span className="band-tag">04.1s · Moderate Anomaly Signal</span>
                </button>
              </div>
            )}

            {/* Bonafide Natural Continuity Line (When human sample selected) */}
            {activeSpecimen === 'bonafide' && (
              <div className="bonafide-indicator-strip">
                <span className="bonafide-line" />
                <span className="bonafide-label">
                  <Icon name="check" /> NATURAL TEMPORAL CONTINUITY · NO ELEVATED SYNTHETIC REGIONS
                </span>
              </div>
            )}
          </div>

          {/* Anomaly Detail Card */}
          {activeSpecimen === 'synthetic' ? (
            <div className="artifact-detail-panel">
              <div className="artifact-detail-header">
                <div className="artifact-badge-row">
                  <span className={`risk-badge ${selectedArtifact.type}`}>
                    {selectedArtifact.type === 'high' ? 'High Model Activation' : 'Moderate Likelihood'}
                  </span>
                  <span className="artifact-time">{selectedArtifact.timeRange}</span>
                  <span className="artifact-category">{selectedArtifact.category}</span>
                </div>
                <h3>{selectedArtifact.title}</h3>
              </div>
              <p className="artifact-description">{selectedArtifact.description}</p>
              <div className="artifact-spectrogram-hint">
                <span className="hint-label">Model Observation:</span>
                <span className="hint-text">{selectedArtifact.spectrogramHint}</span>
              </div>
            </div>
          ) : (
            <div className="artifact-detail-panel bonafide-panel">
              <div className="artifact-detail-header">
                <div className="artifact-badge-row">
                  <span className="risk-badge bonafide">Human Voice</span>
                  <span className="artifact-time">00:00.0 – 00:06.0</span>
                  <span className="artifact-category">Human Reference Sample</span>
                </div>
                <h3>Consistent Human Speech Patterns</h3>
              </div>
              <p className="artifact-description">
                Model scores across all overlapping windows remain low and uniform, showing no localized peaks of synthetic likelihood.
              </p>
              <div className="artifact-spectrogram-hint">
                <span className="hint-label">Model Observation:</span>
                <span className="hint-text">Zero segments exceed the synthetic decision threshold across the duration.</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
