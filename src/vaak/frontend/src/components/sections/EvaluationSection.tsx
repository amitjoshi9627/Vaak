import { Icon } from '../ui/Icon';

export function EvaluationSection() {
  const coverageData = [
    {
      name: 'Neural Vocoders & Resynthesis',
      models: 'BigVGAN · HiFi-GAN · WaveGlow',
      type: 'Vocoders',
      badgeClass: 'theme-vocoder'
    },
    {
      name: 'Zero-Shot Voice Cloners',
      models: 'ElevenLabs · Tortoise · XTTS v2',
      type: 'Cloning',
      badgeClass: 'theme-cloner'
    },
    {
      name: 'Diffusion & Flow Matching',
      models: 'StyleTTS 2 · Voicebox · F5-TTS',
      type: 'Diffusion',
      badgeClass: 'theme-diffusion'
    },
    {
      name: 'Voice Conversion & Morphing',
      models: 'RVC v2 · FreeVC · SoftVC',
      type: 'Conversion',
      badgeClass: 'theme-conversion'
    },
  ];

  const benchmarks = [
    {
      source: 'Neural Voice Clone (3s prompt)',
      type: 'Synthetic Detected',
      score: '0.96 / 1.0',
      finding: 'Elevated synthetic scores concentrated across sustained voiced segments',
      status: 'detected'
    },
    {
      source: 'Casual Phone Recording (Compressed)',
      type: 'Human Voice',
      score: '0.08 / 1.0',
      finding: 'Scores remain consistently within human distribution despite acoustic compression',
      status: 'clean'
    },
    {
      source: 'Voice Conversion Audio Stream',
      type: 'Synthetic Detected',
      score: '0.89 / 1.0',
      finding: 'Model activation spikes across converted phoneme transitions',
      status: 'detected'
    }
  ];

  return (
    <section className="evaluation-section" id="evaluation">
      {/* Visual Narrative Connector */}
      <div className="section-narrative-cue evaluation-cue" aria-hidden="true">
        <span className="narrative-line" />
        <span className="narrative-node">05 / DETECTOR RELIABILITY</span>
        <span className="narrative-line" />
      </div>

      <div className="section-intro evaluation-intro centered-intro">
        <h2>Benchmarked on real and<br />synthetic speech datasets.</h2>
        <p className="eval-lead-copy">
          Evaluation is an ongoing empirical process. Rather than claiming absolute perfection, Vaak is benchmarked across diverse speech datasets, compression levels, and voice generation architectures to measure genuine generalization.
        </p>
      </div>

      {/* Primary Performance Pillars Grid */}
      <div className="eval-performance-grid">
        <div className="eval-metric-card card-empirical">
          <div className="metric-header">
            <span className="metric-tag">01 / EMPIRICAL</span>
            <span className="metric-pill">Active</span>
          </div>
          <h3 className="eval-card-title">Ongoing Benchmarks</h3>
          <p className="metric-desc">
            Evaluated continuously across diverse human and synthetic speech datasets to evaluate real-world generalization.
          </p>
          <div className="eval-card-footer">
            <span className="eval-card-badge">Diverse Speech Corpora</span>
          </div>
        </div>

        <div className="eval-metric-card card-sensitivity">
          <div className="metric-header">
            <span className="metric-tag">02 / SENSITIVITY</span>
            <span className="metric-pill safe">Strictly Guarded</span>
          </div>
          <h3 className="eval-card-title">Guarded Human Speech</h3>
          <p className="metric-desc">
            Operating thresholds are tuned to protect authentic human voices from false positive accusations, even in compressed audio.
          </p>
          <div className="eval-card-footer">
            <span className="eval-card-badge">Low False Alarm Tolerance</span>
          </div>
        </div>

        <div className="eval-metric-card card-resolution">
          <div className="metric-header">
            <span className="metric-tag">03 / RESOLUTION</span>
            <span className="metric-pill">Segment-Level</span>
          </div>
          <h3 className="eval-card-title">Temporal Windows</h3>
          <p className="metric-desc">
            Speech is evaluated across sliding 40ms frames to surface localized anomalies rather than collapsing audio into one guess.
          </p>
          <div className="eval-card-footer">
            <span className="eval-card-badge">40ms Windows · 20ms Hop</span>
          </div>
        </div>

        <div className="eval-metric-card card-uncertainty">
          <div className="metric-header">
            <span className="metric-tag">04 / UNCERTAINTY</span>
            <span className="metric-pill">Probabilistic</span>
          </div>
          <h3 className="eval-card-title">Visible Uncertainty</h3>
          <p className="metric-desc">
            Model likelihoods are exposed as graded analytical signals, never marketed as 100% infallible proof.
          </p>
          <div className="eval-card-footer">
            <span className="eval-card-badge">0.0 – 1.0 Signal Scale</span>
          </div>
        </div>
      </div>

      {/* Dual Analysis: Score Distribution + Model Coverage */}
      <div className="eval-deepdive-grid">
        {/* Panel A: Score Separation Curve */}
        <div className="eval-panel distribution-panel">
          <div className="panel-topline">
            <h3>Score Distribution</h3>
            <span className="panel-badge">Empirical Calibration</span>
          </div>
          <p className="panel-desc">
            Model response distributions across human and synthetic speech test segments, illustrating threshold separation.
          </p>

          {/* SVG Score Distribution Curves with Gradients & Graph Grid */}
          <div className="distribution-chart-wrap" aria-hidden="true">
            <svg viewBox="0 0 600 230" className="distribution-svg" fill="none">
              <defs>
                <linearGradient id="humanCurveGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#244d3b" stopOpacity="0.22" />
                  <stop offset="100%" stopColor="#244d3b" stopOpacity="0.02" />
                </linearGradient>
                <linearGradient id="synthCurveGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#e65f35" stopOpacity="0.26" />
                  <stop offset="100%" stopColor="#e65f35" stopOpacity="0.02" />
                </linearGradient>
                <pattern id="chartGrid" width="40" height="24" patternUnits="userSpaceOnUse">
                  <path d="M 40 0 L 0 0 0 24" fill="none" stroke="rgba(23,32,30,0.035)" strokeWidth="1" />
                </pattern>
              </defs>

              {/* Technical Chart Background Grid */}
              <rect x="30" y="20" width="550" height="160" fill="url(#chartGrid)" rx="3" />

              {/* Threshold Divider */}
              <line x1="300" y1="36" x2="300" y2="180" stroke="#b07a3a" strokeDasharray="4 4" strokeWidth="1.5" />
              <rect x="195" y="8" width="210" height="24" rx="3" fill="#ffffff" stroke="rgba(176,122,58,0.35)" />
              <text x="300" y="24" textAnchor="middle" fill="#844712" fontSize="9.5" fontWeight="600" letterSpacing="0.08em">Decision Threshold (0.50)</text>

              {/* Human Speech Distribution (Left Curve) */}
              <path
                d="M 40 180 C 100 180, 120 45, 160 45 C 200 45, 220 180, 280 180 Z"
                fill="url(#humanCurveGrad)"
              />
              <path
                d="M 40 180 C 100 180, 120 45, 160 45 C 200 45, 220 180, 280 180"
                stroke="#244d3b"
                strokeWidth="2.5"
                fill="none"
              />

              {/* Synthetic Speech Distribution (Right Curve) */}
              <path
                d="M 320 180 C 380 180, 400 40, 450 40 C 500 40, 520 180, 580 180 Z"
                fill="url(#synthCurveGrad)"
              />
              <path
                d="M 320 180 C 380 180, 400 40, 450 40 C 500 40, 520 180, 580 180"
                stroke="#e65f35"
                strokeWidth="2.5"
                fill="none"
              />

              {/* Axis Line */}
              <line x1="30" y1="180" x2="590" y2="180" stroke="rgba(23,32,30,0.22)" strokeWidth="1" />

              {/* Axis Labels */}
              <text x="40" y="202" fill="#244d3b" fontSize="10.5" fontWeight="600">0.0 · Human Distribution Baseline</text>
              <text x="580" y="202" textAnchor="end" fill="#c2471f" fontSize="10.5" fontWeight="600">1.0 · Synthetic Likelihood Threshold</text>
            </svg>

            <div className="chart-legend-row">
              <div className="legend-item">
                <span className="legend-swatch human" />
                <span>Human Speech Profile (Clustered &lt; 0.20)</span>
              </div>
              <div className="legend-item">
                <span className="legend-swatch synthetic" />
                <span>Synthetic Speech Profile (Clustered &gt; 0.80)</span>
              </div>
            </div>
          </div>
        </div>

        {/* Panel B: Speech Generator Coverage Matrix */}
        <div className="eval-panel coverage-panel">
          <div className="panel-topline">
            <h3>Synthetic Voice Coverage</h3>
            <span className="panel-badge">Evaluated Families</span>
          </div>
          <p className="panel-desc">
            Modern speech synthesis architectures and generative model families evaluated in our benchmark suite.
          </p>

          <div className="coverage-list">
            {coverageData.map(item => (
              <div key={item.name} className="coverage-row">
                <div className="coverage-info">
                  <div className="coverage-name-row">
                    <strong>{item.name}</strong>
                    <span className={`coverage-tag ${item.badgeClass}`}>{item.type}</span>
                  </div>
                  <span className="coverage-models">{item.models}</span>
                </div>
                <span className="coverage-badge">
                  <span className="coverage-dot" /> Evaluated
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Representative Evaluation Results */}
      <div className="eval-benchmarks-shelf">
        <div className="benchmarks-header">
          <div>
            <h3>Observed Behavior on Representative Audio</h3>
            <span className="benchmarks-sub">Segment analysis across varied recording conditions and compression formats</span>
          </div>
          <span className="benchmarks-counter">3 Reference Cases</span>
        </div>

        <div className="benchmarks-grid">
          {benchmarks.map((bench, idx) => (
            <div key={idx} className={`benchmark-card bench-${bench.status}`}>
              <div className="bench-top">
                <span className={`bench-verdict ${bench.status}`}>
                  {bench.status === 'detected' ? (
                    <><i className="alert-dot" /> {bench.type}</>
                  ) : (
                    <><Icon name="check" /> {bench.type}</>
                  )}
                </span>
                <span className="bench-score-pill">Score: {bench.score}</span>
              </div>
              <p className="bench-source">{bench.source}</p>
              <p className="bench-finding">{bench.finding}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
