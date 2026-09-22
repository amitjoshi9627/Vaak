import type { AnalysisResult } from '../../types/prediction';

interface VerdictPanelProps {
  result: AnalysisResult | null;
  status: 'idle' | 'scanning' | 'done' | 'error';
}

/**
 * VerdictPanel — The stamped global verdict and supplementary metrics.
 */
export function VerdictPanel({ result, status }: VerdictPanelProps) {
  if (status !== 'done' || !result) {
    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1.5fr', gap: 'var(--space-6)', padding: '0 var(--space-6)', opacity: 0.3 }}>
        <div style={{ border: '1px solid var(--grid-line)', height: 160, background: 'var(--bg-panel)' }} />
        <div style={{ border: '1px solid var(--grid-line)', height: 160, background: 'var(--bg-panel)' }} />
        <div style={{ border: '1px solid var(--grid-line)', height: 160, background: 'var(--bg-panel)' }} />
      </div>
    );
  }

  const isSynthetic = result.verdict === 'SYNTHETIC';
  const stampColor = isSynthetic ? 'var(--forensic-red)' : 'var(--ink-primary)';

  // Calculate mock log-likelihood based on confidence for display purposes
  const likelihood = isSynthetic ? (result.confidence / 5).toFixed(2) : ((100 - result.confidence) / -5).toFixed(2);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1.5fr', padding: '0 var(--space-6)', borderTop: '1px solid var(--grid-line)' }}>

      {/* 03 / GLOBAL VERDICT */}
      <div style={{ padding: 'var(--space-4)', borderRight: '1px solid var(--grid-line)', display: 'flex', flexDirection: 'column' }}>
        <p style={{ fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '0.1em', color: 'var(--ink-secondary)', marginBottom: 'var(--space-4)' }}>
          03 / GLOBAL VERDICT
        </p>

        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          {/* THE STAMP */}
          <div
            style={{
              border: `3px solid ${stampColor}`,
              padding: '8px 16px',
              animation: 'stamp-verdict 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) both',
              transform: 'rotate(-2deg)'
            }}
          >
            <p style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: stampColor, letterSpacing: '0.2em' }}>
              FORENSIC RESULT
            </p>
            <h1 style={{ fontFamily: 'var(--font-serif)', fontSize: 36, color: stampColor, lineHeight: 1, margin: '4px 0' }}>
              {result.verdict}
            </h1>
            <p style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: stampColor, letterSpacing: '0.1em' }}>
              {result.confidence}% CONFIDENCE
            </p>
          </div>
        </div>
      </div>

      {/* LOG-LIKELIHOOD RATIO */}
      <div style={{ padding: 'var(--space-4)', borderRight: '1px solid var(--grid-line)' }}>
        <p style={{ fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '0.1em', color: 'var(--ink-secondary)', marginBottom: 'var(--space-4)' }}>
          LOG-LIKELIHOOD RATIO
        </p>
        <div style={{ marginTop: 'var(--space-6)' }}>
          <h2 style={{ fontFamily: 'var(--font-mono)', fontSize: 32, fontWeight: 400 }}>
            {Number(likelihood) > 0 ? '+' : ''}{likelihood}
          </h2>

          {/* Gauge Line */}
          <div style={{ marginTop: 24, position: 'relative', height: 2, background: 'var(--grid-line-strong)', display: 'flex' }}>
            <div style={{ flex: 1, background: isSynthetic ? 'var(--grid-line-strong)' : 'var(--ink-primary)' }} />
            <div style={{ flex: 1, background: isSynthetic ? 'var(--forensic-red)' : 'var(--grid-line-strong)' }} />

            {/* Indicator */}
            <div
              style={{
                position: 'absolute',
                top: -4,
                width: 4,
                height: 10,
                background: 'var(--ink-primary)',
                left: isSynthetic ? '75%' : '25%',
                transform: 'translateX(-50%)',
                transition: 'left 0.5s ease-out'
              }}
            />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 8 }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--ink-secondary)' }}>-20<br/>HUMAN</span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--forensic-red)', textAlign: 'right' }}>+20<br/>SYNTHETIC</span>
          </div>
        </div>
      </div>

      {/* DETECTED ANOMALIES */}
      <div style={{ padding: 'var(--space-4)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 'var(--space-4)' }}>
          <p style={{ fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '0.1em', color: 'var(--ink-secondary)' }}>
            DETECTED ANOMALIES
          </p>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10 }}>3 / 7</span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 16, marginTop: 'var(--space-6)' }}>
          {[
            { label: 'Phase incoherence', val: 87, sub: '+4.8 sigma' },
            { label: 'F0 pitch flattening', val: 71, sub: '<2.1 Hz var.' },
            { label: 'HF codec cutoff', val: 94, sub: 'at 15.1 kHz' },
          ].map(anomaly => (
            <div key={anomaly.label} style={{ display: 'flex', alignItems: 'center' }}>
              <div style={{ width: 140 }}>
                <p style={{ fontFamily: 'var(--font-sans)', fontSize: 11, fontWeight: 500 }}>{anomaly.label}</p>
                <p style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--ink-muted)' }}>{anomaly.sub}</p>
              </div>
              <div style={{ flex: 1, height: 2, background: 'var(--grid-line)' }}>
                <div style={{ height: '100%', width: `${anomaly.val}%`, background: isSynthetic ? 'var(--forensic-red)' : 'var(--ink-primary)' }} />
              </div>
              <div style={{ width: 40, textAlign: 'right' }}>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, fontWeight: 600 }}>{anomaly.val}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}
