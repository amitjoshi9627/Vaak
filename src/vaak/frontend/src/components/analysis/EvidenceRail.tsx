import { useEffect, useState } from 'react';
import type { AnalysisResult } from '../../types/prediction';

interface EvidenceRailProps {
  result: AnalysisResult | null;
  status: 'idle' | 'scanning' | 'done' | 'error';
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

/**
 * EvidenceRail — The core temporal analysis view.
 * A strict two-track timeline (Amplitude + Synthetic Likelihood)
 */
export function EvidenceRail({ result, status }: EvidenceRailProps) {
  const [playheadPos, setPlayheadPos] = useState(0);

  // Animate a playhead scanning across if in scanning mode
  useEffect(() => {
    if (status === 'scanning') {
      const interval = setInterval(() => {
        setPlayheadPos((p) => (p >= 100 ? 0 : p + 0.5));
      }, 50);
      return () => clearInterval(interval);
    }
    if (status === 'done') setPlayheadPos(100);
  }, [status]);

  const numChunks = result ? result.chunks.length : 60; // default 60 empty slots
  
  // Generate dummy waveform data for visual effect (deterministic based on index)
  const getWaveHeight = (i: number) => {
    if (!result && status !== 'scanning') return 2; // flatline
    const seed = Math.sin(i * 13.37) * 10000;
    return Math.floor(Math.abs(seed - Math.floor(seed)) * 40) + 10;
  };

  return (
    <div style={{ padding: 'var(--space-6)', flex: 1, display: 'flex', flexDirection: 'column' }}>
      
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: 'var(--space-6)' }}>
        <div>
          <p style={{ fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '0.1em', color: 'var(--ink-secondary)', marginBottom: 8 }}>
            02 / TEMPORAL ANALYSIS
          </p>
          <h2 style={{ fontFamily: 'var(--font-serif)', fontSize: 28 }}>
            Acoustic evidence rail
          </h2>
        </div>
        
        {status === 'done' && result && (
          <div style={{ textAlign: 'right' }}>
            <p style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--ink-secondary)' }}>MODEL CONFIDENCE</p>
            <p style={{ fontFamily: 'var(--font-mono)', fontSize: 24, fontWeight: 700 }}>
              {result.confidence}%
            </p>
          </div>
        )}
      </div>

      {/* The Rail Container */}
      <div 
        style={{ 
          border: '1px solid var(--grid-line-strong)', 
          background: '#FFF', 
          position: 'relative',
          display: 'flex',
          flexDirection: 'column'
        }}
      >
        
        {/* Timeline Axis */}
        <div style={{ display: 'flex', borderBottom: '1px solid var(--grid-line)', padding: '4px 0', paddingLeft: 60 }}>
          {[0, 1, 2, 3, 4, 5].map(m => (
            <div key={m} style={{ flex: 1, borderLeft: '1px solid var(--grid-line)', paddingLeft: 4, height: 16 }}>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--ink-muted)' }}>
                {String(m).padStart(2, '0')}:00
              </span>
            </div>
          ))}
        </div>

        {/* Track 1: Amplitude */}
        <div style={{ display: 'flex', height: 120, borderBottom: '1px solid var(--grid-line)' }}>
          <div style={{ width: 60, borderRight: '1px solid var(--grid-line)', padding: 8 }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--ink-muted)' }}>01<br/>AMPLITUDE</span>
          </div>
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 2, padding: '0 4px' }}>
            {Array.from({ length: numChunks }).map((_, i) => (
              <div 
                key={`amp-${i}`} 
                style={{
                  flex: 1,
                  height: getWaveHeight(i) + '%',
                  background: 'var(--ink-primary)',
                  opacity: (status === 'done' || (status === 'scanning' && (i/numChunks)*100 <= playheadPos)) ? 1 : 0.1,
                  transition: 'opacity var(--dur-fast)'
                }}
              />
            ))}
          </div>
        </div>

        {/* Track 2: Synthetic Likelihood */}
        <div style={{ display: 'flex', height: 80, borderBottom: '1px solid var(--grid-line)' }}>
          <div style={{ width: 60, borderRight: '1px solid var(--grid-line)', padding: 8 }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--ink-muted)' }}>02<br/>SYNTHETIC<br/>LIKELIHOOD</span>
          </div>
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 2, padding: '0 4px' }}>
            {Array.from({ length: numChunks }).map((_, i) => {
              const chunk = result?.chunks[i];
              const prob = chunk?.probability ?? 0;
              const isHighRisk = prob > 0.5;
              
              return (
                <div 
                  key={`lik-${i}`} 
                  style={{
                    flex: 1,
                    height: '60%',
                    background: isHighRisk ? 'var(--forensic-red)' : 'var(--grid-line)',
                    opacity: (status === 'done' || (status === 'scanning' && (i/numChunks)*100 <= playheadPos)) ? (isHighRisk ? prob : 0.2) : 0.05,
                    transition: 'opacity var(--dur-fast)'
                  }}
                />
              );
            })}
          </div>
        </div>

        {/* Playhead Line */}
        {(status === 'scanning' || status === 'done') && (
          <div 
            style={{
              position: 'absolute',
              top: 0,
              bottom: 0,
              left: `calc(60px + (100% - 60px) * ${playheadPos / 100})`,
              width: 1,
              background: 'var(--ink-primary)',
              zIndex: 10,
              transition: status === 'done' ? 'left 0.5s ease-out' : 'none'
            }}
          >
            {/* Playhead Marker */}
            <div style={{ position: 'absolute', top: 22, left: -24, background: 'var(--ink-primary)', color: '#fff', padding: '2px 4px', fontSize: 9, fontFamily: 'var(--font-mono)' }}>
              {status === 'done' ? formatTime(numChunks * 4) : formatTime((playheadPos/100) * (numChunks * 4))}
            </div>
          </div>
        )}

        {/* Footer controls (mock) */}
        <div style={{ display: 'flex', alignItems: 'center', padding: '8px 16px', background: 'var(--bg-panel)' }}>
          <span style={{ fontSize: 14, marginRight: 16 }}>▶</span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11 }}>
             {status === 'done' ? formatTime(numChunks * 4) : '00:00'} / {formatTime(numChunks * 4)}
          </span>
        </div>
      </div>
    </div>
  );
}
