import { useMemo, useState, type RefObject } from 'react';
import { Icon } from '../ui/Icon';
import type { Evidence } from './types';
import { formatTime } from './types';
import type { AnalysisResult } from '../../types/prediction';
import {
  DECISION_SCORE,
  HUMAN_THRESHOLD,
  LIKELY_SYNTHETIC_THRESHOLD,
  getVerdictSummary,
} from '../../constants/thresholds';

interface ResultStateProps {
  duration: number;
  evidence: Evidence[];
  result: AnalysisResult | null;
  audioRef: RefObject<HTMLAudioElement | null>;
  audioUrl: string;
  isPlaying: boolean;
  setIsPlaying: (val: boolean) => void;
  playhead: number;
  setPlayhead: (val: number) => void;
  playRegion: (region: Evidence) => void;
  togglePlayback: () => void;
  reset: () => void;
}

/** Waveform visualization from real chunk probabilities */
function WaveformBars({ chunks, duration, playhead }: {
  chunks: AnalysisResult['chunks'];
  duration: number;
  playhead: number;
}) {
  const barCount = 120;
  const bars = useMemo(() => {
    return Array.from({ length: barCount }, (_, i) => {
      const t = (i / barCount) * duration;
      const chunk = chunks.find(c => t >= c.startSec && t < c.endSec);
      const prob = chunk?.probability ?? 0.35;
      const organic = Math.abs(Math.sin(i * 0.41) * 0.28 + Math.cos(i * 0.19) * 0.18);
      const height = Math.max(0.07, Math.min(1, prob * 0.6 + organic * 0.4));
      return { height, prob };
    });
  }, [chunks, duration]);

  const playheadPct = (playhead / Math.max(duration, 1)) * 100;

  return (
    <div className="waveform-bars" aria-hidden="true">
      {bars.map((bar, i) => {
        const pct = (i / barCount) * 100;
        const isPast = pct < playheadPct;
        let cls = 'wbar';
        if (bar.prob >= LIKELY_SYNTHETIC_THRESHOLD) cls += ' wbar--high';
        else if (bar.prob >= HUMAN_THRESHOLD) cls += ' wbar--moderate';
        if (isPast) cls += ' wbar--past';
        return (
          <span key={i} className={cls} style={{ height: `${bar.height * 100}%` }} />
        );
      })}
      <div className="waveform-playhead" style={{ left: `${playheadPct}%` }} />
    </div>
  );
}

/** Spectrum score gauge */
function ScoreGauge({ score }: { score: number }) {
  const pct = Math.min(100, Math.max(0, score));
  const summary = getVerdictSummary(score);
  return (
    <div className="score-gauge">
      <div className="score-spectrum">
        <div className="score-spectrum-track" />
        <div className="score-needle" style={{ left: `${pct}%` }}>
          <div className="score-needle-label">{score}</div>
          <div className="score-needle-line" />
        </div>
        <div className="score-center-tick" style={{ left: `${DECISION_SCORE}%` }} />
      </div>
      <div className="score-zones">
        <span>bonafide</span>
        <span>ambiguous (50)</span>
        <span>synthetic</span>
      </div>
      <div className="score-meta">
        <div className="score-meta-left">
          <span className="score-zone-badge" data-zone={summary.isSpoof ? 'synthetic' : 'bonafide'}>
            {summary.displayVerdict}
          </span>
        </div>
        <div className="score-meta-right">
          <span className="score-conf-label">confidence</span>
          <span className={`score-conf-value ${summary.confidenceTier.toLowerCase()}`}>
            {summary.confidenceTier}
          </span>
        </div>
      </div>
    </div>
  );
}

export function ResultState({
  duration,
  evidence,
  result,
  audioRef,
  audioUrl,
  isPlaying,
  setIsPlaying,
  playhead,
  setPlayhead,
  playRegion,
  togglePlayback,
  reset,
}: ResultStateProps) {
  const [hoverPct, setHoverPct] = useState<number | null>(null);
  const [activeRegion, setActiveRegion] = useState<number | null>(null);
  const score = result ? result.confidence : 0;
  const summary = getVerdictSummary(score);
  const isSynthetic = summary.isSpoof;
  const chunks = result?.chunks ?? [];

  const handleTrackMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    setHoverPct(Math.min(100, Math.max(0, (x / rect.width) * 100)));
  };

  const handleTrackClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    const targetTime = ratio * duration;
    if (audioRef.current) {
      audioRef.current.currentTime = targetTime;
      void audioRef.current.play();
      setIsPlaying(true);
    }
    setPlayhead(targetTime);
  };

  const handleRegionPlay = (region: Evidence, index: number, e: React.MouseEvent) => {
    e.stopPropagation();
    setActiveRegion(index);
    playRegion(region);
    setTimeout(() => setActiveRegion(null), (region.end - region.start) * 1000 + 300);
  };

  return (
    <div className="result-state">

      {/* ══ VERDICT + GAUGE ══════════════════════════════════════ */}
      <div className="result-summary">
        <div className="verdict">
          <div className="verdict-header">
            <p className="verdict-overline">
              <span className={`verdict-dot ${isSynthetic ? 'verdict-dot--synthetic' : 'verdict-dot--bonafide'}`} />
              Overall judgment
            </p>
          </div>
          <div className="verdict-primary">
            <h3 className="verdict-title" style={{ color: isSynthetic ? 'var(--orange)' : '#3a7d5c' }}>
              {summary.displayVerdict}
            </h3>
          </div>
          <ScoreGauge score={score} />
          <p className="score-disclaimer">Model score — not a calibrated probability.</p>
        </div>

        <div className="result-meta-grid">
          {[
            ['Duration', `${duration.toFixed(1)} s`],
            ['Windows', `${chunks.length} × 4 s`],
            ['Coverage', 'Full recording'],
            ['Signal quality', 'Good'],
            ['Model', 'WavLM-Base'],
            ['Format', '16 kHz · Mono'],
          ].map(([label, value]) => (
            <div key={label} className="meta-row">
              <dt>{label}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </div>
      </div>

      {/* ══ TEMPORAL EVIDENCE PANEL ══════════════════════════════ */}
      <div className="temporal-panel">
        <div className="temporal-heading">
          <div>
            <p>Temporal Evidence Timeline</p>
            <h4>Where Vaak noticed more</h4>
          </div>
          <span>Click timeline to seek · Click highlighted region to hear</span>
        </div>

        <audio
          ref={audioRef}
          src={audioUrl || undefined}
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
          onTimeUpdate={(e) => setPlayhead(e.currentTarget.currentTime)}
          onEnded={() => setIsPlaying(false)}
        />

        {/* ── Premium player controls bar ── */}
        <div className="tp-player-bar">
          <button
            className={`tp-playpause ${isPlaying ? 'is-playing' : ''}`}
            onClick={togglePlayback}
            disabled={!audioUrl}
            aria-label={isPlaying ? 'Pause' : 'Play'}
          >
            <Icon name={isPlaying ? 'pause' : 'play'} />
          </button>

          <div className="tp-time-block">
            <span className="tp-time-current">{formatTime(playhead)}</span>
            <span className="tp-time-sep">/</span>
            <span className="tp-time-total">{formatTime(duration)}</span>
          </div>

          <div className="tp-legend">
            <span className="tp-legend-swatch tp-legend-swatch--high" />
            <span>strong synthetic (≥65%)</span>
            <span className="tp-legend-swatch tp-legend-swatch--moderate" />
            <span>ambiguous (35–64%)</span>
          </div>
        </div>

        {/* ── Waveform track ── */}
        <div
          className="evidence-track"
          onClick={handleTrackClick}
          onMouseMove={handleTrackMouseMove}
          onMouseLeave={() => setHoverPct(null)}
          title="Click to play from here"
        >
          {chunks.length > 0 ? (
            <WaveformBars chunks={chunks} duration={duration} playhead={playhead} />
          ) : (
            <div className="track-threads">
              {Array.from({ length: 48 }, (_, i) => <i key={i} style={{ opacity: .12 + ((i * 7) % 10) / 25 }} />)}
              <div className="track-playhead" style={{ left: `${(playhead / Math.max(duration, 1)) * 100}%` }} />
            </div>
          )}

          {/* Hover scrub line */}
          {hoverPct !== null && (
            <div className="waveform-hoverhead" style={{ left: `${hoverPct}%` }}>
              <span className="waveform-hover-time">{formatTime((hoverPct / 100) * duration)}</span>
            </div>
          )}

          {/* Clickable evidence overlays */}
          {evidence.map((region, i) => (
            <button
              key={i}
              className={`evidence-region high ${activeRegion === i ? 'is-active' : ''}`}
              style={{
                left: `${(region.start / Math.max(duration, 1)) * 100}%`,
                width: `${((region.end - region.start) / Math.max(duration, 1)) * 100}%`,
              }}
              onClick={(e) => handleRegionPlay(region, i, e)}
              aria-label={`Play synthetic evidence: ${formatTime(region.start)} → ${formatTime(region.end)}`}
            />
          ))}
        </div>

        {/* Time axis */}
        <div className="track-time-axis">
          {Array.from({ length: 9 }, (_, i) => {
            const t = (i / 8) * duration;
            return <span key={i} style={{ left: `${(i / 8) * 100}%` }}>{formatTime(t)}</span>;
          })}
        </div>

        {/* Evidence list */}
        {evidence.length > 0 ? (
          <div className="evidence-list">
            {evidence.map((region, i) => (
              <button
                key={i}
                className={`ev-row ${activeRegion === i ? 'is-playing' : ''}`}
                onClick={(e) => handleRegionPlay(region, i, e as unknown as React.MouseEvent)}
              >
                <span className="ev-row-dot" />
                <span className="ev-row-time">
                  <strong>{formatTime(region.start)}</strong>
                  <span className="ev-row-arrow">→</span>
                  <strong>{formatTime(region.end)}</strong>
                </span>
                <em className="ev-row-label">SYNTHETIC SIGNAL DETECTED</em>
                <span className="ev-row-hear">
                  <Icon name={activeRegion === i ? 'pause' : 'play'} />
                  {activeRegion === i ? 'Playing…' : 'Hear span'}
                </span>
              </button>
            ))}
          </div>
        ) : (
          <div className="evidence-empty">
            <p>No strong synthetic evidence (≥65%) found in this recording.</p>
          </div>
        )}
      </div>

      {/* ══ FOOTER ═════════════════════════════════════════════ */}
      <div className="result-footer">
        <p>Test the reading against your own ear — Vaak is most useful when its judgment is questioned, not taken on faith.</p>
        <div className="result-footer-actions">
          <button className="result-action-btn result-action-btn--secondary" onClick={reset}>
            <Icon name="replay" /> Examine another
          </button>
        </div>
      </div>

      <p className="result-disclaimer">
        public preview — detection is probabilistic and imperfect. a single reading should never be treated as proof.
      </p>
    </div>
  );
}
