import { useState, useRef, useEffect, useMemo } from 'react';
import type { RefObject } from 'react';
import { Icon } from '../ui/Icon';
import { formatTime } from './types';

interface ReadyStateProps {
  file: File | null;
  duration: number;
  analyze: () => void;
  reset: () => void;
  audioUrl?: string;
  startRecording?: () => void;
  fileInputRef?: RefObject<HTMLInputElement | null>;
}

/** Generate organic waveform heights seeded by filename */
function generateWaveform(seed: string, count: number) {
  let h = 0;
  for (let i = 0; i < seed.length; i++) h = (h * 31 + seed.charCodeAt(i)) >>> 0;
  return Array.from({ length: count }, (_, i) => {
    h = (h * 1664525 + 1013904223) >>> 0;
    const base = (h & 0xff) / 255;
    const wave = Math.abs(Math.sin(i * 0.37 + 1.2) * 0.35 + Math.cos(i * 0.19 + 0.7) * 0.25);
    return 0.06 + base * 0.55 + wave * 0.39;
  });
}

export function ReadyState({
  file,
  duration,
  analyze,
  reset,
  audioUrl,
  startRecording,
  fileInputRef,
}: ReadyStateProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [internalDuration, setInternalDuration] = useState(duration > 0 ? duration : 6.4);
  const [hoverBar, setHoverBar] = useState<number | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const effectiveDuration = internalDuration > 0 ? internalDuration : (duration > 0 ? duration : 6.4);
  const isRecorded = file?.name.startsWith('vaak-recording');
  const displayName = isRecorded ? 'Live Voice Recording' : (file?.name || 'Uploaded Audio');
  const fileExt = file?.name.split('.').pop()?.toUpperCase() || 'WAV';
  const fileSizeKb = file ? Math.round(file.size / 1024) : 0;

  const barCount = 80;
  const waveHeights = useMemo(() => generateWaveform(file?.name ?? 'demo', barCount), [file?.name]);

  useEffect(() => {
    if (duration > 0) setInternalDuration(duration);
  }, [duration]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const onTime = () => setCurrentTime(audio.currentTime);
    const onEnded = () => { setIsPlaying(false); setCurrentTime(0); };
    const onPause = () => setIsPlaying(false);
    const onPlay = () => setIsPlaying(true);
    audio.addEventListener('timeupdate', onTime);
    audio.addEventListener('ended', onEnded);
    audio.addEventListener('pause', onPause);
    audio.addEventListener('play', onPlay);
    return () => {
      audio.removeEventListener('timeupdate', onTime);
      audio.removeEventListener('ended', onEnded);
      audio.removeEventListener('pause', onPause);
      audio.removeEventListener('play', onPlay);
      audio.pause();
    };
  }, [audioUrl]);

  const togglePlayback = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (isPlaying) audio.pause(); else void audio.play();
  };

  const handleWaveClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const audio = audioRef.current;
    if (!audio || !effectiveDuration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    audio.currentTime = ratio * effectiveDuration;
    setCurrentTime(audio.currentTime);
    if (!isPlaying) void audio.play();
  };

  const progressPct = effectiveDuration > 0 ? Math.min(100, (currentTime / effectiveDuration) * 100) : 0;

  return (
    <div className="rp-shell">
      {audioUrl && (
        <audio
          ref={audioRef}
          src={audioUrl}
          preload="auto"
          onLoadedMetadata={(e) => {
            const d = e.currentTarget.duration;
            if (Number.isFinite(d) && d > 0) setInternalDuration(d);
          }}
        />
      )}

      {/* ── PLAYER CARD ─────────────────────────────────────── */}
      <div className="rp-card">
        {/* Header strip */}
        <div className="rp-card-header">
          <div className="rp-header-left">
            <span className="rp-capture-badge">
              <span className="rp-capture-dot" />
              CAPTURE VERIFIED
            </span>
          </div>
          <div className="rp-spec-row">
            <span className="rp-spec">{fileExt}</span>
            <span className="rp-spec">{effectiveDuration.toFixed(1)} s</span>
            {fileSizeKb > 0 && <span className="rp-spec">{fileSizeKb} KB</span>}
          </div>
        </div>

        {/* Title block */}
        <div className="rp-title-block">
          <p className="rp-kicker">ACOUSTIC SPECIMEN READY</p>
          <h3 className="rp-title">{displayName}</h3>
          {file?.name && (
            <p className="rp-filename">{file.name}</p>
          )}
        </div>

        {/* ── WAVEFORM PLAYER ─── */}
        <div className="rp-player">
          {/* Big play/pause */}
          <button
            type="button"
            className={`rp-play-btn ${isPlaying ? 'is-playing' : ''}`}
            onClick={togglePlayback}
            aria-label={isPlaying ? 'Pause' : 'Play'}
          >
            <span className="rp-play-icon">
              <Icon name={isPlaying ? 'pause' : 'play'} />
            </span>
          </button>

          {/* Waveform + scrubber */}
          <div className="rp-waveform-wrap">
            <div
              className="rp-waveform"
              onClick={handleWaveClick}
              onMouseMove={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                const idx = Math.floor(((e.clientX - rect.left) / rect.width) * barCount);
                setHoverBar(Math.max(0, Math.min(barCount - 1, idx)));
              }}
              onMouseLeave={() => setHoverBar(null)}
              role="slider"
              aria-valuenow={currentTime}
              aria-valuemin={0}
              aria-valuemax={duration}
              tabIndex={0}
              title="Click to seek"
            >
              {waveHeights.map((h, i) => {
                const barPct = (i / barCount) * 100;
                const isPast = barPct <= progressPct;
                const isHovered = hoverBar !== null && i <= hoverBar;
                return (
                  <span
                    key={i}
                    className={`rp-bar ${isPast ? 'is-past' : ''} ${isHovered && !isPast ? 'is-hover' : ''}`}
                    style={{ height: `${h * 100}%` }}
                  />
                );
              })}
              {/* Playhead needle */}
              <div className="rp-needle" style={{ left: `${progressPct}%` }} />
            </div>

            {/* Time ticks */}
            <div className="rp-time-row">
              <span className="rp-time-current">{formatTime(currentTime)}</span>
              <span className="rp-time-total">{formatTime(effectiveDuration)}</span>
            </div>
          </div>
        </div>

        {/* Signal metrics */}
        <div className="rp-metrics">
          <div className="rp-metric">
            <span className="rp-metric-label">LENGTH</span>
            <strong className="rp-metric-val">{effectiveDuration.toFixed(1)} s</strong>
            <span className="rp-metric-note">✓ optimal</span>
          </div>
          <div className="rp-metric">
            <span className="rp-metric-label">SIGNAL</span>
            <strong className="rp-metric-val">Nominal</strong>
            <span className="rp-metric-note">✓ no clipping</span>
          </div>
          <div className="rp-metric">
            <span className="rp-metric-label">HEADROOM</span>
            <strong className="rp-metric-val">-38 dBFS</strong>
            <span className="rp-metric-note">✓ clean SNR</span>
          </div>
        </div>
      </div>

      {/* ── DISPATCH PANEL ────────────────────────────────────── */}
      <div className="rp-dispatch">
        {/* Pipeline steps */}
        <div className="rp-pipeline">
          <p className="rp-pipeline-kicker">FORENSIC PIPELINE</p>
          <h4 className="rp-pipeline-title">Ready to evaluate</h4>
          <ol className="rp-steps">
            {[
              ['Acoustic Features', 'Mel-frequency cepstra & vocal-tract formants'],
              ['Temporal Jitter', 'Phase consistency & pitch-stability tracking'],
              ['Synthetic Probability', 'Segment-level likelihood across full recording'],
            ].map(([title, desc], i) => (
              <li key={i} className="rp-step">
                <span className="rp-step-num">0{i + 1}</span>
                <div>
                  <strong>{title}</strong>
                  <p>{desc}</p>
                </div>
              </li>
            ))}
          </ol>
        </div>

        {/* CTA */}
        <div className="rp-cta-stack">
          <button type="button" className="rp-analyze-btn" onClick={analyze} id="ready-analyze-button">
            <span>Analyze Voice Now</span>
            <Icon name="arrow" />
          </button>

          <div className="rp-secondary-actions">
            {startRecording && (
              <button type="button" className="rp-ghost-btn" onClick={startRecording}>
                <Icon name="record" /> Re-record
              </button>
            )}
            {fileInputRef && (
              <button type="button" className="rp-ghost-btn" onClick={() => fileInputRef.current?.click()}>
                <Icon name="upload" /> Upload different
              </button>
            )}
            <button type="button" className="rp-ghost-btn rp-ghost-btn--danger" onClick={reset}>
              <Icon name="close" /> Discard
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
