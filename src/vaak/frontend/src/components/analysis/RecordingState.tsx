import { memo, useEffect, useRef } from 'react';
import { formatTime } from './types';
import { Icon } from '../ui/Icon';

interface RecordingStateProps {
  recordTime: number;
  stopRecording: () => void;
  cancelRecording?: () => void;
  stream: MediaStream | null;
}

/**
 * LiveAudioDeck
 * Encapsulates the live oscilloscope canvas, FFT frequency bars, and segmented VU meter.
 * Uses a dedicated, non-re-rendering 60fps RAF loop that updates DOM nodes directly via refs,
 * completely avoiding AudioContext teardown and React reconciliation delays.
 */
const LiveAudioDeck = memo(function LiveAudioDeck({
  stream,
}: {
  stream: MediaStream | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const meterBarsRef = useRef<(HTMLSpanElement | null)[]>([]);
  const dbTextRef = useRef<HTMLSpanElement>(null);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let audioCtx: AudioContext | null = null;
    let analyser: AnalyserNode | null = null;
    let source: MediaStreamAudioSourceNode | null = null;
    let timeData: Uint8Array<ArrayBuffer> | null = null;
    let freqData: Uint8Array<ArrayBuffer> | null = null;

    if (stream) {
      try {
        const AudioCtx =
          window.AudioContext ||
          (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        audioCtx = new AudioCtx();
        if (audioCtx.state === 'suspended') {
          void audioCtx.resume();
        }
        source = audioCtx.createMediaStreamSource(stream);
        analyser = audioCtx.createAnalyser();
        analyser.fftSize = 2048;
        analyser.smoothingTimeConstant = 0.60; // Smooth, laboratory-grade response
        analyser.minDecibels = -85;
        analyser.maxDecibels = -15;
        source.connect(analyser);
        timeData = new Uint8Array(analyser.frequencyBinCount) as Uint8Array<ArrayBuffer>;
        freqData = new Uint8Array(analyser.frequencyBinCount) as Uint8Array<ArrayBuffer>;
      } catch (err) {
        console.warn('AudioContext initialization failed', err);
      }
    }

    const resumeCtx = () => {
      if (audioCtx && audioCtx.state === 'suspended') {
        void audioCtx.resume();
      }
    };
    window.addEventListener('click', resumeCtx, { passive: true });
    window.addEventListener('keydown', resumeCtx, { passive: true });

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let phase = 0;
    let smoothedRms = 0;
    let autoGain = 1.5;

    const render = () => {
      rafRef.current = requestAnimationFrame(render);
      phase += 0.035;

      if (audioCtx && audioCtx.state === 'suspended') {
        void audioCtx.resume();
      }

      // Handle HiDPI Canvas Scaling
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      const displayW = Math.max(300, Math.floor(rect.width));
      const displayH = Math.max(160, Math.floor(rect.height));

      if (canvas.width !== displayW * dpr || canvas.height !== displayH * dpr) {
        canvas.width = displayW * dpr;
        canvas.height = displayH * dpr;
      }

      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, displayW, displayH);

      const W = displayW;
      const H = displayH;
      const midY = H / 2;

      // 1. Grid lines
      ctx.strokeStyle = 'rgba(23, 32, 30, 0.06)';
      ctx.lineWidth = 1;

      const colStep = W / 8;
      for (let x = colStep; x < W; x += colStep) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, H);
        ctx.stroke();
      }

      [-0.7, -0.35, 0.35, 0.7].forEach((factor) => {
        ctx.beginPath();
        ctx.moveTo(0, midY + factor * midY);
        ctx.lineTo(W, midY + factor * midY);
        ctx.stroke();
      });

      // Center baseline
      ctx.strokeStyle = 'rgba(230, 95, 53, 0.25)';
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(0, midY);
      ctx.lineTo(W, midY);
      ctx.stroke();
      ctx.setLineDash([]);

      // 2. Audio Processing
      let rawRms = 0;
      let peakDev = 0;

      if (analyser && timeData && freqData) {
        analyser.getByteTimeDomainData(timeData);
        analyser.getByteFrequencyData(freqData);

        let sumSquares = 0;
        for (let i = 0; i < timeData.length; i++) {
          const dev = (timeData[i] - 128) / 128;
          const abs = Math.abs(dev);
          if (abs > peakDev) peakDev = abs;
          sumSquares += dev * dev;
        }
        rawRms = Math.sqrt(sumSquares / timeData.length);

        // Background Spectral Bars (speech frequency 80Hz - 4000Hz)
        const numBars = 36;
        const barWidth = W / numBars;
        ctx.fillStyle = 'rgba(230, 95, 53, 0.08)';
        for (let b = 0; b < numBars; b++) {
          const freqIndex = Math.floor((b / numBars) * (freqData.length * 0.25));
          const val = freqData[freqIndex] / 255;
          const barHeight = Math.pow(val, 1.4) * (H * 0.35);
          if (barHeight > 1.5) {
            ctx.fillRect(b * barWidth + 1.5, H - barHeight, barWidth - 3, barHeight);
          }
        }
      }

      // Ballistics smoothing: professional studio meter response
      if (rawRms > smoothedRms) {
        smoothedRms = smoothedRms * 0.70 + rawRms * 0.30;
      } else {
        smoothedRms = smoothedRms * 0.94 + rawRms * 0.06;
      }

      // Controlled Noise Gate & Gain Calibration
      const NOISE_FLOOR = 0.015;
      const isVoiceActive = peakDev > NOISE_FLOOR || smoothedRms > 0.008;

      if (isVoiceActive) {
        const targetHeight = midY * 0.60;
        const instantGain = targetHeight / Math.max(1, peakDev * midY);
        autoGain = autoGain * 0.90 + Math.min(2.5, Math.max(1.2, instantGain)) * 0.10;
      } else {
        autoGain = autoGain * 0.95 + 1.2 * 0.05;
      }

      // Calibrated dBFS Calculation
      let dbfs = -48;
      if (smoothedRms > 0.0005) {
        dbfs = Math.min(0, Math.max(-48, Math.round(20 * Math.log10(smoothedRms * 2.2))));
      }

      // Update Meter Bars directly in DOM (ZERO React re-renders)
      const numSegments = 16;
      let activeCount = 0;
      if (dbfs > -47) {
        const fraction = Math.min(1, Math.max(0, (dbfs - (-48)) / 48));
        activeCount = Math.min(numSegments, Math.max(1, Math.round(Math.pow(fraction, 0.85) * numSegments)));
      } else if (rawRms === 0) {
        // Subtle ambient idle indication if completely silent
        activeCount = 1;
      }

      for (let i = 0; i < numSegments; i++) {
        const seg = meterBarsRef.current[i];
        if (seg) {
          if (i < activeCount) {
            seg.classList.add('is-active');
          } else {
            seg.classList.remove('is-active');
          }
        }
      }

      // Update dB text directly in DOM
      if (dbTextRef.current) {
        dbTextRef.current.textContent = dbfs > -45 ? `${dbfs} dBFS` : 'IDLE';
      }

      // Update right panel radar reticle directly in DOM
      const orbCore = document.getElementById('rec-sensor-core');
      const orbRing = document.getElementById('rec-sensor-ring-2');
      if (orbCore) {
        const coreScale = 1 + Math.min(0.25, smoothedRms * 1.5);
        orbCore.style.transform = `translate(-50%, -50%) scale(${coreScale.toFixed(3)})`;
      }
      if (orbRing) {
        const ringScale = 1 + Math.min(0.18, smoothedRms * 1.0);
        orbRing.style.transform = `translate(-50%, -50%) scale(${ringScale.toFixed(3)})`;
      }

      // 3. Draw Dynamic Waveform
      const points: { x: number; y: number }[] = [];
      const count = 140;
      const sliceW = W / (count - 1);

      for (let i = 0; i < count; i++) {
        const x = i * sliceW;
        let amp = 0;

        if (isVoiceActive && timeData && timeData.length > 0) {
          const sampleIdx = Math.floor((i / count) * (timeData.length * 0.7));
          const raw = (timeData[sampleIdx] - 128) / 128;
          amp = raw * midY * autoGain;
        } else {
          // Ambient gentle breathing baseline
          amp = Math.sin(phase + i * 0.1) * 2.5 + Math.cos(phase * 0.6 + i * 0.04) * 1.2;
        }

        const clampedAmp = Math.max(-midY * 0.94, Math.min(midY * 0.94, amp));
        points.push({ x, y: midY + clampedAmp });
      }

      // Draw Gradient Fill under / mirrored
      const fillGrad = ctx.createLinearGradient(0, 0, 0, H);
      fillGrad.addColorStop(0, 'rgba(230, 95, 53, 0.0)');
      fillGrad.addColorStop(0.5, `rgba(230, 95, 53, ${Math.min(0.38, 0.12 + smoothedRms * 0.8)})`);
      fillGrad.addColorStop(1, 'rgba(230, 95, 53, 0.0)');

      ctx.fillStyle = fillGrad;
      ctx.beginPath();
      ctx.moveTo(0, midY);
      for (let i = 0; i < points.length; i++) {
        ctx.lineTo(points[i].x, points[i].y);
      }
      ctx.lineTo(W, midY);
      for (let i = points.length - 1; i >= 0; i--) {
        const mirroredY = midY - (points[i].y - midY);
        ctx.lineTo(points[i].x, mirroredY);
      }
      ctx.closePath();
      ctx.fill();

      // Draw High-Precision Waveform Curve
      ctx.save();
      ctx.shadowColor = 'rgba(230, 95, 53, 0.45)';
      ctx.shadowBlur = 5;
      ctx.strokeStyle = '#e65f35';
      ctx.lineWidth = 2.4;
      ctx.beginPath();

      for (let i = 0; i < points.length; i++) {
        if (i === 0) ctx.moveTo(points[i].x, points[i].y);
        else {
          const prev = points[i - 1];
          const curr = points[i];
          const cx = (prev.x + curr.x) / 2;
          const cy = (prev.y + curr.y) / 2;
          ctx.quadraticCurveTo(prev.x, prev.y, cx, cy);
        }
      }
      ctx.stroke();
      ctx.restore();

      ctx.restore();
    };

    render();

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      window.removeEventListener('click', resumeCtx);
      window.removeEventListener('keydown', resumeCtx);
      source?.disconnect();
      audioCtx?.close();
    };
  }, [stream]);

  const numSegments = 16;

  return (
    <div className="rec-main-panel">
      <div className="rec-telemetry-bar">
        <div className="rec-live-tag">
          <span className="rec-live-bullet" />
          <span>LIVE MICROPHONE STREAM</span>
        </div>
        <div className="rec-spec-chips">
          <span className="rec-chip">48.0 kHz</span>
          <span className="rec-chip">PCM</span>
          <span className="rec-chip">MONO</span>
        </div>
      </div>

      {/* Live Canvas Oscilloscope */}
      <div className="rec-oscilloscope-wrapper">
        <div className="rec-canvas-frame">
          <canvas ref={canvasRef} className="rec-canvas-element" />
        </div>
      </div>

      {/* Live Segmented VU Meter */}
      <div className="rec-meter-container">
        <div className="rec-meter-header">
          <span className="rec-meter-label">ACOUSTIC ENERGY</span>
          <span ref={dbTextRef} className="rec-meter-val">
            IDLE
          </span>
        </div>
        <div className="rec-meter-track" aria-hidden="true">
          {Array.from({ length: numSegments }).map((_, i) => {
            let tier = 'low';
            if (i >= 13) tier = 'high';
            else if (i >= 9) tier = 'mid';

            return (
              <span
                key={i}
                ref={(el) => {
                  meterBarsRef.current[i] = el;
                }}
                className={`rec-meter-seg rec-meter-seg--${tier}`}
              />
            );
          })}
        </div>
        <div className="rec-meter-scale">
          <span>-48dB</span>
          <span>-24dB</span>
          <span>-12dB</span>
          <span>-6dB</span>
          <span>0dB</span>
        </div>
      </div>

      <p className="rec-footer-guidance">
        Speak in natural conversational tone. Vaak examines phonetic boundaries and temporal micro-spectral artifacts.
      </p>
    </div>
  );
});

export function RecordingState({ recordTime, stopRecording, cancelRecording, stream }: RecordingStateProps) {
  return (
    <div className="recording-console-layout">
      {/* LEFT: Master Oscilloscope & Telemetry (Memoized, completely decoupled from timer ticks) */}
      <LiveAudioDeck stream={stream} />

      {/* RIGHT: Acoustic Sensor & Mission Control */}
      <div className="rec-side-panel">
        {/* Radar / Sensor Node */}
        <div className="rec-sensor-node" aria-hidden="true">
          <div className="rec-sensor-reticle">
            <span className="rec-reticle-ring rec-reticle-ring--1" />
            <span id="rec-sensor-ring-2" className="rec-reticle-ring rec-reticle-ring--2" />
            <span className="rec-reticle-crosshair-h" />
            <span className="rec-reticle-crosshair-v" />
            <div id="rec-sensor-core" className="rec-sensor-core">
              <span className="rec-sensor-inner-dot" />
            </div>
          </div>
          <span className="rec-sensor-caption">ACOUSTIC CHAMBER 01</span>
        </div>

        {/* Big Stopwatch Display */}
        <div className="rec-timer-card">
          <span className="rec-timer-kicker">DURATION</span>
          <div className="rec-timer-digits">
            {formatTime(recordTime)}
          </div>
          <span className="rec-timer-window">
            {recordTime < 3.0 ? 'Minimum 3 seconds recommended' : 'Ideal sample length reached'}
          </span>
        </div>

        {/* Action Controls */}
        <div className="rec-button-stack">
          <button 
            className="rec-stop-button" 
            onClick={stopRecording}
            id="stop-recording-btn"
          >
            <span className="rec-stop-mark" />
            <span>Stop &amp; Analyze Voice</span>
            <Icon name="arrow" />
          </button>

          {cancelRecording && (
            <button 
              className="rec-cancel-button" 
              onClick={cancelRecording}
              id="cancel-recording-btn"
            >
              <Icon name="close" />
              <span>Cancel recording</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
