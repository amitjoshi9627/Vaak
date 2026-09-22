import { useEffect, useMemo, useRef, useState } from 'react';
import type { ChangeEvent as ReactChangeEvent } from 'react';
import { Icon } from '../ui/Icon';
import type { AppState, Evidence } from './types';
import { InputState } from './InputState';
import { RecordingState } from './RecordingState';
import { ReadyState } from './ReadyState';
import { ProcessingState } from './ProcessingState';
import { ResultState } from './ResultState';
import { ErrorState } from './ErrorState';
import { useAudioAnalysis } from '../../hooks/useAudioAnalysis';
import { LIKELY_SYNTHETIC_THRESHOLD } from '../../constants/thresholds';
import sampleAudioUrl from '../../assets/sample.wav';
import { scrollToListen } from '../../utils/scroll';

/** Synthesizes an audible, realistic 16-bit PCM voice-like WAV fallback */
function createFallbackAudioBlob(): Blob {
  const sampleRate = 16000;
  const numSeconds = 6.4;
  const numSamples = Math.floor(sampleRate * numSeconds);
  const buffer = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(buffer);

  const writeString = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + numSamples * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, numSamples * 2, true);

  for (let i = 0; i < numSamples; i++) {
    const t = i / sampleRate;
    const f0 = 175 + Math.sin(t * 3.5) * 28;
    const s = 0.55 * Math.sin(2 * Math.PI * f0 * t) +
              0.28 * Math.sin(2 * Math.PI * f0 * 2 * t) +
              0.14 * Math.sin(2 * Math.PI * f0 * 3 * t);
    const envelope = Math.max(0, Math.sin(t * 4)) * Math.min(1, Math.min(t * 2, (numSeconds - t) * 2));
    const sample = Math.max(-1, Math.min(1, s * envelope * 0.7));
    view.setInt16(44 + i * 2, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

export function ListenSection() {
  const [localState, setLocalState] = useState<AppState>("empty");
  const [file, setFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState("");
  const [duration, setDuration] = useState(6.4);
  const [recordTime, setRecordTime] = useState(0);
  const [processingStep, setProcessingStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playhead, setPlayhead] = useState(0);
  const [localErrorMessage, setLocalErrorMessage] = useState("");
  const audioRef = useRef<HTMLAudioElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const recordTimeRef = useRef<number>(0);
  const sectionRef = useRef<HTMLElement>(null);
  const consoleRef = useRef<HTMLDivElement>(null);
  const playRegionTimeoutRef = useRef<number | null>(null);

  const { state: analysisState, analyze: triggerAnalysis, reset: resetAnalysis } = useAudioAnalysis();

  // Bridge hook state to local AppState
  const state = useMemo<AppState>(() => {
    if (analysisState.status === 'scanning') return 'processing';
    if (analysisState.status === 'done') return 'result';
    if (analysisState.status === 'error') return 'error';
    return localState;
  }, [analysisState.status, localState]);

  // Point out playable evidence spans only for high synthetic likelihood segments (65%+ mark)
  const evidence: Evidence[] = useMemo(() => {
    if (analysisState.status !== 'done') return [];

    return analysisState.result.chunks
      .filter((c) => c.probability >= LIKELY_SYNTHETIC_THRESHOLD)
      .map((c) => ({
        start: c.startSec,
        end: c.endSec,
        strength: 'high' as const,
      }));
  }, [analysisState]);

  useEffect(() => () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    if (timerRef.current) window.clearInterval(timerRef.current);
  }, []);

  useEffect(() => () => {
    if (audioUrl) URL.revokeObjectURL(audioUrl);
  }, [audioUrl]);

  // Update processing steps while scanning
  useEffect(() => {
    if (analysisState.status === 'scanning') {
      let step = 0;
      setProcessingStep(step);
      const interval = setInterval(() => {
        step++;
        if (step < 4) setProcessingStep(step);
      }, 500);
      return () => clearInterval(interval);
    }
  }, [analysisState.status]);

  const readDuration = (url: string) => {
    const probe = new Audio(url);
    probe.addEventListener("loadedmetadata", () => {
      if (Number.isFinite(probe.duration) && probe.duration > 0) {
        setDuration(probe.duration);
      }
    }, { once: true });
    probe.addEventListener("error", () => {
      setDuration(6.4);
    }, { once: true });
  };

  const acceptFile = (selected: File, knownDuration?: number) => {
    const isAudioType = selected.type.startsWith("audio/") || selected.type.includes("webm") || selected.type.includes("ogg") || selected.type.includes("mp4");
    const isAudioExt = /\.(wav|mp3|m4a|flac|aac|ogg|opus|webm|mp4)$/i.test(selected.name);
    if (!isAudioType && !isAudioExt) {
      setLocalState("error");
      return;
    }
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    const url = URL.createObjectURL(selected);
    setAudioUrl(url);
    setFile(selected);
    setPlayhead(0);
    setLocalState("ready");
    if (knownDuration && knownDuration > 0) {
      setDuration(knownDuration);
    } else {
      readDuration(url);
    }
  };

  const onFile = (event: ReactChangeEvent<HTMLInputElement>) => {
    const selected = event.target.files?.[0];
    if (selected) acceptFile(selected);
    event.target.value = "";
  };

  const startRecording = async () => {
    try {
      const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(audioStream);
      streamRef.current = audioStream;
      mediaRecorderRef.current = recorder;
      setStream(audioStream);
      chunksRef.current = [];
      setRecordTime(0);
      recordTimeRef.current = 0;
      recorder.ondataavailable = (event) => event.data.size && chunksRef.current.push(event.data);
      recorder.onstop = () => {
        const finalDuration = Math.max(1, Math.round(recordTimeRef.current * 10) / 10);
        const mimeType = recorder.mimeType || "audio/webm";
        const blob = new Blob(chunksRef.current, { type: mimeType });
        const ext = mimeType.includes("mp4") ? "mp4" : "webm";
        const recorded = new File([blob], `vaak-recording-${Date.now()}.${ext}`, { type: mimeType });
        acceptFile(recorded, finalDuration);
        audioStream.getTracks().forEach((track) => track.stop());
        setStream(null);
      };
      recorder.start();
      setLocalState("recording");
      timerRef.current = window.setInterval(() => {
        recordTimeRef.current += 0.1;
        setRecordTime(recordTimeRef.current);
      }, 100);
    } catch (err: any) {
      if (err.name === 'NotAllowedError') {
        setLocalErrorMessage("Microphone access was denied.");
      } else {
        setLocalErrorMessage("Microphone access requires HTTPS or localhost.");
      }
      setLocalState("error");
    }
  };

  const stopRecording = () => {
    if (timerRef.current) window.clearInterval(timerRef.current);
    timerRef.current = null;
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
  };

  const cancelRecording = () => {
    if (timerRef.current) window.clearInterval(timerRef.current);
    timerRef.current = null;
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.onstop = null;
      mediaRecorderRef.current.stop();
    }
    streamRef.current?.getTracks().forEach((track) => track.stop());
    setStream(null);
    setRecordTime(0);
    setLocalState("empty");
  };

  const analyze = () => {
    if (file) {
      triggerAnalysis(file);
    }
  };

  const reset = () => {
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl("");
    setFile(null);
    setLocalState("empty");
    setIsPlaying(false);
    setPlayhead(0);
    resetAnalysis();
    // Anchor viewport to Give Vaak a voice intro to frame headline and console
    setTimeout(() => {
      scrollToListen();
    }, 50);
  };

  const togglePlayback = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (playRegionTimeoutRef.current) {
      window.clearTimeout(playRegionTimeoutRef.current);
      playRegionTimeoutRef.current = null;
    }
    if (audio.paused) audio.play().catch(console.warn); else audio.pause();
  };

  const playRegion = (region: Evidence) => {
    const audio = audioRef.current;
    if (!audio) return;
    if (playRegionTimeoutRef.current) {
      window.clearTimeout(playRegionTimeoutRef.current);
    }
    audio.currentTime = region.start;
    audio.play().catch(console.warn);
    playRegionTimeoutRef.current = window.setTimeout(() => {
      audio.pause();
      playRegionTimeoutRef.current = null;
    }, Math.max(400, (region.end - region.start) * 1000));
  };

  const sampleDemo = async () => {
    try {
      let blob: Blob | null = null;
      try {
        const response = await fetch(sampleAudioUrl);
        if (response.ok) blob = await response.blob();
      } catch {
        // Continue to static path fallback
      }

      if (!blob) {
        try {
          const response = await fetch('/sample.wav');
          if (response.ok) blob = await response.blob();
        } catch {
          // Continue to programmatic fallback
        }
      }

      if (!blob) {
        blob = createFallbackAudioBlob();
      }

      const demoFile = new File([blob], "field-interview-07.wav", { type: "audio/wav" });
      Object.assign(demoFile, { isSampleDemo: true });
      acceptFile(demoFile);
    } catch (e) {
      console.error('Sample demo load error:', e);
      const fallbackBlob = createFallbackAudioBlob();
      const demoFile = new File([fallbackBlob], "field-interview-07.wav", { type: "audio/wav" });
      Object.assign(demoFile, { isSampleDemo: true });
      acceptFile(demoFile);
    }
  };

  return (
    <section ref={sectionRef} className="listen-section">
      {/* Visual Narrative Connector */}
      <div className="section-narrative-cue listen-cue" aria-hidden="true">
        <span className="narrative-line" />
        <span className="narrative-node">01 / LISTEN</span>
        <span className="narrative-line" />
      </div>

      <div id="listen" className="section-intro listen-intro centered-intro">
        <h2>Give Vaak a voice.</h2>
        <p className="listen-lead-copy">
          Upload a recording or record one here. No settings required.
        </p>
      </div>
      <div ref={consoleRef} className={`analysis-console state-${state}`}>
        <div className="console-topline">
          <span>VAAK / LISTEN</span>
          <span className="system-status">
            <i /> {state === "processing" ? "Analyzing" : state === "recording" ? "Microphone active" : "Public preview v0.5.0"}
          </span>
        </div>

        {state === "empty" && <InputState fileInputRef={fileInputRef} onFile={onFile} startRecording={startRecording} sampleDemo={sampleDemo} />}
        {state === "recording" && (
          <RecordingState
            recordTime={recordTime}
            stopRecording={stopRecording}
            cancelRecording={cancelRecording}
            stream={stream || streamRef.current}
          />
        )}
        {state === "ready" && (
          <ReadyState
            file={file}
            duration={duration}
            analyze={analyze}
            reset={reset}
            audioUrl={audioUrl}
            startRecording={startRecording}
            fileInputRef={fileInputRef}
          />
        )}
        {state === "processing" && <ProcessingState processingStep={processingStep} />}
        {state === "result" && (
          <ResultState
            duration={duration}
            evidence={evidence}
            result={analysisState.status === 'done' ? analysisState.result : null}
            audioRef={audioRef}
            audioUrl={audioUrl}
            isPlaying={isPlaying}
            setIsPlaying={setIsPlaying}
            playhead={playhead}
            setPlayhead={setPlayhead}
            playRegion={playRegion}
            togglePlayback={togglePlayback}
            reset={reset}
          />
        )}
        {state === "error" && (
          <ErrorState
            reset={reset}
            message={analysisState.status === 'error' ? analysisState.message : localErrorMessage || undefined}
            sampleDemo={sampleDemo}
            startRecording={startRecording}
            fileInputRef={fileInputRef}
          />
        )}
      </div>
      <p className="privacy-line">
        <Icon name="shield" /> Your audio is used only for the requested analysis and is not used for model training.
      </p>
    </section>
  );
}
