import type { RefObject } from 'react';
import { Icon } from '../ui/Icon';

interface ErrorStateProps {
  reset: () => void;
  message?: string;
  sampleDemo?: () => void;
  startRecording?: () => void;
  fileInputRef?: RefObject<HTMLInputElement | null>;
}

export function ErrorState({
  reset,
  message,
  sampleDemo,
  startRecording,
  fileInputRef,
}: ErrorStateProps) {
  const handleChooseFile = () => {
    reset();
    setTimeout(() => fileInputRef?.current?.click(), 60);
  };

  const handleStartRecording = () => {
    reset();
    setTimeout(() => startRecording?.(), 60);
  };

  const handleSample = () => {
    reset();
    setTimeout(() => sampleDemo?.(), 60);
  };

  return (
    <div className="err-shell">
      {/* ── LEFT: DIAGNOSTIC & SPECIMEN GUIDANCE ── */}
      <div className="err-diagnostic">
        <div>
          <div className="err-badge-row">
            <span className="err-badge">
              <span className="err-badge-dot" />
              INGESTION ANOMALY
            </span>
            <span className="err-code">STATUS / UNRESOLVED</span>
          </div>

          <div className="err-headline-block">
            <p className="err-kicker">ACOUSTIC INGESTION FAILED</p>
            <h3 className="err-title">The recording could not be evaluated.</h3>
            <p className="err-desc">
              {message || "The audio stream could not be decoded or was interrupted before acoustic verification could complete."}
            </p>
          </div>
        </div>

        <div className="err-guidance">
          <p className="err-guidance-kicker">SPECIMEN CRITERIA</p>
          <ul className="err-guidance-list">
            <li>
              <strong>Supported Formats:</strong> WAV (16-bit PCM), MP3, FLAC, M4A, AAC, OGG
            </li>
            <li>
              <strong>Signal Length:</strong> Minimum 1.0 second continuous speech recommended
            </li>
            <li>
              <strong>Microphone Input:</strong> Ensure browser audio permission is allowed
            </li>
          </ul>
        </div>
      </div>

      {/* ── RIGHT: RECOVERY ACTIONS ── */}
      <div className="err-actions-panel">
        <div>
          <div className="err-actions-header">
            <p className="err-actions-kicker">RECOVERY PIPELINE</p>
            <h4 className="err-actions-title">Try another source</h4>
            <p className="err-actions-sub">Select an alternative input method to continue evaluation.</p>
          </div>

          <div className="err-action-stack">
            <button
              type="button"
              className="err-btn-primary"
              onClick={handleChooseFile}
            >
              <Icon name="upload" />
              <span>Choose New Audio File</span>
            </button>

            {startRecording && (
              <button
                type="button"
                className="err-btn-secondary"
                onClick={handleStartRecording}
              >
                <Icon name="record" />
                <span>Record Live Speech</span>
              </button>
            )}

            {sampleDemo && (
              <button
                type="button"
                className="err-sample-btn"
                onClick={handleSample}
              >
                <span className="err-sample-spark">✦</span>
                <span>Test with illustrative sample instead</span>
              </button>
            )}
          </div>
        </div>

        <div className="err-footer">
          <button type="button" className="err-reset-link" onClick={reset}>
            <Icon name="replay" />
            <span>Reset to listening room</span>
          </button>
        </div>
      </div>
    </div>
  );
}
