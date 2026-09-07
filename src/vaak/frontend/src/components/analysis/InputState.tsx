import type { ChangeEvent, RefObject } from 'react';
import { Icon } from '../ui/Icon';

interface InputStateProps {
  fileInputRef: RefObject<HTMLInputElement | null>;
  onFile: (event: ChangeEvent<HTMLInputElement>) => void;
  startRecording: () => void;
  sampleDemo: () => void;
}

export function InputState({ fileInputRef, onFile, startRecording, sampleDemo }: InputStateProps) {
  return (
    <div className="input-state">
      <div className="input-prompt">
        <span className="prompt-number">01</span>
        <h3>A voice recording<br />is enough.</h3>
        <p>Phone recordings, interviews, voice notes, or social clips.</p>
      </div>
      <div className="input-actions">
        <button className="primary-action" onClick={() => fileInputRef.current?.click()}>
          <Icon name="upload" /> Choose audio
        </button>
        <button className="secondary-action" onClick={startRecording}>
          <Icon name="record" /> Record here
        </button>
        <input ref={fileInputRef} type="file" accept="audio/*,.flac,.opus" onChange={onFile} hidden />
        <p>WAV, MP3, M4A, FLAC, AAC, OGG/Opus</p>
        <button className="sample-link" onClick={sampleDemo}>
          or explore with an illustrative sample
        </button>
      </div>
    </div>
  );
}
