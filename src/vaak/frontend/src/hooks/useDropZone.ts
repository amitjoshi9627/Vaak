import { useCallback, useRef, useState } from 'react';

const ACCEPTED_TYPES = new Set(['.wav', '.flac', 'audio/wav', 'audio/x-wav', 'audio/flac']);

function isValidAudio(file: File): boolean {
  const ext = '.' + file.name.split('.').pop()?.toLowerCase();
  return ACCEPTED_TYPES.has(ext) || ACCEPTED_TYPES.has(file.type);
}

interface UseDropZoneOptions {
  onFile: (file: File) => void;
  onError: (msg: string) => void;
}

/**
 * useDropZone
 *
 * Provides ref attachment and drag-event handlers for a file drop zone.
 * Validates file type before calling onFile.
 */
export function useDropZone({ onFile, onError }: UseDropZoneOptions) {
  const [isDragging, setIsDragging] = useState(false);
  const dragCountRef = useRef(0); // track nested drag-enter/leave

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCountRef.current++;
    if (dragCountRef.current === 1) setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCountRef.current--;
    if (dragCountRef.current === 0) setIsDragging(false);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCountRef.current = 0;
    setIsDragging(false);

    const file = e.dataTransfer.files[0];
    if (!file) return;

    if (!isValidAudio(file)) {
      onError('Unsupported file type. Please drop a .wav or .flac file.');
      return;
    }

    onFile(file);
  }, [onFile, onError]);

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!isValidAudio(file)) {
      onError('Unsupported file type. Please select a .wav or .flac file.');
      return;
    }

    onFile(file);
    // Reset input so the same file can be re-selected
    e.target.value = '';
  }, [onFile, onError]);

  return {
    isDragging,
    handleDragEnter,
    handleDragLeave,
    handleDragOver,
    handleDrop,
    handleInputChange,
  };
}
