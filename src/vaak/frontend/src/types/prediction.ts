/** Raw API response from POST /predict */
export interface PredictionResponse {
  is_spoof: boolean;
  spoof_probability: number;
  chunk_probabilities: number[];
}

/** Derived chunk with computed metadata */
export interface AnalyzedChunk {
  index: number;
  startSec: number;
  endSec: number;
  probability: number;
  risk: 'low' | 'medium' | 'high';
}

/** Final result handed to Result components */
export interface AnalysisResult {
  raw: PredictionResponse;
  chunks: AnalyzedChunk[];
  verdict: 'SYNTHETIC' | 'BONAFIDE';
  confidence: number; // 0–100
}

/** App-level view mode — drives which screen is rendered */
export type ViewMode = 'upload' | 'scanning' | 'results';
