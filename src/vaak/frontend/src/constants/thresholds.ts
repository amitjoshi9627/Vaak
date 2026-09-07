/**
 * Centralized classification & risk threshold constants for Vaak.
 * Synchronized with Python backend constants (vaak/config/constants.py).
 */

// Normalized score thresholds (0.0 to 1.0)
export const DECISION_THRESHOLD = 0.50;
export const HUMAN_THRESHOLD = 0.35;
export const LIKELY_SYNTHETIC_THRESHOLD = 0.65;
export const STRONG_SYNTHETIC_THRESHOLD = 0.80;

// Percentage thresholds (0 to 100)
export const DECISION_SCORE = 50;
export const HUMAN_SCORE = 35;
export const LIKELY_SYNTHETIC_SCORE = 65;
export const STRONG_SYNTHETIC_SCORE = 80;

export type VerdictZone = 'synthetic-strong' | 'synthetic-likely' | 'ambiguous' | 'human-likely';
export type ConfidenceTier = 'HIGH' | 'MEDIUM' | 'LOW';

export interface VerdictSummary {
  verdict: 'SYNTHETIC' | 'BONAFIDE';
  displayVerdict: string;
  confidenceTier: ConfidenceTier;
  zone: VerdictZone;
  isSpoof: boolean;
}

/**
 * Derives full verdict summary and visual zone from a confidence score (0–100).
 */
export function getVerdictSummary(score: number): VerdictSummary {
  if (score >= STRONG_SYNTHETIC_SCORE) {
    return {
      verdict: 'SYNTHETIC',
      displayVerdict: 'strong synthetic evidence',
      confidenceTier: 'HIGH',
      zone: 'synthetic-strong',
      isSpoof: true,
    };
  }
  if (score >= LIKELY_SYNTHETIC_SCORE) {
    return {
      verdict: 'SYNTHETIC',
      displayVerdict: 'likely synthetic',
      confidenceTier: 'MEDIUM',
      zone: 'synthetic-likely',
      isSpoof: true,
    };
  }
  if (score > HUMAN_SCORE) {
    const isSpoof = score >= DECISION_SCORE;
    return {
      verdict: isSpoof ? 'SYNTHETIC' : 'BONAFIDE',
      displayVerdict: 'ambiguous signal',
      confidenceTier: 'LOW',
      zone: 'ambiguous',
      isSpoof,
    };
  }
  return {
    verdict: 'BONAFIDE',
    displayVerdict: 'likely human',
    confidenceTier: 'HIGH',
    zone: 'human-likely',
    isSpoof: false,
  };
}

/**
 * Classifies an individual chunk probability (0.0 to 1.0) into a risk tier.
 */
export function getChunkRisk(probability: number): 'high' | 'medium' | 'low' {
  if (probability >= LIKELY_SYNTHETIC_THRESHOLD) return 'high';
  if (probability >= HUMAN_THRESHOLD) return 'medium';
  return 'low';
}
