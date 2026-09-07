"""Centralized threshold constants for Vaak classification & evaluation."""

# Normalized decision thresholds (0.0 to 1.0)
DECISION_THRESHOLD: float = 0.50
HUMAN_THRESHOLD: float = 0.35
LIKELY_SYNTHETIC_THRESHOLD: float = 0.65
STRONG_SYNTHETIC_THRESHOLD: float = 0.80

# Score percentage equivalents (0 to 100)
DECISION_SCORE: int = 50
HUMAN_SCORE: int = 35
LIKELY_SYNTHETIC_SCORE: int = 65
STRONG_SYNTHETIC_SCORE: int = 80


def get_risk_tier(probability: float) -> str:
    """Classify chunk spoof probability into risk tier."""
    if probability >= LIKELY_SYNTHETIC_THRESHOLD:
        return "high"
    elif probability >= HUMAN_THRESHOLD:
        return "medium"
    return "low"


def get_verdict_summary(score: float) -> dict[str, str | bool]:
    """Derive global classification summary from score (0-100)."""
    if score >= STRONG_SYNTHETIC_SCORE:
        return {
            "verdict": "SYNTHETIC",
            "label": "Strong Synthetic Evidence",
            "zone": "synthetic-strong",
            "confidence_tier": "HIGH",
            "is_spoof": True,
        }
    elif score >= LIKELY_SYNTHETIC_SCORE:
        return {
            "verdict": "SYNTHETIC",
            "label": "Likely Synthetic",
            "zone": "synthetic-likely",
            "confidence_tier": "MEDIUM",
            "is_spoof": True,
        }
    elif score > HUMAN_SCORE:
        is_spoof = score >= DECISION_SCORE
        return {
            "verdict": "SYNTHETIC" if is_spoof else "BONAFIDE",
            "label": "Ambiguous Signal",
            "zone": "ambiguous",
            "confidence_tier": "LOW",
            "is_spoof": is_spoof,
        }
    else:
        return {
            "verdict": "BONAFIDE",
            "label": "Likely Human",
            "zone": "human-likely",
            "confidence_tier": "HIGH",
            "is_spoof": False,
        }
