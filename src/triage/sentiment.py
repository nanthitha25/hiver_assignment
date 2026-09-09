"""Sentiment, customer frustration, and churn threat analyzer."""

import re
from typing import Tuple, List
from src.config import FRUSTRATION_THRESHOLD

# Threat keywords (legal action, regulatory reporting, brand churn)
LEGAL_CHURN_KEYWORDS = [
    "lawyer", "attorney", "lawsuit", "sue", "suing", "police", "fraud",
    "scam", "scammers", "stolen money", "stole my money", "better business bureau",
    "bbb", "consumer protection", "switching to android", "switching to samsung",
    "never buying apple again", "canceling everything",
]

# Anger and profanity markers
ANGER_KEYWORDS = [
    "fucking", "fuck", "shit", "bullshit", "garbage", "trash", "useless",
    "pathetic", "disaster", "horrible", "worst service", "idiots",
    "unacceptable", "furious", "outraged", "ripoff", "rip off",
]

LEGAL_PATTERN = re.compile(r"\b(" + "|".join(re.escape(k) for k in LEGAL_CHURN_KEYWORDS) + r")\b", re.IGNORECASE)
ANGER_PATTERN = re.compile(r"\b(" + "|".join(re.escape(k) for k in ANGER_KEYWORDS) + r")\b", re.IGNORECASE)
EXCLAMATION_PATTERN = re.compile(r"!{2,}|\?{2,}")


class SentimentAnalyzer:
    """Analyzes text for emotional distress, high customer frustration, and legal/churn risk."""

    def __init__(self, threshold: float = FRUSTRATION_THRESHOLD):
        self.threshold = threshold

    def compute_frustration(self, text: str) -> Tuple[float, List[str]]:
        """Calculates a normalized customer frustration score [0.0, 1.0] and triggered markers."""
        score = 0.0
        markers = []

        # 1. Check legal or churn threats (severe liability risk)
        legal_matches = LEGAL_PATTERN.findall(text)
        if legal_matches:
            unique_threats = list(set(m.lower() for m in legal_matches))
            score += 0.55 + min(0.20, (len(unique_threats) - 1) * 0.10)
            markers.append(f"LEGAL_OR_CHURN_THREAT: {unique_threats}")

        # 2. Check anger / profanity (+0.40)
        anger_matches = ANGER_PATTERN.findall(text)
        if anger_matches:
            score += 0.40
            markers.append(f"ANGER_KEYWORDS: {list(set(anger_matches))}")

        # 3. Check punctuation intensity (+0.15)
        if EXCLAMATION_PATTERN.search(text):
            score += 0.15
            markers.append("EXCESSIVE_PUNCTUATION")

        # 4. Check uppercase shouting (+0.15 if >40% uppercase on >20 char string)
        letters = [c for c in text if c.isalpha()]
        if len(letters) >= 15:
            upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if upper_ratio >= 0.40:
                score += 0.20
                markers.append(f"UPPERCASE_SHOUTING: {round(upper_ratio*100)}%")

        normalized_score = min(1.0, round(score, 2))
        return normalized_score, markers

    def is_severe_frustration(self, text: str) -> Tuple[bool, float, List[str]]:
        """Returns True if frustration exceeds the threshold."""
        score, markers = self.compute_frustration(text)
        return score >= self.threshold, score, markers
