"""Deterministic pattern matching rules for immediate safety and PII escalation."""

import re
from typing import Optional, Tuple, List
from src.models import EscalationReasonCode

# Regex patterns for sensitive PII
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_REGEX = re.compile(r"\b(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
CREDIT_CARD_REGEX = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
SSN_REGEX = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Regex patterns for physical hardware and safety risks
BATTERY_HAZARD_REGEX = re.compile(
    r"\b(battery|phone|device|macbook|ipad)\s+(is\s+)?(swollen|swelling|bulg\w+|expanded|puffed|smoking|smokes?|spark\w*|fire|exploded?|burned?)\b|"
    r"\b(swollen|bulging|expanding)\s+battery\b",
    re.IGNORECASE,
)

PHYSICAL_DAMAGE_REGEX = re.compile(
    r"\b(shattered|smashed)\s+(glass|screen|display)\b|"
    r"\b(dropped\s+(in|into)\s+(water|toilet|pool|ocean|bath)|liquid\s+damage|water\s+damage)\b",
    re.IGNORECASE,
)

# Regex patterns for explicit human agent requests
HUMAN_REQUEST_REGEX = re.compile(
    r"\b(speak|talk|connect|transfer|need|want)\s+(to|with)?\s+(a\s+)?(human|person|real person|agent|representative|advisor|operator|manager)\b|"
    r"\b(stop\s+(this\s+)?bot|not\s+a\s+bot|hate\s+bots)\b",
    re.IGNORECASE,
)


class RuleMatcher:
    """Evaluates text against high-precision deterministic regex rules."""

    @staticmethod
    def detect_pii(text: str) -> Tuple[bool, List[str]]:
        """Scans for sensitive personal identification information."""
        matched = []
        if EMAIL_REGEX.search(text):
            matched.append("EMAIL_ADDRESS_DETECTED")
        if PHONE_REGEX.search(text):
            matched.append("PHONE_NUMBER_DETECTED")
        if CREDIT_CARD_REGEX.search(text):
            matched.append("CREDIT_CARD_DETECTED")
        if SSN_REGEX.search(text):
            matched.append("SSN_DETECTED")
        return len(matched) > 0, matched

    @staticmethod
    def detect_hardware_hazard(text: str) -> Tuple[bool, List[str]]:
        """Scans for physical battery swelling, smoke, fire, or shattered hardware."""
        matched = []
        if BATTERY_HAZARD_REGEX.search(text):
            matched.append("BATTERY_THERMAL_HAZARD")
        if PHYSICAL_DAMAGE_REGEX.search(text):
            matched.append("PHYSICAL_DAMAGE_INSPECTION_REQUIRED")
        return len(matched) > 0, matched

    @staticmethod
    def detect_human_request(text: str) -> bool:
        """Scans for explicit user demands to speak with a human agent."""
        return bool(HUMAN_REQUEST_REGEX.search(text))
