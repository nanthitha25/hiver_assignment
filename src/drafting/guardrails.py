"""Output guardrails for drafted replies ensuring length, link safety, and privacy."""

import re
from typing import List, Tuple
from src.config import MAX_TWEET_CHARS

# Allowed official Apple domain prefixes and Twitter official link wrapper (t.co)
WHITELISTED_URL_PATTERN = re.compile(
    r"^https?://(apple\.co|support\.apple\.com|iforgot\.apple\.com|reportaproblem\.apple\.com|t\.co)/",
    re.IGNORECASE,
)

# Regex to detect dangerous solicitation of sensitive data in public tweets
PII_SOLICITATION_PATTERN = re.compile(
    r"(send|dm|give|reply with|share)\s+(us\s+)?(your\s+)?(password|passcode|credit card|cvv|security code|social security|ssn)",
    re.IGNORECASE,
)

URL_EXTRACTOR = re.compile(r"https?://\S+|apple\.co/\S+|reportaproblem\.apple\.com\S*|iforgot\.apple\.com\S*")


class OutputGuardrail:
    """Validates generated drafts against strict brand safety and formatting constraints."""

    def __init__(self, max_chars: int = MAX_TWEET_CHARS):
        self.max_chars = max_chars

    def check_length(self, text: str) -> bool:
        """Returns True if within Twitter length limit."""
        return len(text.strip()) <= self.max_chars

    def check_pii_solicitation(self, text: str) -> bool:
        """Returns True if no sensitive PII solicitation is detected."""
        return not bool(PII_SOLICITATION_PATTERN.search(text))

    def validate_urls(self, text: str) -> Tuple[bool, List[str]]:
        """Returns True if all included URLs belong to whitelisted Apple domains."""
        urls = URL_EXTRACTOR.findall(text)
        invalid_urls = []
        for url in urls:
            normalized = url if url.startswith("http") else f"https://{url}"
            if not WHITELISTED_URL_PATTERN.match(normalized):
                invalid_urls.append(url)
        return len(invalid_urls) == 0, invalid_urls

    def evaluate(self, text: str) -> Tuple[bool, List[str]]:
        """Evaluates all guardrail checks."""
        violations = []

        if not self.check_length(text):
            violations.append(f"LENGTH_EXCEEDED: Draft is {len(text)} characters (max {self.max_chars})")

        if not self.check_pii_solicitation(text):
            violations.append("PII_SOLICITATION: Draft requests sensitive credentials or passwords")

        url_ok, invalid_urls = self.validate_urls(text)
        if not url_ok:
            violations.append(f"UNAUTHORIZED_URL: Draft contains non-whitelisted URLs: {invalid_urls}")

        return len(violations) == 0, violations
