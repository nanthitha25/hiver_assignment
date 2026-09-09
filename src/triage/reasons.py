"""Standardized escalation reason codes and explainable justification strings."""

from typing import Dict, Optional
from src.models import EscalationReasonCode

REASON_EXPLANATIONS: Dict[EscalationReasonCode, str] = {
    EscalationReasonCode.HARDWARE_PHYSICAL_DAMAGE: (
        "Physical damage or battery safety hazard detected (e.g., swelling battery, shattered glass, "
        "liquid immersion, smoke). Requires hands-on inspection and reservation at an Apple Store Genius Bar."
    ),
    EscalationReasonCode.PII_SECURITY_SENSITIVE: (
        "Sensitive customer PII or authentication credentials detected in public tweet (e.g. email, "
        "phone number, credit card). Escalated to secure private channel to safeguard account privacy."
    ),
    EscalationReasonCode.HIGH_FRUSTRATION_CHURN_RISK: (
        "Severe customer frustration, aggressive sentiment, or threat of legal/churn action detected. "
        "Routing directly to a Tier-2 human specialist for empathetic conflict de-escalation."
    ),
    EscalationReasonCode.HUMAN_AGENT_REQUESTED: (
        "Customer explicitly requested to interact with a human agent or representative. "
        "Honoring user preference by routing ticket to support queue."
    ),
    EscalationReasonCode.LOW_CONFIDENCE_AMBIGUOUS: (
        "Classification or retrieval confidence fell below the safety threshold (tau < 0.65). "
        "Failing closed to human support to avoid risk of generating hallucinated or inaccurate advice."
    ),
    EscalationReasonCode.GENERATION_GUARDRAIL_FAILED: (
        "Drafted response violated safety guardrails (length constraint, unverified external link, or "
        "PII solicitation). Withheld automated reply and routed to human review."
    ),
    EscalationReasonCode.SYSTEM_EXCEPTION_FAIL_CLOSED: (
        "An unexpected pipeline or upstream model exception occurred. System failed closed to human "
        "routing to maintain continuous service reliability."
    ),
}


def format_stated_reason(code: EscalationReasonCode, custom_detail: Optional[str] = None) -> str:
    """Returns a structured, human-readable stated explanation for the escalation decision."""
    base_explanation = REASON_EXPLANATIONS.get(code, "Ticket escalated to human specialist.")
    if custom_detail:
        return f"[{code.value}] {custom_detail}. {base_explanation}"
    return f"[{code.value}] {base_explanation}"
