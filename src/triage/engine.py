"""Cascading Triage & Escalation Engine implementing fail-closed policy gates."""

from typing import List, Optional
from src.models import (
    TweetInput,
    IntentResult,
    RetrievalResult,
    TriageDecision,
    TriageAction,
    EscalationReasonCode,
    AppleIntentEnum,
)
from src.config import MIN_INTENT_CONFIDENCE, MIN_RETRIEVAL_SIMILARITY
from src.triage.rules import RuleMatcher
from src.triage.sentiment import SentimentAnalyzer
from src.triage.reasons import format_stated_reason


class TriageEngine:
    """Evaluates customer inquiries through cascading deterministic and probabilistic gates."""

    def __init__(
        self,
        min_intent_confidence: float = MIN_INTENT_CONFIDENCE,
        min_retrieval_similarity: float = MIN_RETRIEVAL_SIMILARITY,
    ):
        self.min_intent_confidence = min_intent_confidence
        self.min_retrieval_similarity = min_retrieval_similarity
        self.rule_matcher = RuleMatcher()
        self.sentiment_analyzer = SentimentAnalyzer()

    def evaluate(
        self,
        tweet: TweetInput,
        intent_res: IntentResult,
        rag_res: Optional[RetrievalResult] = None,
        drafted_reply: Optional[str] = None,
        guardrail_passed: bool = True,
        guardrail_violations: Optional[List[str]] = None,
    ) -> TriageDecision:
        """Executes cascading priority gates to determine AUTO_HANDLE vs ESCALATE."""
        text = tweet.text
        guardrail_violations = guardrail_violations or []

        # Gate 1: Physical hardware hazard (battery swelling, thermal hazard, liquid damage)
        is_hazard, hazard_rules = self.rule_matcher.detect_hardware_hazard(text)
        if is_hazard:
            return TriageDecision(
                action=TriageAction.ESCALATE,
                stated_reason=format_stated_reason(
                    EscalationReasonCode.HARDWARE_PHYSICAL_DAMAGE,
                    f"Triggered safety rules: {hazard_rules}"
                ),
                reason_code=EscalationReasonCode.HARDWARE_PHYSICAL_DAMAGE,
                risk_score=1.0,
                triggered_rules=hazard_rules,
            )

        # Gate 2: Sensitive PII in public tweet
        has_pii, pii_rules = self.rule_matcher.detect_pii(text)
        if has_pii:
            return TriageDecision(
                action=TriageAction.ESCALATE,
                stated_reason=format_stated_reason(
                    EscalationReasonCode.PII_SECURITY_SENSITIVE,
                    f"Customer posted sensitive PII: {pii_rules}"
                ),
                reason_code=EscalationReasonCode.PII_SECURITY_SENSITIVE,
                risk_score=0.95,
                triggered_rules=pii_rules,
            )

        # Gate 3: Explicit Human Agent Request
        if self.rule_matcher.detect_human_request(text):
            return TriageDecision(
                action=TriageAction.ESCALATE,
                stated_reason=format_stated_reason(
                    EscalationReasonCode.HUMAN_AGENT_REQUESTED,
                    "Customer explicitly asked to speak with a human support agent"
                ),
                reason_code=EscalationReasonCode.HUMAN_AGENT_REQUESTED,
                risk_score=0.75,
                triggered_rules=["HUMAN_REQUEST_REGEX"],
            )

        # Gate 4: High Frustration, Aggression, or Legal/Churn Threats
        is_frustrated, frustration_score, sentiment_markers = self.sentiment_analyzer.is_severe_frustration(text)
        if is_frustrated:
            return TriageDecision(
                action=TriageAction.ESCALATE,
                stated_reason=format_stated_reason(
                    EscalationReasonCode.HIGH_FRUSTRATION_CHURN_RISK,
                    f"Frustration score {frustration_score:.2f} exceeded threshold ({sentiment_markers})"
                ),
                reason_code=EscalationReasonCode.HIGH_FRUSTRATION_CHURN_RISK,
                risk_score=frustration_score,
                triggered_rules=sentiment_markers,
            )

        # Gate 5: Intent Uncertainty / Ambiguous Topic
        if intent_res.primary_intent == AppleIntentEnum.OUT_OF_SCOPE_AMBIGUOUS or intent_res.confidence < self.min_intent_confidence:
            return TriageDecision(
                action=TriageAction.ESCALATE,
                stated_reason=format_stated_reason(
                    EscalationReasonCode.LOW_CONFIDENCE_AMBIGUOUS,
                    f"Intent '{intent_res.primary_intent.value}' has low confidence ({intent_res.confidence:.2f} < {self.min_intent_confidence:.2f})"
                ),
                reason_code=EscalationReasonCode.LOW_CONFIDENCE_AMBIGUOUS,
                risk_score=0.70,
                triggered_rules=["INTENT_CONFIDENCE_THRESHOLD_UNMET"],
            )

        # Gate 6: Low Historical Grounding / Retrieval Similarity
        if rag_res and rag_res.max_similarity < self.min_retrieval_similarity:
            return TriageDecision(
                action=TriageAction.ESCALATE,
                stated_reason=format_stated_reason(
                    EscalationReasonCode.LOW_CONFIDENCE_AMBIGUOUS,
                    f"Historical grounding similarity ({rag_res.max_similarity:.2f} < {self.min_retrieval_similarity:.2f}) insufficient for auto-handling"
                ),
                reason_code=EscalationReasonCode.LOW_CONFIDENCE_AMBIGUOUS,
                risk_score=0.65,
                triggered_rules=["GROUNDING_SIMILARITY_THRESHOLD_UNMET"],
            )

        # Gate 7: Generation Guardrail Failure
        if not guardrail_passed:
            return TriageDecision(
                action=TriageAction.ESCALATE,
                stated_reason=format_stated_reason(
                    EscalationReasonCode.GENERATION_GUARDRAIL_FAILED,
                    f"Safety violations: {guardrail_violations}"
                ),
                reason_code=EscalationReasonCode.GENERATION_GUARDRAIL_FAILED,
                risk_score=0.85,
                triggered_rules=guardrail_violations,
            )

        # Gate 8: Safe Auto-Handle Clearance
        return TriageDecision(
            action=TriageAction.AUTO_HANDLE,
            stated_reason="High confidence standard resolution grounded in historical brand data",
            reason_code=None,
            risk_score=0.10,
            triggered_rules=[],
        )
