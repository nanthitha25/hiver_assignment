"""Unit tests for Grounded Reply Drafting and Guardrails."""

import pytest
from src.drafting.retriever import HistoricalRetriever
from src.drafting.guardrails import OutputGuardrail
from src.drafting.generator import GroundedReplyGenerator


@pytest.fixture(scope="module")
def retriever():
    return HistoricalRetriever()


@pytest.fixture(scope="module")
def guardrail():
    return OutputGuardrail()


def test_retriever_returns_relevant_battery_resolutions(retriever):
    res = retriever.retrieve("My iPhone battery dies in two hours", intent="HARDWARE_AND_BATTERY", k=2)
    assert len(res.snippets) == 2
    assert res.max_similarity > 0.50
    assert any("battery" in s.lower() for s in res.snippets)


def test_retriever_returns_wifi_resolutions(retriever):
    res = retriever.retrieve("Wi-Fi keeps dropping on iOS update", intent="OS_SOFTWARE_TROUBLESHOOTING", k=2)
    assert len(res.snippets) == 2
    assert res.max_similarity > 0.50


def test_guardrail_length_pass_and_fail(guardrail):
    short_text = "We'd like to help. Check apple.co/batteryhealth."
    assert guardrail.check_length(short_text) is True

    long_text = "A" * 285
    assert guardrail.check_length(long_text) is False


def test_guardrail_catches_pii_solicitation(guardrail):
    bad_draft = "Please DM us your password and Apple ID security code so we can reset it."
    assert guardrail.check_pii_solicitation(bad_draft) is False

    clean_draft = "You can reset your password securely at iforgot.apple.com."
    assert guardrail.check_pii_solicitation(clean_draft) is True


def test_guardrail_url_whitelisting(guardrail):
    valid_text = "Visit apple.co/forcerestart or support.apple.com/iphone for help."
    passed, invalid = guardrail.validate_urls(valid_text)
    assert passed is True
    assert len(invalid) == 0

    phishing_text = "Click here to claim your free iPhone: http://free-apple-scam.xyz/claim"
    passed, invalid = guardrail.validate_urls(phishing_text)
    assert passed is False
    assert len(invalid) > 0


def test_generator_drafts_safe_reply(retriever):
    gen = GroundedReplyGenerator(provider="mock")
    ret_res = retriever.retrieve("How do I back up my iPhone?", intent="HOW_TO_CONFIGURATION")
    reply, passed, violations = gen.generate("How do I back up my iPhone?", "HOW_TO_CONFIGURATION", ret_res)

    assert reply is not None
    assert len(reply) <= 280
    assert passed is True
    assert len(violations) == 0
