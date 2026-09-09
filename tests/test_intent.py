"""Unit tests for Intent Classification Engine and Baselines."""

import time
import pytest
from src.models import AppleIntentEnum
from src.intent.baselines import TrivialMajorityClassifier, SimpleTfidfClassifier
from src.intent.classifier import SemanticCentroidClassifier


@pytest.fixture(scope="module")
def classifier():
    return SemanticCentroidClassifier()


def test_majority_baseline_classifier():
    clf = TrivialMajorityClassifier()
    res = clf.predict("My screen is completely shattered")
    assert res.primary_intent == AppleIntentEnum.OS_SOFTWARE_TROUBLESHOOTING
    assert res.confidence == 0.50


def test_simple_tfidf_classifier():
    clf = SimpleTfidfClassifier()
    res = clf.predict("My battery dies in two hours after charging")
    assert res.primary_intent in [AppleIntentEnum.HARDWARE_AND_BATTERY, AppleIntentEnum.OS_SOFTWARE_TROUBLESHOOTING]
    assert 0.0 <= res.confidence <= 1.0


def test_semantic_classifier_hardware_intent(classifier):
    text = "My iPhone 11 battery percentage drops from 100% to 20% in 30 minutes"
    res = classifier.predict(text)
    assert res.primary_intent == AppleIntentEnum.HARDWARE_AND_BATTERY
    assert res.confidence >= 0.60


def test_semantic_classifier_account_intent(classifier):
    text = "My Apple ID is locked for security reasons and I cannot reset my password"
    res = classifier.predict(text)
    assert res.primary_intent == AppleIntentEnum.ACCOUNT_BILLING_ICLOUD
    assert res.confidence >= 0.60


def test_semantic_classifier_software_intent(classifier):
    text = "Ever since the new iOS update my phone keeps rebooting into a boot loop"
    res = classifier.predict(text)
    assert res.primary_intent == AppleIntentEnum.OS_SOFTWARE_TROUBLESHOOTING
    assert res.confidence >= 0.60


def test_semantic_classifier_how_to_intent(classifier):
    text = "How do I transfer photos from my iPhone to a Windows PC?"
    res = classifier.predict(text)
    assert res.primary_intent == AppleIntentEnum.HOW_TO_CONFIGURATION
    assert res.confidence >= 0.60


def test_semantic_classifier_ambiguous_fallback(classifier):
    text = "xyz 123 random meaningless gibberish test message"
    res = classifier.predict(text)
    assert res.primary_intent == AppleIntentEnum.OUT_OF_SCOPE_AMBIGUOUS


def test_semantic_classifier_latency(classifier):
    # Warm-up call to initialize PyTorch runtime and caches
    classifier.predict("Warmup device")
    start = time.perf_counter()
    classifier.predict("My phone is frozen")
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    # Steady-state inference must be under 60ms on CPU
    assert elapsed_ms < 60.0
