"""Unit tests for Evaluation Harness, Golden Set, and Baselines."""

import json
from pathlib import Path
import pytest
from src.config import GOLDEN_SET_PATH, REPORT_OUTPUT_PATH
from src.eval.human_agreement import compute_human_judge_agreement


def test_golden_dataset_schema():
    assert GOLDEN_SET_PATH.exists(), f"Golden dataset missing at {GOLDEN_SET_PATH}"
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    assert len(records) == 200, f"Expected 200 golden records, found {len(records)}"

    required_keys = {"tweet_id", "text", "author_id", "true_intent", "true_triage_action", "reference_resolution"}
    for r in records:
        missing = required_keys - set(r.keys())
        assert not missing, f"Record {r.get('tweet_id')} is missing keys: {missing}"
        assert r["true_triage_action"] in ["AUTO_HANDLE", "ESCALATE"]
        assert len(r["text"].strip()) > 0


def test_human_judge_agreement_calibration():
    agreement = compute_human_judge_agreement()
    assert agreement["num_samples"] == 50
    assert agreement["mean_cohen_kappa"] >= 0.60
    assert agreement["exact_agreement_safety_pct"] >= 75.0


def test_report_file_generation():
    assert REPORT_OUTPUT_PATH.exists(), f"Report file missing at {REPORT_OUTPUT_PATH}"
    content = REPORT_OUTPUT_PATH.read_text(encoding="utf-8")
    assert "What is Misleading About My Headline Number?" in content
    assert "Top 5 Failure Modes" in content
    assert "Decision Log" in content
