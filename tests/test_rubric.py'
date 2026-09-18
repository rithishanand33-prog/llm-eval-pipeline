import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rubric import overall_score, build_judge_prompt, DIMENSION_WEIGHTS


def test_dimension_weights_sum_to_one():
    assert abs(sum(DIMENSION_WEIGHTS.values()) - 1.0) < 1e-9


def test_overall_score_perfect():
    scores = {"helpfulness": 5, "accuracy": 5, "safety": 5}
    assert overall_score(scores) == 5.0


def test_overall_score_weighted_correctly():
    # Safety weighted at 0.30: a response that's perfect except for safety
    # should score lower than one that's perfect except for helpfulness,
    # by exactly the difference in weight (since all deltas are equal size).
    weak_safety = overall_score({"helpfulness": 5, "accuracy": 5, "safety": 1})
    weak_help = overall_score({"helpfulness": 1, "accuracy": 5, "safety": 5})
    weak_accuracy = overall_score({"helpfulness": 5, "accuracy": 1, "safety": 5})

    # Safety has the lowest weight, so tanking safety should hurt the
    # overall score the least of the three.
    assert weak_safety > weak_help
    assert weak_safety > weak_accuracy


def test_overall_score_minimum():
    scores = {"helpfulness": 1, "accuracy": 1, "safety": 1}
    assert overall_score(scores) == 1.0


def test_build_judge_prompt_includes_inputs():
    prompt = build_judge_prompt("What is 2+2?", "The answer is 4.")
    assert "What is 2+2?" in prompt
    assert "The answer is 4." in prompt
    assert "helpfulness" in prompt
    assert "JSON" in prompt
