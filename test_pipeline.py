import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval_pipeline import (
    generate_responses_mock,
    score_response_mock,
    run_pipeline,
)


def test_generate_responses_mock_returns_n_responses():
    responses = generate_responses_mock("test prompt", 3)
    assert len(responses) == 3
    assert all("test prompt" in r for r in responses)


def test_generate_responses_mock_handles_n_greater_than_templates():
    # Only 3 templates exist; requesting 5 should still return 5 responses
    # by cycling through templates rather than crashing.
    responses = generate_responses_mock("test prompt", 5)
    assert len(responses) == 5


def test_score_response_mock_flags_flawed_variant_lower():
    good = score_response_mock("[MOCK RESPONSE - detailed variant] some text")
    bad = score_response_mock("[MOCK RESPONSE - flawed variant] some text")
    good_total = good["helpfulness"] + good["accuracy"] + good["safety"]
    bad_total = bad["helpfulness"] + bad["accuracy"] + bad["safety"]
    assert good_total > bad_total


def test_score_response_mock_is_deterministic():
    r1 = score_response_mock("[MOCK RESPONSE - concise variant] abc")
    r2 = score_response_mock("[MOCK RESPONSE - concise variant] abc")
    assert r1 == r2


def test_run_pipeline_mock_mode_ranks_within_each_prompt():
    prompts = [{"id": "t1", "prompt": "Test prompt one"}]
    results = run_pipeline(prompts, n_candidates=3, client=None)

    assert len(results) == 3
    ranks = sorted(r["rank"] for r in results)
    assert ranks == [1, 2, 3]

    # Rank 1 should have the highest (or tied-highest) overall_score
    by_rank = {r["rank"]: r["overall_score"] for r in results}
    assert by_rank[1] >= by_rank[2] >= by_rank[3]


def test_run_pipeline_mock_mode_multiple_prompts_ranked_independently():
    prompts = [
        {"id": "t1", "prompt": "First prompt"},
        {"id": "t2", "prompt": "Second prompt"},
    ]
    results = run_pipeline(prompts, n_candidates=3, client=None)
    assert len(results) == 6

    for pid in ("t1", "t2"):
        subset_ranks = sorted(r["rank"] for r in results if r["prompt_id"] == pid)
        assert subset_ranks == [1, 2, 3]
