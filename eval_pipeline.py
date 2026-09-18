"""
LLM Response Evaluation Pipeline
=================================

Generates multiple candidate responses per prompt, scores each one against
a rubric (helpfulness / accuracy / safety), and ranks them -- automating
the kind of RLHF-style preference evaluation done manually in LLM data QA.

Two modes:
  - LIVE mode: if ANTHROPIC_API_KEY is set, calls Claude to generate
    candidate responses (at different temperatures) and to act as the
    judge that scores them against the rubric.
  - MOCK mode: if no API key is found, generates synthetic variant
    responses and scores them with a deterministic heuristic scorer.
    Every mock output is clearly labeled as mock data. This lets the
    pipeline be tested end-to-end without an API key.

Usage:
    python eval_pipeline.py --prompts sample_prompts.json --candidates 3
"""

import argparse
import csv
import json
import os
import random
import re
import statistics
import sys

from rubric import build_judge_prompt, overall_score

MODEL = "claude-sonnet-4-5"


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------

def get_client():
    """Return an Anthropic client if a key is available, else None (mock mode)."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
    except ImportError:
        print("anthropic package not installed; falling back to mock mode. "
              "Run `pip install anthropic --break-system-packages` to enable live mode.")
        return None
    return anthropic.Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# Response generation
# ---------------------------------------------------------------------------

def generate_responses_live(client, prompt: str, n: int) -> list[str]:
    responses = []
    temperatures = [0.2, 0.7, 1.0][:n] or [0.7] * n
    for temp in temperatures:
        msg = client.messages.create(
            model=MODEL,
            max_tokens=300,
            temperature=temp,
            messages=[{"role": "user", "content": prompt}],
        )
        responses.append(msg.content[0].text.strip())
    return responses


MOCK_VARIANT_TEMPLATES = [
    "[MOCK RESPONSE - concise variant] A short, direct answer to: '{prompt}'. "
    "This variant prioritizes brevity over depth.",
    "[MOCK RESPONSE - detailed variant] A thorough answer to: '{prompt}'. "
    "This variant includes more context, examples, and caveats than the concise one.",
    "[MOCK RESPONSE - flawed variant] A partially off-topic or slightly inaccurate "
    "answer to: '{prompt}', included to test whether the scorer catches quality issues.",
]


def generate_responses_mock(prompt: str, n: int) -> list[str]:
    templates = (MOCK_VARIANT_TEMPLATES * ((n // len(MOCK_VARIANT_TEMPLATES)) + 1))[:n]
    return [t.format(prompt=prompt) for t in templates]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_response_live(client, prompt: str, response: str) -> dict:
    judge_prompt = build_judge_prompt(prompt, response)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=200,
        temperature=0,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    text = msg.content[0].text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Judge did not return valid JSON: {text}")
    return json.loads(match.group(0))


def score_response_mock(response: str) -> dict:
    """Deterministic heuristic scorer used only when no API key is available.

    Not a real quality judgment -- it exists purely so the pipeline can be
    run and demoed end-to-end offline.
    """
    random.seed(len(response))  # deterministic given the same input
    if "flawed variant" in response:
        base = {"helpfulness": 2, "accuracy": 2, "safety": 4}
    elif "detailed variant" in response:
        base = {"helpfulness": 5, "accuracy": 4, "safety": 5}
    else:
        base = {"helpfulness": 3, "accuracy": 4, "safety": 5}
    base["rationale"] = "Heuristic mock score (no LLM judge available)."
    return base


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def check_scoring_consistency(client, prompt: str, response: str, repeats: int) -> dict:
    """Re-run the judge multiple times on the same response and measure
    score variance. High variance flags a response (or a rubric) where the
    judge isn't applying criteria consistently -- the automated analogue
    of computing inter-annotator agreement across human reviewers."""
    totals = []
    for _ in range(repeats):
        s = score_response_live(client, prompt, response)
        totals.append(s["helpfulness"] + s["accuracy"] + s["safety"])
    scores = totals
    return {
        "mean_total_score": round(statistics.mean(scores), 2),
        "stdev": round(statistics.pstdev(scores), 2) if len(scores) > 1 else 0.0,
    }


def run_pipeline(prompts: list[dict], n_candidates: int, client) -> list[dict]:
    mode = "LIVE" if client else "MOCK"
    results = []

    for item in prompts:
        pid, prompt = item["id"], item["prompt"]
        print(f"[{mode}] Generating {n_candidates} candidates for {pid}...")

        if client:
            responses = generate_responses_live(client, prompt, n_candidates)
        else:
            responses = generate_responses_mock(prompt, n_candidates)

        scored = []
        for idx, resp in enumerate(responses):
            if client:
                scores = score_response_live(client, prompt, resp)
            else:
                scores = score_response_mock(resp)

            entry = {
                "prompt_id": pid,
                "prompt": prompt,
                "candidate_index": idx,
                "response": resp,
                "helpfulness": scores["helpfulness"],
                "accuracy": scores["accuracy"],
                "safety": scores["safety"],
                "overall_score": overall_score(scores),
                "rationale": scores.get("rationale", ""),
                "mode": mode,
            }
            scored.append(entry)

        scored.sort(key=lambda r: r["overall_score"], reverse=True)
        for rank, entry in enumerate(scored, start=1):
            entry["rank"] = rank
        results.extend(scored)

    return results


def write_csv(results: list[dict], path: str):
    fieldnames = [
        "prompt_id", "rank", "candidate_index", "overall_score",
        "helpfulness", "accuracy", "safety", "rationale", "mode",
        "prompt", "response",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_summary(results: list[dict]):
    print("\n=== Ranking Summary ===")
    by_prompt = {}
    for r in results:
        by_prompt.setdefault(r["prompt_id"], []).append(r)

    for pid, entries in by_prompt.items():
        print(f"\nPrompt {pid}: {entries[0]['prompt'][:70]}...")
        for e in sorted(entries, key=lambda x: x["rank"]):
            print(f"  #{e['rank']} candidate {e['candidate_index']} "
                  f"-> overall {e['overall_score']} "
                  f"(help={e['helpfulness']} acc={e['accuracy']} safe={e['safety']})")


def main():
    parser = argparse.ArgumentParser(description="LLM Response Evaluation Pipeline")
    parser.add_argument("--prompts", default="sample_prompts.json",
                         help="Path to JSON file of prompts")
    parser.add_argument("--candidates", type=int, default=3,
                         help="Number of candidate responses to generate per prompt")
    parser.add_argument("--out", default="results.csv",
                         help="Output CSV path")
    parser.add_argument("--check-consistency", action="store_true",
                         help="LIVE mode only: re-score the top-ranked response per "
                              "prompt multiple times to measure judge score variance")
    parser.add_argument("--consistency-repeats", type=int, default=3,
                         help="Number of repeat scorings for --check-consistency")
    args = parser.parse_args()

    try:
        with open(args.prompts, "r", encoding="utf-8") as f:
            prompts = json.load(f)
    except FileNotFoundError:
        print(f"Error: prompts file not found at '{args.prompts}'", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: '{args.prompts}' is not valid JSON ({e})", file=sys.stderr)
        sys.exit(1)

    if not isinstance(prompts, list) or not prompts:
        print(f"Error: '{args.prompts}' must contain a non-empty JSON list of prompts",
              file=sys.stderr)
        sys.exit(1)

    for item in prompts:
        if "id" not in item or "prompt" not in item:
            print(f"Error: each prompt entry needs 'id' and 'prompt' fields, got: {item}",
                  file=sys.stderr)
            sys.exit(1)

    if args.candidates < 1:
        print("Error: --candidates must be at least 1", file=sys.stderr)
        sys.exit(1)

    client = get_client()
    if client is None:
        print("No ANTHROPIC_API_KEY found (or package missing) -- running in MOCK mode.\n"
              "Set ANTHROPIC_API_KEY and `pip install anthropic --break-system-packages` "
              "to run in LIVE mode against Claude.\n")

    results = run_pipeline(prompts, args.candidates, client)
    write_csv(results, args.out)
    print_summary(results)
    print(f"\nFull results written to {args.out}")

    if args.check_consistency:
        if not client:
            print("\n--check-consistency requires LIVE mode (ANTHROPIC_API_KEY not set) -- skipped.")
        else:
            print("\n=== Judge Consistency Check (top-ranked response per prompt) ===")
            top_per_prompt = {}
            for r in results:
                if r["rank"] == 1:
                    top_per_prompt[r["prompt_id"]] = r
            for pid, entry in top_per_prompt.items():
                stats = check_scoring_consistency(
                    client, entry["prompt"], entry["response"], args.consistency_repeats
                )
                print(f"  {pid}: mean total score {stats['mean_total_score']} "
                      f"(stdev {stats['stdev']} across {args.consistency_repeats} re-scorings)")


if __name__ == "__main__":
    main()
