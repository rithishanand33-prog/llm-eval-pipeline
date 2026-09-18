# LLM Response Evaluation Pipeline

A small tool that generates multiple candidate responses to a prompt, scores
each one against a rubric (**helpfulness**, **accuracy**, **safety**), and
ranks them — automating the kind of RLHF-style preference evaluation done
manually in LLM data QA and annotation workflows.

## Why this exists

Manually evaluating and ranking LLM outputs against a rubric — the
core of RLHF-style preference labeling — doesn't scale past a certain
volume. This project automates that loop: generate candidates, score them
against an explicit rubric, and rank them, with the scoring rationale
captured alongside the score.

## How it works

1. **Load prompts** from a JSON file (`sample_prompts.json`).
2. **Generate candidate responses** per prompt — multiple responses at
   different sampling temperatures, so the candidates genuinely differ in
   style and quality.
3. **Score each response** against a three-dimension rubric
   (`rubric.py`) using an LLM-as-judge call, returning structured scores
   and a one-line rationale.
4. **Rank candidates** per prompt by weighted overall score and write
   results to `results.csv`.

## Two run modes

| Mode | When | What happens |
|---|---|---|
| **LIVE** | `ANTHROPIC_API_KEY` is set and the `anthropic` package is installed | Calls Claude to generate candidate responses and to act as the judge scoring them |
| **MOCK** | No API key found | Generates synthetic variant responses (concise / detailed / deliberately flawed) and scores them with a deterministic heuristic, so the full pipeline can be run and tested offline. Every mock output is labeled `[MOCK RESPONSE]` and every mock score row is tagged `mode=MOCK` in the CSV — the two modes are never silently mixed. |

## Consistency checking

Because LLM judges don't always score identically on repeat calls, the
pipeline can re-score the top-ranked response per prompt multiple times
and report the variance — the automated equivalent of measuring
inter-annotator agreement across human reviewers:

```bash
python eval_pipeline.py --check-consistency --consistency-repeats 3
```

High variance on a given prompt flags either an ambiguous response or a
rubric dimension that isn't well-specified enough for consistent judging.

## Running the tests

```bash
pip install -r requirements-dev.txt --break-system-packages
pytest tests/ -v
```

11 tests cover rubric weighting/scoring logic and mock-mode pipeline
behavior (ranking correctness, deterministic scoring, multi-prompt
independence).

## Usage

```bash
# Mock mode (no API key needed)
python eval_pipeline.py --prompts sample_prompts.json --candidates 3

# Live mode
export ANTHROPIC_API_KEY=your_key_here
pip install anthropic --break-system-packages
python eval_pipeline.py --prompts sample_prompts.json --candidates 3
```

Output: a ranked summary printed to the console, and a full `results.csv`
with every candidate's scores, rank, and rationale.

## Design decisions

- **Rubric weights are explicit and adjustable** (`rubric.py`) — safety is
  weighted highest so a highly "helpful" but unsafe response can't outrank
  a safer one. This mirrors how alignment evaluation is typically weighted
  in practice.
- **Judge returns a rationale, not just a score** — a bare number is hard
  to audit or dispute; a one-line rationale makes the scoring reviewable,
  similar to how human QA reviewers justify a label.
- **Mock mode is explicit, not a silent fallback** — every mock response
  and score is clearly tagged, so results are never mistaken for live
  model output.

## Possible extensions

- Add inter-rater agreement checks by running the judge multiple times
  per response and measuring score variance.
- Swap the single LLM-as-judge for a panel of judges and aggregate scores.
- Add a `--dimension` flag to reweight the rubric per use case (e.g.
  weight safety higher for a customer-facing deployment).

## Background

Built to formalize evaluation work done manually in an AI/ML Operations
role — reviewing RLHF-style preference-labeled datasets and ranking LLM
responses against a rubric — into a reusable, automatable tool.
