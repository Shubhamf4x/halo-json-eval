import argparse
import json
import os
import re
from typing import Dict, Any
from pydantic import BaseModel, ValidationError
from openai import OpenAI


class BookingSchema(BaseModel):
    name: str
    destination: str
    tickets: int
    date: str  # YYYY-MM-DD
    max_budget: int


SYSTEM_PROMPT = (
    "Extract booking details into a single valid JSON object with exact keys: "
    "name (str), destination (str), tickets (int), date (YYYY-MM-DD), max_budget (int). "
    "Return ONLY raw JSON. No markdown, no explanation."
)


def extract_json(text: str) -> str:
    """Strip markdown fences if a model wraps output in ```json ... ```."""
    cleaned = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    return match.group(1) if match else cleaned


def score_prediction(raw_output: str, expected: Dict[str, Any]) -> Dict[str, bool]:
    """Programmatic verifier (reusable as a GRPO reward signal)."""
    try:
        parsed = json.loads(extract_json(raw_output))
        validated = BookingSchema(**parsed).model_dump()
        exact_match = validated == expected
        return {"valid_json": True, "schema_pass": True, "exact_match": exact_match}
    except json.JSONDecodeError:
        return {"valid_json": False, "schema_pass": False, "exact_match": False}
    except ValidationError:
        return {"valid_json": True, "schema_pass": False, "exact_match": False}


def run_eval(model: str, base_url: str, dataset_path: str, dry_run: bool = False):
    client = None if dry_run else OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        base_url=base_url,
    )

    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    totals = {"valid_json": 0, "schema_pass": 0, "exact_match": 0}

    for case in cases:
        if dry_run:
            raw_output = json.dumps(case["expected"])
        else:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": case["input"]},
                ],
            )
            raw_output = resp.choices[0].message.content or ""

        res = score_prediction(raw_output, case["expected"])
        for k in totals:
            totals[k] += int(res[k])
        print(f"[Case {case['id']}] valid_json={res['valid_json']} schema={res['schema_pass']} exact={res['exact_match']}")

    n = len(cases)
    print("\n--- Eval Summary ---")
    print(f"Model:        {model}")
    print(f"Total Cases:  {n}")
    print(f"Valid JSON:   {totals['valid_json']}/{n} ({totals['valid_json']/n:.1%})")
    print(f"Schema Pass:  {totals['schema_pass']}/{n} ({totals['schema_pass']/n:.1%})")
    print(f"Exact Match:  {totals['exact_match']}/{n} ({totals['exact_match']/n:.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structured JSON extraction eval harness for Halo post-training.")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--dataset", default="dataset.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="Verify pipeline without calling an API.")
    args = parser.parse_args()

    run_eval(args.model, args.base_url, args.dataset, args.dry_run)