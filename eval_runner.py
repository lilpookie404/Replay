"""Collect model responses for review without executing generated code."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from replay import request_strategy


def run_evals(cases_path: Path, output_path: Path) -> int:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Eval cases must be a nonempty JSON list.")
    seen_ids = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each eval case must be a JSON object.")
        for field in ("id", "idea"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError(f"Each eval case needs a nonempty {field} string.")
        if case["id"] in seen_ids:
            raise ValueError("Eval case IDs must be unique.")
        seen_ids.add(case["id"])

    load_dotenv(Path(__file__).with_name(".env"))
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Add it to your local .env file.")

    with output_path.open("x", encoding="utf-8") as output:
        with OpenAI(api_key=api_key, max_retries=0, timeout=120.0) as client:
            for case in cases:
                result = {
                    **case,
                    "response": None,
                    "response_status": None,
                    "error": None,
                    "review_status": "pending",
                }
                try:
                    response = request_strategy(client, case["idea"].strip())
                    result["response"] = response.output_text
                    result["response_status"] = response.status
                except OpenAIError as error:
                    # Raw API error messages can contain sensitive request details.
                    result["error"] = type(error).__name__
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
                print(
                    f"{case['id']}: {result['error'] or 'response saved'}; "
                    "semantic review pending"
                )
    return len(cases)


def main() -> int:
    folder = Path(__file__).resolve().parent
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_path = folder / f"eval_results_{timestamp}.jsonl"
    try:
        count = run_evals(folder / "eval_cases.json", output_path)
    except (OSError, ValueError) as error:
        print(f"Error: {error}")
        return 1
    except OpenAIError as error:
        print(f"Error: Could not initialize OpenAI ({type(error).__name__}).")
        return 1
    print(f"Saved {count} eval records to {output_path}")
    print("Semantic review is pending for this run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
