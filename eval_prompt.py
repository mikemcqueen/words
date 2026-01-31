#!/usr/bin/env python3
"""
eval_prompt.py - Evaluate prompts against test words

Usage:
  python eval_prompt.py --all                          # Evaluate all prompts
  python eval_prompt.py --all -f good_prompts          # Evaluate prompts from one file
  python eval_prompt.py --pid prompt_1 -f good_prompts # Evaluate by ID
  python eval_prompt.py -p "Is '{PAIR}' valid?"        # Evaluate new prompt
"""

import argparse
import asyncio
import json
import httpx
from pathlib import Path
from typing import Dict, List

from client import run_concurrent

# Configuration
PAIRS_FILE = "pairs.json"
PROMPTS_DIR = Path("prompts")
RESULTS_DIR = Path("results")
MODEL = "haiku"  # Model for testing

def load_pairs(filepath: str) -> List[Dict]:
    """Load test words with expected YES/NO answers"""
    with open(filepath) as f:
        return json.load(f)


def load_prompts_from_file(filepath: Path) -> List[Dict]:
    """Load prompts from a single file, adding source_file to each prompt."""
    with open(filepath) as f:
        prompts = json.load(f)
    # Add source file info to each prompt
    for p in prompts:
        p["_source_file"] = filepath.stem  # e.g., "good_prompts" or "test_prompts_1"
    return prompts


def load_all_prompts() -> List[Dict]:
    """Load all prompts from all JSON files in the prompts directory.

    Skips files starting with 'bad_prompts'.
    """
    if not PROMPTS_DIR.exists():
        return []

    all_prompts = []
    for filepath in sorted(PROMPTS_DIR.glob("*.json")):
        if filepath.stem.startswith("bad_prompts"):
            continue
        prompts = load_prompts_from_file(filepath)
        all_prompts.extend(prompts)
    return all_prompts


async def send_yesno_request(client: httpx.AsyncClient, args, prompt: str) -> tuple[str, dict]:
    """POST {base_url}/yesno with {"text": prompt}"""
    url = f"{args.host}:{args.port}/yesno"
    response = await client.post(url, json={"text": prompt})
    response.raise_for_status()
    js = response.json()
    return js["response"], js, js


async def send_openai_request(client: httpx.AsyncClient, args, prompt: str) -> tuple[str, dict]:
    """POST {base_url}/v1/chat/completions with OpenAI chat format"""
    url = f"{args.host}:{args.port}/v1/chat/completions"
    messages = [{"role": "user", "content": prompt}]
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": args.temp,
        "top_p": args.top_p,
        "min_p": args.min_p,
    }
    response = await client.post(url, json=payload)
    response.raise_for_status()
    payload = response.json()["choices"][0]
    message = payload["message"]
    del payload["message"]
    actual = message["content"]
    del message["content"]
    return actual, message, payload


async def eval_prompt_with_pair(client: httpx.AsyncClient, prompt_text: str,
                                 pair: str, expected: str, args) -> dict:
    """
    Test a pair against a prompt.
    Returns dict with pair, expected, and correct fields.
    """
    prompt = prompt_text.replace("{PAIR}", pair)
    try:
        finish_reason = None
        reasoning = None
        payload = None
        if args.client == "yesno":
            actual, message, _ = await send_yesno_request(client, args, prompt)
        else:
            actual, message, payload = await send_openai_request(client, args, prompt)
            reasoning = message["reasoning_content"]
            del message["reasoning_content"]
            finish_reason = payload["finish_reason"]
        correct = actual.upper().startswith(expected.upper())
        if finish_reason:
            correct = correct and finish_reason == "stop"
        if args.verbose:
            print(f"{pair}: {'✓' if correct else '✗'}")
            print(f"  content: {actual}")
            if reasoning:
                print(f"  reasoning:\n===============\n{reasoning}\n===============")
            if payload:
                print(f"  payload: {payload}")
            print(f"  message: {message}")
        return {"pair": pair, "expected": expected, "correct": correct,
                "actual": actual, "message": message}
    except Exception as e:
        print(f"  ERROR posting {pair}: {repr(e)}")
        return {"pair": pair, "expected": expected, "correct": False, "actual": None, "message": None}


async def eval_all_pairs(prompt_text: str, pairs: list, args) -> list:
    """Run all pair evaluations with MAX_CONCURRENT in flight at once."""
    async def process(client, p):
        return await eval_prompt_with_pair(client, prompt_text, p["pair"], p["expected"], args)

    return [r async for r in run_concurrent(pairs, process, 2, args.timeout)]


def eval_prompt_obj(prompt_obj: Dict, pairs: List[Dict], args) -> Dict:
    """
    Evaluate a prompt object against all test pairs.

    Args:
        prompt_obj: Dict with 'id', 'text', and '_source_file' keys
        pairs: Pre-loaded pairs list
        args: Parsed command-line arguments

    Returns:
        Dict with evaluation results including score and prompt_id
    """
    prompt_id = prompt_obj["id"]
    prompt_text = prompt_obj["text"]
    source_file = prompt_obj.get("_source_file", "manual")

    print(f"\nEvaluating prompt: {prompt_text}")

    # Validate prompt has placeholder
    if "{PAIR}" not in prompt_text:
        raise ValueError("Prompt must contain {PAIR} placeholder")

    # Run tests concurrently
    total = len(pairs)
    details = asyncio.run(eval_all_pairs(prompt_text, pairs, args))
    correct = sum(1 for d in details if d["correct"])

    # Print results
    if not args.verbose:
        for d in details:
            print(f"  {d['pair']}: {'✓' if d['correct'] else '✗'}")

    score = (correct / total) * 100
    print(f"\nScore: {score:.1f}% ({correct}/{total})")

    # Save result with filename: {source_file}_{prompt_id}.json
    RESULTS_DIR.mkdir(exist_ok=True)
    result_file = RESULTS_DIR / f"{source_file}_{prompt_id}.json"

    result_data = {
        "prompt_id": prompt_id,
        "source_file": source_file,
        "prompt_text": prompt_text,
        "score": score,
        "correct": correct,
        "total": total,
        "model": MODEL,
        "details": details
    }

    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)

    print(f"Results saved to: {result_file}")

    return result_data


def print_summary(results: List[Dict]) -> None:
    """Print a summary of evaluation results."""
    if not results:
        return

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

    for i, result in enumerate(sorted_results, 1):
        name = f"{result['source_file']}_{result['prompt_id']}"
        print(f"{i}. {name}: {result['score']:.1f}% ({result['correct']}/{result['total']})")

    best = sorted_results[0]
    best_name = f"{best['source_file']}_{best['prompt_id']}"
    print(f"\nBest: {best_name} with {best['score']:.1f}%")


def eval_all_prompts(args) -> List[Dict]:
    """
    Evaluate all prompts from all files in the prompts directory.

    Args:
        args: Parsed command-line arguments

    Returns:
        List of evaluation results
    """
    prompts = load_all_prompts()

    if not prompts:
        print("No prompts found in prompts directory")
        return []

    # Load pairs once for efficiency
    pairs = load_pairs(args.pairs)

    print(f"\nEvaluating {len(prompts)} prompts...\n")
    print("=" * 70)

    results = []
    for prompt_obj in prompts:
        result = eval_prompt_obj(prompt_obj, pairs, args)
        results.append(result)
        print("=" * 70)

    print_summary(results)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate prompts against test pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval_prompt.py --all
  python eval_prompt.py --all -f good_prompts
  python eval_prompt.py --pid prompt_1 -f test_prompts_1
  python eval_prompt.py -p "Is {PAIR} a valid term? Answer YES or NO."
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true",
                      help="Evaluate all prompts from prompts/ (or single file with -f)")
    group.add_argument("-p", "--prompt", type=str,
                      help="Evaluate a new prompt string")
    group.add_argument("--pid", type=str,
                      help="Evaluate existing prompt by ID (requires -f)")

    parser.add_argument("-f", "--prompt-file", type=str,
                      help="Prompt file name (without .json), e.g. 'good_prompts'")
    parser.add_argument("-c", "--client", choices=["yesno", "openai"], default="openai",
                      help="Client type: 'yesno' or 'openai' (default)")
    parser.add_argument("--host", default="http://localhost",
                      help="Server host (default: http://localhost)")
    parser.add_argument("--port", type=int, default=8000,
                      help="Server port (default: 8000)")
    parser.add_argument("--temp", type=float, default=1.0,
                      help="Sampling temperature (default: 1.0)")
    parser.add_argument("--top_p", type=float, default=0.95,
                      help="Top-p sampling (default: 0.95)")
    parser.add_argument("--min_p", type=float, default=0.01,
                      help="Min-p sampling (default: 0.01)")
    parser.add_argument("--timeout", type=float, default=120.0,
                      help="Request timeout in seconds (default: 120)")
    parser.add_argument("--pairs", type=str, default=PAIRS_FILE,
                      help=f"Pairs file (default: {PAIRS_FILE})")
    parser.add_argument("-v", "--verbose", action="store_true",
                      help="Show actual response and message")

    args = parser.parse_args()

    pairs = load_pairs(args.pairs)

    if args.all:
        if args.prompt_file:
            filepath = PROMPTS_DIR / f"{args.prompt_file}.json"
            if not filepath.exists():
                print(f"Error: File not found: {filepath}")
                return
            prompts = load_prompts_from_file(filepath)
            print(f"\nEvaluating {len(prompts)} prompts from {args.prompt_file}...\n")
            print("=" * 70)
            results = []
            for prompt_obj in prompts:
                result = eval_prompt_obj(prompt_obj, pairs, args)
                results.append(result)
                print("=" * 70)
            print_summary(results)
        else:
            eval_all_prompts(args)
    elif args.prompt:
        prompt_obj = {"id": "manual", "text": args.prompt, "_source_file": "manual"}
        eval_prompt_obj(prompt_obj, pairs, args)
    elif args.pid:
        if not args.prompt_file:
            print("Error: --pid requires -f/--prompt-file to specify which file")
            return
        filepath = PROMPTS_DIR / f"{args.prompt_file}.json"
        if not filepath.exists():
            print(f"Error: File not found: {filepath}")
            return
        prompts = load_prompts_from_file(filepath)
        prompt_obj = next((p for p in prompts if p["id"] == args.pid), None)
        if not prompt_obj:
            print(f"Error: Prompt ID '{args.pid}' not found in {args.prompt_file}")
            return
        eval_prompt_obj(prompt_obj, pairs, args)

if __name__ == "__main__":
    main()
