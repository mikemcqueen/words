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
import time

import httpx
from pathlib import Path
from typing import Dict, List

from client import run_concurrent, get_inference_params, send_yesno_request, send_openai_request, query_model_id, resolve_host, get_server_name
from model import supports_thinking

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
        start_time = time.time()
        if args.client == "yesno":
            actual, message, _ = await send_yesno_request(client, args, prompt)
        else:
            actual, message, payload = await send_openai_request(client, args, prompt, MODEL)
            reasoning = message["reasoning_content"]
            del message["reasoning_content"]
            finish_reason = payload["finish_reason"]
        seconds_elapsed = time.time() - start_time
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
                "actual": actual, "message": message, "reasoning": reasoning,
                "finish_reason": finish_reason, "seconds_elapsed": seconds_elapsed}
    except Exception as e:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}]   ERROR posting {pair}: {repr(e)}")
        if "504" in str(e):
            await asyncio.sleep(5)
        return {"pair": pair, "expected": expected, "correct": False, "actual": None,
                "message": None, "reasoning": None, "finish_reason": repr(e),
                "seconds_elapsed": time.time() - start_time}


async def eval_all_pairs(prompt_text: str, pairs: list, args) -> list:
    """Run all pair evaluations with max_concurrent in flight at once."""
    async def process(client, p):
        return await eval_prompt_with_pair(client, prompt_text, p["pair"], p["expected"], args)

    return [r async for r in run_concurrent(pairs, process, args.max_concurrent, args.timeout)]


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
    correct = sum(1 for d in details if d['correct'])

    # Print results
    for d in details:
        status = d["actual"] if d["actual"] else "None"
        status += ' '
        if d["correct"]:
            status += '✓'
        elif d["finish_reason"] == "stop":
            status += '✗'
        else:
            status += f"[{d['finish_reason']}]"
            
        print(f"  {d['pair']}: {status}")

    score = (correct / total) * 100
    print(f"\nScore: {score:.1f}% ({correct}/{total})")

    # Save result with filename: {source_file}_{prompt_id}[_{server}].json
    RESULTS_DIR.mkdir(exist_ok=True)
    base_name = f"{source_file}_{prompt_id}"
    server_name = get_server_name(args.host)
    if server_name:
        base_name += f"_{server_name}"
    if args.tag:
        base_name += f"_{args.tag}"
    result_file = RESULTS_DIR / f"{base_name}.json"

    result_data = {
        "prompt_id": prompt_id,
        "source_file": source_file,
        "prompt_text": prompt_text,
        "score": score,
        "correct": correct,
        "total": total,
        "model": args.model_id,
        "inference_params": get_inference_params(args),
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


def make_prompts_file_path(prompt_file: str) -> Path:
    """Make a full path to a prompts file.

    - If no .json suffix, add it
    - If no directory specified, use PROMPTS_DIR
    """
    if not prompt_file.endswith(".json"):
        prompt_file += ".json"
    if '/' not in prompt_file:
        return PROMPTS_DIR / prompt_file
    return Path(prompt_file)


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
    group.add_argument("-p", "--prompt", type=str, help="Evaluate a new prompt string")
    group.add_argument("--pid", type=str, help="Evaluate existing prompt by ID (requires -f)")

    parser.add_argument("--key", type=str, help="API key")
    parser.add_argument("-f", "--prompt-file", type=str,
                      help="Prompt file name (without .json), e.g. 'good_prompts'")
    parser.add_argument("-s", "--system-prompt", type=str,
                      help="System prompt (optional)")
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
    parser.add_argument("--repeat-penalty", "--rp", type=float, default=1.0,
                      help="Repeat penalty (default: 1.0)")
    parser.add_argument("--repeat-last-n", "--rln", type=int, default=32,
                      help="Repeat last n tokens (default: 32)")
    parser.add_argument("--timeout", type=float, default=300.0,
                      help="Request timeout in seconds (default: 300)")
    parser.add_argument("--pairs", type=str, default=PAIRS_FILE,
                      help=f"Pairs file (default: {PAIRS_FILE})")
    parser.add_argument("--think", action="store_true",
                      help="Enable thinking output (for models that support it)")
    parser.add_argument("-v", "--verbose", action="store_true",
                      help="Show actual response and message")
    parser.add_argument("--max-concurrent", "--mc", type=int, default=1,
                      help="Max concurrent requests (default: 1)")
    parser.add_argument("--tag", type=str,
                      help="Tag to append to result filename")

    args = parser.parse_args()
    args.host = resolve_host(args.host)

    args.model_id = query_model_id(args.host, args.port, args.key)
    if args.think and not supports_thinking(args.model_id):
        print(f"Error: --think not supported for model '{args.model_id}'")
        return

    pairs = load_pairs(args.pairs)

    if args.all:
        if args.prompt_file:
            filepath = make_prompts_file_path(args.prompt_file)
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
        filepath = make_prompts_file_path(args.prompt_file)
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
