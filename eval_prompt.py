#!/usr/bin/env python3
"""
eval_prompt.py - Evaluate prompts against test words

Usage:
  python eval_prompt.py --all -f good_prompts.json     # Evaluate all prompts from file
  python eval_prompt.py --pid prompt_1 -f good_prompts.json  # Evaluate by ID
  python eval_prompt.py -p "Is '{PAIR}' valid?"        # Evaluate new prompt
"""

import argparse
import asyncio
import json
import time

import httpx
from pathlib import Path
from typing import Dict, List

from client import add_inference_args, run_concurrent, get_inference_params, send_yesno_request, send_openai_request, query_model_id, resolve_host, get_server_name, auto_detect_max_concurrent
from common import load_prompts_from_file
# Configuration
PAIRS_FILE = "pairs.json"
RESULTS_DIR = Path("results")
MODEL = "haiku"  # Model for testing

def load_pairs(filepath: str) -> List[Dict]:
    """Load test words with expected YES/NO answers"""
    with open(filepath) as f:
        return json.load(f)


def load_expected_pairs(filepath: str, expected: str) -> List[Dict]:
    """Load pairs from a text file (one comma-separated pair per line), all with the same expected value."""
    with open(filepath) as f:
        return [{"pair": line.strip().replace(",", " "), "expected": expected}
                for line in f if line.strip()]


def load_pairs_from_args(args) -> List[Dict]:
    if args.yes_pairs:
        return load_expected_pairs(args.yes_pairs, "YES")
    elif args.no_pairs:
        return load_expected_pairs(args.no_pairs, "NO")
    elif args.any_pairs:
        return load_expected_pairs(args.any_pairs, "ANY")
    else:
        return load_pairs(args.pairs)



def parse_yesno_response(yesno: str) -> str | None:
    """Return normalized YES/NO for exact matches, else None."""
    if yesno:
        yesno = yesno.strip().upper()
        if yesno.startswith("YES"):
            return "YES"
        elif yesno.startswith("NO"):
            return "NO"
    return None


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
            response, message, _ = await send_yesno_request(client, args, prompt)
        else:
            response, message, payload = await send_openai_request(client, args, prompt, MODEL)
            reasoning = message.get("reasoning_content")
            message.pop("reasoning_content", None)
            finish_reason = payload["finish_reason"]
        seconds_elapsed = time.time() - start_time
        yesno = parse_yesno_response(response)
        expected = expected.upper()
        if expected == "ANY":
            correct = yesno in ("YES", "NO")
        else:
            correct = yesno == expected
        if finish_reason:
            correct = correct and finish_reason == "stop"
        if args.verbose:
            print(f"{pair}: {'✓' if correct else '✗'}")
            print(f"  content: {response}")
            if reasoning:
                print(f"  reasoning:\n===============\n{reasoning}\n===============")
            if payload:
                print(f"  payload: {payload}")
            print(f"  message: {message}")
        return {"pair": pair, "expected": expected, "correct": correct,
                "actual": response, "message": message, "reasoning": reasoning,
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

    return [r async for r in run_concurrent(pairs, process, args)]


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
    wall_start = time.time()
    details = asyncio.run(eval_all_pairs(prompt_text, pairs, args))
    wall_elapsed = time.time() - wall_start
    correct = sum(1 for d in details if d['correct'])
    score = (correct / total) * 100

    # Print results
    if not args.quiet or args.compact:
        compact_details = []
        for d in details:
            status = d["actual"] if d["actual"] else "None"
            status += ' '
            if d["correct"]:
                status += '✓'
            elif d["finish_reason"] == "stop":
                status += '✗'
            else:
                status += f"[{d['finish_reason']}]"

            line = f"{d['pair']}: {status}"
            if not args.quiet:
                print(f"  {line}")
            if args.compact:
                compact_details.append(line)
        if args.compact:
            details = compact_details

    print(f"\nScore: {score:.1f}% ({correct}/{total})")

    # Save results
    args.results_dir.mkdir(exist_ok=True)
    if args.yes_pairs:
        base_name = f"{Path(args.yes_pairs).name}"
    elif args.no_pairs:
        base_name = f"{Path(args.no_pairs).name}"
    elif args.any_pairs:
        base_name = f"{Path(args.any_pairs).name}"
    else:
        base_name = f"{Path(args.pairs).stem}"
    base_name += f"_{source_file}_{prompt_id}"
    server_name = get_server_name(args.host)
    if server_name:
        base_name += f"_{server_name}"
    if args.system_prompt_filename:
        base_name += f"_{Path(args.system_prompt_filename).stem}"
    #base_name += f"_mc{args.max_concurrent}"
    if args.tag:
        base_name += f".{args.tag}"
    result_file = args.results_dir / f"{base_name}.json"

    result_data = {
        "prompt_id": prompt_id,
        "source_file": source_file,
        "prompt_text": prompt_text,
        "score": score,
        "correct": correct,
        "total": total,
        "system_prompt": args.system_prompt_filename,
        "model": args.model_id,
        "max_concurrent": args.max_concurrent,
        "seconds_elapsed": wall_elapsed,
        "inference_params": get_inference_params(args),
        "results": details
    }

    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)

    print(f"Results saved to: {result_file}")

    return result_data


def print_summary(results: List[Dict]) -> None:
    """Print a summary of evaluation results."""
    if not results:
        return

    print("SUMMARY")

    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

    for i, result in enumerate(sorted_results, 1):
        name = f"{result['source_file']}_{result['prompt_id']}"
        print(f"{i}. {name}: {result['score']:.1f}% ({result['correct']}/{result['total']})")

    best = sorted_results[0]
    best_name = f"{best['source_file']}_{best['prompt_id']}"
    print(f"\nBest: {best_name} with {best['score']:.1f}%")



def main():
    parser = argparse.ArgumentParser(
        description="Evaluate prompts against test pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval_prompt.py --all -f good_prompts.json
  python eval_prompt.py --pid prompt_1 -f test_prompts_1.jsonl
  python eval_prompt.py -p "Is {PAIR} a valid term? Answer YES or NO."
        """
    )

    parser.add_argument("--all", action="store_true",
                      help="Evaluate all prompts from the file specified with -f")
    parser.add_argument("-p", "--prompt", type=str, help="Evaluate a new prompt string")
    parser.add_argument("--pid", type=str, help="Evaluate existing prompt by ID (requires -f)")

    parser.add_argument("--key", type=str, help="API key")
    parser.add_argument("-f", "--prompt-file", type=str,
                      help="Prompt file path (.json array or .jsonl)")
    parser.add_argument("-s", "--system-prompt", type=str,
                      help="System prompt file (optional)")
    parser.add_argument("-c", "--client", choices=["yesno", "openai"], default="openai",
                      help="Client type: 'yesno' or 'openai' (default)")
    parser.add_argument("--host", default="localhost",
                      help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000,
                      help="Server port (default: 8000)")
    add_inference_args(parser)
    parser.add_argument("--timeout", type=float, default=300.0,
                      help="Request timeout in seconds (default: 300)")
    pairs_group = parser.add_mutually_exclusive_group()
    pairs_group.add_argument("--pairs", type=str,
                      help=f"Pairs JSON file with expected values (default: {PAIRS_FILE})")
    pairs_group.add_argument("--yes-pairs", type=str,
                      help="Pairs text file (one per line) — all expected YES")
    pairs_group.add_argument("--no-pairs", type=str,
                      help="Pairs text file (one per line) — all expected NO")
    pairs_group.add_argument("--any-pairs", type=str,
                      help="Pairs text file (one per line) — accept either YES or NO, but not None")
    parser.add_argument("-v", "--verbose", action="store_true",
                      help="Show actual response and message")
    parser.add_argument("-q", "--quiet", action="store_true",
                      help="Suppress REQUEST START/END messages and per-pair details")
    parser.add_argument("--compact", action="store_true",
                      help="Save details as compact strings instead of full objects")
    parser.add_argument("--max-concurrent", "--mc", type=int, default=1,
                      help="Max concurrent requests (default: 1)")
    parser.add_argument("--tag", type=str,
                      help="Tag to append to result filename")
    parser.add_argument("-r", "--results-dir", type=Path, default=RESULTS_DIR,
                      help=f"Directory for result files (default: {RESULTS_DIR})")

    args = parser.parse_args()

    if args.prompt_file:
        if not (args.pid or args.all):
            parser.error("--prompt-file requires --pid or --all")
    elif args.prompt:
        if args.pid or args.all:
            parser.error("--pid and --all not allowed with --prompt")
    else:
        parser.error("either -f/--prompt-file or -p/--prompt is required")

    if not args.pairs and not args.yes_pairs and not args.no_pairs and not args.any_pairs:
        args.pairs = PAIRS_FILE

    if args.system_prompt:
        p = Path(args.system_prompt)
        if not p.is_file():
            print(f"Error: system prompt file not found: {args.system_prompt}", file=sys.stderr)
            sys.exit(1)
        args.system_prompt_filename = p.name
        args.system_prompt = p.read_text().strip()
    else:
        args.system_prompt_filename = None

    args.host = resolve_host(args.host)
    auto_detect_max_concurrent(args)

    args.model_id = query_model_id(args.host, args.port, args.key)

    pairs = load_pairs_from_args(args)
    if not pairs:
        print("Error: no pairs loaded")
        return

    if args.all:
        filepath = Path(args.prompt_file)
        if not filepath.exists():
            print(f"Error: File not found: {filepath}")
            return
        prompts = load_prompts_from_file(filepath)
        print(f"\nEvaluating {len(prompts)} prompts from {args.prompt_file}...\n")
        results = []
        for prompt_obj in prompts:
            result = eval_prompt_obj(prompt_obj, pairs, args)
            results.append(result)
        print_summary(results)
    elif args.prompt:
        prompt_obj = {"id": "manual", "text": args.prompt, "_source_file": "manual"}
        eval_prompt_obj(prompt_obj, pairs, args)
    elif args.pid:
        filepath = Path(args.prompt_file)
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
