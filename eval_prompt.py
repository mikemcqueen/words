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

from client import MAX_CONCURRENT, run_concurrent

# Configuration
PAIRS_FILE = "pairs.json"
PROMPTS_DIR = Path("prompts")
RESULTS_DIR = Path("results")
MODEL = "haiku"  # Model for testing

def load_pairs() -> List[Dict]:
    """Load test words with expected YES/NO answers"""
    with open(PAIRS_FILE) as f:
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
    """Load all prompts from all JSON files in the prompts directory."""
    if not PROMPTS_DIR.exists():
        return []

    all_prompts = []
    for filepath in sorted(PROMPTS_DIR.glob("*.json")):
        prompts = load_prompts_from_file(filepath)
        all_prompts.extend(prompts)
    return all_prompts

async def eval_prompt_with_pair(client: httpx.AsyncClient,
                                 prompt_text: str, pair: str, expected: str) -> dict:
    """
    Test a pair against a prompt.
    Returns dict with pair, expected, and correct fields.
    """
    prompt = prompt_text.replace("{PAIR}", pair)
    payload = {"text": prompt}
    try:
        response = await client.post("http://localhost/yesno", json=payload)
        response.raise_for_status()
        js = response.json()
        actual = js['response']
        return {"pair": pair, "expected": expected, "correct": actual == expected}
    except Exception as e:
        print(f"  ERROR posting {pair}: {e}")
        return {"pair": pair, "expected": expected, "correct": False}


async def eval_all_pairs(prompt_text: str, pairs: list) -> list:
    """Run all pair evaluations with MAX_CONCURRENT in flight at once."""
    async def process(client, p):
        return await eval_prompt_with_pair(client, prompt_text, p["pair"], p["expected"])

    return [r async for r in run_concurrent(pairs, process, MAX_CONCURRENT)]

def eval_prompt_obj(prompt_obj: Dict, pairs: List[Dict] = None) -> Dict:
    """
    Evaluate a prompt object against all test pairs.

    Args:
        prompt_obj: Dict with 'id', 'text', and '_source_file' keys
        pairs: Optional pre-loaded pairs list (for efficiency)

    Returns:
        Dict with evaluation results including score and prompt_id
    """
    prompt_id = prompt_obj["id"]
    prompt_text = prompt_obj["text"]
    source_file = prompt_obj.get("_source_file", "manual")

    print(f"\nEvaluating prompt: {prompt_text[:60]}...")

    # Validate prompt has placeholder
    if "{PAIR}" not in prompt_text:
        raise ValueError("Prompt must contain {PAIR} placeholder")

    # Load test words if not provided
    if pairs is None:
        pairs = load_pairs()

    # Run tests concurrently
    total = len(pairs)
    details = asyncio.run(eval_all_pairs(prompt_text, pairs))
    correct = sum(1 for d in details if d["correct"])

    # Print results
    for d in details:
        print(f"  {d['pair']}: {'✓' if d['correct'] else '✗'}")

    score = (correct / total) * 100
    print(f"\nScore: {score:.1f}% ({correct}/{total})")

    # Save result with filename: {source_file}_{prompt_id}.json
    RESULTS_DIR.mkdir(exist_ok=True)
    result_file = RESULTS_DIR / f"{source_file}_{prompt_id}.json"

    result_data = {
        "prompt_id": prompt_id,
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

def eval_all_prompts() -> List[Dict]:
    """
    Evaluate all prompts from all files in the prompts directory.

    Returns:
        List of evaluation results
    """
    prompts = load_all_prompts()

    if not prompts:
        print("No prompts found in prompts directory")
        return []

    # Load pairs once for efficiency
    pairs = load_pairs()

    print(f"\nEvaluating {len(prompts)} prompts...\n")
    print("=" * 70)

    results = []
    for prompt_obj in prompts:
        result = eval_prompt_obj(prompt_obj, pairs)
        results.append(result)
        print("=" * 70)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

    for i, result in enumerate(sorted_results, 1):
        print(f"{i}. {result['prompt_id']}: {result['score']:.1f}% ({result['correct']}/{result['total']})")

    if results:
        best = sorted_results[0]
        print(f"\nBest: {best['prompt_id']} with {best['score']:.1f}%")

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
  python eval_prompt.py -p "Is {PAIR} a valid term?"
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

    args = parser.parse_args()

    if args.all:
        if args.prompt_file:
            filepath = PROMPTS_DIR / f"{args.prompt_file}.json"
            if not filepath.exists():
                print(f"Error: File not found: {filepath}")
                return
            prompts = load_prompts_from_file(filepath)
            pairs = load_pairs()
            print(f"\nEvaluating {len(prompts)} prompts from {args.prompt_file}...\n")
            print("=" * 70)
            results = []
            for prompt_obj in prompts:
                result = eval_prompt_obj(prompt_obj, pairs)
                results.append(result)
                print("=" * 70)
        else:
            eval_all_prompts()
    elif args.prompt:
        prompt_obj = {"id": "manual", "text": args.prompt, "_source_file": "manual"}
        eval_prompt_obj(prompt_obj)
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
        eval_prompt_obj(prompt_obj)

if __name__ == "__main__":
    main()
