#!/usr/bin/env python3
"""
eval_prompt.py - Evaluate prompts against test words

Usage:
  python eval_prompt.py --all                    # Evaluate all prompts
  python eval_prompt.py -p "Is {WORD} valid?"   # Evaluate new prompt
  python eval_prompt.py --pid prompt_042         # Evaluate by ID
"""

import argparse
import json
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Configuration
PAIRS_FILE = "pairs.json"
PROMPTS_FILE = "prompts.json"
RESULTS_DIR = Path("results")
MODEL = "haiku"  # Model for testing

def load_pairs() -> List[Dict]:
    """Load test words with expected YES/NO answers"""
    with open(PAIRS_FILE) as f:
        return json.load(f)

def load_prompts() -> List[Dict]:
    """Load all prompts from prompts.json"""
    if not Path(PROMPTS_FILE).exists():
        return []
    with open(PROMPTS_FILE) as f:
        return json.load(f)

def save_prompts(prompts: List[Dict]):
    """Save prompts to prompts.json"""
    with open(PROMPTS_FILE, 'w') as f:
        json.dump(prompts, f, indent=2)

def get_next_prompt_id(prompts: List[Dict]) -> str:
    """Generate next sequential prompt ID"""
    if not prompts:
        return "prompt_001"
    
    # Extract numeric part from existing IDs
    max_num = max(
        int(p["id"].split("_")[1]) 
        for p in prompts 
        if p["id"].startswith("prompt_")
    )
    return f"prompt_{max_num + 1:03d}"

def eval_prompt_with_pair(prompt_text: str, pair: str, expected: str) -> bool:
    """
    Test a pair against a prompt.
    Returns True if response matches expected YES/NO.
    """
    prompt = prompt_text.replace("{PAIR}", pair)

    url = "http://localhost:8000/yesno"
    payload = {
        "text": prompt
    }

    try:
        # Sending the POST request
        response = requests.post(url, json=payload)
        response.raise_for_status()
    except Exception as e:
        print(f"  ERROR posting {prompt}: {e}")
        return False
        
    js = response.json()

    # Checking the results
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {js}")

    actual = js['response']
    return actual == expected

def eval_prompt(prompt: str, source: str = "manual") -> Dict:
    """
    Evaluate a prompt against all test words.
    Updates prompts.json and saves timestamped results.
    
    Args:
        prompt: The prompt string to evaluate (must contain {WORD})
        source: Where this prompt came from ("manual", "generated", "original")
    
    Returns:
        Dict with evaluation results including score and prompt_id
    """
    print(f"\nEvaluating prompt: {prompt[:60]}...")
    
    # Validate prompt has placeholder
    if "{PAIR}" not in prompt:
        raise ValueError("Prompt must contain {WORD} placeholder")
    
    # Load test words
    # TODO load once; move to global or pass through
    pairs = load_pairs()
    
    # Run tests
    correct = 0
    total = len(pairs)
    details = []
    
    for pair_data in pairs:
        pair = pair_data["pair"]
        expected = pair_data["expected"]
        
        is_correct = eval_prompt_with_pair(prompt, pair, expected)
        if is_correct:
            correct += 1
        
        details.append({
            "pair": pair,
            "expected": expected,
            "correct": is_correct
        })
        
        print(f"  {pair}: {'✓' if is_correct else '✗'}")
    
    score = (correct / total) * 100
    print(f"\nScore: {score:.1f}% ({correct}/{total})")
    
    # Update prompts.json
    prompts = load_prompts()
    
    # Check if this exact prompt already exists
    existing = next((p for p in prompts if p["text"] == prompt), None)
    
    if existing:
        prompt_id = existing["id"]
        existing["score"] = score
        existing["last_tested"] = datetime.now().isoformat()
        existing["correct"] = correct
        existing["total"] = total
        print(f"Updated existing prompt: {prompt_id}")
    else:
        prompt_id = get_next_prompt_id(prompts)
        new_prompt = {
            "id": prompt_id,
            "text": prompt,
            "score": score,
            "correct": correct,
            "total": total,
            "created": datetime.now().isoformat(),
            "last_tested": datetime.now().isoformat(),
            "source": source
        }
        prompts.append(new_prompt)
        print(f"Added new prompt: {prompt_id}")
    
    save_prompts(prompts)
    
    # Save timestamped result
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_file = RESULTS_DIR / f"eval_{prompt_id}_{timestamp}.json"
    
    result_data = {
        "prompt_id": prompt_id,
        "prompt_text": prompt,
        "score": score,
        "correct": correct,
        "total": total,
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "details": details
    }
    
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"Results saved to: {result_file}")
    
    return result_data

def eval_prompt_id(pid: str) -> Optional[Dict]:
    """
    Evaluate an existing prompt by its ID.
    Calls eval_prompt which handles all updates.
    
    Args:
        pid: Prompt ID (e.g., "prompt_001")
    
    Returns:
        Dict with evaluation results, or None if ID not found
    """
    prompts = load_prompts()
    prompt_obj = next((p for p in prompts if p["id"] == pid), None)
    
    if not prompt_obj:
        print(f"Error: Prompt ID '{pid}' not found")
        return None
    
    print(f"Found prompt {pid}: {prompt_obj['text'][:60]}...")
    return eval_prompt(prompt_obj["text"], source=prompt_obj.get("source", "unknown"))

def eval_all_prompts() -> List[Dict]:
    """
    Evaluate all prompts in prompts.json.
    Calls eval_prompt_id for each, which handles updates.
    
    Returns:
        List of evaluation results
    """
    prompts = load_prompts()
    
    if not prompts:
        print("No prompts found in prompts.json")
        return []
    
    print(f"\nEvaluating {len(prompts)} prompts...\n")
    print("=" * 70)
    
    results = []
    for prompt_obj in prompts:
        result = eval_prompt_id(prompt_obj["id"])
        if result:
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
  python eval_prompt.py -p "Is {PAIR} a valid term?"
  python eval_prompt.py --pid prompt_042
  python eval_prompt.py --all
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", 
                      help="Evaluate all prompts in prompts.json")
    group.add_argument("-p", "--prompt", type=str,
                      help="Evaluate a new prompt string")
    group.add_argument("--pid", type=str,
                      help="Evaluate existing prompt by ID")
    
    args = parser.parse_args()
    
    if args.all:
        eval_all_prompts()
    elif args.prompt:
        eval_prompt(args.prompt, source="manual")
    elif args.pid:
        eval_prompt_id(args.pid)

if __name__ == "__main__":
    main()
