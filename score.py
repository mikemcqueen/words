#!/usr/bin/env python3
"""
score.py - Score evalpair JSONL output against expected values.

Usage:
  python score.py results/foo.jsonl --pairs pairs.json
  python score.py results/foo.jsonl --pairs pairs.json --method any-yes -v
"""

import argparse
import json
import sys
from pathlib import Path

from common import parse_yesno_response

METHODS = {}

def method(name):
    def decorator(fn):
        METHODS[name] = fn
        return fn
    return decorator


def extract_top_token(logprobs: list) -> str | None:
    """Return normalized YES/NO/None from the first (highest-prob) entry in a logprobs list."""
    if not logprobs:
        return None
    top_token = next(iter(logprobs[0]))
    return parse_yesno_response(top_token)


@method("any-yes")
def method_any_yes(fwd: str | None, rvs: str | None) -> str | None:
    """YES if fwd=YES or rvs=YES, else NO if either is NO, else None."""
    if fwd == "YES" or rvs == "YES":
        return "YES"
    if fwd == "NO" or rvs == "NO":
        return "NO"
    return None



def load_expected_pairs(filepath: str) -> dict:
    """Load a pairs JSON file and return a {pair: expected} dict (space-separated keys)."""
    with open(filepath) as f:
        data = json.load(f)
    return {entry["pair"].replace(",", " "): entry["expected"].upper() for entry in data}


def score_record(record: dict, method_fn) -> dict:
    """Score a single evalpair record. Returns None if pair not in expected."""
    pair = record["pair"]
    logprobs = record.get("logprobs", {})
    fwd = extract_top_token(logprobs.get("fwd", []))
    rvs = extract_top_token(logprobs.get("rvs", []))
    actual = method_fn(fwd, rvs)
    return {"pair": pair, "fwd": fwd, "rvs": rvs, "actual": actual}


def main():
    parser = argparse.ArgumentParser(
        description="Score evalpair JSONL output against expected values",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python score.py results/foo.jsonl --pairs pairs.json
  python score.py results/foo.jsonl --pairs pairs.json --method any-yes -v
        """
    )
    parser.add_argument("input", type=Path, help="evalpair JSONL file to score")
    parser.add_argument("--pairs", type=str, required=True,
                        help="Pairs JSON file with expected values")
    parser.add_argument("--method", type=str, default="any-yes",
                        choices=list(METHODS.keys()),
                        help="Scoring methodology (default: any-yes)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print per-pair results")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    expected = load_expected_pairs(args.pairs)
    method_fn = METHODS[args.method]

    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Sort by seq so output order matches input order
    records.sort(key=lambda r: r.get("seq", 0))

    def fmt_tok(tok: str | None) -> str:
        if tok == "YES":
            return "YES"
        if tok == "NO":
            return "NO "
        return "---"

    correct = 0
    total = 0
    fp = 0
    fn = 0
    rows = []  # (pair, row_str | None) — None means skipped

    for record in records:
        pair = record["pair"]
        lookup_key = pair.replace(",", " ")
        if lookup_key not in expected:
            rows.append((pair, None))
            continue

        exp = expected[lookup_key]
        result = score_record(record, method_fn)
        actual = result["actual"]

        if exp == "ANY":
            is_correct = actual in ("YES", "NO")
        else:
            is_correct = actual == exp

        total += 1
        if is_correct:
            correct += 1
        elif actual == "YES":
            fp += 1
        elif actual == "NO" or actual is None:
            fn += 1

        mark = "\033[32m✓\033[0m" if is_correct else "\033[31m✗\033[0m"
        rows.append((pair, (result["fwd"], result["rvs"], actual, mark)))

    if args.verbose:
        max_pair = max((len(p) for p, _ in rows), default=0)
        for pair, data in rows:
            p = pair.ljust(max_pair)
            if data is None:
                print(f"  {p}  [skipped]")
            else:
                fwd, rvs, actual, mark = data
                print(f"  {p}  fwd={fmt_tok(fwd)}  rvs={fmt_tok(rvs)}  {fmt_tok(actual)}  {mark}")

    skipped = sum(1 for _, d in rows if d is None)
    if skipped and not args.verbose:
        print(f"Skipped {skipped} pairs not found in {args.pairs}")

    if total == 0:
        print("No pairs scored.")
        sys.exit(1)

    score = (correct / total) * 100
    print(f"Score: {score:.1f}% ({correct}/{total})  FP: {fp}  FN: {fn}")


if __name__ == "__main__":
    main()
