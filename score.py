#!/usr/bin/env python3
"""
score.py - Score evalpair JSONL output against expected values.

Usage:
  python score.py results/foo.jsonl --pairs pairs.json
  python score.py results/foo.jsonl --pairs pairs.json --method any-yes -v
  python score.py results/            --pairs pairs.json          (discover all .jsonl; ranked table)
"""

import argparse
import json
import sys
from collections import namedtuple
from pathlib import Path

from common import (load_expected_pairs, load_eval_results, parse_yesno_response,
                    discover_files_all, compute_stats, print_discovery_ranked)

ScoreResult = namedtuple("ScoreResult", ["score", "correct", "total", "fp", "fn"])

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
def method_any_yes(logprobs: dict) -> str | None:
    """YES if fwd=YES or rvs=YES, else NO if either is NO, else None."""
    fwd = extract_top_token(logprobs.get("fwd", []))
    rvs = extract_top_token(logprobs.get("rvs", []))
    if fwd == "YES" or rvs == "YES":
        return "YES"
    if fwd == "NO" or rvs == "NO":
        return "NO"
    return None


def label_eval_results(eval_results: dict, expected: dict, method: str) -> ScoreResult:
    """Score eval_results against expected, adding a 'label' field to each matched pair.

    Label values: 'correct', 'fp', or 'fn'. Pairs not in expected are skipped (no label added).
    Returns a ScoreResult with running totals.
    """
    method_fn = METHODS[method]
    correct = 0
    total = 0
    fp = 0
    fn = 0

    for pair, data in eval_results.items():
        lookup_key = pair.replace(",", " ")
        if lookup_key not in expected:
            continue

        exp = expected[lookup_key]
        actual = method_fn(data.get("logprobs", {}))
        is_correct = actual == exp

        data["actual"] = actual

        total += 1
        if is_correct:
            correct += 1
            data["label"] = "correct"
        elif actual == "YES":
            fp += 1
            data["label"] = "fp"
        elif actual == "NO" or actual is None:
            fn += 1
            data["label"] = "fn"

    score = (correct / total * 100) if total > 0 else 0.0
    return ScoreResult(score=score, correct=correct, total=total, fp=fp, fn=fn)


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
    parser.add_argument("input", type=Path, help="evalpair JSONL file to score, or a directory")
    parser.add_argument("--pairs", type=str, required=True,
                        help="Pairs JSON file with expected values")
    parser.add_argument("--method", type=str, default="any-yes",
                        choices=list(METHODS.keys()),
                        help="Scoring methodology (default: any-yes)")
    parser.add_argument("-s", "--sort", default="score", metavar="FIELD",
                        help="sort field for directory mode: score (default), FP, FN")
    parser.add_argument("--pk", "--print-keys", dest="print_keys", type=int, default=None,
                        metavar="N", help="print top N keys as a comma-separated list (directory mode only)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print per-pair results (single-file mode only)")
    args = parser.parse_args()

    if args.print_keys is not None:
        if args.print_keys <= 0:
            parser.error("--print-keys N must be > 0")

    if not args.input.exists():
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.input.is_dir():
        files = discover_files_all(args.input)
        if not files:
            print("No .jsonl files found.", file=sys.stderr)
            sys.exit(1)
        expected = load_expected_pairs(args.pairs)
        for eval_results in files.values():
            label_eval_results(eval_results, expected, args.method)
        print_discovery_ranked(files, args.sort)
        if args.print_keys is not None:
            rows = [(key, compute_stats(records)) for key, records in files.items()]
            sort_lc = args.sort.lower()
            if sort_lc == 'fp':
                rows.sort(key=lambda x: (x[1]['fp'], -x[1]['pct']))
            elif sort_lc == 'fn':
                rows.sort(key=lambda x: (x[1]['fn'], -x[1]['pct']))
            else:
                rows.sort(key=lambda x: x[1]['pct'], reverse=True)
            print(','.join(key for key, _ in rows[:args.print_keys]))
        return

    if args.print_keys is not None:
        parser.error("--print-keys requires a directory input")

    expected = load_expected_pairs(args.pairs)
    eval_results = load_eval_results(args.input)
    sr = label_eval_results(eval_results, expected, args.method)

    def fmt_tok(tok: str | None) -> str:
        if tok == "YES":
            return "YES"
        if tok == "NO":
            return "NO "
        return "---"

    rows = []  # (pair, row_str | None) — None means skipped

    for pair, data in eval_results.items():
        if "label" not in data:
            rows.append((pair, None))
            continue
        logprobs = data.get("logprobs", {})
        fwd = extract_top_token(logprobs.get("fwd", []))
        rvs = extract_top_token(logprobs.get("rvs", []))
        mark = "\033[32m✓\033[0m" if data["label"] == "correct" else "\033[31m✗\033[0m"
        rows.append((pair, (fwd, rvs, data["actual"], mark)))

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
    if skipped:
        print(f"Skipped {skipped} pairs not found in {args.pairs}")

    if sr.total == 0:
        print("No pairs scored.")
        sys.exit(1)

    print(f"Score: {sr.score:.1f}% ({sr.correct}/{sr.total})  FP: {sr.fp}  FN: {sr.fn}")


if __name__ == "__main__":
    main()
