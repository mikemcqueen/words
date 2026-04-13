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
import signal
import sys
from collections import namedtuple
from pathlib import Path

signal.signal(signal.SIGPIPE, signal.SIG_DFL)

from common import (load_expected_pairs, load_eval_results, parse_yesno_response,
                    discover_files_all, compute_stats, print_discovery_ranked, resolve_key)

TokenLabel = namedtuple("TokenLabel", ["token", "label"])

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



@method("top-token")
def method_top_token(logprobs: dict) -> dict:
    """Return dict mapping each logprobs key to TokenLabel(token, label=None)."""
    result = dict(logprobs=logprobs)
    for key, value in logprobs.items():
        result[key] = TokenLabel(token=extract_top_token(value), label=None)
    return result


def any_correct(result: dict) -> str:
    """Return 'correct' if any key has label='correct', else the first key's label."""
    for key in result['logprobs']:
        if result[key].label == "correct":
            return "correct"
    first_key = next(iter(result['logprobs']))
    return result[first_key].label


def apply_any_correct(eval_results: dict) -> None:
    """Set data["label"] = any_correct(result) for each scored pair."""
    for data in eval_results.values():
        if "result" in data:
            data["label"] = any_correct(data["result"])


def label_eval_results(eval_results: dict, expected: dict, method: str) -> None:
    """Fill in per-key TokenLabel labels for each scored pair, stored in data["result"].

    Label values: 'correct', 'fp', or 'fn'. Pairs not in expected are skipped.
    Call apply_any_correct() afterward to set data["label"] from the per-key results.
    """
    method_fn = METHODS[method]

    for pair, data in eval_results.items():
        lookup_key = pair.replace(",", " ")
        if lookup_key not in expected:
            continue

        exp = expected[lookup_key]
        result = method_fn(data.get("logprobs", {}))
        for key in result['logprobs']:
            tl = result[key]
            if tl.token == exp:
                label = "correct"
            elif tl.token == "YES":
                label = "fp"
            else:
                label = "fn"
            result[key] = tl._replace(label=label)
        data["result"] = result


def parse_args():
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
    parser.add_argument("--method", type=str, default="top-token",
                        choices=list(METHODS.keys()),
                        help="Scoring methodology (default: top-token)")
    parser.add_argument("-s", "--sort", default="score", metavar="FIELD",
                        help="sort field for directory mode: score (default), FP, FN")
    limit_group = parser.add_mutually_exclusive_group()
    limit_group.add_argument("--top-k", type=int, default=None, metavar="K",
                             help="limit display to top K results (directory mode only)")
    limit_group.add_argument("--min-score", type=float, default=None, metavar="M.N",
                             help="limit display to results with score >= M.N (directory mode only)")
    limit_group.add_argument("-k", "--keys", type=str, default=None, metavar="KEYS",
                             help="comma-separated discovery keys to score (directory mode only)")
    parser.add_argument("--pk", "--print-keys", dest="print_keys", action="store_true",
                        help="print displayed keys as a comma-separated list (directory mode only)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print per-pair results (single-file mode only)")
    parser.add_argument("--bad", action="store_true",
                        help="Only display incorrect results; implies --verbose (single-file mode only)")
    args = parser.parse_args()

    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top-k K must be > 0")
    if not args.input.exists():
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if not args.input.is_dir():
        if args.top_k is not None:
            parser.error("--top-k requires a directory input")
        if args.min_score is not None:
            parser.error("--min-score requires a directory input")
        if args.keys is not None:
            parser.error("--keys requires a directory input")
        if args.print_keys:
            parser.error("--print-keys requires a directory input")
    if args.bad:
        if args.input.is_dir():
            one_key = args.keys is not None and len([k for k in args.keys.split(',') if k.strip()]) == 1
            if not (args.top_k == 1 or one_key):
                parser.error("--bad in directory mode requires --top-k 1 or -k with exactly one key")
        args.verbose = True
    return args


def print_discovery_table(args):
    if args.keys is not None:
        keys = [k.strip() for k in args.keys.split(',')]
        expected = load_expected_pairs(args.pairs)
        files = {}
        for key in keys:
            path = resolve_key(args.input, key)
            files[key] = load_eval_results(path)
    else:
        files = discover_files_all(args.input)
        if not files:
            print("No .jsonl files found.", file=sys.stderr)
            sys.exit(1)
        expected = load_expected_pairs(args.pairs)
    for eval_results in files.values():
        label_eval_results(eval_results, expected, args.method)
        apply_any_correct(eval_results)

    rows = [(key, compute_stats(records)) for key, records in files.items()]
    sort_lc = args.sort.lower()
    if sort_lc == 'fp':
        rows.sort(key=lambda x: (x[1]['fp'], -x[1]['pct']))
    elif sort_lc == 'fn':
        rows.sort(key=lambda x: (x[1]['fn'], -x[1]['pct']))
    else:
        rows.sort(key=lambda x: x[1]['pct'], reverse=True)
    if args.top_k is not None:
        rows = rows[:args.top_k]
    elif args.min_score is not None:
        rows = [(k, s) for k, s in rows if s['pct'] >= args.min_score]
    files = {key: files[key] for key, _ in rows}

    print_discovery_ranked(files, args.sort)
    if args.print_keys:
        print(','.join(key for key, _ in rows))
    if args.bad:
        print_details(next(iter(files.values())), args)


def print_details(eval_results, args):
    """Print per-pair verbose table if requested. Returns count of skipped pairs."""
    def fmt_tok(tok: str | None) -> str:
        if tok == "YES":
            return "YES"
        if tok == "NO":
            return "NO "
        return "---"

    rows = []  # (pair, (result, mark, correct) | None) — None means skipped
    for pair, data in eval_results.items():
        if "label" not in data:
            rows.append((pair, None))
            continue
        result = data["result"]
        correct = data["label"] == "correct"
        mark = "\033[32m✓\033[0m" if correct else "\033[31m✗\033[0m"
        rows.append((pair, (result, mark, correct)))

    if args.verbose:
        display = [(p, d) for p, d in rows if not args.bad or (d is not None and not d[2])]
        max_pair = max((len(p) for p, _ in display), default=0)
        for pair, data in display:
            p = pair.ljust(max_pair)
            if data is None:
                print(f"  {p}  [skipped]")
            else:
                result, mark, _ = data
                tokens_str = "  ".join(
                    f"{k}={fmt_tok(result[k].token)}"
                    for k in result['logprobs']
                )
                print(f"  {p}  {tokens_str}  {mark}")

    return sum(1 for _, d in rows if d is None)


def main():
    args = parse_args()

    if args.input.is_dir():
        print_discovery_table(args)
        return

    expected = load_expected_pairs(args.pairs)
    eval_results = load_eval_results(args.input)
    label_eval_results(eval_results, expected, args.method)
    apply_any_correct(eval_results)
    skipped = print_details(eval_results, args)
    if skipped:
        print(f"Skipped {skipped} pairs not found in {args.pairs}")

    stats = compute_stats(eval_results)
    if stats['total'] == 0:
        print("No pairs scored.")
    else:
        print(f"Score: {stats['pct']:.1f}% ({stats['correct']}/{stats['total']})  FP: {stats['fp']}  FN: {stats['fn']}")


if __name__ == "__main__":
    main()
