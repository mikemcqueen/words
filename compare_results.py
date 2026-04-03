#!/usr/bin/env python3
"""Compare two prompt result files and report individual + ensemble statistics."""

import json
import sys
from pathlib import Path


def parse_result_file(path):
    """Parse result file. Returns (metadata_dict, {pair: {answer, expected}})."""
    with open(path) as f:
        data = json.load(f)
    results = {}
    for entry in data['results']:
        # Format: "word1,word2: YES ✓" or "word1,word2: NO ✗"
        pair, rest = entry.split(': ', 1)
        parts = rest.split()
        answer = parts[0]   # YES or NO
        marker = parts[1]   # ✓ or ✗
        correct = marker == '\u2713'
        if correct:
            expected = answer
        else:
            expected = 'NO' if answer == 'YES' else 'YES'
        results[pair] = {'answer': answer, 'expected': expected}
    return data, results


def compute_stats(pair_results):
    tp = tn = fp = fn = 0
    for r in pair_results.values():
        a, e = r['answer'], r['expected']
        if   e == 'YES' and a == 'YES': tp += 1
        elif e == 'NO'  and a == 'NO':  tn += 1
        elif e == 'NO'  and a == 'YES': fp += 1
        elif e == 'YES' and a == 'NO':  fn += 1
    total   = tp + tn + fp + fn
    correct = tp + tn
    pct     = 100 * correct / total if total else 0.0
    return dict(correct=correct, total=total, pct=pct, tp=tp, tn=tn, fp=fp, fn=fn)


def print_stats(label, stats):
    print(f"  {label:<20s}  {stats['correct']:3d}/{stats['total']:3d} ({stats['pct']:5.1f}%)  "
          f"FP={stats['fp']:3d}  FN={stats['fn']:3d}")


def apply_ensemble(results_a, results_b, rule):
    """Apply pairwise rule(a_answer, b_answer) -> 'YES'/'NO' over common pairs."""
    combined = {}
    for pair in set(results_a) & set(results_b):
        a = results_a[pair]
        b = results_b[pair]
        if a['expected'] != b['expected']:
            print(f"  WARNING: expected mismatch for '{pair}': "
                  f"{a['expected']} vs {b['expected']}", file=sys.stderr)
        combined[pair] = {'answer': rule(a['answer'], b['answer']),
                          'expected': a['expected']}
    return combined


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <result_file_a> <result_file_b>")
        sys.exit(1)

    path_a, path_b = sys.argv[1], sys.argv[2]
    data_a, results_a = parse_result_file(path_a)
    data_b, results_b = parse_result_file(path_b)

    pid_a = data_a.get('prompt_id', '?')
    pid_b = data_b.get('prompt_id', '?')

    print(f"\nA: {Path(path_a).name}")
    print(f"   prompt={pid_a}  model={data_a.get('model', '?')}")
    print(f"B: {Path(path_b).name}")
    print(f"   prompt={pid_b}  model={data_b.get('model', '?')}")

    n_common = len(set(results_a) & set(results_b))
    if n_common < len(results_a) or n_common < len(results_b):
        print(f"\n  (A has {len(results_a)} pairs, B has {len(results_b)}, "
              f"{n_common} in common — ensemble uses common pairs only)")

    print(f"\n{'':>22s}  {'correct':>9s}  {'FP':>5s}  {'FN':>5s}")
    print(f"  {'─'*60}")

    print("  Individual:")
    print_stats(f"A ({pid_a})", compute_stats(results_a))
    print_stats(f"B ({pid_b})", compute_stats(results_b))

    print("  Ensemble (common pairs):")
    ensembles = [
        ("A OR B",      lambda a, b: 'YES' if a == 'YES' or  b == 'YES' else 'NO'),
        ("A AND B",     lambda a, b: 'YES' if a == 'YES' and b == 'YES' else 'NO'),
        ("A=YES, B=NO", lambda a, b: 'YES' if a == 'YES' and b == 'NO'  else 'NO'),
        ("A=NO, B=YES", lambda a, b: 'YES' if a == 'NO'  and b == 'YES' else 'NO'),
    ]
    for label, rule in ensembles:
        combined = apply_ensemble(results_a, results_b, rule)
        print_stats(label, compute_stats(combined))

    print()


if __name__ == '__main__':
    main()
