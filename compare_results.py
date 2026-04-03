#!/usr/bin/env python3
"""Compare prompt result files and report individual + ensemble statistics.

Usage:
  compare_results.py <file_a> <file_b>   — compare two files explicitly
  compare_results.py <file>              — auto-discover sibling files by prompt id,
                                           report all baselines + pairwise ensembles
                                           sorted by # correct
"""

import json
import sys
from itertools import combinations
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
    print(f"  {label:<24s}  {stats['correct']:3d}/{stats['total']:3d} ({stats['pct']:5.1f}%)  "
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


ENSEMBLE_RULES = [
    ("OR",      lambda a, b: 'YES' if a == 'YES' or  b == 'YES' else 'NO'),
    ("AND",     lambda a, b: 'YES' if a == 'YES' and b == 'YES' else 'NO'),
    ("A=Y,B=N", lambda a, b: 'YES' if a == 'YES' and b == 'NO'  else 'NO'),
    ("A=N,B=Y", lambda a, b: 'YES' if a == 'NO'  and b == 'YES' else 'NO'),
]


def discover_files(seed_path, seed_pid):
    """Return {pid: (data, results)} for all p1, p2, ... files found sequentially."""
    path = Path(seed_path)
    marker = f'_{seed_pid}_'
    if marker not in path.name:
        print(f"Error: '{marker}' not found in filename '{path.name}'", file=sys.stderr)
        sys.exit(1)

    found = {}
    for n in range(1, 100):
        pid = f'p{n}'
        candidate = path.parent / path.name.replace(marker, f'_{pid}_', 1)
        if candidate.exists():
            found[pid] = parse_result_file(candidate)
        else:
            break
    return found


def run_two_file(path_a, path_b):
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

    print(f"\n{'':>26s}  {'correct':>9s}  {'FP':>5s}  {'FN':>5s}")
    print(f"  {'─'*64}")

    print("  Individual:")
    print_stats(f"A ({pid_a})", compute_stats(results_a))
    print_stats(f"B ({pid_b})", compute_stats(results_b))

    print("  Ensemble (common pairs):")
    for label, rule in ENSEMBLE_RULES:
        combined = apply_ensemble(results_a, results_b, rule)
        print_stats(label, compute_stats(combined))

    print()


def run_discovery(seed_path):
    data0, _ = parse_result_file(seed_path)
    seed_pid  = data0.get('prompt_id', '')
    if not seed_pid:
        print("Error: could not determine prompt_id from file", file=sys.stderr)
        sys.exit(1)

    files = discover_files(seed_path, seed_pid)
    if not files:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    pids = sorted(files.keys())
    print(f"\nPattern: {Path(seed_path).name.replace(f'_{seed_pid}_', '_pN_', 1)}")
    print(f"Found: {', '.join(pids)}\n")

    # Collect all rows: label -> stats
    rows = {}

    # Baselines
    for pid in pids:
        d, r = files[pid]
        rows[pid] = compute_stats(r)

    # Pairwise ensembles over every combination
    for pid_a, pid_b in combinations(pids, 2):
        _, results_a = files[pid_a]
        _, results_b = files[pid_b]
        pair_key = f"{pid_a},{pid_b}"
        for method, rule in ENSEMBLE_RULES:
            combined = apply_ensemble(results_a, results_b, rule)
            rows[f"{pair_key} {method}"] = compute_stats(combined)

    # Sort by correct count descending
    sorted_rows = sorted(rows.items(), key=lambda x: x[1]['correct'], reverse=True)

    print(f"  {'label':<24s}  {'correct':>9s}  {'FP':>5s}  {'FN':>5s}")
    print(f"  {'─'*64}")
    for label, stats in sorted_rows:
        print_stats(label, stats)
    print()


def main():
    if len(sys.argv) == 2:
        run_discovery(sys.argv[1])
    elif len(sys.argv) == 3:
        run_two_file(sys.argv[1], sys.argv[2])
    else:
        print(f"Usage: {sys.argv[0]} <file_a> [file_b]")
        sys.exit(1)


if __name__ == '__main__':
    main()
