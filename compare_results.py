#!/usr/bin/env python3
"""Compare prompt result files and report statistics.

Usage:
  compare_results.py <file_a> <file_b>          — compare two files explicitly
  compare_results.py <file>                     — auto-discover sibling files by prompt id;
                                                   default: Fixed/New FP/FN + score per file
  compare_results.py -e [<file_a> <file_b>]     — ensemble mode: test all combinations
  compare_results.py -e -3 <file>               — ensemble mode with 3-way combos
"""

import argparse
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


def compute_pair_diff(results_1, results_2):
    """Compute Fixed/New FP/FN counts and score comparing results_1 (anchor) to results_2."""
    common = set(results_1) & set(results_2)
    fixed_fp = fixed_fn = new_fp = new_fn = 0
    for pair in common:
        r1, r2 = results_1[pair], results_2[pair]
        was_fp = r1['answer'] == 'YES' and r1['expected'] == 'NO'
        was_fn = r1['answer'] == 'NO'  and r1['expected'] == 'YES'
        is_fp  = r2['answer'] == 'YES' and r2['expected'] == 'NO'
        is_fn  = r2['answer'] == 'NO'  and r2['expected'] == 'YES'
        was_correct = not was_fp and not was_fn
        if was_fp and not is_fp:     fixed_fp += 1
        if was_fn and not is_fn:     fixed_fn += 1
        if was_correct and is_fp:    new_fp += 1
        if was_correct and is_fn:    new_fn += 1
    s1 = compute_stats(results_1)
    s2 = compute_stats(results_2)
    score = (fixed_fp + fixed_fn) - (new_fp + new_fn)
    return dict(fixed_fp=fixed_fp, fixed_fn=fixed_fn, new_fp=new_fp, new_fn=new_fn,
                score=score, s1=s1, s2=s2)


def apply_ensemble(results_a, results_b, rule):
    """Apply rule(a, b) -> 'YES'/'NO' over common pairs."""
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


def apply_ensemble_3(results_a, results_b, results_c, rule):
    """Apply rule(a, b, c) -> 'YES'/'NO' over common pairs."""
    combined = {}
    for pair in set(results_a) & set(results_b) & set(results_c):
        a, b, c = results_a[pair], results_b[pair], results_c[pair]
        expected = a['expected']
        for other in (b, c):
            if other['expected'] != expected:
                print(f"  WARNING: expected mismatch for '{pair}'", file=sys.stderr)
        combined[pair] = {'answer': rule(a['answer'], b['answer'], c['answer']),
                          'expected': expected}
    return combined


ENSEMBLE_RULES_2 = [
    ("OR",      lambda a, b: 'YES' if a == 'YES' or  b == 'YES' else 'NO'),
    ("AND",     lambda a, b: 'YES' if a == 'YES' and b == 'YES' else 'NO'),
    ("A=Y,B=N", lambda a, b: 'YES' if a == 'YES' and b == 'NO'  else 'NO'),
    ("A=N,B=Y", lambda a, b: 'YES' if a == 'NO'  and b == 'YES' else 'NO'),
]

ENSEMBLE_RULES_3 = [
    ("OR",       lambda a, b, c: 'YES' if 'YES' in (a, b, c)           else 'NO'),
    ("AND",      lambda a, b, c: 'YES' if a == b == c == 'YES'          else 'NO'),
    ("MAJORITY", lambda a, b, c: 'YES' if (a, b, c).count('YES') >= 2   else 'NO'),
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


# --- default (diff) output ---

def print_explicit_default(pid_a, results_a, pid_b, results_b):
    d = compute_pair_diff(results_a, results_b)
    s1, s2 = d['s1'], d['s2']
    print(f"\n{pid_a}:  FP: {s1['fp']}  FN: {s1['fn']}")
    print(f"{pid_b}:  FP: {s2['fp']}  FN: {s2['fn']}")
    print(f"\nFixedFP: {d['fixed_fp']}  FixedFN: {d['fixed_fn']}  "
          f"NewFP: {d['new_fp']}  NewFN: {d['new_fn']}")
    print(f"Score: {d['score']:+d}")
    print()


def print_discovery_default(seed_pid, results0, files, pids):
    rows = []
    for pid in pids:
        if pid == seed_pid:
            continue
        _, results_other = files[pid]
        d = compute_pair_diff(results0, results_other)
        rows.append((pid, d))
    rows.sort(key=lambda x: x[1]['score'], reverse=True)

    print(f"  {'label':<10s}  {'score':>6s}  {'FixFP':>6s}  {'FixFN':>6s}  "
          f"{'NewFP':>6s}  {'NewFN':>6s}  {'A corr':>8s}  {'B corr':>8s}")
    print(f"  {'─'*72}")
    for pid, d in rows:
        s1, s2 = d['s1'], d['s2']
        print(f"  {seed_pid}→{pid:<7s}  {d['score']:>+6d}  {d['fixed_fp']:>6d}  "
              f"{d['fixed_fn']:>6d}  {d['new_fp']:>6d}  {d['new_fn']:>6d}  "
              f"{s1['correct']:3d}/{s1['total']:3d}  {s2['correct']:3d}/{s2['total']:3d}")
    print()


# --- ensemble output ---

def print_explicit_ensemble(path_a, data_a, results_a, path_b, data_b, results_b):
    pid_a = data_a.get('prompt_id', '?')
    pid_b = data_b.get('prompt_id', '?')

    print(f"\nA: {Path(path_a).name}")
    print(f"   prompt={pid_a}  model={data_a.get('model', '?')}")
    print(f"B: {Path(path_b).name}")
    print(f"   prompt={pid_b}  model={data_b.get('model', '?')}")

    n_common = len(set(results_a) & set(results_b))
    if n_common < len(results_a) or n_common < len(results_b):
        print(f"\n  (A has {len(results_a)} pairs, B has {len(results_b)}, "
              f"{n_common} in common)")

    print(f"\n{'':>26s}  {'correct':>9s}  {'FP':>5s}  {'FN':>5s}")
    print(f"  {'─'*64}")
    print("  Individual:")
    print_stats(f"A ({pid_a})", compute_stats(results_a))
    print_stats(f"B ({pid_b})", compute_stats(results_b))
    print("  Ensemble (common pairs):")
    for label, rule in ENSEMBLE_RULES_2:
        combined = apply_ensemble(results_a, results_b, rule)
        print_stats(label, compute_stats(combined))
    print()


def print_discovery_ensemble(seed_pid, files, pids, three_way):
    rows = {}
    for pid in pids:
        _, r = files[pid]
        rows[pid] = compute_stats(r)

    for pid_a, pid_b in combinations(pids, 2):
        _, results_a = files[pid_a]
        _, results_b = files[pid_b]
        pair_key = f"{pid_a},{pid_b}"
        for method, rule in ENSEMBLE_RULES_2:
            combined = apply_ensemble(results_a, results_b, rule)
            rows[f"{pair_key} {method}"] = compute_stats(combined)

    if three_way:
        for pid_a, pid_b, pid_c in combinations(pids, 3):
            _, results_a = files[pid_a]
            _, results_b = files[pid_b]
            _, results_c = files[pid_c]
            triple_key = f"{pid_a},{pid_b},{pid_c}"
            for method, rule in ENSEMBLE_RULES_3:
                combined = apply_ensemble_3(results_a, results_b, results_c, rule)
                rows[f"{triple_key} {method}"] = compute_stats(combined)

    sorted_rows = sorted(rows.items(), key=lambda x: x[1]['correct'], reverse=True)
    print(f"  {'label':<24s}  {'correct':>9s}  {'FP':>5s}  {'FN':>5s}")
    print(f"  {'─'*64}")
    for label, stats in sorted_rows:
        print_stats(label, stats)
    print()


# --- top-level runners ---

def run_explicit(args):
    data_a, results_a = parse_result_file(args.files[0])
    data_b, results_b = parse_result_file(args.files[1])
    pid_a = data_a.get('prompt_id', '?')
    pid_b = data_b.get('prompt_id', '?')
    if not args.ensemble:
        print_explicit_default(pid_a, results_a, pid_b, results_b)
    else:
        print_explicit_ensemble(args.files[0], data_a, results_a,
                                args.files[1], data_b, results_b)


def run_discovery(args):
    data0, results0 = parse_result_file(args.files[0])
    seed_pid = data0.get('prompt_id', '')
    if not seed_pid:
        print("Error: could not determine prompt_id from file", file=sys.stderr)
        sys.exit(1)

    files = discover_files(args.files[0], seed_pid)
    if not files:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    pids = sorted(files.keys())
    print(f"\nPattern: {Path(args.files[0]).name.replace(f'_{seed_pid}_', '_pN_', 1)}")
    print(f"Found: {', '.join(pids)}\n")

    if not args.ensemble:
        print(f"Anchor (file1): {seed_pid}\n")
        print_discovery_default(seed_pid, results0, files, pids)
    else:
        print_discovery_ensemble(seed_pid, files, pids, args.three_way)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='+', metavar='file')
    parser.add_argument('-3', '--three-way', action='store_true',
                        help='include 3-way ensemble combinations (ensemble discovery mode only)')
    parser.add_argument('-e', '--ensemble', action='store_true',
                        help='ensemble mode: show all combination statistics')
    args = parser.parse_args()

    if len(args.files) == 1:
        run_discovery(args)
    elif len(args.files) == 2:
        run_explicit(args)
    else:
        parser.error('specify 1 or 2 files')


if __name__ == '__main__':
    main()
