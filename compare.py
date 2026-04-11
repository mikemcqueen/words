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
import heapq
import json
import re
import signal
import sys

signal.signal(signal.SIGPIPE, signal.SIG_DFL)
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations, islice
from pathlib import Path

import numpy as np

from common import load_expected_pairs, load_eval_results
from score import label_eval_results


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


def compute_stats(eval_results):
    correct = fp = fn = 0
    for data in eval_results.values():
        label = data.get('label')
        if label == 'correct': correct += 1
        elif label == 'fp':    fp += 1
        elif label == 'fn':    fn += 1
    total = correct + fp + fn
    pct   = 100 * correct / total if total else 0.0
    return dict(correct=correct, total=total, pct=pct, fp=fp, fn=fn)


def print_stats(label, stats, w=24):
    print(f"  {label:<{w}s}  {stats['correct']:3d}/{stats['total']:3d} ({stats['pct']:5.1f}%)  "
          f"FP={stats['fp']:3d}  FN={stats['fn']:3d}")


def compute_pair_diff(results_1, results_2):
    """Compute Fixed/New FP/FN counts and score comparing results_1 (anchor) to results_2."""
    common = set(results_1) & set(results_2)
    fixed_fp = fixed_fn = new_fp = new_fn = 0
    for pair in common:
        r1, r2 = results_1[pair], results_2[pair]
        if 'label' not in r1 or 'label' not in r2:
            continue
        was_fp      = r1['label'] == 'fp'
        was_fn      = r1['label'] == 'fn'
        was_correct = r1['label'] == 'correct'
        is_fp       = r2['label'] == 'fp'
        is_fn       = r2['label'] == 'fn'
        if was_fp and not is_fp:     fixed_fp += 1
        if was_fn and not is_fn:     fixed_fn += 1
        if was_correct and is_fp:    new_fp += 1
        if was_correct and is_fn:    new_fn += 1
    s1 = compute_stats(results_1)
    s2 = compute_stats(results_2)
    score = 2 * (fixed_fn - new_fn) + (fixed_fp - new_fp)
    return dict(fixed_fp=fixed_fp, fixed_fn=fixed_fn, new_fp=new_fp, new_fn=new_fn,
                score=score, s1=s1, s2=s2)


def apply_ensemble(results_a, results_b, rule):
    """Apply rule(a_actual, b_actual) -> 'YES'/'NO' over common labeled pairs."""
    combined = {}
    for pair in set(results_a) & set(results_b):
        a = results_a[pair]
        b = results_b[pair]
        if 'label' not in a or 'label' not in b:
            continue
        actual = rule(a['actual'], b['actual'])
        a_label = a['label']
        expected = a['actual'] if a_label == 'correct' else ('NO' if a_label == 'fp' else 'YES')
        label = 'correct' if actual == expected else ('fp' if actual == 'YES' else 'fn')
        combined[pair] = {'actual': actual, 'label': label}
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
    ("AND",     lambda a, b: 'YES' if a == 'YES' and b == 'YES' else 'NO')
#    ("A=Y,B=N", lambda a, b: 'YES' if a == 'YES' and b == 'NO'  else 'NO'),
#    ("A=N,B=Y", lambda a, b: 'YES' if a == 'NO'  and b == 'YES' else 'NO'),
]

ENSEMBLE_RULES_3 = [
    ("OR",       lambda a, b, c: 'YES' if 'YES' in (a, b, c)           else 'NO'),
    ("AND",      lambda a, b, c: 'YES' if a == b == c == 'YES'          else 'NO'),
    ("MAJORITY", lambda a, b, c: 'YES' if (a, b, c).count('YES') >= 2   else 'NO'),
]


def _build_vecs(files_dict):
    """Return (exp_vec, yes_vecs, n_pairs) using common labeled pairs.
    exp_vec and yes_vecs values are numpy bool arrays."""
    all_keys = list(files_dict.keys())
    pair_sets = [set(p for p, d in files_dict[k].items() if 'label' in d) for k in all_keys]
    common = sorted(set.intersection(*pair_sets))
    n = len(common)
    first_results = files_dict[all_keys[0]]
    exp_vec = np.array(
        [first_results[p]['label'] == 'fn' or
         (first_results[p]['label'] == 'correct' and first_results[p].get('actual') == 'YES')
         for p in common], dtype=np.bool_)
    yes_vecs = {
        k: np.array([files_dict[k][p].get('actual') == 'YES' for p in common], dtype=np.bool_)
        for k in all_keys
    }
    return exp_vec, yes_vecs, n


def _stats_from_vec(yes_vec, exp_vec, n):
    """Compute stats dict from boolean vectors."""
    tp = int((yes_vec & exp_vec).sum())
    tn = int((~yes_vec & ~exp_vec).sum())
    fp = int((yes_vec & ~exp_vec).sum())
    fn = int((~yes_vec & exp_vec).sum())
    correct = tp + tn
    pct = 100 * correct / n if n else 0.0
    return dict(correct=correct, total=n, pct=pct, tp=tp, tn=tn, fp=fp, fn=fn)


def _combo_batches(n, r, batch_size):
    """Yield lists of index tuples, batch_size at a time, for combinations(range(n), r)."""
    gen = combinations(range(n), r)
    while True:
        batch = list(islice(gen, batch_size))
        if not batch:
            break
        yield batch


def _batch_stats_from_mat(mat, exp_vec, n):
    """Compute stats for each row of mat (bool, shape batch x n_pairs).
    Returns list of stats dicts."""
    tp = ( mat &  exp_vec).sum(axis=1)
    tn = (~mat & ~exp_vec).sum(axis=1)
    fp = ( mat & ~exp_vec).sum(axis=1)
    fn = (~mat &  exp_vec).sum(axis=1)
    correct = tp + tn
    pct = 100.0 * correct / n if n else np.zeros(len(correct), dtype=np.float64)
    return [dict(correct=int(correct[i]), total=n, pct=float(pct[i]),
                 tp=int(tp[i]), tn=int(tn[i]), fp=int(fp[i]), fn=int(fn[i]))
            for i in range(len(correct))]


def parse_result_filename(name):
    """Parse {pair_file}_{prompt_file}_{prompt_id}_{host}.json.
    Returns (pair_file, prompt_file, prompt_id, host) or None if no match."""
    stem = Path(name).stem

    #m = re.match(r'^(.+)_([^_]+)_(p\d+)_(.+)$', stem)
    m =  re.match(r'^(.+)_([^_]+)_(p\d+)_(.+?)(?:\.(.+))?$', stem)
    return (m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)) if m else None


def discover_files_all(seed_path):
    """Return {prompt_file.prompt_id[.tag]: eval_results} for all .jsonl files in seed's directory."""
    directory = Path(seed_path).parent
    candidates = []
    for jsonl_file in sorted(directory.glob('*.jsonl')):
        parsed = parse_result_filename(jsonl_file.name)
        if parsed is None:
            continue
        _, prompt_file, prompt_id, _, tag = parsed
        if not tag:
            tag = ""
        key = f"{prompt_file}.{prompt_id}.{tag}"
        candidates.append((key, jsonl_file))

    def _load(item):
        key, jsonl_file = item
        try:
            return key, load_eval_results(jsonl_file)
        except Exception:
            return key, None

    found = {}
    with ThreadPoolExecutor() as executor:
        for key, result in executor.map(_load, candidates):
            if result is not None:
                found[key] = result
    return found


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
        #print(f"trying: {candidate}")
        if candidate.exists():
            found[pid] = parse_result_file(candidate)
        else:
            break
    return found


# --- default (diff) output ---

def print_default_table(seed_pid, rows):
    """Print diff table. rows is a list of (complement_pid, diff_dict)."""
    wa = max(len(seed_pid), len('anchor'))
    wc = max((len(pid) for pid, _ in rows), default=len('complement'))
    wc = max(wc, len('complement'))
    print(f"  {'anchor':<{wa}s} {'corr%':>6s} {'FP':>4s} {'FN':>4s}  "
          f"{'complement':<{wc}s} {'corr%':>6s} {'FP':>4s} {'FN':>4s} | "
          f"{'score':>5s} {'FixFP':>5s} {'FixFN':>5s} {'NewFP':>5s} {'NewFN':>5s}")
    print(f"  {'─'*(wa + wc + 68)}")
    for pid, d in rows:
        s1, s2 = d['s1'], d['s2']
        print(f"  {seed_pid:<{wa}s} {s1['pct']:5.1f}% {s1['fp']:>4d} {s1['fn']:>4d}  "
              f"{pid:<{wc}s} {s2['pct']:5.1f}% {s2['fp']:>4d} {s2['fn']:>4d} | "
              f"{d['score']:>+5d} {d['fixed_fp']:>5d} {d['fixed_fn']:>5d} "
              f"{d['new_fp']:>5d} {d['new_fn']:>5d}")
    print()


def print_explicit_default(pid_a, results_a, pid_b, results_b):
    d = compute_pair_diff(results_a, results_b)
    print_default_table(pid_a, [(pid_b, d)])


SORT_DEFAULT_KEYS = {
    'score':  ('score',    True),
    'fixfp':  ('fixed_fp', True),
    'fixfn':  ('fixed_fn', True),
    'newfp':  ('new_fp',   False),
    'newfn':  ('new_fn',   False),
}

SORT_ENSEMBLE_KEYS = {
    'score': ('correct', True),
    'fp':    ('fp',      False),
    'fn':    ('fn',      False),
}


def print_discovery_default(seed_pid, results0, files, pids, sort='score'):
    rows = []
    for pid in pids:
        if pid == seed_pid:
            continue
        _, results_other = files[pid]
        d = compute_pair_diff(results0, results_other)
        rows.append((pid, d))
    sort_key, sort_rev = SORT_DEFAULT_KEYS[sort.lower()]
    if sort_key == 'score':
        rows.sort(key=lambda x: x[1]['score'], reverse=True)
    else:
        rows.sort(key=lambda x: (x[1][sort_key] * (-1 if sort_rev else 1), -x[1]['score']))

    print_default_table(seed_pid, rows)


# --- ensemble output ---

def print_explicit_ensemble(path_a, pid_a, results_a, path_b, pid_b, results_b):
    print(f"\nA: {Path(path_a).name}")
    print(f"   prompt={pid_a}")
    print(f"B: {Path(path_b).name}")
    print(f"   prompt={pid_b}")

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


def print_discovery_ensemble(args, files):
    pids = list(files.keys())
    exp_vec, yes_vecs, n_pairs = _build_vecs(files)
    rows = {}

    for pid in pids:
        rows[pid] = _stats_from_vec(yes_vecs[pid], exp_vec, n_pairs)

    if not args.rank:
        for pid_a, pid_b in combinations(pids, 2):
            ya, yb = yes_vecs[pid_a], yes_vecs[pid_b]
            pair_key = f"{pid_a},{pid_b}"
            rows[f"{pair_key} OR"]  = _stats_from_vec(ya | yb,  exp_vec, n_pairs)
            rows[f"{pair_key} AND"] = _stats_from_vec(ya & yb,  exp_vec, n_pairs)

        if args.three_way:
            M = np.stack([yes_vecs[p] for p in pids])  # (n_files, n_pairs)
            pid_list = list(pids)
            sort_key, sort_rev = SORT_ENSEMBLE_KEYS[args.sort.lower()]
            top_k = args.top
            heap = []   # min-heap of (heap_val, counter, label, stats)
            counter = 0
            for batch in _combo_batches(len(pid_list), 3, 10_000):
                idx = np.array(batch, dtype=np.intp)
                ya = M[idx[:, 0]]
                yb = M[idx[:, 1]]
                yc = M[idx[:, 2]]
                and_mat = ya & yb & yc
                or_mat  = ya | yb | yc
                maj_mat = (ya.view(np.uint8) + yb.view(np.uint8) + yc.view(np.uint8)) >= 2
                and_stats = _batch_stats_from_mat(and_mat, exp_vec, n_pairs)
                or_stats  = _batch_stats_from_mat(or_mat,  exp_vec, n_pairs)
                maj_stats = _batch_stats_from_mat(maj_mat, exp_vec, n_pairs)
                del ya, yb, yc, and_mat, or_mat, maj_mat
                for i, (ia, ib, ic) in enumerate(batch):
                    triple_key = f"{pid_list[ia]},{pid_list[ib]},{pid_list[ic]}"
                    for suffix, s in ((" OR", or_stats[i]), (" AND", and_stats[i]),
                                      (" MAJORITY", maj_stats[i])):
                        hv = s[sort_key] if sort_rev else -s[sort_key]
                        if len(heap) < top_k:
                            heapq.heappush(heap, (hv, counter, triple_key + suffix, s))
                        elif hv > heap[0][0]:
                            heapq.heapreplace(heap, (hv, counter, triple_key + suffix, s))
                        counter += 1
            for _, _, label, stats in heap:
                rows[label] = stats

    sort_key, sort_rev = SORT_ENSEMBLE_KEYS[args.sort.lower()]
    if sort_key == 'correct':
        sorted_rows = sorted(rows.items(), key=lambda x: x[1]['correct'], reverse=True)
    else:
        sorted_rows = sorted(rows.items(), key=lambda x: (x[1][sort_key] * (-1 if sort_rev else 1), -x[1]['correct']))
    w = max((len(label) for label, _ in sorted_rows), default=5)
    w = max(w, len('label'))
    print(f"  {'label':<{w}s}  {'correct':>9s}  {'FP':>5s}  {'FN':>5s}")
    print(f"  {'─'*(w + 28)}")
    for label, stats in sorted_rows:
        print_stats(label, stats, w)
    print()


# --- ranked discovery output (jsonl path) ---

def print_discovery_ranked(files_dict, sort='score'):
    """Print a ranked table of pre-labeled eval results: key | score% | correct/total | FP | FN."""
    rows = [(key, compute_stats(records)) for key, records in files_dict.items()]

    sort_lc = sort.lower()
    if sort_lc == 'fp':
        rows.sort(key=lambda x: (x[1]['fp'], -x[1]['pct']))
    elif sort_lc == 'fn':
        rows.sort(key=lambda x: (x[1]['fn'], -x[1]['pct']))
    else:
        rows.sort(key=lambda x: x[1]['pct'], reverse=True)

    w = max((len(k) for k, _ in rows), default=5)
    w = max(w, len('key'))
    print(f"  {'key':<{w}s}  {'score':>7s}  {'corr':>9s}  {'FP':>4s}  {'FN':>4s}")
    print(f"  {'─'*(w + 32)}")
    for key, s in rows:
        print(f"  {key:<{w}s}  {s['pct']:6.1f}%  {s['correct']:4d}/{s['total']:<4d}  {s['fp']:4d}  {s['fn']:4d}")
    print()


# --- top-level runners ---

def run_explicit(args):
    expected = load_expected_pairs(args.pairs)
    results_a = load_eval_results(args.files[0])
    results_b = load_eval_results(args.files[1])
    label_eval_results(results_a, expected, args.method)
    label_eval_results(results_b, expected, args.method)
    parsed_a = parse_result_filename(args.files[0])
    parsed_b = parse_result_filename(args.files[1])
    pid_a = parsed_a[2] if parsed_a else Path(args.files[0]).stem
    pid_b = parsed_b[2] if parsed_b else Path(args.files[1]).stem

    if not args.ensemble:
        print_explicit_default(pid_a, results_a, pid_b, results_b)
    else:
        print_explicit_ensemble(args.files[0], pid_a, results_a,
                                args.files[1], pid_b, results_b)


def run_discovery(args):
    expected = load_expected_pairs(args.pairs)

    files = discover_files_all(args.files[0])
    if not files:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    for eval_results in files.values():
        label_eval_results(eval_results, expected, args.method)

    print(f"\nDirectory: {Path(args.files[0]).parent}")
    print(f"Found: {len(files)} file(s)\n")

    if args.ensemble:
        print_discovery_ensemble(args, files)
    else:
        print_discovery_ranked(files, args.sort)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='+', metavar='file')
    parser.add_argument('--pairs', required=True, metavar='PAIRS_JSON',
                        help='pairs JSON file with expected values')
    parser.add_argument('--method', default='any-yes', metavar='METHOD',
                        help='scoring method (default: any-yes)')
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('-e', '--ensemble', action='store_true',
                            help='ensemble mode: show all combination statistics')
    mode_group.add_argument('-r', '--rank', action='store_true',
                            help='rank mode: show only per-file statistics (discovery --all mode only)')
    parser.add_argument('-3', '--three-way', action='store_true',
                        help='include 3-way ensemble combinations (ensemble discovery mode only)')
    parser.add_argument('-a', '--all', action='store_true',
                        help='discover all .jsonl files in same directory as FILE '
                             '(single-file mode only; keys shown as prompt_file.prompt_id)')
    parser.add_argument('-s', '--sort', default='score', metavar='FIELD',
                        help='sort field: score (default), FP, FN')
    parser.add_argument('--top', type=int, default=1000, metavar='N',
                        help='max 3-way combo results to retain (default: 1000)')
    args = parser.parse_args()

    if args.all and len(args.files) != 1:
        parser.error('--all requires exactly one positional FILE')
    if args.rank and not args.all:
        parser.error('--rank requires --all')
    if args.three_way and not args.ensemble:
        parser.error('--three-way requires --ensemble')

    sort_lc = args.sort.lower()
    valid_sort = {'score', 'fp', 'fn'}
    if sort_lc not in valid_sort:
        parser.error(f'--sort: invalid value {args.sort!r}; choices: score, FP, FN')

    if len(args.files) == 1:
        run_discovery(args)
    elif len(args.files) == 2:
        run_explicit(args)
    else:
        parser.error('specify 1 or 2 files')


if __name__ == '__main__':
    main()
