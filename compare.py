#!/usr/bin/env python3
"""Compare prompt result files and report statistics.

Usage:
  compare.py <dir>                                   — discover all .jsonl files; pairwise 2-way diff table (all ordered pairs)
  compare.py <file>                                  — discover all .jsonl files in same directory; anchor-based 2-way diff table
  compare.py <file_a> <file_b>                      — compare two files: Fixed/New FP/FN table
  compare.py -e <file>                               — discover + show pairwise ensemble combinations
  compare.py -e <file_a> <file_b>                   — explicit 2-way ensemble (OR, AND)
  compare.py -e -3 <file>                            — discover + include 3-way ensemble combinations
  compare.py -k key1,key2[,...] -2 <dir>            — explicit 2-way ensemble across all key combinations
  compare.py -k key1,key2,key3[,...] -3 <dir>       — explicit 3-way ensemble across all key combinations
  compare.py -k key1,...,key5[,...] -5 <dir>        — explicit 5-way ensemble across all key combinations
"""

import argparse
import heapq
import json
import signal
import sys

signal.signal(signal.SIGPIPE, signal.SIG_DFL)
from itertools import combinations, islice, permutations
from pathlib import Path

import numpy as np

from common import (load_expected_pairs, load_eval_results,
                    parse_result_filename, discover_files_all,
                    compute_stats, print_stats, resolve_key, print_bad_pairs)
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
    score = 2 * fixed_fn + fixed_fp - new_fp
    or_correct = s1['correct'] + fixed_fn - new_fp
    or_fp      = s1['fp'] + new_fp
    or_fn      = s1['fn'] - fixed_fn
    or_total   = s1['total']
    or_pct     = 100.0 * or_correct / or_total if or_total else 0.0
    return dict(fixed_fp=fixed_fp, fixed_fn=fixed_fn, new_fp=new_fp, new_fn=new_fn,
                score=score, s1=s1, s2=s2,
                or_pct=or_pct, or_fp=or_fp, or_fn=or_fn)


def apply_ensemble_labeled(results_list, rule_name):
    """Apply OR/AND/MAJORITY across N labeled result dicts. Returns combined labeled dict."""
    n = len(results_list)
    common = set.intersection(*(set(r) for r in results_list))
    majority_threshold = (n + 1) // 2
    combined = {}
    for pair in common:
        datas = [r[pair] for r in results_list]
        if any('label' not in d for d in datas):
            continue
        yes_count = sum(1 for d in datas if d['actual'] == 'YES')
        if rule_name == 'OR':
            actual = 'YES' if yes_count > 0 else 'NO'
        elif rule_name == 'AND':
            actual = 'YES' if yes_count == n else 'NO'
        else:  # MAJORITY
            actual = 'YES' if yes_count >= majority_threshold else 'NO'
        first_label = datas[0]['label']
        expected = datas[0]['actual'] if first_label == 'correct' else ('NO' if first_label == 'fp' else 'YES')
        label = 'correct' if actual == expected else ('fp' if actual == 'YES' else 'fn')
        combined[pair] = {'actual': actual, 'label': label}
    return combined


ENSEMBLE_RULES_2 = ["OR", "AND"]
ENSEMBLE_RULES_3 = ["OR", "AND", "MAJORITY"]


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

def print_default_table(rows):
    """Print diff table. rows is a list of (anchor_pid, complement_pid, diff_dict)."""
    wa = max((len(a) for a, _, _ in rows), default=len('anchor'))
    wa = max(wa, len('anchor'))
    wc = max((len(c) for _, c, _ in rows), default=len('complement'))
    wc = max(wc, len('complement'))
    print(f"{'anchor':<{wa}s} {'corr%':>6s} {'FP':>4s} {'FN':>4s}  "
          f"{'complement':<{wc}s} {'corr%':>6s} {'FP':>4s} {'FN':>4s} | "
          f"{'score':>5s} {'FixFP':>5s} {'FixFN':>5s} {'NewFP':>5s} {'NewFN':>5s} | "
          f"{'Or%':>6s} {'OrFP':>4s} {'OrFN':>4s}")
    print(f"{'─'*(wa + wc + 84)}")
    for anchor, complement, d in rows:
        s1, s2 = d['s1'], d['s2']
        print(f"{anchor:<{wa}s} {s1['pct']:5.1f}% {s1['fp']:>4d} {s1['fn']:>4d}  "
              f"{complement:<{wc}s} {s2['pct']:5.1f}% {s2['fp']:>4d} {s2['fn']:>4d} | "
              f"{d['score']:>+5d} {d['fixed_fp']:>5d} {d['fixed_fn']:>5d} "
              f"{d['new_fp']:>5d} {d['new_fn']:>5d} | "
              f"{d['or_pct']:5.1f}% {d['or_fp']:>4d} {d['or_fn']:>4d}")
    print()


def print_explicit_default(pid_a, results_a, pid_b, results_b):
    d = compute_pair_diff(results_a, results_b)
    print_default_table([(pid_a, pid_b, d)])


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


def _key_from_path(path):
    """Parse a discovery key from a result filename. Exits on failure."""
    parsed = parse_result_filename(Path(path).name)
    if parsed is None:
        print(f"Error: could not parse key from filename: {Path(path).name}", file=sys.stderr)
        sys.exit(1)
    _, prompt_file, prompt_id, _, tag = parsed
    return f"{prompt_file}.{prompt_id}.{tag}" if tag else f"{prompt_file}.{prompt_id}"


def _sort_diff_rows(rows, sort):
    """Sort list of (anchor, complement, diff) by the given sort field. Returns sorted list."""
    sort_key, sort_rev = SORT_DEFAULT_KEYS[sort.lower()]
    if sort_key == 'score':
        return sorted(rows, key=lambda x: (x[2]['or_pct'], -x[2]['or_fp'], -x[2]['or_fn']), reverse=True)
    else:
        return sorted(rows, key=lambda x: (x[2][sort_key] * (-1 if sort_rev else 1), -x[2]['or_pct']))


def print_discovery_default(seed_key, files, args):
    """Print anchor-based diff table: seed_key vs all other discovered files."""
    rows = []
    for key, results_other in files.items():
        if key == seed_key:
            continue
        d = compute_pair_diff(files[seed_key], results_other)
        rows.append((seed_key, key, d))
    sorted_rows = _sort_diff_rows(rows, args.sort)
    print_default_table(sorted_rows[:args.top] if args.top else sorted_rows)


def print_discovery_all_pairs(files, args):
    """Print diff table for all ordered pairs of discovered files."""
    rows = []
    for a, b in permutations(files, 2):
        d = compute_pair_diff(files[a], files[b])
        rows.append((a, b, d))
    sorted_rows = _sort_diff_rows(rows, args.sort)
    print_default_table(sorted_rows[:args.top] if args.top else sorted_rows)


# --- ensemble output ---

def print_explicit_ensemble(path_a, pid_a, results_a, path_b, pid_b, results_b, sort, ensemble):
    n_common = len(set(results_a) & set(results_b))
    if n_common < len(results_a) or n_common < len(results_b):
        print(f"\n  ({pid_a} has {len(results_a)} pairs, {pid_b} has {len(results_b)}, "
              f"{n_common} in common)")

    ind_rows = [(pid_a, compute_stats(results_a)), (pid_b, compute_stats(results_b))]
    rules = [lbl for lbl in ENSEMBLE_RULES_2 if ensemble == 'ALL' or lbl == ensemble]
    if ensemble != 'ALL':
        combined = apply_ensemble_labeled([results_a, results_b], rules[0])
        ens_rows = [(rules[0], compute_stats(combined))]
    else:
        combined = None
        ens_rows = [(lbl, compute_stats(apply_ensemble_labeled([results_a, results_b], lbl)))
                    for lbl in rules]

    sort_key, sort_rev = SORT_ENSEMBLE_KEYS[sort.lower()]
    ind_rows.sort(key=lambda x: x[1][sort_key], reverse=sort_rev)
    ens_rows.sort(key=lambda x: x[1][sort_key], reverse=sort_rev)

    w = max(len(lbl) + 2 for lbl, _ in ind_rows + ens_rows)

    print(f"\n{'':>{w}s}  {'correct':>9s}  {'FP':>5s}  {'FN':>5s}")
    print(f"{'─'*(w + 28)}")
    print("Individual:")
    for lbl, stats in ind_rows:
        print_stats(f"  {lbl}", stats, w)
    print("Ensemble:")
    for lbl, stats in ens_rows:
        print_stats(f"  {lbl}", stats, w)
    print()

    if combined is not None:
        print_bad_pairs(combined)


def print_explicit_ensemble_3(keys, results_list, sort, ensemble):
    sizes = [len(r) for r in results_list]
    n_common = len(set(results_list[0]) & set(results_list[1]) & set(results_list[2]))
    if n_common < min(sizes):
        print(f"\n  ({keys[0]} has {sizes[0]}, {keys[1]} has {sizes[1]}, {keys[2]} has {sizes[2]} pairs; "
              f"{n_common} in common)")

    ind_rows = [(k, compute_stats(r)) for k, r in zip(keys, results_list)]
    rules = [lbl for lbl in ENSEMBLE_RULES_3 if ensemble == 'ALL' or lbl == ensemble]
    if ensemble != 'ALL':
        combined = apply_ensemble_labeled(results_list, rules[0])
        ens_rows = [(rules[0], compute_stats(combined))]
    else:
        combined = None
        ens_rows = [(lbl, compute_stats(apply_ensemble_labeled(results_list, lbl)))
                    for lbl in rules]

    sort_key, sort_rev = SORT_ENSEMBLE_KEYS[sort.lower()]
    ind_rows.sort(key=lambda x: x[1][sort_key], reverse=sort_rev)
    ens_rows.sort(key=lambda x: x[1][sort_key], reverse=sort_rev)

    w = max(len(lbl) + 2 for lbl, _ in ind_rows + ens_rows)

    print(f"\n{'':>{w}s}  {'correct':>9s}  {'FP':>5s}  {'FN':>5s}")
    print(f"{'─'*(w + 28)}")
    print("Individual:")
    for lbl, stats in ind_rows:
        print_stats(f"  {lbl}", stats, w)
    print("Ensemble:")
    for lbl, stats in ens_rows:
        print_stats(f"  {lbl}", stats, w)
    print()

    if combined is not None:
        print_bad_pairs(combined)


def print_discovery_ensemble(args, files):
    pids = list(files.keys())
    exp_vec, yes_vecs, n_pairs = _build_vecs(files)
    rows = {}

    if args.ensemble == 'ALL':
        for pid in pids:
            rows[pid] = _stats_from_vec(yes_vecs[pid], exp_vec, n_pairs)

    for pid_a, pid_b in combinations(pids, 2):
        ya, yb = yes_vecs[pid_a], yes_vecs[pid_b]
        pair_key = f"{pid_a},{pid_b}"
        if args.ensemble in ('ALL', 'OR'):
            rows[f"{pair_key} OR"]  = _stats_from_vec(ya | yb,  exp_vec, n_pairs)
        if args.ensemble in ('ALL', 'AND'):
            rows[f"{pair_key} AND"] = _stats_from_vec(ya & yb,  exp_vec, n_pairs)

    if args.n_way >= 3:
        M = np.stack([yes_vecs[p] for p in pids])  # (n_files, n_pairs)
        pid_list = list(pids)
        r = args.n_way
        majority_threshold = (r + 1) // 2
        sort_key, sort_rev = SORT_ENSEMBLE_KEYS[args.sort.lower()]
        top_k = args.heap_size
        heap = []   # min-heap of (heap_val, counter, label, stats)
        counter = 0
        for batch in _combo_batches(len(pid_list), r, 10_000):
            idx = np.array(batch, dtype=np.intp)
            sub = M[idx]  # (batch_size, r, n_pairs)
            batch_results = []
            if args.ensemble in ('ALL', 'OR'):
                or_mat = sub.any(axis=1)
                batch_results.append((" OR", _batch_stats_from_mat(or_mat, exp_vec, n_pairs)))
                del or_mat
            if args.ensemble in ('ALL', 'AND'):
                and_mat = sub.all(axis=1)
                batch_results.append((" AND", _batch_stats_from_mat(and_mat, exp_vec, n_pairs)))
                del and_mat
            if args.ensemble in ('ALL', 'MAJORITY'):
                maj_mat = sub.astype(np.uint8).sum(axis=1) >= majority_threshold
                batch_results.append((" MAJORITY", _batch_stats_from_mat(maj_mat, exp_vec, n_pairs)))
                del maj_mat
            del sub
            for i, combo in enumerate(batch):
                combo_key = ','.join(pid_list[j] for j in combo)
                for suffix, stats_arr in batch_results:
                    s = stats_arr[i]
                    hv = s[sort_key] if sort_rev else -s[sort_key]
                    if len(heap) < top_k:
                        heapq.heappush(heap, (hv, counter, combo_key + suffix, s))
                    elif hv > heap[0][0]:
                        heapq.heapreplace(heap, (hv, counter, combo_key + suffix, s))
                    counter += 1
        for _, _, label, stats in heap:
            rows[label] = stats

    sort_key, sort_rev = SORT_ENSEMBLE_KEYS[args.sort.lower()]
    if sort_key == 'correct':
        sorted_rows = sorted(rows.items(), key=lambda x: (x[1]['correct'], -x[1]['fp'], -x[1]['fn']), reverse=True)
    else:
        sorted_rows = sorted(rows.items(), key=lambda x: (x[1][sort_key] * (-1 if sort_rev else 1), -x[1]['correct']))
    w = max((len(label) for label, _ in sorted_rows), default=5)
    w = max(w, len('label'))
    print(f"{'label':<{w}s}  {'correct':>16s}  {'FP':>3s}  {'FN':>3s}")
    print(f"{'─'*(w + 28)}")
    for label, stats in sorted_rows[:args.top]:
        print_stats(label, stats, w)
    print()

    if args.bad and sorted_rows:
        top_label, _ = sorted_rows[0]
        parts = top_label.rsplit(' ', 1)
        if len(parts) == 2:
            combo_pids, rule_name = parts[0].split(','), parts[1]
            combined = apply_ensemble_labeled([files[p] for p in combo_pids], rule_name)
            print_bad_pairs(combined)


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
                                args.files[1], pid_b, results_b, args.sort,
                                args.ensemble)


def load_files_from_keys(args):
    """Resolve keys to files, load and label eval results. Returns {key: labeled_results}."""
    keys = [k.strip() for k in args.keys.split(',')]
    directory = args.files[0]
    expected = load_expected_pairs(args.pairs)
    files = {}
    for key in keys:
        path = resolve_key(directory, key)
        r = load_eval_results(path)
        label_eval_results(r, expected, args.method)
        files[key] = r
    return files


def run_explicit_nway(args):
    files = load_files_from_keys(args)
    print_discovery_ensemble(args, files)


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
    elif Path(args.files[0]).is_dir():
        print_discovery_all_pairs(files, args)
    else:
        seed_key = _key_from_path(args.files[0])
        print_discovery_default(seed_key, files, args)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='+', metavar='file')
    parser.add_argument('--pairs', required=True, metavar='PAIRS_JSON',
                        help='pairs JSON file with expected values')
    parser.add_argument('--method', default='any-yes', metavar='METHOD',
                        help='scoring method (default: any-yes)')
    parser.add_argument('-e', '--ensemble', type=str.upper,
                        choices=['ALL', 'AND', 'OR', 'MAJORITY'],
                        default=None, metavar='TYPE',
                        help='ensemble type: ALL (default), AND, OR, MAJORITY (case-insensitive)')
    parser.add_argument('-3', '--three-way', action='store_true',
                        help='include 3-way combinations in discovery ensemble mode')
    parser.add_argument('-2', '--two-way', action='store_true',
                        help='2-way ensemble across all pairwise key combinations (use with -k)')
    parser.add_argument('-5', '--five-way', action='store_true',
                        help='5-way ensemble across all quintuple key combinations (use with -k, requires >= 5 keys)')
    parser.add_argument('-k', '--keys', metavar='KEYS',
                        help='comma-separated discovery keys for explicit ensemble; '
                             'positional arg must be a directory')
    parser.add_argument('-s', '--sort', default='score', metavar='FIELD',
                        help='sort field: score (default), FP, FN')
    parser.add_argument('--top', type=int, default=50, metavar='K',
                        help='max rows to display in output tables (default: 50)')
    parser.add_argument('--heap-size', type=int, default=100, metavar='N',
                        help='max N-way discovery results to retain (default: 100)')
    parser.add_argument('--bad', action='store_true',
                        help='show FP and FN pairs for the single table entry; requires --top 1')
    args = parser.parse_args()

    # Resolve n_way from the mutually exclusive -2/-3/-5 flags.
    n_ways = sum([args.two_way, args.three_way, args.five_way])
    if n_ways > 1:
        parser.error('-2, -3, and -5 are mutually exclusive')
    if n_ways == 1:
        if args.five_way:
            args.n_way = 5
        elif args.three_way:
            args.n_way = 3
        else:
            args.n_way = 2

    if args.keys:
        if len(args.files) != 1:
            parser.error('--keys requires exactly one positional argument (a directory)')
        if not Path(args.files[0]).is_dir():
            parser.error(f'--keys: {args.files[0]!r} is not a directory')
        if n_ways == 0:
            parser.error('-k requires -2, -3, or -5')
        keys = [k.strip() for k in args.keys.split(',')]
        if len(keys) < args.n_way:
            parser.error(f'-{args.n_way} requires at least {args.n_way} keys, got {len(keys)}')
        if args.ensemble == 'MAJORITY' and args.n_way % 2 == 0:
            parser.error(f'--ensemble MAJORITY is not valid with -{args.n_way}')
        args.ensemble = args.ensemble or 'ALL'
        run_explicit_nway(args)
        return

    if args.ensemble and n_ways == 0:
        parser.error('--ensemble requires -2, -3, or -5')

    if (args.two_way or args.three_way or args.five_way) and not args.ensemble:
        parser.error('-2/-3/-5 requires --ensemble')

    if args.ensemble == 'MAJORITY' and args.n_way % 2 == 0:
        parser.error('--ensemble MAJORITY requires -3 or -5')

    if args.bad and args.top != 1:
        parser.error('--bad requires --top 1')

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
