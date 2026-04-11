#!/usr/bin/env python3
"""Compare prompt result files and report statistics.

Usage:
  compare.py <file>                           — discover all .jsonl files in same directory; ranked scores
  compare.py <file_a> <file_b>               — compare two files: Fixed/New FP/FN table
  compare.py -e <file>                        — discover + show pairwise ensemble combinations
  compare.py -e <file_a> <file_b>            — explicit 2-way ensemble (OR, AND)
  compare.py -e -3 <file>                     — discover + include 3-way ensemble combinations
  compare.py -k key1,key2[,...] -2 <dir>     — explicit 2-way ensemble across all key combinations
  compare.py -k key1,key2,key3[,...] -3 <dir> — explicit 3-way ensemble across all key combinations
"""

import argparse
import heapq
import json
import signal
import sys
import types

signal.signal(signal.SIGPIPE, signal.SIG_DFL)
from itertools import combinations, islice
from pathlib import Path

import numpy as np

from common import (load_expected_pairs, load_eval_results,
                    parse_result_filename, discover_files_all,
                    compute_stats, print_stats, print_discovery_ranked)
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


def apply_ensemble_3_labeled(results_a, results_b, results_c, rule):
    """Apply rule(a_actual, b_actual, c_actual) -> 'YES'/'NO' over common labeled pairs."""
    combined = {}
    for pair in set(results_a) & set(results_b) & set(results_c):
        a, b, c = results_a[pair], results_b[pair], results_c[pair]
        if 'label' not in a or 'label' not in b or 'label' not in c:
            continue
        actual = rule(a['actual'], b['actual'], c['actual'])
        a_label = a['label']
        expected = a['actual'] if a_label == 'correct' else ('NO' if a_label == 'fp' else 'YES')
        label = 'correct' if actual == expected else ('fp' if actual == 'YES' else 'fn')
        combined[pair] = {'actual': actual, 'label': label}
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



def resolve_key(directory, key):
    """Given a directory and a discovery key like 'crosswd2.p81.qwen35', find the matching .jsonl file.
    Splits key into (prompt_file, pid, tag) and globs for a unique match."""
    parts = key.split('.', 2)
    if len(parts) == 3:
        prompt_file, pid, tag = parts
    elif len(parts) == 2:
        prompt_file, pid, tag = parts[0], parts[1], ''
    else:
        print(f"Error: invalid key {key!r} (expected prompt_file.pid[.tag])", file=sys.stderr)
        sys.exit(1)

    d = Path(directory)
    pattern = f'*_{prompt_file}_{pid}_*.{tag}.jsonl' if tag else f'*_{prompt_file}_{pid}_*.jsonl'
    candidates = []
    for f in sorted(d.glob(pattern)):
        parsed = parse_result_filename(f.name)
        if parsed is None:
            continue
        _, pf, ppid, _, ptag = parsed
        if pf == prompt_file and ppid == pid and (ptag or '') == tag:
            candidates.append(f)

    if len(candidates) == 0:
        print(f"Error: no file found for key {key!r} in {directory}", file=sys.stderr)
        sys.exit(1)
    if len(candidates) > 1:
        print(f"Error: multiple files found for key {key!r}:", file=sys.stderr)
        for c in candidates:
            print(f"  {c.name}", file=sys.stderr)
        sys.exit(1)
    return candidates[0]



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
    print(f"{'anchor':<{wa}s} {'corr%':>6s} {'FP':>4s} {'FN':>4s}  "
          f"{'complement':<{wc}s} {'corr%':>6s} {'FP':>4s} {'FN':>4s} | "
          f"{'score':>5s} {'FixFP':>5s} {'FixFN':>5s} {'NewFP':>5s} {'NewFN':>5s}")
    print(f"{'─'*(wa + wc + 68)}")
    for pid, d in rows:
        s1, s2 = d['s1'], d['s2']
        print(f"{seed_pid:<{wa}s} {s1['pct']:5.1f}% {s1['fp']:>4d} {s1['fn']:>4d}  "
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

def print_explicit_ensemble(path_a, pid_a, results_a, path_b, pid_b, results_b, sort, ensemble):
    n_common = len(set(results_a) & set(results_b))
    if n_common < len(results_a) or n_common < len(results_b):
        print(f"\n  ({pid_a} has {len(results_a)} pairs, {pid_b} has {len(results_b)}, "
              f"{n_common} in common)")

    ind_rows = [(pid_a, compute_stats(results_a)), (pid_b, compute_stats(results_b))]
    rules = [(lbl, rule) for lbl, rule in ENSEMBLE_RULES_2
             if ensemble == 'ALL' or lbl == ensemble]
    if ensemble != 'ALL':
        lbl, rule = rules[0]
        combined = apply_ensemble(results_a, results_b, rule)
        ens_rows = [(lbl, compute_stats(combined))]
    else:
        combined = None
        ens_rows = [(lbl, compute_stats(apply_ensemble(results_a, results_b, rule)))
                    for lbl, rule in rules]

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
        fp_pairs = sorted(pair for pair, data in combined.items() if data['label'] == 'fp')
        fn_pairs = sorted(pair for pair, data in combined.items() if data['label'] == 'fn')
        if fp_pairs:
            print("FP:")
            for pair in fp_pairs:
                print(f"  {pair}")
        if fn_pairs:
            print("FN:")
            for pair in fn_pairs:
                print(f"  {pair}")
        print()


def print_explicit_ensemble_3(keys, results_list, sort, ensemble):
    sizes = [len(r) for r in results_list]
    n_common = len(set(results_list[0]) & set(results_list[1]) & set(results_list[2]))
    if n_common < min(sizes):
        print(f"\n  ({keys[0]} has {sizes[0]}, {keys[1]} has {sizes[1]}, {keys[2]} has {sizes[2]} pairs; "
              f"{n_common} in common)")

    ind_rows = [(k, compute_stats(r)) for k, r in zip(keys, results_list)]
    rules = [(lbl, rule) for lbl, rule in ENSEMBLE_RULES_3
             if ensemble == 'ALL' or lbl == ensemble]
    if ensemble != 'ALL':
        lbl, rule = rules[0]
        combined = apply_ensemble_3_labeled(*results_list, rule)
        ens_rows = [(lbl, compute_stats(combined))]
    else:
        combined = None
        ens_rows = [(lbl, compute_stats(apply_ensemble_3_labeled(*results_list, rule)))
                    for lbl, rule in rules]

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
        fp_pairs = sorted(pair for pair, data in combined.items() if data['label'] == 'fp')
        fn_pairs = sorted(pair for pair, data in combined.items() if data['label'] == 'fn')
        if fp_pairs:
            print("FP:")
            for pair in fp_pairs:
                print(f"  {pair}")
        if fn_pairs:
            print("FN:")
            for pair in fn_pairs:
                print(f"  {pair}")
        print()


def print_discovery_ensemble(args, files):
    pids = list(files.keys())
    exp_vec, yes_vecs, n_pairs = _build_vecs(files)
    rows = {}

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
        sort_key, sort_rev = SORT_ENSEMBLE_KEYS[args.sort.lower()]
        top_k = args.top
        heap = []   # min-heap of (heap_val, counter, label, stats)
        counter = 0
        for batch in _combo_batches(len(pid_list), 3, 10_000):
            idx = np.array(batch, dtype=np.intp)
            ma = M[idx[:, 0]]
            mb = M[idx[:, 1]]
            mc = M[idx[:, 2]]
            batch_results = []
            if args.ensemble in ('ALL', 'OR'):
                or_mat = ma | mb | mc
                batch_results.append((" OR", _batch_stats_from_mat(or_mat, exp_vec, n_pairs)))
                del or_mat
            if args.ensemble in ('ALL', 'AND'):
                and_mat = ma & mb & mc
                batch_results.append((" AND", _batch_stats_from_mat(and_mat, exp_vec, n_pairs)))
                del and_mat
            if args.ensemble in ('ALL', 'MAJORITY'):
                maj_mat = (ma.view(np.uint8) + mb.view(np.uint8) + mc.view(np.uint8)) >= 2
                batch_results.append((" MAJORITY", _batch_stats_from_mat(maj_mat, exp_vec, n_pairs)))
                del maj_mat
            del ma, mb, mc
            for i, (ia, ib, ic) in enumerate(batch):
                triple_key = f"{pid_list[ia]},{pid_list[ib]},{pid_list[ic]}"
                for suffix, stats_arr in batch_results:
                    s = stats_arr[i]
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
    print(f"{'label':<{w}s}  {'correct':>9s}  {'FP':>5s}  {'FN':>5s}")
    print(f"{'─'*(w + 28)}")
    for label, stats in sorted_rows:
        print_stats(label, stats, w)
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


def run_explicit_3way(args):
    files = load_files_from_keys(args)
    ens_args = types.SimpleNamespace(
        ensemble=args.ensemble or 'ALL',
        n_way=3,
        sort=args.sort,
        top=args.top,
    )
    print_discovery_ensemble(ens_args, files)


def run_explicit_2way(args):
    files = load_files_from_keys(args)
    ens_args = types.SimpleNamespace(
        ensemble=args.ensemble or 'ALL',
        n_way=2,
        sort=args.sort,
        top=args.top,
    )
    print_discovery_ensemble(ens_args, files)


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
        args.n_way = 3 if args.three_way else 2
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
    parser.add_argument('-e', '--ensemble', type=str.upper,
                        choices=['ALL', 'AND', 'OR', 'MAJORITY'],
                        default=None, metavar='TYPE',
                        help='ensemble type: ALL (default), AND, OR, MAJORITY (case-insensitive)')
    parser.add_argument('-3', '--three-way', action='store_true',
                        help='include 3-way combinations in discovery ensemble mode')
    parser.add_argument('-2', '--two-way', action='store_true',
                        help='2-way ensemble across all pairwise key combinations (use with -k)')
    parser.add_argument('-k', '--keys', metavar='KEYS',
                        help='comma-separated discovery keys for explicit ensemble; '
                             'positional arg must be a directory')
    parser.add_argument('-s', '--sort', default='score', metavar='FIELD',
                        help='sort field: score (default), FP, FN')
    parser.add_argument('--top', type=int, default=1000, metavar='N',
                        help='max 3-way results to retain in discovery ensemble mode (default: 1000)')
    args = parser.parse_args()

    if args.keys:
        if len(args.files) != 1:
            parser.error('--keys requires exactly one positional argument (a directory)')
        if not Path(args.files[0]).is_dir():
            parser.error(f'--keys: {args.files[0]!r} is not a directory')
        if args.two_way and args.three_way:
            parser.error('-2 and -3 are mutually exclusive')
        if not args.two_way and not args.three_way:
            parser.error('-k requires -2 or -3')
        keys = [k.strip() for k in args.keys.split(',')]
        if args.two_way:
            if len(keys) < 2:
                parser.error(f'-2 requires at least 2 keys, got {len(keys)}')
            if args.ensemble == 'MAJORITY':
                parser.error('--ensemble MAJORITY is not valid with -2')
            run_explicit_2way(args)
        else:
            if len(keys) < 3:
                parser.error(f'-3 requires at least 3 keys, got {len(keys)}')
            run_explicit_3way(args)
        return

    if args.three_way and not args.ensemble:
        parser.error('--three-way requires --ensemble')

    if args.ensemble == 'MAJORITY' and not args.three_way:
        parser.error('--ensemble MAJORITY requires --three-way or --keys')

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
