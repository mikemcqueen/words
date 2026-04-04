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
import re
import signal
import sys

signal.signal(signal.SIGPIPE, signal.SIG_DFL)
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations, islice
from pathlib import Path

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False


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


def print_stats(label, stats, w=24):
    print(f"  {label:<{w}s}  {stats['correct']:3d}/{stats['total']:3d} ({stats['pct']:5.1f}%)  "
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
    score = 2 * (fixed_fn - new_fn) + (fixed_fp - new_fp)
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
    """Return (exp_vec, yes_vecs, n_pairs) using a common pair ordering.
    exp_vec and yes_vecs values are numpy bool arrays (or plain lists if numpy absent)."""
    all_keys = list(files_dict.keys())
    pair_sets = [set(files_dict[k][1].keys()) for k in all_keys]
    common = sorted(set.intersection(*pair_sets))
    n = len(common)
    first_results = files_dict[all_keys[0]][1]
    if _NUMPY:
        exp_vec = np.array([first_results[p]['expected'] == 'YES' for p in common], dtype=np.bool_)
        yes_vecs = {
            k: np.array([files_dict[k][1][p]['answer'] == 'YES' for p in common], dtype=np.bool_)
            for k in all_keys
        }
    else:
        exp_vec = [first_results[p]['expected'] == 'YES' for p in common]
        yes_vecs = {
            k: [files_dict[k][1][p]['answer'] == 'YES' for p in common]
            for k in all_keys
        }
    return exp_vec, yes_vecs, n


def _stats_from_vec(yes_vec, exp_vec, n):
    """Compute stats dict from boolean vectors."""
    if _NUMPY:
        tp = int((yes_vec & exp_vec).sum())
        tn = int((~yes_vec & ~exp_vec).sum())
        fp = int((yes_vec & ~exp_vec).sum())
        fn = int((~yes_vec & exp_vec).sum())
    else:
        tp = sum(y and e for y, e in zip(yes_vec, exp_vec))
        tn = sum(not y and not e for y, e in zip(yes_vec, exp_vec))
        fp = sum(y and not e for y, e in zip(yes_vec, exp_vec))
        fn = sum(not y and e for y, e in zip(yes_vec, exp_vec))
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
    m = re.match(r'^(.+)_([^_]+)_(p\d+)_(.+)$', stem)
    return (m.group(1), m.group(2), m.group(3), m.group(4)) if m else None


def discover_files_all(seed_path):
    """Return {prompt_file.prompt_id: (data, results)} for all .json files in seed's directory."""
    directory = Path(seed_path).parent
    candidates = []
    for json_file in sorted(directory.glob('*.json')):
        parsed = parse_result_filename(json_file.name)
        if parsed is None:
            continue
        _, prompt_file, prompt_id, _ = parsed
        key = f"{prompt_file}.{prompt_id}"
        candidates.append((key, json_file))

    def _load(item):
        key, json_file = item
        try:
            return key, parse_result_file(json_file)
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


def print_discovery_ensemble(seed_pid, files, pids, three_way, sort='score'):
    exp_vec, yes_vecs, n_pairs = _build_vecs({p: files[p] for p in pids})
    rows = {}

    for pid in pids:
        rows[pid] = _stats_from_vec(yes_vecs[pid], exp_vec, n_pairs)

    for pid_a, pid_b in combinations(pids, 2):
        ya, yb = yes_vecs[pid_a], yes_vecs[pid_b]
        pair_key = f"{pid_a},{pid_b}"
        if _NUMPY:
            rows[f"{pair_key} OR"]  = _stats_from_vec(ya | yb,  exp_vec, n_pairs)
            rows[f"{pair_key} AND"] = _stats_from_vec(ya & yb,  exp_vec, n_pairs)
        else:
            rows[f"{pair_key} OR"]  = _stats_from_vec([a or b  for a, b in zip(ya, yb)], exp_vec, n_pairs)
            rows[f"{pair_key} AND"] = _stats_from_vec([a and b for a, b in zip(ya, yb)], exp_vec, n_pairs)

    if three_way:
        if _NUMPY:
            M = np.stack([yes_vecs[p] for p in pids])  # (n_files, n_pairs)
            pid_list = list(pids)
            for batch in _combo_batches(len(pid_list), 3, 100_000):
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
                for i, (ia, ib, ic) in enumerate(batch):
                    triple_key = f"{pid_list[ia]},{pid_list[ib]},{pid_list[ic]}"
                    rows[f"{triple_key} OR"]       = or_stats[i]
                    rows[f"{triple_key} AND"]      = and_stats[i]
                    rows[f"{triple_key} MAJORITY"] = maj_stats[i]
        else:
            for pid_a, pid_b, pid_c in combinations(pids, 3):
                ya, yb, yc = yes_vecs[pid_a], yes_vecs[pid_b], yes_vecs[pid_c]
                triple_key = f"{pid_a},{pid_b},{pid_c}"
                rows[f"{triple_key} OR"]       = _stats_from_vec([a or b or c for a,b,c in zip(ya,yb,yc)], exp_vec, n_pairs)
                rows[f"{triple_key} AND"]      = _stats_from_vec([a and b and c for a,b,c in zip(ya,yb,yc)], exp_vec, n_pairs)
                rows[f"{triple_key} MAJORITY"] = _stats_from_vec([(a+b+c)>=2 for a,b,c in zip(ya,yb,yc)], exp_vec, n_pairs)

    sort_key, sort_rev = SORT_ENSEMBLE_KEYS[sort.lower()]
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

    if args.all:
        parsed = parse_result_filename(Path(args.files[0]).name)
        if parsed is None:
            print(f"Error: cannot parse filename '{Path(args.files[0]).name}' "
                  f"as {{pair_file}}_{{prompt_file}}_{{prompt_id}}_{{host}}.json",
                  file=sys.stderr)
            sys.exit(1)
        _, prompt_file0, prompt_id0, _ = parsed
        seed_key = f"{prompt_file0}.{prompt_id0}"

        files = discover_files_all(args.files[0])
        if not files:
            print("No files found.", file=sys.stderr)
            sys.exit(1)

        keys = sorted(files.keys())
        print(f"\nDirectory: {Path(args.files[0]).parent}")
        print(f"Found: {', '.join(keys)}\n")

        if not args.ensemble:
            print(f"Anchor: {seed_key}\n")
            print_discovery_default(seed_key, results0, files, keys, args.sort)
        else:
            print_discovery_ensemble(seed_key, files, keys, args.three_way, args.sort)
    else:
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
            print_discovery_default(seed_pid, results0, files, pids, args.sort)
        else:
            print_discovery_ensemble(seed_pid, files, pids, args.three_way, args.sort)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='+', metavar='file')
    parser.add_argument('-3', '--three-way', action='store_true',
                        help='include 3-way ensemble combinations (ensemble discovery mode only)')
    parser.add_argument('-a', '--all', action='store_true',
                        help='discover all .json files in same directory as FILE '
                             '(single-file mode only; keys shown as prompt_file.prompt_id)')
    parser.add_argument('-e', '--ensemble', action='store_true',
                        help='ensemble mode: show all combination statistics')
    parser.add_argument('-s', '--sort', default='score', metavar='FIELD',
                        help='sort field: score (default); FP/FN (ensemble only); '
                             'FixFP/FixFN/NewFP/NewFN (default mode only)')
    args = parser.parse_args()

    if args.all and len(args.files) != 1:
        parser.error('--all requires exactly one positional FILE')

    sort_lc = args.sort.lower()
    ensemble_only = {'fp', 'fn'}
    default_only  = {'fixfp', 'fixfn', 'newfp', 'newfn'}
    valid_sort    = {'score'} | ensemble_only | default_only
    if sort_lc not in valid_sort:
        parser.error(f'--sort: invalid value {args.sort!r}; '
                     f'choices: score, FP, FN, FixFP, FixFN, NewFP, NewFN')
    if sort_lc in ensemble_only and not args.ensemble:
        parser.error(f'--sort {args.sort} is only valid in ensemble mode (-e)')
    if sort_lc in default_only and args.ensemble:
        parser.error(f'--sort {args.sort} is not valid in ensemble mode')

    if len(args.files) == 1:
        run_discovery(args)
    elif len(args.files) == 2:
        run_explicit(args)
    else:
        parser.error('specify 1 or 2 files')


if __name__ == '__main__':
    main()
