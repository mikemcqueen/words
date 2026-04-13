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
from score import label_eval_results, resolve_all_pair_labels


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
    """Compute Fixed/New FP/FN counts and score comparing results_1 (anchor) to OR-ensemble of both.

    OR is applied per-direction across both files: a pair is 'correct' under OR if any direction
    in either file is correct. Directions are derived from each pair's logprobs keys.
    """
    common = set(results_1) & set(results_2)
    fixed_fp = fixed_fn = new_fp = new_fn = 0
    for pair in common:
        d1, d2 = results_1[pair], results_2[pair]
        if 'label' not in d1 or 'result' not in d1 or 'result' not in d2:
            continue
        r1, r2 = d1['result'], d2['result']
        dirs = [d for d in r1['logprobs'] if d in r2]
        if not dirs:
            continue

        # Derive expected from the first direction's label+token in r1
        first = r1[dirs[0]]
        if first.label == 'correct':
            expected = first.token
        elif first.label == 'fp':
            expected = 'NO'
        else:
            expected = 'YES'

        # OR ensemble: correct if any direction in either file is correct
        or_correct = any(r1[d].label == 'correct' or r2[d].label == 'correct' for d in dirs)
        if or_correct:
            or_label = 'correct'
        elif expected == 'NO':
            or_label = 'fp'   # OR said YES (both files wrong on a NO pair)
        else:
            or_label = 'fn'   # OR said NO (both files wrong on a YES pair)

        old_label = d1['label']
        if old_label == 'fp'      and or_label != 'fp':   fixed_fp += 1
        if old_label == 'fn'      and or_label != 'fn':   fixed_fn += 1
        if old_label == 'correct' and or_label == 'fp':   new_fp += 1
        if old_label == 'correct' and or_label == 'fn':   new_fn += 1

    s1 = compute_stats(results_1)
    s2 = compute_stats(results_2)
    score = 2 * fixed_fn + fixed_fp - new_fp
    or_correct_count = s1['correct'] + fixed_fn - new_fp
    or_fp      = s1['fp'] + new_fp
    or_fn      = s1['fn'] - fixed_fn
    or_total   = s1['total']
    or_pct     = 100.0 * or_correct_count / or_total if or_total else 0.0
    return dict(fixed_fp=fixed_fp, fixed_fn=fixed_fn, new_fp=new_fp, new_fn=new_fn,
                score=score, s1=s1, s2=s2,
                or_pct=or_pct, or_fp=or_fp, or_fn=or_fn)


def _expected(data) -> str | None:
    """Derive the expected YES/NO value from a labeled result entry."""
    result = data.get('result')
    if result is None:
        return None
    first_dir = next(iter(result['logprobs']))
    tl = result[first_dir]
    if tl.label == 'fp':
        return 'NO'
    elif tl.label == 'fn':
        return 'YES'
    else:  # correct
        return tl.token


def apply_ensemble_labeled(results_list, rule_name):
    """Apply OR/AND/MAJORITY per direction across N files. Returns combined labeled dict.
    Final pair label uses any_correct semantics across directions."""
    n = len(results_list)
    common = set.intersection(*(set(r) for r in results_list))
    majority_threshold = (n + 1) // 2
    combined = {}
    for pair in common:
        datas = [r[pair] for r in results_list]
        if any('result' not in d for d in datas):
            continue
        results = [d['result'] for d in datas]
        dirs = list(results[0]['logprobs'].keys())
        expected = _expected(datas[0])
        if expected is None:
            continue

        # Apply ensemble rule per direction; pair is correct if any direction matches expected
        pair_correct = False
        for dir_ in dirs:
            tokens = [r[dir_].token for r in results]
            yes_count = sum(1 for t in tokens if t == 'YES')
            if rule_name == 'OR':
                combined_token = 'YES' if yes_count > 0 else 'NO'
            elif rule_name == 'AND':
                combined_token = 'YES' if yes_count == n else 'NO'
            else:  # MAJORITY
                combined_token = 'YES' if yes_count >= majority_threshold else 'NO'
            if combined_token == expected:
                pair_correct = True

        # fallback: expected=NO → all dirs said YES (fp); expected=YES → all dirs said NO (fn)
        label = 'correct' if pair_correct else ('fp' if expected == 'NO' else 'fn')
        combined[pair] = {'label': label}
    return combined


ENSEMBLE_RULES_2 = ["OR", "AND"]
ENSEMBLE_RULES_3 = ["OR", "AND", "MAJORITY"]


def _build_vecs(files_dict):
    """Return (exp_vec, yes_vecs, dirs, n_pairs).
    yes_vecs: {key: {dir: bool_array_over_common_pairs}}
    Directions derived from the first labeled pair's result['logprobs'] keys."""
    all_keys = list(files_dict.keys())
    pair_sets = [set(p for p, d in files_dict[k].items() if 'label' in d) for k in all_keys]
    common = sorted(set.intersection(*pair_sets))
    n = len(common)
    first_results = files_dict[all_keys[0]]
    first_pair = next(p for p in common if 'result' in first_results[p])
    dirs = list(first_results[first_pair]['result']['logprobs'].keys())
    exp_vec = np.array([_expected(first_results[p]) == 'YES' for p in common], dtype=np.bool_)
    yes_vecs = {
        k: {
            d: np.array(
                [files_dict[k][p]['result'][d].token == 'YES'
                 if 'result' in files_dict[k][p] else False
                 for p in common], dtype=np.bool_)
            for d in dirs
        }
        for k in all_keys
    }
    return exp_vec, yes_vecs, dirs, n


def _resolve_labels_np(combined_per_dir, exp_vec):
    """Vectorized equivalent of resolve_pair_label() in score.py.

    Per-direction label logic:
      fp      = cv & ~exp  (said YES on NO-expected pair)
      correct = (cv & exp) | (~cv & ~exp)
      fn      = ~cv & exp  (said NO on YES-expected pair)
    Priority across directions: fp > correct > fn.

    Works for both single and batch cases (stacks along axis=0):
      _stats_from_dirs:       combined_per_dir elements shape (n_pairs,)
      _batch_stats_from_dirs: combined_per_dir elements shape (batch_size, n_pairs)
    """
    dir_fp      = np.stack([cv & ~exp_vec for cv in combined_per_dir], axis=0)
    dir_correct = np.stack([(cv & exp_vec) | (~cv & ~exp_vec) for cv in combined_per_dir], axis=0)
    pair_fp      = np.any(dir_fp,      axis=0)
    pair_correct = np.any(dir_correct, axis=0) & ~pair_fp  # fp takes priority
    pair_fn      = ~pair_fp & ~pair_correct
    return pair_fp, pair_correct, pair_fn


def _stats_from_dirs(combined_per_dir, exp_vec, n):
    """Compute stats from per-direction combined bool arrays.
    combined_per_dir: list of bool arrays (one per direction), each shape (n_pairs,).
    Uses resolve_pair_label semantics (see score.py)."""
    pair_fp, pair_correct, pair_fn = _resolve_labels_np(combined_per_dir, exp_vec)
    correct = int(pair_correct.sum())
    fp      = int(pair_fp.sum())
    fn      = int(pair_fn.sum())
    pct = 100 * correct / n if n else 0.0
    return dict(correct=correct, total=n, pct=pct, fp=fp, fn=fn)


def _combo_batches(n, r, batch_size):
    """Yield lists of index tuples, batch_size at a time, for combinations(range(n), r)."""
    gen = combinations(range(n), r)
    while True:
        batch = list(islice(gen, batch_size))
        if not batch:
            break
        yield batch


def _batch_stats_from_dirs(combined_per_dir, exp_vec, n):
    """Compute stats for a batch of combos using resolve_pair_label semantics (see score.py).
    combined_per_dir: list of (batch_size, n_pairs) bool arrays, one per direction.
    Returns list of stats dicts."""
    pair_fp, pair_correct, pair_fn = _resolve_labels_np(combined_per_dir, exp_vec)
    correct = pair_correct.sum(axis=1)
    fp      = pair_fp.sum(axis=1)
    fn      = pair_fn.sum(axis=1)
    pct = 100.0 * correct / n if n else np.zeros(len(correct), dtype=np.float64)
    return [dict(correct=int(correct[i]), total=n, pct=float(pct[i]),
                 fp=int(fp[i]), fn=int(fn[i]))
            for i in range(len(correct))]


# --- bitmask-accelerated n-way ensemble (n_pairs <= 64) ---

# 16-bit popcount lookup table (65536 entries, covers 2 bytes at a time)
_POPCOUNT_TABLE_16 = np.array([bin(i).count('1') for i in range(65536)], dtype=np.int32)


def _popcount(arr):
    """Vectorized popcount for uint64 array using 16-bit lookup table."""
    result = np.zeros(arr.shape, dtype=np.int32)
    for shift in range(0, 64, 16):
        result += _POPCOUNT_TABLE_16[(arr >> shift).astype(np.uint16)]
    return result


def _build_bitmasks(files_dict):
    """Like _build_vecs but encodes as uint64 bitmasks. Requires n_pairs <= 64.
    Returns (exp_bits, yes_bits, dirs, n_pairs, pid_list).
    exp_bits: uint64 scalar.
    yes_bits: {dir: (n_files,) uint64 array}, indexed parallel to pid_list."""
    all_keys = list(files_dict.keys())
    pair_sets = [set(p for p, d in files_dict[k].items() if 'label' in d) for k in all_keys]
    common = sorted(set.intersection(*pair_sets))
    n = len(common)
    assert n <= 64, f"_build_bitmasks requires n_pairs <= 64, got {n}"
    first_results = files_dict[all_keys[0]]
    first_pair = next(p for p in common if 'result' in first_results[p])
    dirs = list(first_results[first_pair]['result']['logprobs'].keys())

    # Build expected bitmask
    exp_bits = np.uint64(0)
    for i, p in enumerate(common):
        if _expected(first_results[p]) == 'YES':
            exp_bits |= np.uint64(1 << i)

    # Build per-file per-direction bitmasks
    yes_bits = {}
    for d in dirs:
        col = np.zeros(len(all_keys), dtype=np.uint64)
        for fi, k in enumerate(all_keys):
            bits = np.uint64(0)
            for i, p in enumerate(common):
                data = files_dict[k][p]
                if 'result' in data and data['result'][d].token == 'YES':
                    bits |= np.uint64(1 << i)
            col[fi] = bits
        yes_bits[d] = col

    return exp_bits, yes_bits, dirs, n, all_keys


def _bitmask_stats(combined_per_dir, exp_bits, not_exp, n_mask):
    """Compute (correct, fp, fn) int32 arrays from per-direction combined uint64 arrays.
    combined_per_dir: list of (batch_size,) uint64 arrays, one per direction."""
    dir_fp = [cv & not_exp for cv in combined_per_dir]
    dir_correct = [(cv & exp_bits) | ((~cv & n_mask) & not_exp) for cv in combined_per_dir]
    pair_fp = dir_fp[0]
    for dfp in dir_fp[1:]:
        pair_fp = pair_fp | dfp
    pair_correct = dir_correct[0]
    for dc in dir_correct[1:]:
        pair_correct = pair_correct | dc
    pair_correct = pair_correct & ~pair_fp
    pair_fn = ~pair_fp & ~pair_correct & n_mask
    return _popcount(pair_correct), _popcount(pair_fp), _popcount(pair_fn)


def _screen_bits(combined_per_dir, exp_bits, not_exp, n_mask):
    """Compute (pair_correct_bits, pair_fp_bits) arrays without popcount.
    These are the raw uint64 bitmask arrays needed for screening and deferred stats."""
    dir_fp = [cv & not_exp for cv in combined_per_dir]
    dir_correct = [(cv & exp_bits) | ((~cv & n_mask) & not_exp) for cv in combined_per_dir]
    pair_fp = dir_fp[0]
    for dfp in dir_fp[1:]:
        pair_fp |= dfp
    pair_correct = dir_correct[0]
    for dc in dir_correct[1:]:
        pair_correct |= dc
    pair_correct &= ~pair_fp
    return pair_correct, pair_fp


def _scalar_fp_fn(combined_per_dir, ci, exp_bits, not_exp, n_mask):
    """Compute (fp, fn) for a single combo element from batch arrays."""
    fp_bits = np.uint64(0)
    correct_bits = np.uint64(0)
    for cv in combined_per_dir:
        v = cv[ci]
        fp_bits |= v & not_exp
        correct_bits |= (v & exp_bits) | ((~v & n_mask) & not_exp)
    correct_bits &= ~fp_bits
    fp = bin(int(fp_bits)).count('1')
    fn = bin(int(~fp_bits & ~correct_bits & n_mask)).count('1')
    return fp, fn


def _combo_batches_np(n, r, batch_size):
    """Yield (batch_idx,) numpy index arrays for combinations(range(n), r).
    Uses numpy-native generation for r=3 and r=5, itertools fallback otherwise.
    batch_idx: shape (batch_size, r), dtype=np.intp."""
    if r == 3:
        yield from _combo_batches_np_3(n, batch_size)
    elif r == 5:
        yield from _combo_batches_np_5(n, batch_size)
    else:
        # Fallback: itertools with larger batches
        gen = combinations(range(n), r)
        while True:
            batch = list(islice(gen, batch_size))
            if not batch:
                break
            yield np.array(batch, dtype=np.intp)


def _combo_batches_np_3(n, batch_size):
    """Yield numpy index arrays for 3-way combinations, generated without Python tuples."""
    buf = []
    buf_len = 0
    for i in range(n - 2):
        remaining = n - i - 1
        if remaining < 2:
            continue
        jj, kk = np.triu_indices(remaining, k=1)
        jj = jj + (i + 1)
        kk = kk + (i + 1)
        chunk = np.column_stack([np.full(len(jj), i, dtype=np.intp), jj, kk])
        buf.append(chunk)
        buf_len += len(chunk)
        if buf_len >= batch_size:
            merged = np.concatenate(buf)
            # Yield full batches
            pos = 0
            while pos + batch_size <= len(merged):
                yield merged[pos:pos + batch_size]
                pos += batch_size
            # Keep remainder
            if pos < len(merged):
                buf = [merged[pos:]]
                buf_len = len(merged) - pos
            else:
                buf = []
                buf_len = 0
    if buf_len > 0:
        yield np.concatenate(buf)


def _combo_batches_np_5(n, batch_size):
    """Yield numpy index arrays for 5-way combinations, generated without Python tuples."""
    buf = []
    buf_len = 0
    for i in range(n - 4):
        for j in range(i + 1, n - 3):
            remaining = n - j - 1
            if remaining < 3:
                continue
            # Generate all 3-way combos from [j+1, n)
            aa, bb, cc = [], [], []
            for a in range(remaining - 2):
                rem2 = remaining - a - 1
                b_idx, c_idx = np.triu_indices(rem2, k=1)
                aa.append(np.full(len(b_idx), a + j + 1, dtype=np.intp))
                bb.append(b_idx + (a + j + 2))
                cc.append(c_idx + (a + j + 2))
            if not aa:
                continue
            aa = np.concatenate(aa)
            bb = np.concatenate(bb)
            cc = np.concatenate(cc)
            chunk = np.column_stack([
                np.full(len(aa), i, dtype=np.intp),
                np.full(len(aa), j, dtype=np.intp),
                aa, bb, cc
            ])
            buf.append(chunk)
            buf_len += len(chunk)
            if buf_len >= batch_size:
                merged = np.concatenate(buf)
                pos = 0
                while pos + batch_size <= len(merged):
                    yield merged[pos:pos + batch_size]
                    pos += batch_size
                if pos < len(merged):
                    buf = [merged[pos:]]
                    buf_len = len(merged) - pos
                else:
                    buf = []
                    buf_len = 0
    if buf_len > 0:
        yield np.concatenate(buf)


def _nway_ensemble_vecs(files, pids, yes_vecs, exp_vec, dirs, n_pairs, args):
    """Fallback n-way ensemble for n_pairs > 64, using the original numpy bool approach."""
    M_per_dir = {d: np.stack([yes_vecs[p][d] for p in pids]) for d in dirs}
    pid_list = list(pids)
    r = args.n_way
    majority_threshold = (r + 1) // 2
    sort_key, sort_rev = SORT_ENSEMBLE_KEYS[args.sort.lower()]
    top_k = args.heap_size
    heap = []
    counter = 0
    for batch in _combo_batches(len(pid_list), r, 10_000):
        idx = np.array(batch, dtype=np.intp)
        sub_per_dir = [M_per_dir[d][idx] for d in dirs]
        batch_results = []
        if args.ensemble in ('ALL', 'OR'):
            or_per_dir = [s.any(axis=1) for s in sub_per_dir]
            batch_results.append((" OR", _batch_stats_from_dirs(or_per_dir, exp_vec, n_pairs)))
        if args.ensemble in ('ALL', 'AND'):
            and_per_dir = [s.all(axis=1) for s in sub_per_dir]
            batch_results.append((" AND", _batch_stats_from_dirs(and_per_dir, exp_vec, n_pairs)))
        if args.ensemble in ('ALL', 'MAJORITY'):
            maj_per_dir = [s.astype(np.uint8).sum(axis=1) >= majority_threshold for s in sub_per_dir]
            batch_results.append((" MAJORITY", _batch_stats_from_dirs(maj_per_dir, exp_vec, n_pairs)))
        del sub_per_dir
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
    rows = {}
    for _, _, label, stats in heap:
        rows[label] = stats
    return rows


def _heap_update(heap, counter, top_k, hv_arr, combined, n_pairs,
                 exp_bits, not_exp, n_mask,
                 idx, pid_list, suffix, prefix_pids=None):
    """Vectorized top-k: filter candidates in numpy, only do Python heap ops for winners.
    prefix_pids: optional tuple of pid strings to prepend to combo labels."""
    bs = len(hv_arr)
    if len(heap) >= top_k:
        candidates = np.where(hv_arr > heap[0][0])[0]
    else:
        candidates = np.arange(min(bs, top_k - len(heap)))

    prefix = ','.join(prefix_pids) + ',' if prefix_pids else ''
    for ci in candidates:
        hv = int(hv_arr[ci])
        if len(heap) < top_k or hv > heap[0][0]:
            combo_key = prefix + ','.join(pid_list[j] for j in idx[ci]) + suffix
            fp, fn = _scalar_fp_fn(combined, ci, exp_bits, not_exp, n_mask)
            correct = n_pairs - fp - fn
            s = dict(correct=correct, total=n_pairs,
                     pct=100.0 * correct / n_pairs if n_pairs else 0.0,
                     fp=fp, fn=fn)
            if len(heap) < top_k:
                heapq.heappush(heap, (hv, counter, combo_key, s))
            else:
                heapq.heapreplace(heap, (hv, counter, combo_key, s))
            counter += 1
    return counter


def _bitmask_combine_colwise(bits_per_dir, dirs, idx, rule):
    """Combine bitmasks using column-wise indexing (avoids (batch, r) intermediate)."""
    r = idx.shape[1]
    combined = []
    for d in dirs:
        yb = bits_per_dir[d]
        if rule == 'OR':
            c = yb[idx[:, 0]]
            for j in range(1, r):
                c = c | yb[idx[:, j]]
        elif rule == 'AND':
            c = yb[idx[:, 0]]
            for j in range(1, r):
                c = c & yb[idx[:, j]]
        combined.append(c)
    return combined


def _bitmask_majority_colwise(bits_per_dir, dirs, idx, n_pairs, threshold):
    """Majority vote using column-wise indexing and per-bit counting."""
    r = idx.shape[1]
    bs = len(idx)
    bit_masks = np.array([np.uint64(1 << b) for b in range(n_pairs)], dtype=np.uint64)
    combined = []
    for d in dirs:
        yb = bits_per_dir[d]
        accum = np.zeros(bs, dtype=np.uint64)
        for b in range(n_pairs):
            bm = bit_masks[b]
            yes_count = np.zeros(bs, dtype=np.int32)
            for j in range(r):
                yes_count += ((yb[idx[:, j]] & bm) > 0).astype(np.int32)
            accum |= np.where(yes_count >= threshold, bm, np.uint64(0))
        combined.append(accum)
    return combined


def _nway_ensemble_bitmask(files, pids, exp_bits, yes_bits, dirs, n_pairs, args):
    """Run n-way ensemble search using bitmask-accelerated computation.
    Returns dict of {label: stats_dict} for top results."""
    n_mask = np.uint64((1 << n_pairs) - 1)
    not_exp = ~exp_bits & n_mask
    bits_per_dir = {d: yes_bits[d] for d in dirs}
    pid_list = list(pids)
    r = args.n_way
    majority_threshold = (r + 1) // 2
    sort_key, sort_rev = SORT_ENSEMBLE_KEYS[args.sort.lower()]
    top_k = args.heap_size
    heap = []
    counter = 0
    n = len(pid_list)

    def _screen_hv(combined):
        """Compute screening heap-value array from combined bitmasks (1 popcount)."""
        pc, pf = _screen_bits(combined, exp_bits, not_exp, n_mask)
        if sort_key == 'correct':
            return _popcount(pc)
        elif sort_key == 'fp':
            return -_popcount(pf)
        else:  # fn
            return -_popcount(~pf & ~pc & n_mask)

    def run_rules(idx, batch_results):
        if args.ensemble in ('ALL', 'OR'):
            combined = _bitmask_combine_colwise(bits_per_dir, dirs, idx, 'OR')
            batch_results.append((' OR', combined, _screen_hv(combined)))
        if args.ensemble in ('ALL', 'AND'):
            combined = _bitmask_combine_colwise(bits_per_dir, dirs, idx, 'AND')
            batch_results.append((' AND', combined, _screen_hv(combined)))
        if args.ensemble in ('ALL', 'MAJORITY'):
            combined = _bitmask_majority_colwise(bits_per_dir, dirs, idx, n_pairs, majority_threshold)
            batch_results.append((' MAJORITY', combined, _screen_hv(combined)))

    if r == 5:
        # 3+2 decomposition with precomputation.
        # Precompute all C(n,3) three-way and all C(n,2) two-way combined bitmasks.
        # Then 5-way = 3way | 2way (for OR), 3way & 2way (for AND).
        # Group 3-way combos by max index c; pair with 2-way combos (d,e) where d > c.
        # Each 5-way combo counted exactly once.
        do_or  = args.ensemble in ('ALL', 'OR')
        do_and = args.ensemble in ('ALL', 'AND')
        do_maj = args.ensemble in ('ALL', 'MAJORITY')

        # --- Precompute 3-way bitmasks, grouped by max index ---
        # three_by_max[c] = (indices_ab, {d: combined_array})
        # where indices_ab is (A, 2) array of (a,b) pairs with a<b<c
        three_by_max = {}
        for c in range(2, n):
            a_idx, b_idx = np.triu_indices(c, k=1)
            if len(a_idx) == 0:
                continue
            entry = {}
            if do_or:
                entry['or'] = {d: bits_per_dir[d][a_idx] | bits_per_dir[d][b_idx] | bits_per_dir[d][c]
                               for d in dirs}
            if do_and:
                entry['and'] = {d: bits_per_dir[d][a_idx] & bits_per_dir[d][b_idx] & bits_per_dir[d][c]
                                for d in dirs}
            if do_maj:
                entry['maj'] = {d: bits_per_dir[d][a_idx] | bits_per_dir[d][b_idx] | bits_per_dir[d][c]
                                for d in dirs}  # placeholder; majority needs special handling
            entry['ab'] = np.column_stack([a_idx, b_idx])
            three_by_max[c] = entry

        # --- Precompute 2-way bitmasks from each starting index ---
        # two_from[start] = (indices_de, {d: combined_array})
        two_from = {}
        for start in range(3, n - 1):
            pool = n - start
            d_idx, e_idx = np.triu_indices(pool, k=1)
            d_idx = d_idx + start
            e_idx = e_idx + start
            entry = {}
            if do_or:
                entry['or'] = {d: bits_per_dir[d][d_idx] | bits_per_dir[d][e_idx] for d in dirs}
            if do_and:
                entry['and'] = {d: bits_per_dir[d][d_idx] & bits_per_dir[d][e_idx] for d in dirs}
            if do_maj:
                entry['maj'] = {d: bits_per_dir[d][d_idx] | bits_per_dir[d][e_idx] for d in dirs}
            entry['de'] = np.column_stack([d_idx, e_idx])
            two_from[start] = entry

        # --- Iterate: for each max-3-index c, cross-product 3-way × 2-way ---
        batch_size = 1_000_000
        for c in range(2, n - 2):
            if c not in three_by_max or (c + 1) not in two_from:
                continue
            three = three_by_max[c]
            two = two_from[c + 1]
            A = len(three['ab'])     # number of 3-way combos with max index c
            B = len(two['de'])       # number of 2-way combos from [c+1, n)
            if A == 0 or B == 0:
                continue

            # Process cross product in row-batches to limit memory
            batch_rows = max(1, batch_size // B)
            for row_start in range(0, A, batch_rows):
                row_end = min(row_start + batch_rows, A)
                chunk_size = (row_end - row_start) * B
                batch_results = []

                if do_or:
                    combined = []
                    for d in dirs:
                        # (chunk, 1) | (1, B) → (chunk, B) → flatten
                        c5 = three['or'][d][row_start:row_end, None] | two['or'][d][None, :]
                        combined.append(c5.ravel())
                    batch_results.append((' OR', combined, _screen_hv(combined)))

                if do_and:
                    combined = []
                    for d in dirs:
                        c5 = three['and'][d][row_start:row_end, None] & two['and'][d][None, :]
                        combined.append(c5.ravel())
                    batch_results.append((' AND', combined, _screen_hv(combined)))

                if do_maj:
                    # Majority needs full 5-way index for per-bit counting; fall back
                    ab_chunk = three['ab'][row_start:row_end]  # (chunk_rows, 2)
                    de_all = two['de']                          # (B, 2)
                    # Build full index via cross product
                    cr = row_end - row_start
                    ii = np.repeat(ab_chunk, B, axis=0)         # (cr*B, 2)
                    jj = np.tile(de_all, (cr, 1))               # (cr*B, 2)
                    cc = np.full(chunk_size, c, dtype=np.intp)
                    idx5 = np.column_stack([ii, cc[:, None], jj])
                    combined = _bitmask_majority_colwise(bits_per_dir, dirs, idx5, n_pairs, majority_threshold)
                    batch_results.append((' MAJORITY', combined, _screen_hv(combined)))

                # Build index arrays for heap labels only for winning combos
                for suffix, combined, hv in batch_results:
                    if len(heap) >= top_k:
                        candidates = np.where(hv > heap[0][0])[0]
                    else:
                        candidates = np.arange(min(chunk_size, top_k - len(heap)))

                    for ci in candidates:
                        hv_val = int(hv[ci])
                        if len(heap) < top_k or hv_val > heap[0][0]:
                            # Map flat index back to (3way_row, 2way_col)
                            row = ci // B + row_start
                            col = ci % B
                            a, b = int(three['ab'][row, 0]), int(three['ab'][row, 1])
                            d_i, e_i = int(two['de'][col, 0]), int(two['de'][col, 1])
                            combo_key = ','.join(pid_list[x] for x in (a, b, c, d_i, e_i)) + suffix
                            fp, fn = _scalar_fp_fn(combined, ci, exp_bits, not_exp, n_mask)
                            correct = n_pairs - fp - fn
                            s = dict(correct=correct, total=n_pairs,
                                     pct=100.0 * correct / n_pairs if n_pairs else 0.0,
                                     fp=fp, fn=fn)
                            if len(heap) < top_k:
                                heapq.heappush(heap, (hv_val, counter, combo_key, s))
                            else:
                                heapq.heapreplace(heap, (hv_val, counter, combo_key, s))
                            counter += 1
    else:
        for idx in _combo_batches_np(n, r, 1_000_000):
            batch_results = []
            run_rules(idx, batch_results)
            for suffix, combined, hv in batch_results:
                counter = _heap_update(heap, counter, top_k, hv,
                                       combined, n_pairs,
                                       exp_bits, not_exp, n_mask,
                                       idx, pid_list, suffix)

    rows = {}
    for _, _, label, stats in heap:
        rows[label] = stats
    return rows


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


def print_discovery_anchored(seed_key, files, args):
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
        print_bad_pairs(combined, sources=[(pid_a, results_a), (pid_b, results_b)])


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
        print_bad_pairs(combined, sources=list(zip(keys, results_list)))


def print_discovery_ensemble(args, files):
    pids = list(files.keys())
    exp_vec, yes_vecs, dirs, n_pairs = _build_vecs(files)
    rows = {}

    if args.ensemble == 'ALL':
        for pid in pids:
            rows[pid] = _stats_from_dirs([yes_vecs[pid][d] for d in dirs], exp_vec, n_pairs)

    for pid_a, pid_b in combinations(pids, 2):
        ya, yb = yes_vecs[pid_a], yes_vecs[pid_b]
        pair_key = f"{pid_a},{pid_b}"
        if args.ensemble in ('ALL', 'OR'):
            rows[f"{pair_key} OR"]  = _stats_from_dirs([ya[d] | yb[d] for d in dirs], exp_vec, n_pairs)
        if args.ensemble in ('ALL', 'AND'):
            rows[f"{pair_key} AND"] = _stats_from_dirs([ya[d] & yb[d] for d in dirs], exp_vec, n_pairs)

    if args.n_way >= 3:
        if n_pairs <= 64:
            print("fast way")
            exp_bits, yes_bits_bm, _, _, _ = _build_bitmasks(files)
            nway_rows = _nway_ensemble_bitmask(files, pids, exp_bits, yes_bits_bm, dirs, n_pairs, args)
        else:
            print("slow way")
            nway_rows = _nway_ensemble_vecs(files, pids, yes_vecs, exp_vec, dirs, n_pairs, args)
        rows.update(nway_rows)

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
            print_bad_pairs(combined, sources=[(p, files[p]) for p in combo_pids])


# --- top-level runners ---

def run_explicit(args):
    expected = load_expected_pairs(args.pairs)
    results_a = load_eval_results(args.files[0])
    results_b = load_eval_results(args.files[1])
    label_eval_results(results_a, expected, args.method)
    label_eval_results(results_b, expected, args.method)
    resolve_all_pair_labels(results_a)
    resolve_all_pair_labels(results_b)
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
        resolve_all_pair_labels(r)
        files[key] = r
    return files


def run_explicit_nway(args):
    files = load_files_from_keys(args)
    print_discovery_ensemble(args, files)


def run_discovery(args):
    """
    only one file was supplied. discover all other result files in same dir.

    if supplied file is directory, no anchor is specified, do two-way pair-wise
      compare of all files in both orders
    if supplied file is a filename, it's always the anchor, one-way compare with
      all other files

    if ensemble is specified, do the ensemble thing, anchor doesn't matter, i guess?
      seems things like "fixedfp, fixedfn" don't make any sense without an anchor.

    """
    expected = load_expected_pairs(args.pairs)

    files = discover_files_all(args.files[0])
    if not files:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    for eval_results in files.values():
        label_eval_results(eval_results, expected, args.method)
        resolve_all_pair_labels(eval_results)

    print(f"\nDirectory: {Path(args.files[0]).parent}")
    print(f"Found: {len(files)} file(s)\n")

    if args.ensemble:
        print_discovery_ensemble(args, files)
    elif Path(args.files[0]).is_dir():
        print_discovery_all_pairs(files, args)
    else:
        seed_key = _key_from_path(args.files[0])
        print_discovery_anchored(seed_key, files, args)


def parse_args(argv=None):
    """Parse and validate command-line arguments. Returns (args, parser) or exits on error."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='+', metavar='file')
    parser.add_argument('--pairs', required=True, metavar='PAIRS_JSON',
                        help='pairs JSON file with expected values')
    parser.add_argument('--method', default='top-token', metavar='METHOD',
                        help='scoring method (default: top-token)')
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
    args = parser.parse_args(argv)

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
        return args, parser

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

    if len(args.files) > 2:
        parser.error('specify 1 or 2 files')

    return args, parser


def main():
    args, _ = parse_args()

    if args.keys:
        run_explicit_nway(args)
        return

    if len(args.files) == 1:
        run_discovery(args)
    elif len(args.files) == 2:
        run_explicit(args)


if __name__ == '__main__':
    main()
