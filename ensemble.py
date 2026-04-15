import heapq
import sys
from itertools import combinations, islice
from types import SimpleNamespace

import numpy as np

from common import (compute_stats, expected_yesno_from_labeled_result, print_bad_pairs,
                    print_displayed_keys, print_stats)
from score import TokenLabel, resolve_pair_label


ENSEMBLE_RULES_2 = ["OR", "AND"]
ENSEMBLE_RULES_3 = ["OR", "AND", "MAJORITY"]

SORT_ENSEMBLE_KEYS = {
    'score': ('correct', True),
    'fp':    ('fp',      False),
    'fn':    ('fn',      False),
}


def _row_rank(stats, sort_key):
    if sort_key == 'correct':
        return (stats['correct'], -stats['fp'], -stats['fn'])
    if sort_key == 'fp':
        return (-stats['fp'], stats['correct'])
    if sort_key == 'fn':
        return (-stats['fn'], stats['correct'])
    raise ValueError(f"unsupported sort key: {sort_key}")


def apply_ensemble_labeled(results_list, rule_name):
    """Apply OR/AND/MAJORITY per direction across N files. Returns combined labeled dict."""
    n = len(results_list)
    common = set.intersection(*(set(r) for r in results_list))
    if not common:
        print("No common pairs")
        sys.exit(1)
    majority_threshold = (n + 1) // 2
    combined = {}
    for pair in common:
        datas = [r[pair] for r in results_list]
        if any('result' not in d for d in datas):
            continue
        results = [d['result'] for d in datas]
        dirs = list(results[0]['logprobs'].keys())
        expected = expected_yesno_from_labeled_result(datas[0])
        if expected is None:
            continue

        combined_result = {'logprobs': {}}
        for dir_ in dirs:
            tokens = [r[dir_].token for r in results]
            yes_count = sum(1 for t in tokens if t == 'YES')
            if rule_name == 'OR':
                combined_token = 'YES' if yes_count > 0 else 'NO'
            elif rule_name == 'AND':
                combined_token = 'YES' if yes_count == n else 'NO'
            else:
                combined_token = 'YES' if yes_count >= majority_threshold else 'NO'
            if combined_token == expected:
                dir_label = 'correct'
            elif combined_token == 'YES':
                dir_label = 'fp'
            else:
                dir_label = 'fn'
            combined_result['logprobs'][dir_] = []
            combined_result[dir_] = TokenLabel(token=combined_token, label=dir_label)

        combined[pair] = {'label': resolve_pair_label(combined_result)}
    return combined


def _build_vecs(files_dict):
    """Return (exp_vec, yes_vecs, dirs, n_pairs)."""
    all_keys = list(files_dict.keys())
    pair_sets = [set(p for p, d in files_dict[k].items() if 'label' in d) for k in all_keys]
    common = sorted(set.intersection(*pair_sets))
    if not common:
        print("No common pairs")
        sys.exit(1)
    first_results = files_dict[all_keys[0]]
    first_pair = next(p for p in common if 'result' in first_results[p])
    dirs = list(first_results[first_pair]['result']['logprobs'].keys())
    exp_vec = np.array(
        [expected_yesno_from_labeled_result(first_results[p]) == 'YES' for p in common],
        dtype=np.bool_,
    )
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
    return exp_vec, yes_vecs, dirs, len(common)


def _resolve_labels_np(combined_per_dir, exp_vec):
    dir_fp = np.stack([cv & ~exp_vec for cv in combined_per_dir], axis=0)
    dir_correct = np.stack([(cv & exp_vec) | (~cv & ~exp_vec) for cv in combined_per_dir], axis=0)
    pair_fp = np.any(dir_fp, axis=0)
    pair_correct = np.any(dir_correct, axis=0) & ~pair_fp
    pair_fn = ~pair_fp & ~pair_correct
    return pair_fp, pair_correct, pair_fn


def _stats_from_dirs(combined_per_dir, exp_vec, n):
    pair_fp, pair_correct, pair_fn = _resolve_labels_np(combined_per_dir, exp_vec)
    correct = int(pair_correct.sum())
    fp = int(pair_fp.sum())
    fn = int(pair_fn.sum())
    pct = 100 * correct / n if n else 0.0
    return dict(correct=correct, total=n, pct=pct, fp=fp, fn=fn)


def _reduce_any_yes(combined_per_dir):
    any_yes = combined_per_dir[0].copy()
    for cv in combined_per_dir[1:]:
        any_yes |= cv
    return any_yes


def _score_any_yes(any_yes, exp_bits, not_exp, n_mask, n_pairs):
    pair_fp = any_yes & not_exp
    pair_correct = (any_yes & exp_bits) | (((~any_yes) & n_mask) & not_exp)
    correct = np.bitwise_count(pair_correct).sum(axis=-1)
    fp = np.bitwise_count(pair_fp).sum(axis=-1)
    fn = n_pairs - correct - fp
    return correct, fp, fn


def _build_bitmasks(files_dict):
    all_keys = list(files_dict.keys())
    pair_sets = [set(p for p, d in files_dict[k].items() if 'label' in d) for k in all_keys]
    common = sorted(set.intersection(*pair_sets))
    if not common:
        print("No common pairs")
        sys.exit(1)
    n = len(common)
    n_words = (n + 63) // 64
    first_results = files_dict[all_keys[0]]
    first_pair = next(p for p in common if 'result' in first_results[p])
    dirs = list(first_results[first_pair]['result']['logprobs'].keys())

    exp_bits = np.zeros(n_words, dtype=np.uint64)
    for i, p in enumerate(common):
        if expected_yesno_from_labeled_result(first_results[p]) == 'YES':
            exp_bits[i // 64] |= np.uint64(1 << (i % 64))

    yes_bits = {}
    for d in dirs:
        col = np.zeros((len(all_keys), n_words), dtype=np.uint64)
        for fi, k in enumerate(all_keys):
            for i, p in enumerate(common):
                data = files_dict[k][p]
                if 'result' in data and data['result'][d].token == 'YES':
                    col[fi, i // 64] |= np.uint64(1 << (i % 64))
        yes_bits[d] = col

    return exp_bits, yes_bits, dirs, n, all_keys


def _combo_batches_np(n, r, batch_size):
    if r == 3:
        yield from _combo_batches_np_3(n, batch_size)
    elif r == 5:
        yield from _combo_batches_np_5(n, batch_size)
    else:
        gen = combinations(range(n), r)
        while True:
            batch = list(islice(gen, batch_size))
            if not batch:
                break
            yield np.array(batch, dtype=np.intp)


def _combo_batches_np_3(n, batch_size):
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


def _combo_batches_np_5(n, batch_size):
    buf = []
    buf_len = 0
    for i in range(n - 4):
        for j in range(i + 1, n - 3):
            remaining = n - j - 1
            if remaining < 3:
                continue
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


def _screen_counts(combined_per_dir, exp_bits, not_exp, n_mask, n_pairs):
    return _score_any_yes(_reduce_any_yes(combined_per_dir), exp_bits, not_exp, n_mask, n_pairs)


def _screen_candidates(heap, top_k, sort_key, correct, fp, fn):
    bs = len(correct)
    if len(heap) < top_k:
        return np.arange(min(bs, top_k - len(heap)))

    min_rank = heap[0][0]
    if sort_key == 'correct':
        min_correct, min_neg_fp, min_neg_fn = min_rank
        min_fp = -min_neg_fp
        min_fn = -min_neg_fn
        return np.where(
            (correct > min_correct)
            | ((correct == min_correct) & ((fp < min_fp) | ((fp == min_fp) & (fn < min_fn))))
        )[0]
    if sort_key == 'fp':
        min_fp = -min_rank[0]
        min_correct = min_rank[1]
        return np.where((fp < min_fp) | ((fp == min_fp) & (correct > min_correct)))[0]

    min_fn = -min_rank[0]
    min_correct = min_rank[1]
    return np.where((fn < min_fn) | ((fn == min_fn) & (correct > min_correct)))[0]


def _heap_update(heap, counter, top_k, correct_arr, fp_arr, fn_arr,
                 idx, pid_list, suffix, sort_key, prefix_pids=None):
    prefix = ','.join(prefix_pids) + ',' if prefix_pids else ''
    candidates = _screen_candidates(heap, top_k, sort_key, correct_arr, fp_arr, fn_arr)
    for ci in candidates:
        combo_key = prefix + ','.join(pid_list[j] for j in idx[ci]) + suffix
        correct = int(correct_arr[ci])
        fp = int(fp_arr[ci])
        fn = int(fn_arr[ci])
        total = correct + fp + fn
        s = dict(correct=correct, total=total,
                 pct=100.0 * correct / total if total else 0.0,
                 fp=fp, fn=fn)
        rank = _row_rank(s, sort_key)
        if len(heap) < top_k:
            heapq.heappush(heap, (rank, counter, combo_key, s))
            counter += 1
        elif rank > heap[0][0]:
            heapq.heapreplace(heap, (rank, counter, combo_key, s))
            counter += 1
    return counter


def _bitmask_combine_colwise(bits_per_dir, dirs, idx, rule):
    combined = []
    for d in dirs:
        yb = bits_per_dir[d]
        if rule == 'OR':
            c = yb[idx[:, 0]]
            for j in range(1, idx.shape[1]):
                c = c | yb[idx[:, j]]
        elif rule == 'AND':
            c = yb[idx[:, 0]]
            for j in range(1, idx.shape[1]):
                c = c & yb[idx[:, j]]
        combined.append(c)
    return combined


def _bitmask_majority_colwise(bits_per_dir, dirs, idx, threshold):
    r = idx.shape[1]
    if r == 3:
        assert threshold == 2, f"3-way majority threshold should be 2, got {threshold}"
        combined = []
        for d in dirs:
            yb = bits_per_dir[d]
            a = yb[idx[:, 0]]
            b = yb[idx[:, 1]]
            c = yb[idx[:, 2]]
            combined.append((a & b) | (c & (a | b)))
        return combined
    if r == 5:
        assert threshold == 3, f"5-way majority threshold should be 3, got {threshold}"
        combined = []
        for d in dirs:
            yb = bits_per_dir[d]
            a = yb[idx[:, 0]]
            b = yb[idx[:, 1]]
            c = yb[idx[:, 2]]
            d4 = yb[idx[:, 3]]
            e = yb[idx[:, 4]]
            ab = a & b
            cd = c & d4
            x = a | b
            y = c | d4
            combined.append((ab & y) | (cd & x) | (e & (ab | cd | (x & y))))
        return combined
    raise ValueError(f"bitmask majority only supports 3-way and 5-way, got r={r}")


def _build_three_by_max(bits_per_dir, dirs, n, do_or, do_and, do_maj):
    three_by_max = {}
    for c in range(2, n):
        a_idx, b_idx = np.triu_indices(c, k=1)
        if len(a_idx) == 0:
            continue
        entry = {}
        if do_or:
            entry['any_or'] = None
        if do_or or do_maj:
            entry['or'] = {}
        if do_and or do_maj:
            entry['and'] = {}
        if do_maj:
            entry['maj'] = {}
        for d in dirs:
            yb = bits_per_dir[d]
            a = yb[a_idx]
            b = yb[b_idx]
            cc = yb[c]
            ab = a & b
            a_or_b = a | b
            if do_or or do_maj:
                dir_or = a_or_b | cc
                entry['or'][d] = dir_or
                if do_or:
                    if entry['any_or'] is None:
                        entry['any_or'] = dir_or.copy()
                    else:
                        entry['any_or'] |= dir_or
            if do_and or do_maj:
                entry['and'][d] = ab & cc
            if do_maj:
                entry['maj'][d] = ab | (cc & a_or_b)
        entry['ab'] = np.column_stack([a_idx, b_idx])
        three_by_max[c] = entry
    return three_by_max


def _nway_ensemble_bitmask(pids, exp_bits, yes_bits, dirs, n_pairs, args):
    n_mask = np.full(exp_bits.shape, np.uint64(0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
    tail = n_pairs % 64
    if tail:
        n_mask[-1] = np.uint64((1 << tail) - 1)
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

    def _screen_stats(combined):
        return _screen_counts(combined, exp_bits, not_exp, n_mask, n_pairs)

    def run_rules(idx, batch_results):
        if args.ensemble in ('ALL', 'OR'):
            combined = _bitmask_combine_colwise(bits_per_dir, dirs, idx, 'OR')
            batch_results.append((' OR', combined, *_screen_stats(combined)))
        if args.ensemble in ('ALL', 'AND'):
            combined = _bitmask_combine_colwise(bits_per_dir, dirs, idx, 'AND')
            batch_results.append((' AND', combined, *_screen_stats(combined)))
        if args.ensemble in ('ALL', 'MAJORITY'):
            combined = _bitmask_majority_colwise(bits_per_dir, dirs, idx, majority_threshold)
            batch_results.append((' MAJORITY', combined, *_screen_stats(combined)))

    do_or = args.ensemble in ('ALL', 'OR')
    do_and = args.ensemble in ('ALL', 'AND')
    do_maj = args.ensemble in ('ALL', 'MAJORITY')

    if r == 3:
        for c in range(2, n):
            a_idx, b_idx = np.triu_indices(c, k=1)
            A = len(a_idx)
            if A == 0:
                continue
            idx = np.column_stack([a_idx, b_idx, np.full(A, c, dtype=a_idx.dtype)])
            three = {'or': {}, 'and': {}, 'maj': {}, 'any_or': None}
            for d in dirs:
                yb = bits_per_dir[d]
                a = yb[a_idx]
                b = yb[b_idx]
                cc = yb[c]
                ab_and = a & b
                a_or_b = a | b
                if do_or or do_maj:
                    dir_or = a_or_b | cc
                    if do_or:
                        if three['any_or'] is None:
                            three['any_or'] = dir_or.copy()
                        else:
                            three['any_or'] |= dir_or
                if do_and:
                    three['and'][d] = ab_and & cc
                if do_maj:
                    three['maj'][d] = ab_and | (cc & a_or_b)
            batch_results = []
            if do_or:
                any_yes = three['any_or']
                batch_results.append((' OR', *_score_any_yes(any_yes, exp_bits, not_exp, n_mask, n_pairs)))
            if do_and:
                any_yes = None
                for d in dirs:
                    dir_and = three['and'][d]
                    any_yes = dir_and.copy() if any_yes is None else any_yes | dir_and
                batch_results.append((' AND', *_score_any_yes(any_yes, exp_bits, not_exp, n_mask, n_pairs)))
            if do_maj:
                any_yes = None
                for d in dirs:
                    dir_maj = three['maj'][d]
                    any_yes = dir_maj.copy() if any_yes is None else any_yes | dir_maj
                batch_results.append((' MAJORITY', *_score_any_yes(any_yes, exp_bits, not_exp, n_mask, n_pairs)))
            for suffix, correct_arr, fp_arr, fn_arr in batch_results:
                counter = _heap_update(heap, counter, top_k, correct_arr, fp_arr, fn_arr,
                                       idx, pid_list, suffix, sort_key)
    elif r == 5:
        three_by_max = _build_three_by_max(bits_per_dir, dirs, n, do_or, do_and, do_maj)

        two_from = {}
        for start in range(3, n - 1):
            pool = n - start
            d_idx, e_idx = np.triu_indices(pool, k=1)
            d_idx = d_idx + start
            e_idx = e_idx + start
            entry = {}
            if do_or:
                entry['any_or'] = None
            if do_or or do_maj:
                entry['or'] = {}
            if do_and or do_maj:
                entry['and'] = {}
            if do_maj:
                entry['xor'] = {}
            for d in dirs:
                yb = bits_per_dir[d]
                lhs = yb[d_idx]
                rhs = yb[e_idx]
                if do_or or do_maj:
                    dir_or = lhs | rhs
                    entry['or'][d] = dir_or
                    if do_or:
                        if entry['any_or'] is None:
                            entry['any_or'] = dir_or.copy()
                        else:
                            entry['any_or'] |= dir_or
                if do_and or do_maj:
                    entry['and'][d] = lhs & rhs
                if do_maj:
                    entry['xor'][d] = lhs ^ rhs
            entry['de'] = np.column_stack([d_idx, e_idx])
            two_from[start] = entry

        batch_size = 1_000_000
        for c in range(2, n - 2):
            if c not in three_by_max or (c + 1) not in two_from:
                continue
            three = three_by_max[c]
            two = two_from[c + 1]
            A = len(three['ab'])
            B = len(two['de'])
            if A == 0 or B == 0:
                continue
            batch_rows = max(1, batch_size // B)
            for row_start in range(0, A, batch_rows):
                row_end = min(row_start + batch_rows, A)
                chunk_size = (row_end - row_start) * B
                batch_results = []
                if do_or:
                    any_yes = three['any_or'][row_start:row_end, None, :] | two['any_or'][None, :, :]
                    any_yes = any_yes.reshape(-1, any_yes.shape[-1])
                    batch_results.append((' OR', any_yes, *_score_any_yes(any_yes, exp_bits, not_exp, n_mask, n_pairs)))
                if do_and:
                    any_yes = None
                    for d in dirs:
                        dir_and = three['and'][d][row_start:row_end, None, :] & two['and'][d][None, :, :]
                        dir_and = dir_and.reshape(-1, dir_and.shape[-1])
                        if any_yes is None:
                            any_yes = dir_and
                        else:
                            any_yes |= dir_and
                    batch_results.append((' AND', any_yes, *_score_any_yes(any_yes, exp_bits, not_exp, n_mask, n_pairs)))
                if do_maj:
                    any_yes = None
                    for d in dirs:
                        any3 = three['or'][d][row_start:row_end, None, :]
                        all3 = three['and'][d][row_start:row_end, None, :]
                        maj3 = three['maj'][d][row_start:row_end, None, :]
                        any2 = two['or'][d][None, :, :]
                        all2 = two['and'][d][None, :, :]
                        one2 = two['xor'][d][None, :, :]
                        c5 = (all2 & any3) | (one2 & maj3) | (((~any2) & n_mask) & all3)
                        dir_maj = c5.reshape(-1, c5.shape[-1])
                        if any_yes is None:
                            any_yes = dir_maj
                        else:
                            any_yes |= dir_maj
                    batch_results.append((' MAJORITY', any_yes, *_score_any_yes(any_yes, exp_bits, not_exp, n_mask, n_pairs)))
                for suffix, any_yes, correct_arr, fp_arr, fn_arr in batch_results:
                    candidates = _screen_candidates(heap, top_k, sort_key, correct_arr, fp_arr, fn_arr)
                    for ci in candidates:
                        row = ci // B + row_start
                        col = ci % B
                        a, b = int(three['ab'][row, 0]), int(three['ab'][row, 1])
                        d_i, e_i = int(two['de'][col, 0]), int(two['de'][col, 1])
                        combo_key = ','.join(pid_list[x] for x in (a, b, c, d_i, e_i)) + suffix
                        correct = int(correct_arr[ci])
                        fp = int(fp_arr[ci])
                        fn = int(fn_arr[ci])
                        s = dict(correct=correct, total=n_pairs,
                                 pct=100.0 * correct / n_pairs if n_pairs else 0.0,
                                 fp=fp, fn=fn)
                        rank = _row_rank(s, sort_key)
                        if len(heap) < top_k:
                            heapq.heappush(heap, (rank, counter, combo_key, s))
                            counter += 1
                        elif rank > heap[0][0]:
                            heapq.heapreplace(heap, (rank, counter, combo_key, s))
                            counter += 1
    else:
        for idx in _combo_batches_np(n, r, 1_000_000):
            batch_results = []
            run_rules(idx, batch_results)
            for suffix, combined, correct_arr, fp_arr, fn_arr in batch_results:
                counter = _heap_update(heap, counter, top_k, correct_arr, fp_arr, fn_arr,
                                       idx, pid_list, suffix, sort_key)

    rows = {}
    for _, _, label, stats in heap:
        rows[label] = stats
    return rows


def print_2way_ensemble(two_results, args):
    pid_a, results_a = two_results[0]
    pid_b, results_b = two_results[1]

    n_common = len(set(results_a) & set(results_b))
    if n_common < len(results_a) or n_common < len(results_b):
        print(f"\n  ({pid_a} has {len(results_a)} pairs, {pid_b} has {len(results_b)}, "
              f"{n_common} in common)")

    ind_rows = [(pid_a, compute_stats(results_a)), (pid_b, compute_stats(results_b))]
    rules = [lbl for lbl in ENSEMBLE_RULES_2 if args.ensemble == 'ALL' or lbl == args.ensemble]
    if args.ensemble != 'ALL':
        combined = apply_ensemble_labeled([results_a, results_b], rules[0])
        ens_rows = [(rules[0], compute_stats(combined))]
    else:
        combined = None
        ens_rows = [(lbl, compute_stats(apply_ensemble_labeled([results_a, results_b], lbl)))
                    for lbl in rules]

    sort_key, sort_rev = SORT_ENSEMBLE_KEYS[args.sort.lower()]
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
        print_bad_pairs(combined, sources=two_results)


def print_discovery_ensemble(args, files):
    pids = list(files.keys())
    exp_vec, yes_vecs, dirs, n_pairs = _build_vecs(files)
    rows = {}

    def with_ensemble(ensemble):
        d = dict(vars(args))
        d['ensemble'] = ensemble
        return SimpleNamespace(**d)

    def run_nway(ensemble):
        run_args = with_ensemble(ensemble)
        exp_bits, yes_bits_bm, _, _, _ = _build_bitmasks(files)
        return _nway_ensemble_bitmask(pids, exp_bits, yes_bits_bm, dirs, n_pairs, run_args)

    if args.ensemble == 'ALL':
        for pid in pids:
            rows[pid] = _stats_from_dirs([yes_vecs[pid][d] for d in dirs], exp_vec, n_pairs)

    for pid_a, pid_b in combinations(pids, 2):
        ya, yb = yes_vecs[pid_a], yes_vecs[pid_b]
        pair_key = f"{pid_a},{pid_b}"
        if args.ensemble in ('ALL', 'OR'):
            rows[f"{pair_key} OR"] = _stats_from_dirs([ya[d] | yb[d] for d in dirs], exp_vec, n_pairs)
        if args.ensemble in ('ALL', 'AND'):
            rows[f"{pair_key} AND"] = _stats_from_dirs([ya[d] & yb[d] for d in dirs], exp_vec, n_pairs)

    if args.n_way >= 3:
        if args.ensemble == 'ALL':
            rows.update(run_nway('OR'))
            rows.update(run_nway('AND'))
            rows.update(run_nway('MAJORITY'))
        else:
            rows.update(run_nway(args.ensemble))

    sort_key, sort_rev = SORT_ENSEMBLE_KEYS[args.sort.lower()]
    sorted_rows = sorted(rows.items(), key=lambda x: _row_rank(x[1], sort_key), reverse=True)
    w = max((len(label) for label, _ in sorted_rows), default=5)
    w = max(w, len('label'))
    print(f"{'label':<{w}s}  {'correct':>16s}  {'FP':>3s}  {'FN':>3s}")
    print(f"{'─'*(w + 28)}")
    displayed_rows = sorted_rows[:args.top]
    for label, stats in displayed_rows:
        print_stats(label, stats, w)
    print()
    if args.print_keys:
        print_displayed_keys([(label.rsplit(' ', 1)[0], stats) for label, stats in displayed_rows])
    return sorted_rows
