import heapq
import numpy as np

from common import (compute_stats, expected_yesno_from_labeled_result,
                    pair_has_yes, print_displayed_keys)

from ensemble import ENSEMBLE_RULES_2


SORT_DIFF_KEYS = {
    'score':  ('score',    True),
    'fixfp':  ('fixed_fp', True),
    'fixfn':  ('fixed_fn', True),
    'newfp':  ('new_fp',   False),
    'newfn':  ('new_fn',   False),
    'fp':     ('ens_fp',   False),
    'fn':     ('ens_fn',   False),
}


def _common_pair_entries(results_1, results_2):
    """Yield shared pair entries as (pair, data_1, data_2) without materializing intersections."""
    if len(results_1) <= len(results_2):
        items = results_1.items()
        other = results_2
        left_first = True
    else:
        items = results_2.items()
        other = results_1
        left_first = False

    for pair, data in items:
        other_data = other.get(pair)
        if other_data is None:
            continue
        if left_first:
            yield pair, data, other_data
        else:
            yield pair, other_data, data


def _pair_ensemble_label(d1, d2, rule):
    """Return the 2-way ensemble pair label using the given rule (OR or AND)."""
    result_1 = d1.get('result')
    result_2 = d2.get('result')
    if result_1 is None or result_2 is None or 'label' not in d1:
        return None

    expected = expected_yesno_from_labeled_result(d1)
    if expected is None:
        return None

    if rule == 'OR':
        pair_has_yes = False
        for dir_ in result_1['logprobs']:
            if result_1[dir_].token == 'YES' or result_2[dir_].token == 'YES':
                pair_has_yes = True
                break
    else:
        assert rule == 'AND', rule
        pair_has_yes = True
        for dir_ in result_1['logprobs']:
            if result_1[dir_].token != 'YES' or result_2[dir_].token != 'YES':
                pair_has_yes = False
                break

    if expected == 'YES':
        return 'correct' if pair_has_yes else 'fn'
    return 'fp' if pair_has_yes else 'correct'


def _pair_diff_from_counts(fixed_fp, fixed_fn, new_fp, new_fn,
                           correct, fp, fn, *, s1, s2, results=None):
    total = ens_correct + ens_fp + ens_fn
    score = 2 * fixed_fn + fixed_fp - new_fp
    diff = dict(
        fixed_fp=fixed_fp,
        fixed_fn=fixed_fn,
        new_fp=new_fp,
        new_fn=new_fn,
        score=score,
        s1=s1,
        s2=s2,
        ens_pct=100 * correct / total if total else 0.0,
        ens_fp=fp,
        ens_fn=fn,
    )
    if results is not None:
        diff['ens_results'] = results
    return diff


def compute_pair_diff(results_1, results_2, *, rule, include_ens_results=False, s1=None, s2=None):
    """Compare anchor labels to the 2-way ensemble computed with standard label semantics."""
    fixed_fp = fixed_fn = new_fp = new_fn = 0
    ens_results = {} if include_ens_results else None
    ens_correct = ens_fp = ens_fn = 0
    for pair, d1, d2 in _common_pair_entries(results_1, results_2):
        old_label = d1.get('label')
        if old_label is None:
            continue

        ens_label = _pair_ensemble_label(d1, d2, rule)
        if ens_label is None:
            continue

        if old_label == 'fp' and ens_label != 'fp':
            fixed_fp += 1
        elif old_label == 'fn' and ens_label != 'fn':
            fixed_fn += 1
        elif old_label == 'correct':
            if ens_label == 'fp':
                new_fp += 1
            elif ens_label == 'fn':
                new_fn += 1

        if ens_label == 'correct':
            ens_correct += 1
        elif ens_label == 'fp':
            ens_fp += 1
        else:
            ens_fn += 1

        if include_ens_results:
            ens_results[pair] = {'label': ens_label}

    if s1 is None:
        s1 = compute_stats(results_1)
    if s2 is None:
        s2 = compute_stats(results_2)
    return _pair_diff_from_counts(
        fixed_fp, fixed_fn, new_fp, new_fn,
        ens_correct, ens_fp, ens_fn,
        s1=s1, s2=s2, ens_results=ens_results,
    )


def _stats_from_label_masks(valid_mask, correct_mask, fp_mask, fn_mask):
    """Build a stats dict from precomputed label bitmasks."""
    total = valid_mask.bit_count()
    correct = correct_mask.bit_count()
    fp = fp_mask.bit_count()
    fn = fn_mask.bit_count()
    pct = 100 * correct / total if total else 0.0
    return dict(correct=correct, total=total, pct=pct, fp=fp, fn=fn)


def _build_2way_diff_masks(files):
    """Precompute compact per-file masks for fast 2-way diff discovery."""
    pair_bits = {}
    expected_yes_mask = 0

    for results in files.values():
        for pair, data in results.items():
            if data.get('label') is None:
                continue
            bit = pair_bits.get(pair)
            if bit is None:
                bit = 1 << len(pair_bits)
                pair_bits[pair] = bit
                if expected_yesno_from_labeled_result(data) == 'YES':
                    expected_yes_mask |= bit

    masks_by_key = {}
    stats_by_key = {}
    for key, results in files.items():
        valid_mask = correct_mask = fp_mask = fn_mask = any_yes_mask = 0
        for pair, data in results.items():
            label = data.get('label')
            if label is None:
                continue
            bit = pair_bits[pair]
            valid_mask |= bit
            if label == 'correct':
                correct_mask |= bit
            elif label == 'fp':
                fp_mask |= bit
            else:
                fn_mask |= bit

            if pair_has_yes(data):
                any_yes_mask |= bit

        masks_by_key[key] = (valid_mask, correct_mask, fp_mask, fn_mask, any_yes_mask)
        stats_by_key[key] = _stats_from_label_masks(valid_mask, correct_mask, fp_mask, fn_mask)

    return expected_yes_mask, masks_by_key, stats_by_key


def _pair_diff_from_masks(mask_1, mask_2, expected_yes_mask, *, s1, s2):
    """Compute one ordered 2-way diff row from precomputed per-file masks."""
    valid_1, correct_1, fp_1, fn_1, any_yes_1 = mask_1
    valid_2, _, _, _, any_yes_2 = mask_2

    common = valid_1 & valid_2
    ens_yes_mask = common & (any_yes_1 | any_yes_2)
    ens_fp_mask = common & ~expected_yes_mask & ens_yes_mask
    ens_fn_mask = common & expected_yes_mask & ~ens_yes_mask
    ens_fp = ens_fp_mask.bit_count()
    ens_fn = ens_fn_mask.bit_count()
    ens_correct = common.bit_count() - ens_fp - ens_fn

    return _pair_diff_from_counts(
        (fp_1 & common & ~ens_fp_mask).bit_count(),
        (fn_1 & common & ~ens_fn_mask).bit_count(),
        (correct_1 & ens_fp_mask).bit_count(),
        (correct_1 & ens_fn_mask).bit_count(),
        ens_correct, ens_fp, ens_fn,
        s1=s1, s2=s2,
    )


def _pair_diff_both_from_masks(mask_1, mask_2, expected_yes_mask, *, s1, s2):
    """Compute both ordered 2-way diff rows from precomputed per-file masks."""
    valid_1, correct_1, fp_1, fn_1, any_yes_1 = mask_1
    valid_2, correct_2, fp_2, fn_2, any_yes_2 = mask_2

    common = valid_1 & valid_2
    ens_yes_mask = common & (any_yes_1 | any_yes_2)
    ens_fp_mask = common & ~expected_yes_mask & ens_yes_mask
    ens_fn_mask = common & expected_yes_mask & ~ens_yes_mask
    ens_fp = ens_fp_mask.bit_count()
    ens_fn = ens_fn_mask.bit_count()
    ens_correct = common.bit_count() - ens_fp - ens_fn

    return (
        _pair_diff_from_counts(
            (fp_1 & common & ~ens_fp_mask).bit_count(),
            (fn_1 & common & ~ens_fn_mask).bit_count(),
            (correct_1 & ens_fp_mask).bit_count(),
            (correct_1 & ens_fn_mask).bit_count(),
            ens_correct, ens_fp, ens_fn,
            s1=s1, s2=s2,
        ),
        _pair_diff_from_counts(
            (fp_2 & common & ~ens_fp_mask).bit_count(),
            (fn_2 & common & ~ens_fn_mask).bit_count(),
            (correct_2 & ens_fp_mask).bit_count(),
            (correct_2 & ens_fn_mask).bit_count(),
            ens_correct, ens_fp, ens_fn,
            s1=s2, s2=s1,
        ),
    )


def print_2way_diff_table(rows):
    """Print 2-way diff table. rows is a list of (anchor_pid, complement_pid, diff_dict)."""
    wa = max((len(a) for a, _, _ in rows), default=len('anchor'))
    wa = max(wa, len('anchor'))
    wc = max((len(c) for _, c, _ in rows), default=len('complement'))
    wc = max(wc, len('complement'))
    rules = set(d.get('rule', 'OR') for _, _, d in rows)
    multi_rule = len(rules) > 1
    if multi_rule:
        wr = max(len(r) for r in rules)
        wr = max(wr, len('Rule'))
        rule_cols = f"{'Rule':<{wr}s} {'Rule%':>6s} {'FP':>4s} {'FN':>4s}"
        sep_len = wa + wc + 84 + wr + 2
    else:
        rule = next(iter(rules))
        rule_cols = f"{rule + '%':>6s} {'FP':>4s} {'FN':>4s}"
        sep_len = wa + wc + 84
    print(f"{'anchor':<{wa}s} {'corr%':>6s} {'FP':>4s} {'FN':>4s}  "
          f"{'complement':<{wc}s} {'corr%':>6s} {'FP':>4s} {'FN':>4s} | "
          f"{'score':>5s} {'FixFP':>5s} {'FixFN':>5s} {'NewFP':>5s} {'NewFN':>5s} | "
          f"{rule_cols}")
    print(f"{'─'*sep_len}")
    for anchor, complement, d in rows:
        s1, s2 = d['s1'], d['s2']
        rule = d.get('rule', 'OR')
        base = (f"{anchor:<{wa}s} {s1['pct']:5.1f}% {s1['fp']:>4d} {s1['fn']:>4d}  "
                f"{complement:<{wc}s} {s2['pct']:5.1f}% {s2['fp']:>4d} {s2['fn']:>4d} | "
                f"{d['score']:>+5d} {d['fixed_fp']:>5d} {d['fixed_fn']:>5d} "
                f"{d['new_fp']:>5d} {d['new_fn']:>5d} | ")
        if multi_rule:
            print(f"{base}{rule:<{wr}s} {d['ens_pct']:5.1f}% {d['ens_fp']:>4d} {d['ens_fn']:>4d}")
        else:
            print(f"{base}{d['ens_pct']:5.1f}% {d['ens_fp']:>4d} {d['ens_fn']:>4d}")
    print()


def run_2way_nopairs_projected(block_iter, rule, method='top-token'):
    """Process projected native blocks without materializing pair-keyed dicts."""
    import compare_native

    if method != 'top-token':
        raise RuntimeError(f"projected native loader does not support method {method!r}")

    rules = ENSEMBLE_RULES_2 if rule == 'ALL' else [rule]
    total_pairs = 0
    yes_counts = {}
    combined_yes = {r: 0 for r in rules}
    file_keys = None

    for block in block_iter:
        keys = list(block.keys())
        if file_keys is None:
            file_keys = keys
        for fk in keys:
            if fk not in yes_counts:
                yes_counts[fk] = 0

        labels = np.asarray(block.labels())
        if labels.ndim != 3:
            raise RuntimeError("projected native block labels must be 3D")
        if labels.shape[0] != len(keys):
            raise RuntimeError("projected native block key count mismatch")

        file_yes = np.any(labels == compare_native.LABEL_YES, axis=2)
        total_pairs += int(labels.shape[1])

        for file_index, fk in enumerate(keys):
            yes_counts[fk] += int(file_yes[file_index].sum())

        for r in rules:
            if r == 'OR':
                pair_yes = np.any(file_yes, axis=0)
            else:
                assert r == 'AND', r
                pair_yes = np.all(file_yes, axis=0)
            combined_yes[r] += int(pair_yes.sum())

    if file_keys is None:
        file_keys = []
    wa = max(len(file_keys[0]), len('file_a')) if file_keys else len('file_a')
    wc = max(len(file_keys[1]), len('file_b')) if len(file_keys) > 1 else len('file_b')
    header = (f"{'file_a':<{wa}s} {'YES%':>6s} {'YES':>6s} {'NO':>6s}  "
              f"{'file_b':<{wc}s} {'YES%':>6s} {'YES':>6s} {'NO':>6s} | "
              f"{'rule':>5s} {'YES%':>6s} {'YES':>6s} {'NO':>6s}")
    print(header)
    print('─' * len(header))
    s = [{}, {}]
    for i, fk in enumerate(file_keys[:2]):
        s[i]['yes'] = yes_counts[fk]
        s[i]['no'] = total_pairs - s[i]['yes']
        s[i]['pct'] = 100 * s[i]['yes'] / total_pairs if total_pairs else 0.0
    for r in rules:
        c_yes = combined_yes[r]
        c_no = total_pairs - c_yes
        c_pct = 100 * c_yes / total_pairs if total_pairs else 0.0
        print(f"{file_keys[0]:<{wa}s} {s[0]['pct']:5.1f}% {s[0]['yes']:>6d} {s[0]['no']:>6d}  "
              f"{file_keys[1]:<{wc}s} {s[1]['pct']:5.1f}% {s[1]['yes']:>6d} {s[1]['no']:>6d} | "
              f"{r:>5s} {c_pct:5.1f}% {c_yes:>6d} {c_no:>6d}")
    print()


def print_explicit_2way_diff(files, args, ensemble_rule):
    key_a, key_b, *_ = iter(files)
    s1 = compute_stats(files[key_a])
    s2 = compute_stats(files[key_b])
    rules = ENSEMBLE_RULES_2 if ensemble_rule == 'ALL' else [ensemble_rule]
    rows = []
    bad_rows = []
    for rule in rules:
        d = compute_pair_diff(files[key_a], files[key_b],
                              rule=rule, include_ens_results=args.bad, s1=s1, s2=s2)
        d['rule'] = rule
        rows.append((key_a, key_b, d))
        bad_rows.append((f"{key_a},{key_b} {rule}", d))
    print_2way_diff_table(rows)
    return bad_rows


def _diff_sort_value(diff, sort):
    """Return a comparable value where larger is always a better diff row."""
    sort_key, sort_rev = SORT_DIFF_KEYS[sort.lower()]
    if sort_key == 'score':
        return (diff['ens_pct'], -diff['ens_fp'], -diff['ens_fn'])
    primary = diff[sort_key] if sort_rev else -diff[sort_key]
    return (primary, diff['ens_pct'])


def _retain_top_rows(rows, limit, value_fn):
    """Retain the best `limit` rows from an iterable using a min-heap."""
    heap = []
    counter = 0
    for row in rows:
        hv = value_fn(row)
        item = (hv, counter, row)
        if len(heap) < limit:
            heapq.heappush(heap, item)
        elif hv > heap[0][0]:
            heapq.heapreplace(heap, item)
        counter += 1
    return [row for _, _, row in sorted(heap, key=lambda item: item[1])]


def _sort_diff_rows(rows, sort):
    """Sort list of (anchor, complement, diff) by the given sort field. Returns sorted list."""
    return sorted(rows, key=lambda x: _diff_sort_value(x[2], sort), reverse=True)


def _diff_bad_rows(rows):
    """Convert displayed diff rows into the labeled form expected by print_bad_pairs."""
    return [(f"{anchor},{complement} {diff['rule']}", diff) for anchor, complement, diff in rows]


def print_2way_diff_anchored(anchor_key, files, args):
    """Print anchor-based diff table: anchor_key vs all other discovered files."""
    expected_yes_mask, masks_by_key, stats_by_key = _build_2way_diff_masks(files)
    anchor_stats = stats_by_key[anchor_key]
    anchor_masks = masks_by_key[anchor_key]
    rows = []
    for key in files:
        if key == anchor_key:
            continue
        d = _pair_diff_from_masks(
            anchor_masks,
            masks_by_key[key],
            expected_yes_mask,
            s1=anchor_stats,
            s2=stats_by_key[key],
        )
        rows.append((anchor_key, key, d))
    sorted_rows = _sort_diff_rows(rows, args.sort)
    displayed_rows = sorted_rows[:args.top] if args.top else sorted_rows
    print_2way_diff_table(displayed_rows)
    if args.print_keys:
        print_displayed_keys([(complement, diff) for _, complement, diff in displayed_rows])
    return _diff_bad_rows(displayed_rows)


def _iter_2way_diff_all_pair_rows(files, stats_by_key, masks_by_key, expected_yes_mask):
    """Yield ordered all-pairs diff rows while computing each unordered pair only once."""
    items = list(files)
    reverse_rows = {}
    for i, anchor_key in enumerate(items):
        for j, complement_key in enumerate(items):
            if i == j:
                continue
            if j < i:
                yield reverse_rows.pop((i, j))
                continue

            diff_ab, diff_ba = _pair_diff_both_from_masks(
                masks_by_key[anchor_key],
                masks_by_key[complement_key],
                expected_yes_mask,
                s1=stats_by_key[anchor_key],
                s2=stats_by_key[complement_key],
            )
            yield (anchor_key, complement_key, diff_ab)
            reverse_rows[(j, i)] = (complement_key, anchor_key, diff_ba)


def print_2way_diff_all_pairs(files, args):
    """Print diff table for all ordered pairs of discovered files."""
    expected_yes_mask, masks_by_key, stats_by_key = _build_2way_diff_masks(files)
    rows = _retain_top_rows(
        _iter_2way_diff_all_pair_rows(files, stats_by_key, masks_by_key, expected_yes_mask),
        args.heap_size,
        lambda row: _diff_sort_value(row[2], args.sort),
    )
    sorted_rows = _sort_diff_rows(rows, args.sort)
    displayed_rows = sorted_rows[:args.top] if args.top else sorted_rows
    print_2way_diff_table(displayed_rows)
    if args.print_keys:
        print_displayed_keys([((f"{anchor},{complement}"), diff) for anchor, complement, diff in displayed_rows])
    return _diff_bad_rows(displayed_rows)
