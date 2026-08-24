# Plan: Per-direction actual and label in compare.py and score.py

## Context

Each pair result loaded from a evalpair.py result file has `fwd` and `rvs` logprobs which reprent two independent orders, or directions, for that pair.  "fwd" represents the results for the order "one,pair" (pair words in order), and "rvs" represents the results for the order "pair,one" (pair words reversed). Currently both share a single combined `actual`/`label` via the `any-yes` method in score.py, which breaks compare.py's ensemble logic: a file with `fwd=YES` and another with `rvs=YES` appear as two agreeing YESs even though they never agreed in the same direction.

Fix: give each direction its own `actual` and `label` inside new 'fwd`/`rvs` maps. Ensemble logic iterates over directions from `logprobs` keys (generic, no hard-coded fwd/rvs), votes per direction, and the final result is YES if any direction reaches the ensemble threshold. The passing direction is stored in the ensemble result for display purposes.

## Data structures

**Per-file raw result** (after `label_eval_results`):
```python
{
    "logprobs": {"fwd": [...], "rvs": [...]},   # unchanged
    "fwd": {"actual": "YES", "label": "fp"},
    "rvs": {"actual": "NO",  "label": "correct"}
    # no top-level actual/label
}
```

**Ensemble combined result** (output of `apply_ensemble_labeled`):
```python
{
    "actual": "YES",            # ensemble-level YES/NO
    "label": "fp",              # ensemble-level correct/fp/fn
    "direction": "fwd",         # which direction passed (None if ensemble=NO)
}
```

`compute_stats` uses the ensemble-level `actual`/`label`/`direction` (N pairs, not 2N). The FP/FN counts in the score table are naturally direction-driven: a pair is FP/FN based on which direction passed (or didn't), not a cumulative sum across directions.

---

## Critical files

- `score.py` — `label_eval_results`, `extract_top_token`
- `compare.py` — `apply_ensemble_labeled`, `_build_vecs`, `print_discovery_ensemble`, `compute_pair_diff`
- `common.py` — `compute_stats`, `print_bad_pairs`

---

## Changes

### 1. `score.py` — `label_eval_results`

Replace single top-level `actual`/`label` with per-direction entries. Also store `expected` for use downstream:

```python
# Remove: data["actual"] = actual; data["label"] = ...

exp = expected[lookup_key]
directions = list(data.get("logprobs", {}).keys())
for direction in directions:
    dir_actual = extract_top_token(data["logprobs"][direction])
    if dir_actual == exp:       dir_label = "correct"
    elif dir_actual == "YES":   dir_label = "fp"
    else:                       dir_label = "fn"
    data[direction] = {"actual": dir_actual, "label": dir_label}
data["expected"] = exp
```

Stats counters in `label_eval_results` (correct/fp/fn totals returned as `ScoreResult`): count per direction per pair (so totals are N×directions, or keep as N pairs using any-direction-correct logic — TBD, minor).

### 2. `common.py` — `compute_stats`

Add handling for both old shape (top-level `label`) and new shape (per-direction). For new-shape results, use the top-level `label` if present (ensemble results have it), otherwise skip (raw per-file results don't have top-level label).

For raw per-file results, if callers want stats, they must call per-direction. **Simplest approach**: `compute_stats` keeps its current signature but checks for top-level `label` OR falls back to any-direction-correct:

```python
def compute_stats(eval_results):
    correct = fp = fn = 0
    for data in eval_results.values():
        if "label" in data:                         # ensemble result
            label = data["label"]
        else:                                       # raw per-file: any-direction correct
            directions = [k for k in data.get("logprobs", {}) if k in data]
            labels = [data[d]["label"] for d in directions if "label" in data.get(d, {})]
            if not labels: continue
            if any(l == "correct" for l in labels): label = "correct"
            elif all(l == "fp" for l in labels):    label = "fp"
            else:                                   label = "fn"
        if label == "correct": correct += 1
        elif label == "fp":    fp += 1
        elif label == "fn":    fn += 1
    total = correct + fp + fn
    pct = 100 * correct / total if total else 0.0
    return dict(correct=correct, total=total, pct=pct, fp=fp, fn=fn)
```

### 3. `compare.py` — `apply_ensemble_labeled`

Vote per direction across N files. First direction that reaches threshold sets `actual=YES` and records the passing direction:

```python
def apply_ensemble_labeled(results_list, rule_name):
    n = len(results_list)
    common = set.intersection(*(set(r) for r in results_list))
    majority_threshold = (n + 1) // 2
    combined = {}
    for pair in common:
        datas = [r[pair] for r in results_list]
        directions = list(datas[0].get("logprobs", {}).keys())
        if not all(all(d in r for d in directions) for r in datas):
            continue

        expected = datas[0].get("expected")
        if expected is None:         # derive from first direction of first file
            fd = datas[0][directions[0]]
            expected = fd["actual"] if fd["label"] == "correct" else ("NO" if fd["label"] == "fp" else "YES")

        passing_direction = None
        for direction in directions:
            yes_count = sum(1 for d in datas if d[direction]["actual"] == "YES")
            if rule_name == "OR":      passes = yes_count > 0
            elif rule_name == "AND":   passes = yes_count == n
            else:                      passes = yes_count >= majority_threshold  # MAJORITY
            if passes:
                passing_direction = direction
                break

        actual = "YES" if passing_direction else "NO"
        label = "correct" if actual == expected else ("fp" if actual == "YES" else "fn")

        entry = {
            "logprobs": datas[0]["logprobs"],
            "actual": actual,
            "label": label,
            "direction": passing_direction,
            "expected": expected,
        }
        for direction in directions:
            yes_count = sum(1 for d in datas if d[direction]["actual"] == "YES")
            entry[direction] = {"actual": "YES" if yes_count > 0 else "NO"}
        combined[pair] = entry
    return combined
```

### 4. `compare.py` — `_build_vecs`

Build per-direction bool vectors. `yes_vecs` (any-direction) stays for OR and individual rows:

```python
def _build_vecs(files_dict):
    all_keys = list(files_dict.keys())
    # labeled if at least one direction key is present
    first_pair_data = next(iter(files_dict[all_keys[0]].values()))
    directions = list(first_pair_data.get("logprobs", {}).keys())

    pair_sets = [
        set(p for p, d in files_dict[k].items() if all(dir in d for dir in directions))
        for k in all_keys
    ]
    common = sorted(set.intersection(*pair_sets))
    n = len(common)
    first = files_dict[all_keys[0]]

    def expected_yes(d):
        if "expected" in d: return d["expected"] == "YES"
        for direction in directions:
            dd = d.get(direction, {})
            if dd.get("label") == "fn": return True
            if dd.get("label") == "correct" and dd.get("actual") == "YES": return True
        return False

    exp_vec = np.array([expected_yes(first[p]) for p in common], dtype=np.bool_)

    dir_vecs = {
        direction: {
            k: np.array([files_dict[k][p].get(direction, {}).get("actual") == "YES"
                         for p in common], dtype=np.bool_)
            for k in all_keys
        }
        for direction in directions
    }
    yes_vecs = {
        k: np.array(
            [any(files_dict[k][p].get(d, {}).get("actual") == "YES" for d in directions)
             for p in common], dtype=np.bool_)
        for k in all_keys
    }
    return exp_vec, yes_vecs, dir_vecs, directions, n, common
```

Update the single call site in `print_discovery_ensemble` to unpack all six values.

### 5. `compare.py` — `print_discovery_ensemble`: directional AND/MAJORITY

**2-way combinations** (~line 352):
```python
# OR — unchanged (yes_vecs already = any-direction YES)
rows[f"{pair_key} OR"] = _stats_from_vec(ya | yb, exp_vec, n_pairs)

# AND — directional: YES if both=YES in fwd OR both=YES in rvs
and_vec = np.zeros(n_pairs, dtype=np.bool_)
for direction in directions:
    and_vec |= dir_vecs[direction][pid_a] & dir_vecs[direction][pid_b]
rows[f"{pair_key} AND"] = _stats_from_vec(and_vec, exp_vec, n_pairs)
```

**n-way batch loop** (~line 360): add directional stacked matrices:
```python
M = np.stack([yes_vecs[p] for p in pids])               # for OR
M_dirs = {d: np.stack([dir_vecs[d][p] for p in pids]) for d in directions}

# In batch loop:
# OR — unchanged
or_mat = sub.any(axis=1)

# AND — directional
and_mat = np.zeros((len(batch), n_pairs), dtype=np.bool_)
for d in directions:
    sub_d = M_dirs[d][idx]
    and_mat |= sub_d.all(axis=1)

# MAJORITY — directional
maj_mat = np.zeros((len(batch), n_pairs), dtype=np.bool_)
for d in directions:
    sub_d = M_dirs[d][idx]
    maj_mat |= sub_d.astype(np.uint8).sum(axis=1) >= majority_threshold
```

### 6. `compare.py` — `compute_pair_diff`: directional comparison

Compare per-direction labels across two files, each `(pair, direction)` is an independent test case:

```python
def compute_pair_diff(results_1, results_2):
    common = set(results_1) & set(results_2)
    fixed_fp = fixed_fn = new_fp = new_fn = 0
    for pair in common:
        r1, r2 = results_1[pair], results_2[pair]
        directions = [k for k in r1.get("logprobs", {}) if k in r1 and k in r2]
        for direction in directions:
            d1, d2 = r1[direction], r2[direction]
            was_fp = d1["label"] == "fp";    is_fp = d2["label"] == "fp"
            was_fn = d1["label"] == "fn";    is_fn = d2["label"] == "fn"
            was_correct = d1["label"] == "correct"
            if was_fp and not is_fp:    fixed_fp += 1
            if was_fn and not is_fn:    fixed_fn += 1
            if was_correct and is_fp:   new_fp += 1
            if was_correct and is_fn:   new_fn += 1
    s1 = compute_stats(results_1)
    s2 = compute_stats(results_2)
    score = 2 * fixed_fn + fixed_fp - new_fp
    or_correct = s1["correct"] + fixed_fn - new_fp
    or_fp = s1["fp"] + new_fp
    or_fn = s1["fn"] - fixed_fn
    or_total = s1["total"]
    or_pct = 100.0 * or_correct / or_total if or_total else 0.0
    return dict(fixed_fp=fixed_fp, fixed_fn=fixed_fn, new_fp=new_fp, new_fn=new_fn,
                score=score, s1=s1, s2=s2,
                or_pct=or_pct, or_fp=or_fp, or_fn=or_fn)
```

### 7. `common.py` — `print_bad_pairs`

For FP/FN pairs in ensemble output, the `combined` dict has a `direction` field indicating which direction passed. Show it inline with the pair name:

```
FP:
---
avenue,china  [fwd]
  crosswd2.p82  fwd=[YES: 52.5%, NO: 46.5%] rvs=[NO: 67.5%, YES: 31.5%]
  ...
```

For FN (no direction passed), show `[none]` or omit the tag. `print_bad_pairs` receives the `combined` dict which already has `direction` per pair — read it and append to the label.

---

## Verification

```
source ../.torch/bin/activate && python compare.py <dir> --bad -e MAJORITY --n-way 3
```

- `doctor,storm` no longer in FP (fwd:1/3, rvs:1/3 — neither reaches majority=2)
- `avenue,china` — verify per-direction counts confirm its status
- Individual file score %s may shift slightly (compute_stats now any-direction-correct per pair)
