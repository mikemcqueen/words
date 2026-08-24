# Plan: Anchor-filtered n-way discovery (FILE + -3)

## Context
When `compare.py` is called with a single FILE (not DIR) and `-3`, it currently discovers all C(n,3) triple combinations across every file in the parent directory. The user wants to filter so that only combinations that **include the anchor FILE** are considered — i.e., FILE + any 2 other files. There is already a TODO comment at line 77–79 of `compare.py` noting this exact gap.

The design uses a **set** of anchor keys/indices so that it generalises naturally to multiple anchor files (e.g. 2 FILEs + `-3` → all 3 slots must include both supplied files). Constraint: `len(anchor_keys) <= n_way` (validated in `parse_args` when multi-file anchoring is added later).

## Files to modify
- `src/compare.py` — build `anchor_keys` frozenset in `run_discovery`, pass to `print_discovery_ensemble`
- `src/ensemble.py` — `print_discovery_ensemble` filters 2-way and n-way combinations; `nway_ensemble_bitmask` gains an `anchor_indices` frozenset guard for r==3

---

## Changes

### 1. `src/compare.py` — `run_discovery` (line ~283–289)

Build `anchor_keys` as a frozenset (empty = no filter):

```python
def run_discovery(files, expected, args):
    assert len(args.files) == 1

    p = Path(args.files[0])
    print(f"\nDirectory: {p if p.is_dir() else p.parent}")
    print(f"Found: {len(files)} file(s)\n")

    if args.ensemble:
        anchor_keys = frozenset() if p.is_dir() else frozenset([key_from_path(args.files[0])])
        return ensemble.print_discovery_ensemble(args, files, anchor_keys=anchor_keys)
    elif p.is_dir():
        return diff.print_2way_diff_all_pairs(files, args)
    else:
        anchor_key = key_from_path(args.files[0])
        return diff.print_2way_diff_anchored(anchor_key, files, args)
```

### 2. `src/ensemble.py` — `print_discovery_ensemble` signature and 2-way loop

Add `anchor_keys=frozenset()` parameter. Filter the 2-way `combinations` loop and forward `anchor_indices` to bitmask:

```python
def print_discovery_ensemble(args, files, anchor_keys=frozenset()):
    keys = list(files.keys())
    ...
    for key_a, key_b in combinations(keys, 2):
        if anchor_keys and not anchor_keys.issubset({key_a, key_b}):
            continue
        ...

    if args.n_way >= 3:
        anchor_indices = frozenset(keys.index(k) for k in anchor_keys if k in keys)
        if args.ensemble == 'ALL':
            rows.update(run_nway('OR', anchor_indices=anchor_indices))
            rows.update(run_nway('AND', anchor_indices=anchor_indices))
            rows.update(run_nway('MAJORITY', anchor_indices=anchor_indices))
        else:
            rows.update(run_nway(args.ensemble, anchor_indices=anchor_indices))
```

The `run_nway` closure forwards `anchor_indices` to `nway_ensemble_bitmask`.

### 3. `src/ensemble.py` — `nway_ensemble_bitmask` anchor filtering for r==3

Add `anchor_indices=frozenset()` parameter. Inside the `r == 3` branch, before computing `np.triu_indices`:

```python
if r == 3:
    for c_idx in range(2, n_keys):
        if anchor_indices:
            ab_required = []
            skip = False
            for ai in anchor_indices:
                if ai == c_idx:
                    pass            # covered by c_idx
                elif ai > c_idx:
                    skip = True     # ai can never appear (all indices < c_idx)
                    break
                else:               # ai < c_idx — must land in a or b
                    ab_required.append(ai)
            if skip:
                continue
        a_idx, b_idx = np.triu_indices(c_idx, k=1)
        if anchor_indices:
            for ai in ab_required:
                mask = (a_idx == ai) | (b_idx == ai)
                a_idx, b_idx = a_idx[mask], b_idx[mask]
        A = len(a_idx)
        if A == 0:
            continue
        ...  # rest of r==3 logic unchanged
```

---

## Verification

```bash
# DIR mode — should be unchanged (all combos):
python compare.py results/ -p pairs.json -3

# FILE mode — every row label must contain the anchor key:
python compare.py results/some_file.jsonl -p pairs.json -3
```
