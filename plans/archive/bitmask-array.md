# Plan: Remove 64-pair limit from bitmask ensemble

## Context

The "fast way" bitmask-accelerated n-way ensemble in `compare.py` encodes each pair as a single bit in a `np.uint64`, capping at 64 pairs. With >64 pairs it falls back to a much slower boolean-array path. The fix: use arrays of uint64 words (`W = ceil(n_pairs/64)`) instead of single uint64 scalars, so the bitmask approach works for any pair count.

## File: `/home/mike/code/words/compare.py`

### 1. Add `_popcount_total` helper (after `_popcount`, line 257)

```python
def _popcount_total(arr):
    """Popcount summed across the last (word) axis. (..., W) uint64 -> (...,) int32."""
    return _popcount(arr).sum(axis=-1)
```

### 2. Modify `_build_bitmasks` (line 260)

- Remove `assert n <= 64` (line 269)
- Compute `W = (n + 63) // 64`
- `exp_bits`: `np.zeros(W, dtype=np.uint64)`, set bits via `exp_bits[i // 64] |= np.uint64(1 << (i % 64))`
- `yes_bits[d]`: shape `(n_files, W)`, set bits via `col[fi, i // 64] |= np.uint64(1 << (i % 64))`

### 3. Update `n_mask` / `not_exp` in `_nway_ensemble_bitmask` (line 545)

Replace single-uint64 `n_mask`:
```python
W = exp_bits.shape[0]
n_mask = np.full(W, np.uint64(0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
tail = n_pairs % 64
if tail:
    n_mask[-1] = np.uint64((1 << tail) - 1)
```

### 4. Update `_scalar_fp_fn` (line 327)

- Init `fp_bits` / `correct_bits` as `np.zeros_like(exp_bits)` instead of `np.uint64(0)`
- Replace `bin(int(x)).count('1')` with `int(_popcount(x).sum())`

### 5. Update `_screen_hv` (inner fn, line 557)

- Replace `_popcount(x)` with `_popcount_total(x)` in all three branches

### 6. Update `_bitmask_majority_colwise` (line 523)

- `bit_masks`: shape `(n_pairs, W)`, only one word nonzero per row
- `yb[idx[:, j]] & bm` produces `(bs, W)`; use `.any(axis=-1)` to reduce to `(bs,)` bool
- `np.where` needs `yes_count[:, None] >= threshold` to broadcast against `(W,)` mask
- `accum`: shape `(bs, W)`

### 7. Update 5-way decomposition (line 578+)

- 3-way/2-way precomputation: shapes go from `(A,)` to `(A, W)` automatically via broadcasting
- Cross-product: `three[row_start:row_end, None, :] | two[None, :, :]` → `(chunk, B, W)` → `.reshape(-1, W)` (NOT `.ravel()`)
- Three `.ravel()` calls (OR ~line 651, AND ~line 658, majority reshape) all become `.reshape(-1, W)`

### 8. `_bitmask_combine_colwise` (line 505) — no code change needed

`yb[idx[:, j]]` naturally becomes `(batch, W)`. Bitwise ops work element-wise.

### 9. `_bitmask_stats` (line 296) — swap `_popcount` → `_popcount_total`

### 10. `_screen_bits` (line 312) — no change needed, returns raw bitmasks

### 11. Remove the branch at line 911-918

Always use bitmask path. Remove `if n_pairs <= 64` / `else` and the debug prints.

### 12. Delete dead code

- `_nway_ensemble_vecs` (line 436-473)
- `_combo_batches` (line 222-229) — only used by the deleted function
- `_batch_stats_from_dirs` (line 232-243) — check if used elsewhere first

## Verification

1. Run with a dataset that has <=64 pairs and compare output before/after (should be identical)
2. Run with a dataset that has >64 pairs — should now use fast path and produce correct results
3. Spot-check: boundary cases at exactly 64 and 65 pairs
