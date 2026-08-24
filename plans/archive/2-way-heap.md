## Refactor 2-Way Diff Discovery to Use Bounded Retention

### Summary
Refactor 2-way diff discovery to use the same bounded-retention pattern as the existing n-way heap code, but not by forcing it through the current n-way vectorized helpers. Implement a small shared heap utility for "keep best N rows by sort key" and make `print_2way_diff_all_pairs()` stream pairwise diffs through it instead of building the full `rows` list.

Use a separate retention limit for 2-way diff, matching your chosen direction: `--heap-size` is the retained-row budget, while `--top` remains display count.

### Implementation Changes
- Add a shared helper for bounded top-N retention over generic row objects:
  - Input: iterable of candidate rows, retention size, and a function that computes a comparable heap value from a row based on the active sort mode.
  - Output: retained rows, sorted into final display order before printing.
- Keep the existing n-way batched/vectorized search logic intact, but migrate its final heap push/replace logic to the shared helper only if that can be done without disturbing the screening optimization.
- Change `print_2way_diff_all_pairs()` in [compare.py](/home/mike/code/words/compare.py:970) to:
  - iterate ordered pairs as it does now,
  - compute one diff at a time,
  - push `(anchor, complement, diff)` into the bounded-retention helper,
  - avoid storing all pairwise rows.
- Change `compute_pair_diff()` in [compare.py](/home/mike/code/words/compare.py:199) so it can return a compact summary form for table/ranking use:
  - default for discovery/all-pairs should omit `or_results`,
  - keep a mode that includes `or_results` only for explicit 2-file diff or `--bad` paths that actually need per-pair details.
- Preserve current 2-way ordering semantics by using the same comparison logic as `_sort_diff_rows()`:
  - `score`: rank by `(or_pct desc, or_fp asc, or_fn asc)` as today,
  - `fixfp` / `fixfn` / `newfp` / `newfn`: preserve existing ordering and tie-break behavior.
- Reuse `--heap-size` for 2-way discovery retention:
  - retained rows = best `heap_size`,
  - displayed rows = first `top` after final sorting of retained rows,
  - if `top > heap_size`, either cap display to retained rows or validate and reject; choose one explicit policy and document it in help text. Recommended: validation error when `top > heap_size` in this mode.

### Public Interface / Behavior
- No new CLI flag required if `--heap-size` is reused for 2-way diff discovery.
- Update `--heap-size` help text to clarify it now applies to bounded-retention discovery modes, including 2-way all-pairs diff.
- Keep explicit 2-file diff and anchored 2-way diff behavior unchanged unless you also want to optimize anchored mode later.

### Test Plan
- Functional:
  - 2-way all-pairs output with small fixture set matches current output for the top rows under each sort mode.
  - explicit 2-file diff still supports `--bad` and still has access to `or_results`.
  - anchored 2-way diff remains unchanged.
- Limit behavior:
  - `--top <= --heap-size` returns correctly sorted displayed rows.
  - `--top > --heap-size` follows the chosen validation/capping policy.
- Regression:
  - n-way ensemble output is unchanged for existing cases.
  - tie ordering remains stable enough for deterministic tests on fixed fixtures.
- Memory:
  - large synthetic all-pairs run no longer grows with O(number of pairwise rows retained); retained row count stays bounded by `heap_size`.

### Assumptions
- The main goal is to fix discovery-mode OOM, not to preserve the ability to fully sort all pairwise rows when retention is bounded.
- Reusing `--heap-size` for 2-way diff is acceptable.
- The biggest avoidable allocation in 2-way diff is retaining all diff rows plus unnecessary `or_results`; both should be removed from the discovery path.
