# Plan: Add directory-mode filtering keyed by a pairs file

## Context

`src/filter.py` currently filters one `.jsonl` of eval results by per-row
prob/label criteria and emits matching pairs to stdout. The user has eval
campaigns with 10s–100s of `.jsonl` files (each 10s–100s of millions of
rows) and wants to filter all of them down to just the rows whose pair is
in a supplied "interest list" (up to ~5M pairs) — while still applying
the existing prob/label flags on top. Today, doing this would require
running filter once per file with no pair-set restriction and
post-processing — slow and wasteful. We want a second filter path that
takes `(pairs_file, results_dir)` and walks the dir.

## Design

### CLI shape

Keep the current single-file CLI working. Add a new mode triggered by a
flag that supplies the directory; the existing positional `file` becomes
the pairs file in that mode:

```
# unchanged single-file mode:
filter.py RESULTS.jsonl -y [--pm ... --pr ... --any]

# new directory mode:
filter.py PAIRS_FILE --dir RESULTS_DIR -y [--pm ... --pr ... --any]
```

`--dir` (or `-d`) is the mode switch. When present:
- positional is interpreted as the pairs file
- `RESULTS_DIR` is scanned for `*.jsonl` (non-recursive — matches how
  `compare.py` treats discovery dirs)
- prob/label filters still apply as an AND constraint on top of
  pair-membership (per user answer)

### Pairs file format

Same line-oriented `word1,word2` format that `filter.py` already emits
and that `pairs/known-pairs.72`, `pairs/pairlist.72`, etc. use. Loader:
read the file line-by-line, strip, skip blanks, insert into a Python
`set[str]`. No JSON path needed in this mode. 5M short strings in a set
is well under 1 GB and lookup is O(1).

### Filtering loop

Reuse the existing `compare_native.iter_projected_blocks` reader — it
already returns blocks with labels/probs/pair_at. New helper, e.g.
`filter_results_dir(pairs_path, results_dir, yes, out_file, pmin, prng, use_max)`:

1. Load pair set from `pairs_path`.
2. `files = sorted(Path(results_dir).glob("*.jsonl"))` — same shape of
   discovery that `compare.discover_files_all` uses (no need to import
   it; this mode doesn't need the key parsing).
3. For each file:
   - `blocks = compare_native.iter_projected_blocks([file], chunk_size=8192)`
   - Wrap with `prefetch` (see "Reuse" below) so JSON parsing of the
     next chunk overlaps Python-side mask building of the current chunk.
   - For each block, build the prob/label mask exactly as in
     `filter_results` today (lines 17–35 of `src/filter.py`).
   - Build a pair-membership mask:
     `pair_mask = np.fromiter((p in pair_set for p in block.pairs()),
                              dtype=bool, count=block.size)`
   - `mask &= pair_mask`
   - Emit `out_file.write(block.pair_at(idx) + "\n")` for each
     `np.flatnonzero(mask)` index.
4. Errors opening any one file should not abort the whole run — log to
   stderr and continue (these campaigns produce partial dirs sometimes).

Output: concatenated to stdout (per user answer). Duplicates may occur
when the same pair qualifies in multiple files — that is accepted, the
user can `sort -u` downstream if desired.

### Reuse of compare.py prefetch

`src/compare.py` has `_prefetch(iterable)` (lines 173–196) — a one-slot
queue + daemon thread that produces ahead by one block. It is private to
`compare.py` today. Lift it into `src/common.py` as `prefetch(iterable)`
and import from both `compare.py` and `filter.py`. Minimal change to
`compare.py`: replace the local def with the import and update its one
caller at line 254.

### Why this works at scale

- The bottleneck for 100M-row files is JSON parsing in the native
  reader, which already releases the GIL (`gil_scoped_release` in
  `compare_native.cpp:194`). The prefetch thread lets us parse the next
  chunk while Python iterates the current one.
- Pair-membership check is a Python `in` on a `set` — fast and adds
  ~tens of ns per row. At 8192 rows/chunk and a single-threaded
  consumer, this is dwarfed by JSON parsing.
- Memory stays bounded: one chunk at a time + the 5M-entry set.

## Files to change

- `src/filter.py` — add `filter_results_dir`, `_load_pair_set`, extend
  `_parse_args` / `_filter_args` to dispatch on `--dir`.
- `src/common.py` — add `prefetch(iterable)` (lifted from
  `compare.py:_prefetch`).
- `src/compare.py` — replace the local `_prefetch` def with
  `from src.common import prefetch` and update its one caller at
  line 254.

No native (`compare_native.cpp`) changes required.

## Verification

1. Build/import smoke: `python -c "from src import filter, compare"` —
   ensures the prefetch refactor didn't break compare's import.
2. Single-file mode regression: run on an existing result file and
   confirm output is identical to the old code, e.g.
   `python -m src.filter results/yesno.200_third_p3_mini.q35-q2.jsonl -y --pm 0.5` —
   compare against the same command on master.
3. Dir mode happy path: build a small pair set (5–10 pairs) from
   `pairs/known-pairs.72`, point `--dir` at `results/`, run with `-y`
   and confirm only those pairs (and matching prob criteria) appear in
   stdout, drawn from multiple files.
4. Dir mode with a pair that exists in no file: confirm empty output,
   no crash.
5. Dir mode with a missing/empty subset file in `--dir`: confirm a
   stderr warning is printed and other files still process.
6. Sanity: with the pair set = full pair list of one file and prob
   range `--pm 0 --pr 1.0`, the dir-mode output for that single file
   should equal `wc -l` of the file (all rows pass).
7. Scale spot-check: time the new path on the largest available
   `.jsonl` and confirm runtime is within ~20% of the old single-file
   filter on the same file with the same prob criteria (pair-membership
   check should not be the bottleneck).
