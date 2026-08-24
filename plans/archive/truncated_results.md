# Tolerate truncated secondary files in compare_native

## Context

`native/compare_native.cpp` currently throws `pair mismatch ... truncated secondary file` (line 152–155) whenever a secondary JSONL file has fewer rows than the primary inside a chunk. That kills the whole iteration even when the prefix of pairs is perfectly aligned. We want truncation to be a recoverable condition: shrink the chunk's logical pair list to the shortest secondary's length and emit a stderr warning naming the file and the number of rows dropped.

We don't want to pre-read all secondaries before allocating output buffers. Instead, we keep the existing "allocate-once-per-chunk" layout and use the natural fact that truncation only ever drops a *suffix* of rows. We track `live_rows` (the running min across secondaries) and expose only that prefix to Python. The buffer stays sized at `n_rows`; the numpy view returned to Python is sized at `live_rows` with strides that step by the original `n_rows`. Non-contiguous numpy views are first-class, so this costs nothing besides a few unused trailing slots per file.

Pair-mismatch errors (different pair strings within the common prefix) and direction-schema errors continue to throw — only pure suffix truncation is downgraded to a warning.

## Files

- `native/compare_native.cpp` — all logic changes
- `tests/test_compare.py` — add a regression test for truncation tolerance

## Changes

### 1. `ProjectedChunk` (lines 164–204)

Split logical row count from buffer capacity:

- Add a private `size_t capacity_` field alongside `rows_`.
- Constructor signature becomes `ProjectedChunk(keys, directions, rows, capacity, labels, probs)`. `rows_` is what `size` reports and the shape dimension; `capacity_` is the per-file stride in the buffer.
- `labels()` / `probs()` keep shape `[n_keys, rows_, n_dirs]` but compute the outermost stride as `capacity_ * n_dirs * sizeof(...)` instead of `rows_ * n_dirs * sizeof(...)`. Inner two strides are unchanged.
- Add an `else` path so that when `rows_ == 0` we still hand back a valid empty array (existing behavior — just verify the new constructor signature works for the empty-chunk early return at line 252).

### 2. `fill_arrays` (lines 284–296)

Take an explicit `capacity` parameter so the per-file offset doesn't depend on the count of rows actually populated:

```cpp
void fill_arrays(const std::vector<ProjectedRow> &rows, size_t file_index,
                 size_t capacity, std::vector<uint8_t> &labels,
                 std::vector<double> &probs) const {
  const size_t n_dirs = directions_.size();
  const size_t file_offset = file_index * capacity * n_dirs;
  for (size_t row_index = 0; row_index < rows.size(); ++row_index) {
    for (size_t dir_index = 0; dir_index < n_dirs; ++dir_index) {
      const size_t offset = file_offset + row_index * n_dirs + dir_index;
      labels[offset] = rows[row_index].directions[dir_index].label;
      probs[offset] = rows[row_index].directions[dir_index].prob;
    }
  }
}
```

This lets a partially-filled secondary share the same stride as the primary; trailing slots stay at the constructor's zero/`kUnknownLabel` defaults.

### 3. `validate_pair_alignment` (lines 147–162)

Drop the size-mismatch throw entirely. Only walk `min(primary.size(), secondary.size())` and verify pair-string equality on that prefix; differing pair strings still throw `"pair mismatch ... differing pair X"`.

```cpp
const size_t common = std::min(primary.size(), secondary.size());
for (size_t j = 0; j < common; ++j) {
  if (primary[j].pair != secondary[j].pair) {
    throw std::runtime_error("pair mismatch between " + paths[0] + " and " +
                             paths[secondary_index] + ": differing pair " + secondary[j].pair);
  }
}
```

### 4. `read_next_chunk` (lines 249–282)

Allocate buffers at the original `n_rows`, track `live_rows`, accumulate truncation counts per offending file, then emit warnings and construct the chunk with the shrunken logical size:

```cpp
const size_t n_rows = primary.size();           // capacity / stride
size_t live_rows = n_rows;                      // logical size
std::vector<std::pair<size_t, size_t>> truncations;  // (file_index, dropped_count)

std::vector<uint8_t> labels(n_files * n_rows * n_dirs, kUnknownLabel);
std::vector<double>  probs (n_files * n_rows * n_dirs, 0.0);

fill_arrays(primary, 0, n_rows, labels, probs);

for (size_t file_index = 1; file_index < n_files; ++file_index) {
  std::vector<ProjectedRow> rows = read_rows_for_file(file_index, n_rows);
  if (rows.size() < live_rows) {
    truncations.emplace_back(file_index, live_rows - rows.size());
    live_rows = rows.size();
  }
  validate_pair_alignment(paths_, primary, rows, file_index);
  validate_direction_schema(rows, file_index);
  fill_arrays(rows, file_index, n_rows, labels, probs);
}

if (!truncations.empty() && live_rows > 0) {
  for (const auto &t : truncations) {
    std::fprintf(stderr,
                 "WARNING: %s truncated; dropped %zu pair(s) from chunk\n",
                 paths_[t.first].c_str(), t.second);
  }
}

return ProjectedChunk(keys_, directions_, live_rows, n_rows,
                      std::move(labels), std::move(probs));
```

Notes:
- If `live_rows == 0` (a secondary returned no rows at all), we return an empty chunk which will trigger `stop_iteration` on the next `next()` call. Skipping the warning in that case avoids a useless message at end-of-stream when truncation happened to land exactly on a chunk boundary in a *prior* chunk. The misaligned case (secondary ends mid-chunk) still warns.
- `std::fprintf(stderr, ...)` is safe under `gil_scoped_release`; `py::print` is not.
- Add `#include <cstdio>` near the top of the file for `fprintf`.

### 5. Empty-chunk early return (line 252)

Update the call site to match the new constructor: `return ProjectedChunk(keys_, directions_, 0, 0, {}, {});`

### 6. `next()` placeholder (line 237)

Update to `ProjectedChunk chunk({}, {}, 0, 0, {}, {});` to match.

## Verification

1. Build the extension: `make` (or whatever `setup.py build_ext --inplace` invocation the Makefile wires up — confirm by reading `Makefile`).
2. Run the existing native tests: `source ../.torch/bin/activate && python -m pytest tests/test_compare.py -v`. All currently-passing tests must still pass — including `test_projected_loader_pair_mismatch_raises_when_available` (line 121) which checks that *differing* pairs still raise.
3. Add a new test `test_projected_loader_tolerates_truncated_secondary_when_available` in `tests/test_compare.py`:
   - Build two JSONL files with the same `pair` schema, where the secondary has a strict prefix of the primary's rows (e.g. 3 of 5 rows).
   - Capture stderr (`contextlib.redirect_stderr` won't catch C-level fprintf — use `os.dup2` on `sys.__stderr__.fileno()` with a tempfile, or just assert on the chunk shape and skip stderr capture).
   - Iterate the reader and assert `block.size == 3`, that `np.asarray(block.labels()).shape == (2, 3, n_dirs)`, and that the values for both files in the surviving rows match what's in the JSONL.
4. Spot-check end-to-end with a real truncated pair file via `compare.py` against the dataset the user normally uses to confirm the warning prints and the run completes instead of crashing.
