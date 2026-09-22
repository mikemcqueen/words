# cluer/query_index.py optimization opportunities

Context: `query_index.py -f -` is slow when fed around a billion `a,b` pair
lines. All of the query work happens in `query()` in `cluer/index.py`.

## Status (2026-09-21)

| item | state |
|---|---|
| #1 memmap overhead | Done, commit `c03c51d`, always on |
| #1 wasted work in plain `-f` | Done, commit `c03c51d` |
| #2 neighbor sets | Done, commit `c03c51d`, behind `--sorted-input`, two-word `-f` lines (not `-j`) |
| #3 regex per line | Done, commit `865ff56` |
| #3 chunked stdin, `stdout.buffer.write` | Not started |
| #4 `-j` bigram table | Done, commit `c475c92`; index format version 2 |
| #4 `-e` clue set | Not wanted for now |
| #5 parallelism | Not started |

Timings on the 200k bench (`-e` and `--forward` without `-j` not
benchmarked):

| input | mode | original | #1 | #1 + `--sorted-input` | bigram table |
|---|---|---|---|---|---|
| unsorted | plain | 3.61s | 2.39s | >10s (killed) | — |
| sorted | plain | 2.96s | 1.90s | 1.16s | — |
| unsorted | `-j` | 8.25s | 5.31s | >10s (killed) | 1.16s |
| sorted | `-j` | 7.33s | 5.59s | 4.10s | 1.15s |
| sorted | `-j --forward` | — | — | — | 0.99s |

Most of the ~1.1s figures is fixed startup (loading the index, plus the
~0.9s clue→words build for `--sorted-input` or the bigram set for `-j`), so
per-line gains on a billion lines are larger than these ratios suggest.

## Baseline

Benchmark: 200k random `a,b` lines built from the words in `pairs/pairs.1000`.

- Current code: 3.5s wall time, about 17µs per line. At that rate a billion
  lines takes about 5 hours.
- Index size: 3.28M clues, 196,849 words, 12.2M postings (`postings.npy` is
  48MB, and the whole index is about 110MB).

Top of the cProfile output (6s total under the profiler):

| calls | tottime | function |
|---|---|---|
| 2,404,611 | 1.19s | `memmap.__getitem__` (2.0s cumulative) |
| 195,672 | 1.09s | `ndarray.searchsorted` |
| 195,672 | 0.82s | `intersect` |
| 787,021 | 0.58s | `memmap.__array_finalize__` |
| 395,672 | 0.18s | sort-key lambda at `index.py:271` |

## 1. Memmap subclass overhead (easy fix)

`np.load(..., mmap_mode="r")` returns `np.memmap` objects. Every scalar
index or slice goes through `memmap.__getitem__` and `__array_finalize__`,
which accounts for about a third of the runtime. That covers:

- `starts[i + 1] - starts[i]` in the sort key (`index.py:271`), plus the
  `starts[...]` bounds at lines 273 and 276
- every `postings[...]` slice passed to `intersect`

Fixes:

- Convert `starts` to Python ints once (`starts.tolist()`), or build
  `word -> (lo, hi)` straight away.
- Wrap `postings` and the other arrays in `np.asarray(...)`. That keeps the
  mmap but drops the subclass. Or just load `postings` into RAM, since it's
  only 48MB.

A prototype with only these changes, same algorithm: **3.5s → 2.0s**.

### This isn't disk I/O

After the first touch, the mmap'd pages sit in the OS page cache, so the
data is effectively in memory already. The cost is the `np.memmap` Python
subclass: every index or slice runs extra Python code in
`memmap.__getitem__` and `__array_finalize__`, and a plain `ndarray` view of
the same pages doesn't. So `np.asarray(np.load(..., mmap_mode="r"))` gets
most of the benefit without copying anything. Loading into RAM does the same
job and also avoids first-touch page faults.

### What each index file does per input line

| piece | size | when it's touched | worth it? |
|---|---|---|---|
| `postings_start.npy` (`starts`) | 0.8MB | Every line that passes the vocabulary check: about 6 scalar reads in the sort key (`index.py:271`) and the slice bounds (273, 276) | **Yes, most important.** Use `.tolist()` or a `word -> (lo, hi)` dict. Scalar reads from a plain `ndarray` are still slower than from a Python list. |
| `postings.npy` | 48MB | Two slices plus `searchsorted` per line | **Yes.** `np.asarray` or a full load. |
| `clue_off.npy` | 13MB | Once per candidate clue (`index.py:285`) | **Yes for `-j`/`-e`/`--forward`/`-r`**, which loop over candidates. Better still, gather in one call with `clue_off[matches].tolist()` instead of scalar reads in the loop. |
| cluedata (`data`) | 267MB | Clue text slice per candidate (`index.py:286`) | Not really. It's a stdlib `mmap`, not `np.memmap`, so slicing is already cheap. Copying it into `bytes` saves little and costs 267MB. |
| `ref_start.npy`, `refs.npy` | 13MB, 29MB | Only when printing results (`-r`, or a single QUERY) | No, unless running `-r` with lots of matches. `np.asarray` is harmless, though. |
| `tokens.txt`, `answers.txt` | — | — | Already read into Python lists/dicts |

### Wasted work in the plain `-f` path (done, `c03c51d`)

With no `-r` and no filters (`-j`/`-e`/`--forward`), the first matching clue
still reads `clue_off` and slices its clue text (`index.py:285-286`) before
the `break` at line 301. Neither value is used. Printing the query and moving
on as soon as `matches` is non-empty skips that work.

Done: with `-f` and no `-r`, the query is printed as soon as it matches
(`query_only` in `cluer/index.py`).

## 2. Co-occurrence neighbor sets (the big win)

For plain `-f` with two words and no `-r`, the only question is whether the
two words ever appear in the same clue. Right now each line pays for a
`searchsorted` intersection. Instead:

- Build a clue→word-ids forward index once, by inverting `postings`:
  `np.repeat` word ids over the posting counts, then `argsort` by clue id.
  That takes about 0.9s.
- For word A, compute the set of every word that shares a clue with it, as a
  `frozenset` of word ids, and cache it (for example with `lru_cache`).
- Each line then becomes `ids[b] in nbrs(ids[a])`, a single hash lookup.

Prototype: **0.8s of query time for 200k lines**, with the same 30,579
matches as the current code. Most of that is building neighbor sets for the
first time. If the input is grouped by first word, as `pairs.1000` is
(`twine,wolf`, `twine,woman`, …), almost every lookup hits the cache. The
per-line cost then drops to the Python loop itself, about 1–2µs.

The same neighbor set works as a pre-filter for `-j`, `-e` and `--forward`.
Only lines that pass it need the existing clue-by-clue check.

### As implemented

- Behind `--sorted-input`, which requires `-f`. It applies to two-word lines
  in every mode except `-j`, which uses the bigram table instead (#4).
- Only the most recent lead word's neighbor set is kept, stored as a
  `bytes` mask over word ids (197KB). The clue→words index is built on the
  first two-word line.
- Output is identical with or without the flag. Unsorted input is only
  slower: nearly every line changes the lead word, and rebuilding the set
  for a common word costs more than one intersection. So the default path
  keeps the intersection and gets only #1.

## 3. Per-line Python overhead

This is what's left once #2 is done.

- `clean_words` runs a regex plus `replace` and `lower` on every line. For
  `-f` input that's known to be `word,word`,
  `line.rstrip().lower().split(b",")` is cheaper.

  Done, commit `865ff56`: lines are split on commas, and the regex
  (`clean_words` for plain `-f`, the `fullmatch` check for `-j`/`-e`) only
  runs when a part isn't a known word. Output is unchanged, and bad
  `-j`/`-e` lines still raise the same error. The regexes were already
  compiled once at import. Only the pattern choice and error string were
  rebuilt per line, and those now run only on the fallback. On the 200k
  bench: plain unsorted 2.42s → 2.38s, `--sorted-input` 1.16s → 1.08s,
  `-j` sorted 1.10s → 1.05s, `-e` sorted 3.49s → 3.46s (single runs).
- Reading with `sys.stdin.buffer.read(1 << 24)` in chunks plus
  `splitlines()` beats iterating line by line.
- Use `sys.stdout.buffer.write` instead of `print` + `decode`. That only
  matters if many lines match.

## 4. Mode-specific ideas

- **`-e`**: build a set of lowercased clues that have one or two words. Each
  query then becomes one or two set lookups, with no postings at all.
- **`-j`**: a set of adjacent word-id pairs would be very fast. One catch:
  the current check is a substring test (`phrase in clue_lower`), so
  `wood,wood` also matches 'Firewood wood' (the clue contains the whole word
  `wood`, and the substring `wood wood`). A bigram set would change that
  behaviour.

### `-j` bigram table, as implemented

- `build_index.py` writes `bigrams.npy`: sorted `uint64` keys,
  `first_word_id << 32 | second_word_id`. The build collects keys in an
  `array("Q")` (8 bytes each, instead of about 40 for a list of Python ints)
  and then runs `np.unique`.
- Adjacency rule (`adjacent_words` in `cluer/index.py`): delete apostrophes
  and lowercase, as `clean_words` does, then pair `[a-z0-9]+` runs separated
  by exactly one space. `FORMAT_VERSION` is now 2, so older indexes must be
  rebuilt.
- Size on the real cluedata: 2,242,576 pairs, 17.9MB on disk. It adds about
  10s to the build (29s total). At query time the table is loaded into a
  Python `set` (estimated ~130MB) for fast lookups.
- Plain `-f -j` is a table lookup of `(a, b)` and `(b, a)`; with
  `--forward`, only `(a, b)`. Single QUERY mode and `-r` use the table as a
  pre-filter, then apply the same rule to each candidate clue. They still
  print the original clue string.

Output changes from the substring test, `-j` only (every other mode is
byte-identical):

- **Lost:** substring false positives, for example `wood,wood`
  ('Firewood wood', 'Plywood wood'), 'Two twos', 'What a wolf wolfs'.
- **Gained:** apostrophe cases, for example `yankees,white`
  ("Yankee's White"), `woods,wood` ("Wood's wood"), `winds,will`
  ("...the wind's will").

Open trade-off: a single `-j "new york"` query takes 0.81s instead of
0.24s for a plain query. The extra time is converting the table to a `set`,
which only pays off over many lines. Single queries could instead search the
sorted array directly.

## 5. Parallelism

After #2, split stdin into chunks and run them through
`multiprocessing.Pool.imap` to keep output order. The index is small enough
(about 110MB) to share between worker processes.

## Next steps

- Optionally, skip the `set` conversion for single `-j` queries.
- The rest of #3 (chunked stdin, `sys.stdout.buffer.write`): now the
  largest per-line cost for plain `--sorted-input` runs and for `-j`.
- #5 parallelism, if a single core is still too slow.
