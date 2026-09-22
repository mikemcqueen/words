# cluer/query_index.py optimization opportunities

Context: `query_index.py -f -` is slow when fed around a billion `a,b` pair
lines. All of the query work happens in `query()` in `cluer/index.py`.

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

### Wasted work in the plain `-f` path

With no `-r` and no filters (`-j`/`-e`/`--forward`), the first matching clue
still reads `clue_off` and slices its clue text (`index.py:285-286`) before
the `break` at line 301. Neither value is used. Printing the query and moving
on as soon as `matches` is non-empty skips that work.

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

## 3. Per-line Python overhead

This is what's left once #2 is done.

- `clean_words` runs a regex plus `replace` and `lower` on every line. For
  `-f` input that's known to be `word,word`,
  `line.rstrip().lower().split(b",")` is cheaper.
- Reading with `sys.stdin.buffer.read(1 << 24)` in chunks plus
  `splitlines()` beats iterating line by line.
- Use `sys.stdout.buffer.write` instead of `print` + `decode`. That only
  matters if many lines match.

## 4. Mode-specific ideas

- **`-e`**: build a set of lowercased clues that have one or two words. Each
  query then becomes one or two set lookups, with no postings at all.
- **`-j`**: a set of adjacent word-id pairs would be very fast. One catch:
  the current check is a substring test (`phrase in clue_lower`), so
  `new york` also matches `renew yorkshire`. A bigram set would change that
  behaviour, unless that's the behaviour we actually want.

## 5. Parallelism

After #2, split stdin into chunks and run them through
`multiprocessing.Pool.imap` to keep output order. The index is small enough
(about 110MB) to share between worker processes.

## Recommendation

Do #1 plus #2: a fast path in `query()` for two-word `-f` input, with the
neighbor set also used as a pre-filter for `-j`, `-e` and `--forward`. That
should take a billion lines from about 5 hours to well under an hour on one
core.
