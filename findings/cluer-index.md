# Indexing cluedata for bulk queries

Question: given the `cluedata` format described in `cluer/find.py`, what is an
efficient way to load and index it so that one run can check a file of
potentially millions of search terms?

## What the file contains (measured 2026-09-19)

| Section | Size |
|---|---|
| Answers | 395,019 strings, about 3.8 MB of text |
| Clues | 3,279,505 records, about 73 MB of clue text |
| Answer references | 7,356,101 in total; the most on one clue is 3,757 |
| Clue section end | byte 123.5 MB of the 267 MB file; the rest is the metadata region |

Walking the whole clue section once in plain Python takes about 1.5 seconds.
`find.py` does that walk once per query. With a million queries that's roughly
17 days, so the fix is to stop scanning the file for each query. Either scan it
once for the whole batch, or build an index once and look things up in it.

## 1. Exact clue match: no index needed

If the tool matches whole clues (what `--exact` does), load the query file into
a Python `set` of lowercased latin-1 bytes. A few million entries fits in a few
hundred MB. Then walk the clue records once, as `lookup()` already does, and
test `clue.lower() in queries`. Total time is one pass over the file, about 2
to 3 seconds plus reading the queries, whether there are 10 queries or 10
million. Clue text can repeat across records, so collect every hit rather than
stopping at the first one.

If the database will be queried many times, save a sorted table instead. Make a
numpy array of `(64-bit hash of normalized clue, record offset)` pairs, about
40 MB, and store it next to `cluedata`. Load it with `np.load(mmap_mode="r")`,
hash all the queries at once, and look them up with one vectorized
`np.searchsorted`. Millions of lookups finish in well under a second. Check the
actual clue bytes at each offset to rule out hash collisions.

## 2. Word match (the default mode): build an inverted index

The default mode requires every query word to appear in the clue. For that,
use an index from each word to the clues that contain it. The layout that stays
compact in Python is:

- `tokens`: a sorted vocabulary, or a dict from word to id.
- `postings_start`: a uint32 array marking where each word's list begins.
- `postings`: one flat uint32 array of clue indices.

With a whitespace split, the clues have 571,690 distinct words and 12.1M
word-to-clue entries, about 48 MB as uint32. The cleaning described below
splits on punctuation too, so its vocabulary should be similar or smaller. The
largest lists are `of` (343k clues), `a` (310k), `the` (244k) and `in` (236k).
Save the arrays with `np.save` and memory-map them. Opening them is then
instant.

The index records which clues contain each word, regardless of position. A
multi-word query is the intersection of its words' clue lists, so the words
can be anywhere in the clue, in any order, like `grep -w word1 | grep -w word2`.
For each query:

1. Clean the query with the same function used on the clues (see "Cleaning
   text" below). If any word is missing from the index, the query has no
   matches.
2. Start with the shortest list, then intersect it with each longer list in
   turn. Stop early if the result becomes empty.
3. Look up the answers for the surviving clues through `ref_start`/`refs`
   (option 3).

Don't use `np.intersect1d` for step 2, because it re-sorts both arrays. Use
`pos = np.searchsorted(big, small)` and keep the entries of `small` where
`big[pos] == small`, after clipping `pos` to the last valid index. That takes
time proportional to the short list times the log of the long one. A query
like "of paris" then costs about the size of the "paris" list, even though
"of" is in 343k clues.

This is whole-word matching on purpose. `find.py` does substring matching
(`word in clue_lower`), so `art` matches "p**art**y" there but not in this
index.

## Cleaning text the way nutrimatic's make-index does

Clues are cleaned with the same rule `do_line` in
`../nutrimatic/source/make-index.cpp` applies to corpus text:

- Letters and digits are kept and lowercased. `do_line` uses `isalnum` without
  calling `setlocale`, so only `A-Z a-z 0-9` count. Any byte above 127
  separates words, so "Château" becomes `ch teau`.
- Apostrophes are deleted without leaving a space: "don't" becomes `dont`,
  "Aaron's" becomes `aarons`, "rock 'n' roll" becomes `rock n roll`.
- Every other character, hyphens included, ends the current word, and runs of
  separators collapse into one space. "ex-wife" becomes `ex wife`. Hyphenated
  words are split, not discarded.
- The `___` blank in fill-in clues (121k of them) produces no word, so fill-in
  clues can't be searched for as a group.

`do_line`'s 40-character window and the 10× title weighting only affect
nutrimatic's phrase counts, not how words are split, so they don't apply here.

A Python version that produces the same words:

```python
import re

_WORD = re.compile(rb"[a-z0-9]+")


def clean_words(text: bytes) -> list[bytes]:
    """Split text into words the way nutrimatic's make-index do_line does."""
    return _WORD.findall(text.replace(b"'", b"").lower())
```

It works on the raw latin-1 bytes from `cluedata`. `bytes.lower()` changes only
ASCII letters, just as `tolower` in the C locale does, and non-ASCII bytes
never match `[a-z0-9]`, so they separate words. Use the same function on the
queries, so that a query of "Don't" looks up `dont`.

Checked against `cluedata`: no clue contains a NUL byte, which would stop
`do_line` early, and only 153 of the 3.28M clues contain any non-ASCII byte.
Most are accented letters in French, Italian or Spanish quotations.

This is not the cleanup behind the hyphen-skipping in nutrimatic's
dictionaries. `load_dictionary` and `clean_word` in
`../nutrimatic/source/dfs-cli-args.cpp` skip any line that contains `-` and
delete every character outside `a-z0-9` without leaving a space, so
"New York" becomes `newyork`. That rule treats each dictionary line as one
unit and never splits it into words, so it is the wrong one for clue text.

### Answers

All 395,019 answers contain only `A-Z`: no spaces, digits or punctuation.
Multi-word answers are already joined, as in `AAAAUTOCLUB` and `AARONJUDGE`.
The only cleaning they need is lowercasing, and both nutrimatic rules give
the same result. Because they have no word boundaries, a word lookup can't
find `club` inside `AAAAUTOCLUB`. Searching inside answers means a substring
scan of the answer list, which is small enough (about 3.8 MB) to do directly.

## 3. Shared step for both: turn the file into arrays once

Whichever option is used, do a one-time conversion of the clue section into
flat arrays:

- `answers`: a list of str, or a numpy bytes array plus offsets.
- `clue_off`: a uint32 array of each record's byte offset in `cluedata`. The
  clue text can still be read from the mmap.
- `ref_start`, `refs`: uint32 arrays. The references for clue `i` are
  `refs[ref_start[i]:ref_start[i+1]]`. Keep the full values so the unexplained
  low bit survives; `>> 1` still gives the answer index.

This removes Python parsing from every later run. Reversing the direction
(answer → clues) is then one `np.argsort` over `refs >> 1`, if it is ever
needed.

## Alternative: SQLite

A single SQLite file does everything above for less code:

- a `clues(id, text)` table with an index on `lower(text)` for exact matches
- an FTS5 table for word or substring matches
- a `refs(clue_id, answer_id, flag)` table

To run millions of queries, don't issue them one at a time. Load them into a
temporary table and do one `JOIN`. SQLite will be slower than the numpy arrays,
but it's easier to query ad hoc.

## Recommendation

The bulk tool needs multi-word, whole-word lookup where the words need not be
adjacent. Build the index in option 2 on top of the arrays from option 3, and
split clues and queries into words with `clean_words`. Option 1 remains the
simplest approach if an exact-clue mode is ever needed.
