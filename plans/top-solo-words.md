# Frontier = segments + solo-words, and a workflow-managed dictionary

## Context

The BEST inner loop today is: `gen dfs` → `gen top.segments` → `review` →
`complete` (classify) → regen → review again, with a one-off review branch
alongside. Every verdict it produces is a verdict about a *pair*.

A second kind of judgement is needed: some single words in the dictionary are
junk, and they pollute every search that can spell them. `top-segments` already
has `--solo-words` (`nutrimatic/source/top-segments.cpp:37`, "print only
single-word segments"), so the candidate list is one extra call away from a DFS
file we already have.

The rub is that rejecting a word invalidates the dictionary, which invalidates
the DFS, which costs hours — and the operator does not want to re-search on
every rejection. So the design has to make a *frontier regeneration* (seconds)
absorb a word rejection, leaving the re-search as a thing the operator chooses.

The recently added `"dictionary changed"` reason on `_frontier_outdated`
(`workflow/best/state.py:739`) is a placeholder for exactly this: it reports the
dictionary moved but the regen it offers cannot currently answer it. This plan
makes it answerable.

Intended outcome: `top.segments` and `top.solo-words` are one frontier generated
together; rejected words are recorded once, globally, and are gone from both the
frontier and the next search; and the operator picks which review to do.

---

## MAJOR UNRESOLVED ISSUES — do not implement without resolving first

**1. How a `top.solo-words` review is actually conducted.**
The pair review pipes a bundle through `p2` — `submit.P2` → `evaluate.P2` →
Evernote notes → `complete p2` extracts YES/NO. Reviewing a word list is
probably **manual** instead: the operator eyeballs a ranked word list and writes
down the rejects, with no note round at all. Until this is settled, do not build:

- `Review._solo_words` (bundle creation, cutoff, round numbering)
- any `complete` routing for the solo kind
- `_review_needed`'s "either kind counts" rule (see §3 below — it is written
  one-line-ready but must stay `("segments",)` until this is decided)

Sub-questions that fall out of it:
- Do solo rounds live in `p2/queued` / `p2/eval` / `p2/done/in` at all, or
  somewhere of their own? Round discovery (`_rounds_in`) only scans those three.
- Bundle files are named `*.pairs` and `_review_round`
  (`state.py:487-497`) hardcodes that suffix. A word bundle is not pairs.
- Should a solo review in flight block a segments review? `review_locations`
  (`state.py:526-529`) raises on more than one in-flight bundle for a target,
  so today it would, and that may or may not be wanted.
- Does the reviewed bundle carry counts (`  1234 word`) or bare words?

Everything else in this plan is independent of that decision and can be built
now. The verdict-recording path is `wf best exclude-words` (§4), a hand-driven
command that needs no review mechanics.

**2. Whether `-n` should be shared between the two frontier artifacts.**
One `gen top.segments` will make two `top-segments` calls. `--top-count` /
`-n` currently means "frontier rows". Sharing one cutoff for both is the
assumption below; a separate solo cutoff is a CLI change if wanted.

---

## Not in scope

- `top.all-words`. The third kind is designed for (nothing hardcodes two) but
  not wired.
- Any change in `/home/mike/code/nutrimatic`. Notably, a `top-segments`
  word-level reject turns out **not** to be needed: the derived dictionary keeps
  rejected words out of new searches, and the Python post-filter (§3) keeps them
  out of a frontier built from an old one.
- `dict-remove` / `dict_remove.py` is a one-off hack (in-place edit of a
  hardcoded `~/code/nutrimatic/idx/words.big`, numbered `.removed.N` archives).
  **Do not call it.** Reuse its two good ideas only: strip a leading count
  prefix off removal input (`^ *[0-9]+ `), and take the difference with
  `comm -23` under `LC_ALL=C` — which `workflow/setops.py` already provides.

---

## Design

### 1. Review kinds: rename `top` → `segments`, add `solo-words`

`workflow/best/state.py:460`:

```python
REVIEW_KINDS = ("segments", "solo-words", "oneoff")
# The kinds that count as reviewing the frontier. Keep at ("segments",) until
# the solo review mechanics are settled.
FRONTIER_KINDS = ("segments",)
```

`Target.review_prefix` (`state.py:110-118`) needs no change — it already
interpolates the kind, and no kind is a prefix of another, so the trailing-dot
guard still holds. `_review_round`, `_rounds_in`, `review_rounds`, and
`_check_round_ordinals` are already kind-generic.

Replace the four hardcoded `kind == "top"` filters with membership in
`FRONTIER_KINDS`:

- `state.py:862` `_review_queued`
- `state.py:872` `_review_evaluating`
- `state.py:908` `_review_needed`
- `commands.py:81` `_preflight_top_segments`

And the two literals in `Review._top` (`commands.py:384-385`):
`review_rounds(target, archived, "segments")`, `review_prefix('segments')`.

### 2. One frontier, two artifacts, one marker

`generate.gen_top_segments` (`generate.py:210-255`) makes two `top-segments`
calls from the same DFS file and writes the marker once, after both:

| artifact | flags |
|---|---|
| `top.segments` | `--pairs [-n N] --wfroot ROOT -y <dfs>` (unchanged) |
| `top.solo-words` | `--solo-words [-n N] --wfroot ROOT -y <dfs>` |

- Marker stays `.top.segments.gen` and keeps holding the source (`seed`/`best`),
  written by `mark_generated` *after* both placements — the existing
  crash-ordering rule (content before marker, so an interrupted run re-offers
  itself) now covers both files.
- The `gen` stage name stays `top.segments` (`Gen.STAGES`, `commands.py:126`).
  It is a mild wart that one stage name produces two artifacts; renaming the
  stage would churn the marker name the design depends on.
- `_no_frontier` (`state.py:887-903`) requires **both** artifacts and names
  whichever is missing. On existing trees this fires once per target and is
  answered by one regen (seconds).
- `_top_segments_choices`, `gen_top_command`, `prepare` need no change.

### 3. Post-filter both artifacts against `no.solo-words`

This is what makes a word rejection answerable by a regen instead of a
re-search. `top-segments` output is ordered by descending count, not by
collation, so `comm` cannot be used here — filter by streaming in Python:

- Load `no.solo-words` into a set (bare words, one per line).
- `top.solo-words`: each row is `<count> <word>`; strip the count prefix
  (`^ *[0-9]+ `) to get the key, drop the row if the key is rejected.
- `top.segments`: each row is comma-separated words; drop the row if **any**
  word is rejected. (`--wfroot` already does the equivalent for rejected
  *pairs*; this is the word-level parallel.)

`setops._place` runs an argv and captures stdout, so it cannot host a filter
step. Add a sibling that places an already-written file with the same
compare-and-rename tail:

```python
def place_file(src: Path, dst: Path, stable_mtime: bool = False) -> Path
```

so `stable_mtime=True` still means "a byte-identical regeneration leaves the
artifact's mtime alone", which the whole staleness design rests on.

### 4. `best/dict/`: base, verdicts, derived

| file | what it is |
|---|---|
| `words.big` | hand-placed base (today a symlink to nutrimatic's copy). **Never written by the workflow.** |
| `no.solo-words` | union-only global rejection set, sorted-unique bare words |
| `words.dfs` | derived: `words.big` − `no.solo-words`; this is what `--dict` gets |

- `Target.dictionary` (`state.py:86-88`) repoints from `words.big` to
  `words.dfs`. `_dfs_inputs` (`generate.py:94`) then passes the derived file,
  and `Inputs.dictionary` / `frontier_outdated`'s `"dictionary changed"` reason
  date against it — so a rejection that removes nothing (word not in the dict)
  leaves `words.dfs` byte-identical under `stable_mtime=True` and costs nothing.
- Derivation is `setops.diff(words_big, no_solo_words, words_dfs,
  stable_mtime=True)` — `comm -23` under `LC_ALL=C`, which is what
  `dict_remove.py` does by hand. Both inputs are already C-sorted sets
  (`words.big` verified; `no.solo-words` is built by `setops.fold`).
- New command `wf best dict` rebuilds `words.dfs` and reports line counts.
- New command **`wf best exclude-words FILE`**, mirroring `Exclude`
  (`commands.py:311-329`) and `workflow/classify.py`: strip count prefixes from
  FILE, `setops.fold` into `no.solo-words`, then rebuild `words.dfs`. This is
  the verdict-recording path that needs no review mechanics, so a solo review
  can be done by hand today.
- `wf init` is idempotent (`init.py:8-19`), so no layout node needs a
  migration script; `no.solo-words` is created empty by `ensure_file` the way
  `classified/yes|no` are.

### 5. Staleness, after the above

- `_review_needed` (`state.py:906-917`) compares against the frontier's
  **content** clock, which becomes
  `max(top.segments.st_mtime_ns, top.solo-words.st_mtime_ns)`. This preserves
  the termination property exactly: a no-op regen moves neither file, so G3
  stays declined and the loop ends.
- One new row, immediately above `_next_search`: `_dictionary_stale` — fires
  when `no.solo-words` is newer than `words.dfs`, offering `wf best dict`
  (seconds, ahead of the hours). Normally never fires, since `exclude-words`
  rebuilds; it catches a hand-edit of `no.solo-words`.
- Remove the "deliberate and temporary" hedge from `frontier_outdated`'s
  docstring: the regen it offers now genuinely drops rejected words.

The resulting loop: reject words → `no.solo-words` grows → `words.dfs` rebuilt
→ `_frontier_outdated` says "dictionary changed" → regen (seconds) → post-filter
drops the words from both artifacts → content moves → `_review_needed` reopens
→ review. The re-search stays an explicit choice at G7.

---

## Files to change

| file | what |
|---|---|
| `workflow/best/state.py` | `REVIEW_KINDS`/`FRONTIER_KINDS`; four `kind == "top"` filters; `Target.dictionary` → `words.dfs`; `Inputs.top_solo_words` + frontier content clock; `_no_frontier` requires both; `_dictionary_stale` row + `ROWS` entry; `frontier_outdated` docstring |
| `workflow/best/generate.py` | `gen_top_segments` makes two calls, post-filters both, one `mark_generated`; the solo/pairs filter helper |
| `workflow/best/commands.py` | `_preflight_top_segments` filter; `Review._top` literals; new `Dict` and `ExcludeWords` actions + dispatcher entries |
| `workflow/setops.py` | `place_file` (compare-and-rename tail of `_place`, reused) |
| `workflow/config.py` | `_BEST["parts"]["dict"]` gains the managed files; a `solo_words()` accessor beside `classified()` |
| `workflow/init.py` | `ensure_file` for `no.solo-words` |
| `tests/test_workflow_best.py` | bundle-name literals `top.` → `segments.`; note `_complete_files` (line ~73) hand-builds the prefix instead of calling `review_prefix` — fix it to derive |
| `tests/test_workflow_best_rows.py` | `_top` fixture places both artifacts; new tests for `_dictionary_stale`, the two-artifact content clock, `_no_frontier` on a missing solo file |
| `tests/test_workflow_best_e2e.py` | stub `top-segments` twice per gen; assert both artifacts and one marker |

---

## Migration (operator, one time)

`final/.wf` has three targets' worth of archived rounds. The rename touches
three places, not just `done/in` — `wf best notes` re-derives from the
`done/out` names:

```
.wf/p2/done/in/top.*.pairs            → segments.*.pairs
.wf/p2/done/out/top.*.p2.{yes,no}     → segments.*.p2.{yes,no}
.wf/p2/done/out/enex/top.*/           → segments.*/   (directories)
```

Then `wf init` (creates `no.solo-words`), `wf best dict` (creates `words.dfs`),
and one `wf best gen ... top.segments --source <recorded>` per target to place
`top.solo-words`.

---

## Verification

1. `python -m unittest discover -s tests -p 'test_workflow*.py'` — 245 tests
   pass today; expect additions, no removals.
2. Unit: a rejection recorded via `exclude-words` removes the word from
   `words.dfs`, and the next `gen top.segments` drops it from **both**
   artifacts; a rejection of a word not in the dictionary leaves `words.dfs`
   byte-identical and does not mark any search stale.
3. Unit: two regens with no intervening change leave both artifacts' mtimes
   untouched and bump only the marker — `_review_needed` stays declined
   (the loop terminates).
4. E2E: `_no_frontier` fires on a tree with `top.segments` but no
   `top.solo-words`, and one regen clears it.
5. Live, read-only first: `cd final && ./wf best status s7 -a` before and after,
   and confirm `dfs-anagrams` is invoked with `--dict .../words.dfs`
   (`generate.py` `_display_dfs` prints the argv) without actually running the
   hours-long search.
