# Hand-edited best.pairs, and a frontier filtered against the classified sets

## Context

`findings/stale-top-segments.md` (nutrimatic) proposes passing
`--wfroot ROOT -y` to `top-segments` when generating `top.segments`, so the
1000-row review frontier holds 1000 *unanswered* candidates instead of 1000
rows some of which already have verdicts. Its one open decision was what to do
about `best.pairs`, which is built by intersecting the frontier with
`classified/yes` and would come out empty on a new target once `-y` hides the
YES pairs.

Investigating that intersection showed it does no useful work:

- `--pairs` is a pure scoring flag. `dfs-class-list.cpp:281-286` looks each
  multi-word index entry up in a hash set at enumeration time and sets
  `DFS_MEMBER_KNOWN_PAIR`, worth a 1e6 multiplier (`dfs-score.h:12-13`).
  Enumeration is bounded by the letter bag, so a pair whose letters do not fit
  is never emitted and never flagged. An unreachable pair in `--pairs` costs
  one hash-set slot and nothing else.
- The seed leg already relies on exactly that. `gen_dfs` passes the raw seed
  file with no eligibility filter of any kind: 98,250 pairs for s6, of which
  22.7% fit the bag; 26,396 for s7, of which 35.9% fit. The best leg applies
  elaborate machinery to cut 112 candidates down to 2.
- The intersection withholds real signal. Against the current tree:

  | target | YES pairs that fit the bag | in `best.pairs` |
  | --- | --- | --- |
  | `s6/u-toyfastmusketsalvo/m4/g4` | 15 | 2 |
  | `s7/u-vindiesel/m4/g4` | 54 | 52 |
  | `s7/u-vindiesel/m4/g5` | 54 | 48 |
  | `s7/u-vindiesel/m4/g6` | 54 | no `best.pairs` yet |

  s6 searches with 2 of the 15 confirmed-good pairs it could use. The
  withholding is self-reinforcing: the frontier comes from `dfs.seed`, so a
  pair the seed search never ranked highly cannot enter `best.pairs`, and never
  gets the bonus that would have surfaced it.
- The accumulator's unique contribution across every target is one line:
  `tiger,lily` in s6, which is not in `classified/yes`.

So `best.pairs` becomes hand-edited only, `gen dfs.best` unions it with
`classified/yes` at run time, and the finding's open decision disappears rather
than being rerouted. This also removes an asymmetry: NO is applied globally at
search time via `--exclude-pairs <root>`, while YES was applied target-locally
through the frontier. A YES verdict is a judgment about English, not about a
letter bag.

The first bullet also runs the other way. If an unreachable pair in `--pairs`
cannot change a score, then *removing* it cannot either — so the list handed to
`--pairs` is filtered down to what this target's bag can spell, and the
filtered list is published beside the results it produced. That one file is
what tells `status` whether a classify anywhere actually changed anything here,
which is the difference between offering a refine that would find something and
offering hours that reproduce the same file byte for byte.

No C++ changes. `top-segments` already implements `--wfroot` and `-y`
(`source/top-segments.cpp`, `source/pair-exclusions.cpp`), and
`source/test-top-segments.sh:129-160` already covers them.

## Outcome

- `top.segments` holds 1000 candidates with no standing verdict, refilled by
  `gen top.segments` after every classify rather than only when a search lands.
- `dfs.best` searches with every confirmed-YES pair its bag can spell, and
  publishes that list as `dfs.best.pairs` beside its results.
- `dfs.best` goes stale when the pairs it could actually use change, not when
  a shared file is touched.
- `best.pairs` is a hand-edited file, optional, that nothing generates.

---

## 1. `best.pairs` becomes hand-edited

### Delete

- `generate.build_best_pairs()` (`workflow/best/generate.py:240-286`), and
  `review_locations` from `generate.py`'s import at line 11-13, which nothing
  else in that module uses.
- `state.best_pairs_manifest()` (`state.py:513-525`).
- `state.yes_pairs_argv()` (`state.py:455-467`) — already documented as called
  by nothing.
- `Inputs.best_pairs` (`state.py:613-614`) and
  `Inputs.gen_best_pairs_command()` (`state.py:699-700`), whose only callers
  are the two rows below.
- Rows `_best_pairs_missing` (`state.py:805`) and `_best_pairs_out_of_date`
  (`state.py:814`), and their `ROWS` entries. Nothing then declares
  `provides="best.pairs"` or `requires=("best.pairs", ...)`.
- `"best.pairs"` from `Gen.STAGES` (`commands.py:126`), its clause in
  `Gen.__init__`'s `positional_help`, and the `-n is not valid for gen
  best.pairs` refusal plus the `build_best_pairs` call that follows it
  (`commands.py:182-184`).
- The `generate.build_best_pairs(target)` call in `Complete.run`
  (`commands.py:556`).

An absent `best.pairs` is now normal, not a state.

### Migration

Leave the three existing `best.pairs` files byte-for-byte. All but
`tiger,lily` are already in `classified/yes`, so the union `gen dfs.best`
computes is a superset of what each one supplied.

Delete the three now-unread `.best.pairs.gen` markers under
`final/.wf/best/`. (`s7/u-vindiesel/m4/g6` has neither file.)

The three targets holding a `dfs.best` have no `dfs.best.pairs`, so the first
`status` after this change reports `dfs.best out of date (usable pair set
changed)` for all three. That is true, and it is the payoff rather than a
regression: each was searched with its own `best.pairs` and would now search
with everything its bag can spell — s6 with 16 pairs instead of 2, s7/g4 with
54 instead of 52, s7/g5 with 54 instead of 48. `s7/u-vindiesel/m4/g6` has no
`dfs.best` at all and is unaffected.

## 2. The pair list `dfs.best` runs with

`state.py` gains `import filecmp` and `import tempfile`, and `setops` on its
`from workflow import` line. These live in `state.py` rather than `generate.py`
because `Inputs` needs them too and `state.py` cannot import `generate.py` —
`generate.py` already imports `state.py`.

### New in `state.py`

```python
def search_bag(target: Target) -> str:
    """The multiset of letters this target's search may spend, sorted."""
```

Reads `target.letters` and returns `_working_bag(target.letter_set, _bag(...))`
— the existing private helpers at `state.py:225-252`, which already handle both
the `-o` and `-u` forms. `_working_bag` returning `None` means the label names
no proper subset of the sentence; raise `ValueError` naming the target address.
`check_letter_set` guarantees this at creation time but returns early for an
existing `letter_set_dir`, so `gen dfs.best` on an established target is the
path that needs the guard.

```python
def search_pair_sources(target: Target) -> list[Path]:
    """The pair files unioned into --pairs, in display order."""
```

`classified/yes` first, then `best.pairs`. `fs.raise_if_not_file` on
`classified/yes` unconditionally. `best.pairs` is optional, so gate on
`best.exists() or best.is_symlink()`: inside the gate call
`fs.raise_if_not_file` and include it, outside the gate omit it. A directory or
a dangling symlink named `best.pairs` is then an error rather than a silent
omission, which is the distinction `_no_frontier` and `_best_pairs_missing`
both drew and the reason the second half of the gate is not redundant.

```python
def build_search_pairs(target: Target, scratch: Path) -> tuple[Path, int]:
    """The pairs dfs.best may use, and how many stood before the bag filter."""
```

Three steps into `scratch`:

1. `setops.merge(search_pair_sources(target), scratch / "union.pairs")`. The
   merge is what normalises a hand-edited `best.pairs` — it may be unsorted and
   may hold duplicates, and `setops.diff` shells out to `comm`, which requires
   `LC_ALL=C` order on both sides.
2. `setops.diff(union, config.classified(root, "no"), scratch / "standing.pairs")`.
   Redundant against `dfs-anagrams` itself — `emit` tests `exclude_pairs` and
   returns before it reaches the pair flag (`dfs-class-list.cpp:262-286`) — but
   it is what makes the counts below honest, and it keeps the file meaning what
   its name says.
3. A Python pass writing `scratch / "dfs.best.pairs"`, keeping each non-blank
   line for which `_without(bag, _bag(line.replace(",", ""))) is not None`
   against `search_bag(target)`. Every line in the live classified sets and
   every `best.pairs` matches `^[a-z]*,[a-z]*$`, so the comma is the only
   non-letter to strip; `_bag` drops whitespace for anything hand-typed. The
   input is already sorted-unique and a filter preserves order, so the result
   is still a set that `comm` and `filecmp` can both read.

Returns the filtered path and the line count of step 2, so a caller can say
"113 confirmed pairs, none spellable here" without a second pass.

The bag filter is the load-bearing step and it is safe for exactly the reason
the Context section gives: enumeration is bounded by the bag, so a pair that
does not fit is never emitted, never looked up, and cannot affect a score
whether it is in the set or out of it.

### `gen_dfs`

The body moves inside a `tempfile.TemporaryDirectory(prefix="wf-dfs-pairs-")`
so the union outlives input validation and the search.

`_dfs_inputs` keeps its role — everything the run reads, checked before
anything is created — and changes in two ways:

- For `final`, replace the `target.artifact("best.pairs")` lookup and the
  `fs.line_count(pairs) == 0` refusal (`generate.py:104-114`) with
  `state.search_pair_sources(target)`.
- Return that source list rather than one path. The seed leg returns `[seed]`
  and is otherwise untouched.

`gen_dfs` then builds the union (final leg only) and applies the refusal from
§3, both before `target.target_dir.mkdir`, and hands the scratch file to
`--pairs`.

On success, after `scratch.replace(rendered)` and `_publish_link`, place the
same file at `target.artifact("dfs.best.pairs")` with
`setops.merge([pairs], destination)` — a `sort -u` over an already-sorted file,
taken for its atomic write-aside-and-rename rather than for the sort.

**After the search, not before.** Written first, an interrupted run would leave
a `dfs.best.pairs` describing a search that never finished, matching whatever
`Inputs` recomputes, and `status` would call the previous `dfs.best` current.
Written last, a crash between the two leaves the record behind, `status`
re-offers the search, and the state heals — the ordering `gen_top_segments`
already spells out for `top.segments` and its marker (`generate.py:194-199`).

`log.success` gains the published list and its size beside the results line, so
the operator sees what the search was actually weighted by.

### What the operator sees

`_display_dfs` (`generate.py:56-64`) is unchanged, and takes no new parameter.
Its `--pairs` argument is already a real path that exists for the life of the
run, and the earlier idea of printing a `sort`/`comm` reconstruction in its
place is no longer expressible: the bag filter is a Python pass, not a shell
one. It also no longer needs to be. The list is a durable artifact now, so the
answer to "what did that run search with" is `cat .../g4/dfs.best.pairs`, and
the printed argv goes back to being what it is everywhere else — a record of
the process that was started. The existing `displayed[2]` hardcode for the bag
stays as it is.

## 3. The dead-end check and the staleness test

Today an empty `best.pairs` means "this target's review confirmed nothing", and
both `_dfs_inputs` and the `_best_pairs_empty` row refuse on it. Under a global
union that goes green as soon as anyone anywhere says YES, so the test becomes
whether any union pair is spellable from this target's bag — which is now just
the size of the file `build_search_pairs` returns.

A bag-fit count is necessary, not sufficient — a pair can fit the letters and
still not be an index entry, and `-g` may put it out of reach. It is strictly
sharper than "did this pair reach the top 1000", it errs only toward keeping a
pair, and it is the only test available without running the search.

### Refusal in `gen_dfs`

After `build_search_pairs`, before `target.target_dir.mkdir`:

```text
no pair in <yes.pairs> or <best.pairs> fits <address>'s letters;
dfs.best would search without pair bonuses. Review more of top.segments,
or add pairs to <best.pairs>
```

Keep it a `ValueError` in the same place as the check it replaces, for the
reason that check's comment gives: a run that cannot help must not cost hours.

### `Inputs`

```python
@cached_property
def usable_pairs(self) -> tuple[int, int, bool]:
    """(standing pairs, how many this bag can spell, whether dfs.best used them)."""
    with tempfile.TemporaryDirectory(prefix="wf-usable-pairs-") as tmp:
        pairs, standing = build_search_pairs(self.target, Path(tmp))
        stored = self.target.artifact("dfs.best.pairs")
        current = (stored.is_file()
                   and filecmp.cmp(pairs, stored, shallow=False))
        return standing, fs.line_count(pairs), current
```

Two subprocesses and a read over ~2200 lines, once per `Inputs`. `report`
builds two or three of those per target — `derive_state`, `walk_rows` under
`--all`, and its own for `oneoff_in_flight` — so `status --all` over the four
live targets costs a couple of dozen short subprocesses. That is the shape the
existing rows already have and it is not worth caching across instances.

`filecmp` caches by `(path, size, mtime)` on both sides, which is what
`setops._place` has to clear because it reuses one temp path. Here the
directory name is unique per call, so no two comparisons can share a key and
there is nothing to clear.

**`status` never writes this file.** It recomputes the list into a temp
directory and compares; `gen_dfs` is the only writer, and only on a run that
finished.

### `_best_pairs_empty` becomes `_no_usable_pairs`

Same position in `ROWS`, same `requires=("top.segments", "seed")`, same choices
— widen, plus a reseed when one is available — and the same reasoning for them:
widening the frontier is orders of magnitude cheaper than a search, and a wider
frontier means more review candidates, more YES verdicts, and more union
entries. What changes:

- Condition: declines unless the second element of `inputs.usable_pairs` is 0.
- Message: `"no confirmed pair fits this target's letters"`.
- Detail: `f"({standing} confirmed pairs, none spellable here)"`.
- Note: hand-adding to `best.pairs` joins retracting NO verdicts as the prose
  way out.

### `best_search_needed`

`state.py:652-667` currently gates on `best.pairs` being present and non-empty,
and dates against `best.pairs` and `hard_no`. It becomes:

```python
standing, usable, current = self.usable_pairs
if usable == 0:
    return []
dfs_best = self.dfs("best")
if not dfs_best.exists():
    return ["missing"]
reasons = []
if not current:
    reasons.append("usable pair set changed")
if _newer(self.hard_no, dfs_best):
    reasons.append("hard-NO set changed")
return reasons
```

The gate is an unchanged reason for the unchanged conclusion: there is no such
search to offer.

`"usable pair set changed"` is a content comparison, not a clock comparison,
and that is the whole point. `classified/yes` is one file shared by every
target: of the 112 confirmed pairs, 54 fit s7's letters and 15 fit s6's, and 40
fit s7 but not s6. Under an mtime test a YES recorded for s7 marks s6's
`dfs.best` stale and offers hours that would reproduce the same file byte for
byte — and, because `top_segments_behind` declines for a source whose search is
needed (`state.py:681-682`), it would also stop a finished `dfs.best` from ever
being read as a frontier. A content test says nothing changed for s6, because
nothing did.

An absent `dfs.best.pairs` reads as changed. There is no record of what the run
used, and re-running is the only way to get one.

### What this does not fix

`hard_no` stays an mtime test, and it has to: `--exclude-pairs <root>` is read
by `dfs-anagrams` at run time and drops whole segments, so a NO on a pair that
was never in the union still changes what the search emits. `no.pairs` is
`stable_mtime` (`config.py:87,91`) so it only moves when a round actually
records a NO — but rounds mostly do, 2039 NO against 112 YES, so in practice
`dfs.best` will still report stale after most completed rounds.

What Option 2 buys is the YES half: a round that records only YES verdicts, and
a YES that this bag cannot spell, both stop being false alarms. Filtering the
exclusion set the same way would mean building a file and changing the flag,
and is out of scope here.

## 4. Filtering the frontier (the finding)

### `gen_top_segments`

`generate.py:186-190`. Before building argv, `fs.raise_if_not_file` on both
`config.classified(target.root, "yes")` and `config.classified(target.root,
"no")`: `pair-exclusions.cpp:29-36` only warns on a missing classified file,
and a warning would produce an unfiltered frontier that the state machine then
believes is filtered.

Then extend argv with `["--wfroot", str(target.root), "-y"]`. `target.root` is
what `gen_dfs` already passes to `--exclude-pairs`, and `--wfroot DIR` resolves
`DIR/.wf/classified/{yes,no}/`.

The two flags are not symmetric, and both are wanted:

- `-y` *ignores* (`top-segments.cpp:104`): a YES pair is not counted, but its
  line still contributes every other segment on it.
- `--wfroot` *rejects* (`top-segments.cpp:95,102`): a line containing a NO pair
  is dropped whole, so it stops contributing counts for its other segments.
  That reshapes the ranking, not just the membership. It is the same semantics
  `--exclude-pairs` applies at search time, so a regenerated frontier stays
  consistent with a re-run search — and it is what does the work here, since
  the DFS file was written with whatever `no.pairs` held when it ran, and the
  common case is that `no.pairs` has grown since.

### New row `_frontier_behind_classified`

The condition goes on `Inputs`, not in the row, because `_converged` in §5
needs the same test:

```python
@cached_property
def frontier_behind_classified(self) -> list[str]:
    """Classified sets written since the frontier was last generated."""
```

`generated = _generated(self.top_segments)`, then `_newer(self.confirmed_yes,
generated)` → `"confirmed-YES set changed"` and `_newer(self.hard_no,
generated)` → `"hard-NO set changed"` — the same strings
`_best_pairs_out_of_date` used for the same two files. This is the pattern
`Inputs`' own docstring describes: conditions shared by more than one caller
live there so the two cannot drift.

The row declines on an empty list and otherwise returns `State(f"top.segments
behind the classified sets ({', '.join(reasons)})")` with choices
`_top_segments_choices(inputs, (inputs.source,))`, the renderer `_no_frontier`
and `_top_segments_behind_dfs` already share.

Date against `_generated(top_segments)`, not `top_segments.stat()`.
`setops._place(..., stable_mtime=True)` leaves the content mtime behind on a
no-op regen, so a content-clock comparison would report stale forever — the
failure `_generated`'s docstring (`state.py:311`) exists to describe.

The marker clock is also what terminates the loop: a no-op regen bumps
`.top.segments.gen`, this row declines, and `_review_needed` above it still
declines because the content mtime did not move.

**It sits below `_top_segments_behind_dfs`, not above it.** The row offers a
regeneration from `inputs.source`, the source the current frontier was recorded
as coming from. If it outranked `_top_segments_behind_dfs`, then a `dfs.best`
that landed at T1 with the marker at T0 and a classify at T2 would be answered
by regenerating from `seed`, which bumps the marker past T1 — and
`_top_segments_behind_dfs` compares the DFS against that marker, so the
finished search would never become a frontier and the hours would be spent
again to recover it. Generating from the newer DFS satisfies both conditions at
once, so the row that names it goes first. The finding put this row above
`_best_pairs_missing` to keep `best.pairs` from being built and then
immediately demanded again; those rows are gone and that argument with them.

### `ROWS`, after both changes

```text
G0  _letters_missing
    _seed_missing                  provides="seed"
G1  _review_queued
    _review_evaluating
G2  _no_frontier                   provides="top.segments"
G3  _review_needed                 requires=("top.segments",)
G4  _no_usable_pairs               requires=("top.segments", "seed")   was _best_pairs_empty
G5  _top_segments_behind_dfs       requires=("top.segments", "seed")
G6  _frontier_behind_classified    requires=("top.segments",)          NEW
G7  _next_search                   requires=("seed",)
```

G6 sits below `_review_needed` so a freshly generated frontier gets reviewed
rather than immediately regenerated, and above `_next_search` so the seconds
are offered before the hours. Rewrite the group comments in `ROWS`; the old G4
comment ("derived set; everything below has a current best.pairs") goes with
the rows it described.

Right after `wf best complete`, the archived round is newer than
`top.segments`, so `_review_needed` declines and G6 fires — which is the
payoff: the same DFS refills 1000 fresh candidates instead of the frontier
moving only when a search finishes.

## 5. Two consequences elsewhere

### `Review._top`

Keep `setops.diff(remaining, confirmed_yes, ...)` (`commands.py:407-409`). It
is a no-op for a `-y`-filtered frontier and still load-bearing for `_oneoff`,
whose supplied file has been through no filter at all.

Replace its comment (`commands.py:398-405`), which explains the subtraction by
`build_best_pairs` retaining YES pairs across rounds — a rationale that no
longer exists. The rationale now: a YES verdict is global and reaches `--pairs`
directly, so re-asking buys nothing.

### `Review._converged`

Its docstring (`commands.py:459-465`) says status cannot report "every frontier
pair already has a verdict" because "row 6 is an mtime comparison and neither
clock moved". `_frontier_behind_classified` now does report it: the classify
that produced those verdicts moved a classified set past the frontier's marker.

`_converged` stays — it is reached at review time, in the window between a
classify and the next frontier regen — but its guidance changes. It already
builds an `Inputs` (`commands.py:468`), so: when
`inputs.frontier_behind_classified` is non-empty, offer
`inputs.gen_top_command(inputs.source)` ahead of `inputs.search_choices()`.
Same condition as the §4 row and the same renderers, no new command spellings
and no second copy of the test.

## 6. Tests

Smoke coverage only, per the repo's standing preference.

- `tests/test_workflow_best_rows.py` — the `_best_pairs` helper (line 65) and
  every test naming `_best_pairs_missing`, `_best_pairs_out_of_date` or
  `_best_pairs_empty` (lines 216-241, 292-313, 360-399, 406-424, 508-525) need
  reworking against the new table. The `ROWS` name list at line 329 and the
  `unmet` assertions at 508-511 are the two that must match it exactly.
- `_steady()` (line 79) currently reaches a usable pair by accident: the bag
  under `u-cdef` is `ab`, `_classified("yes")` is empty, and the only thing
  putting `a,b` in the union is `_best_pairs("a,b\n")`. Make that deliberate —
  write a bag-fit pair into `classified/yes` — so the fixture does not depend
  on a file the plan makes optional. `_steady` also needs a `dfs.best.pairs`
  matching the union, or every test below G4 fires on `usable pair set
  changed`.
- New: `_frontier_behind_classified` fires when `yes.pairs` post-dates
  `.top.segments.gen` and declines after `mark_generated`; `_no_usable_pairs`
  fires when no union pair fits the bag and declines when one does;
  `_top_segments_behind_dfs` outranks `_frontier_behind_classified` when both
  hold.
- `tests/test_workflow_best_e2e.py` — lines 122, 156, 174, 210-214, 262-270
  assert on generated `best.pairs` content and its `.gen` marker. Rework to
  assert that a classify moves the frontier's staleness instead, and that
  `dfs.best` is invoked with a `--pairs` file holding the bag-filtered union.
- `tests/test_workflow_best.py` — more than the removed stage. The stage list
  at 1009 no longer holds `best.pairs`, and these all exercise deleted code:
  `test_gen_best_pairs_accumulates_manual_entries_and_preserves_mtime` (655),
  `test_gen_best_pairs_uses_target_oneoffs_without_a_frontier` (686),
  `test_best_pairs_write_before_manifest_is_reoffered_and_heals` (715, which
  calls `generate.build_best_pairs` directly),
  `test_complete_selects_target_bundle_and_generates_best_pairs` (847),
  `test_an_empty_best_pairs_refuses_both_final_searches` (1183), the
  `state.yes_pairs_argv` assertion at 1365-1367, and
  `test_the_best_pairs_marker_stays_empty_and_unparsed` (1369).
- New: `gen dfs.best` publishes `dfs.best.pairs` holding the bag-filtered
  union, and publishes it only on a run that finished; a classify of a pair the
  bag cannot spell leaves `best_search_needed` empty, and one it can spell
  fills it with `usable pair set changed`.

## Verification

```bash
cd ~/code/words && python -m pytest tests/test_workflow_best.py \
    tests/test_workflow_best_rows.py tests/test_workflow_best_e2e.py
```

Then against the live tree, read-only first:

```bash
cd ~/code/nutrimatic && ./wf best status --all
```

`classified/{yes,no}.pairs` are both stamped 2026-08-31 16:42. The frontier
markers are s6/g4 2026-08-28 16:49, s7/g4 2026-08-29 23:31, s7/g5 2026-08-31
18:26, s7/g6 2026-09-01 06:57 — so `_frontier_behind_classified` is behind for
the first two and current for the last two, and the `--all` table should read
that way wherever a higher row does not win first. No target should mention
`best.pairs` any more. The three targets holding a `dfs.best` should report
`usable pair set changed`, for the reason §1's migration note gives.

`s7/u-vindiesel/m4/g6` is the interesting one: `dfs.seed` and a 1000-row
`top.segments`, no review yet, no `best.pairs`, no `dfs.best`. It sits at
`_review_needed` before and after. What changes is what comes next — it now
reaches `_next_search` with all 54 bag-fit YES pairs available instead of only
what its first review round happened to surface.

Then the frontier regen, which takes seconds:

```bash
./wf best gen s7 -u vindiesel -g 5 top.segments --source best
```

Confirm 1000 rows still, that `comm -12` against both classified sets is empty,
and that `.top.segments.gen` advanced.

That target is at `review needed (frontier from best)` and the regen replaces a
frontier nobody has reviewed. That is deliberate and it discards nothing.
`top.segments` is derived from `dfs.best` and the classified sets and holds no
human input; `_preflight_top_segments` (`commands.py:70-87`) guards a bundle in
flight because the notes were derived from it and `complete` folds verdicts back
against it, and there is no bundle here — nothing queued, nothing evaluating,
no notes. Reviewing first would mean reviewing the worse frontier: the current
one was built without `--wfroot -y`, so `Review._top` would subtract the
already-answered pairs and hand over fewer than 1000 real candidates. g5 is the
target to pick for exactly that reason — its frontier came from `dfs.best`,
which was weighted toward confirmed pairs, so `-y` shows the largest change
there. The regen moves `top.segments` past the newest archived round, so the
target stays at `_review_needed` and the operator's next step is unchanged. No
discard or supersede operation is wanted for a file this cheap to rebuild.

The union that would feed `--pairs` is checked by running the pipeline itself,
without `wf` and without a search:

```bash
cd "$WFROOT/.wf"
LC_ALL=C sort -u classified/yes/yes.pairs \
        best/s6/u-toyfastmusketsalvo/m4/g4/best.pairs \
    | LC_ALL=C comm -23 - classified/no/no.pairs
```

That yields 113 pairs (the 112 YES plus `tiger,lily`; `yes.pairs` and
`no.pairs` do not overlap). 16 of them fit the s6 bag, against the 2 the old
`best.pairs` supplied, and those 16 are what `dfs.best.pairs` would hold. The
same union against s7 fits 54, against the 52 / 48 / 0 its three `best.pairs`
supplied. Those counts are what `_no_usable_pairs` and the `gen_dfs` refusal
read, so confirming them here confirms both without a run.

The published `dfs.best.pairs` and the argv are covered by
`tests/test_workflow_best_e2e.py`, which asserts on the `--pairs` file handed
to a stubbed `dfs-anagrams`. There is no live check of either.

## What not to do

**Do not run `gen dfs.best` against `$WFROOT` for any purpose in this plan** —
not to verify, not to validate, not to smoke-test the argv, and not with the
intent of interrupting it once `_display_dfs` has printed. `_display_dfs`
prints immediately before `subprocess.run` and there is no dry-run flag, so
there is no way to reach the printed command that does not also start a real
search on the live tree. A `dfs.best` run costs hours, competes with any
`query-index` or `dfs-anagrams` already on the host, and writes into the target
directory.

Every claim this plan makes about `dfs.best` is checkable without running it:
the union and the bag-fit counts from the `sort`/`comm` pipeline above plus the
bag arithmetic, the argv and the published pair list from the e2e tests, and
the state-machine consequences from `wf best status --all`, which is read-only.

`gen top.segments` is the exception and stays in Verification: it takes
seconds, reads the existing DFS file, and rewrites only `top.segments`.

## Documentation

- `findings/stale-top-segments.md` (nutrimatic): replace the "What it breaks"
  section with how the decision was resolved — `best.pairs` stopped being
  derived from the frontier, so `-y` breaks nothing — then move the file to
  `findings/archive/`. Its `top-segments.cpp:101`/`:105` and
  `pair-exclusions.cpp:121` citations are stale; §3 and §4 above carry the
  current ones.
- `findings/arbitrary-best-review.md` (words): issue #1, "`best.pairs`
  currently only admits new entries from `top.segments`", is dissolved rather
  than solved. One-off results land in `classified/yes` globally and reach
  `--pairs` from there, so `build_best_pairs` taking a `src_segments` file is
  no longer a thing to build. Strike it and note why.
- `docs/workflow.md` and `docs/best-pairs-workflow-v2.md` (nutrimatic) are
  dated and out of scope. Neither mentions `best.pairs` by name; v2's steps 3-4
  describe the old "confirm entries, then run DFS with them" shape. Leave both
  alone.
