# Add explicit BEST search-source selection and `prepare`

## Summary

Treat `top.segments` as the single current review frontier. Make its DFS
source explicit, add `prepare` to generate DFS results plus that frontier, and
let status present reseed/refine choices neutrally.

Keep the existing `best.pairs` formula unchanged:

```text
(top.segments ∪ prior best.pairs)
∩ classified/yes
− classified/no
```

## Vocabulary

The plan says *generated* and *generation*, never *extraction*. The step that
produces `top.segments` from a DFS file is `gen top.segments`; name it that.

Three clocks are read, and rows below name which one they use:

- `mt(x)` — `x.stat().st_mtime_ns`, following symlinks. For anything placed
  with `stable_mtime=True` this only advances when content changed.
- `M(x)` — `mt(.x.gen)` when that marker exists, else `mt(x)`. Advances on
  every successful generation, no-op or not. This is `state._generated`.
- `source` — the contents of `.top.segments.gen`: `seed` or `best`.

Which artifacts use which:

- `top.segments`, `best.pairs` — placed with `stable_mtime=True`, so `mt` is a
  content clock and `M` is a generation clock, and the two differ.
- `classified/yes/yes.pairs`, `classified/no/no.pairs` — placed with
  `stable_mtime=True` (`config.py:82-97`), no marker, so `mt` is `M`. Their
  mtimes advance only when a round contributes a pair the set did not hold.
- `dfs.seed`, `dfs.best` — symlinks into `results/`. `_gen_dfs` replaces the
  rendered file on every run and writes no marker, so `mt` is a true
  generation time. This is correct and stays.
- `letters`, the seed pairs file — hand-placed, `mt` only.

## CLI and generation

- Require `--source seed|best` for `wf best gen ... top.segments`; map it to
  `dfs.seed` or `dfs.best`. Retain `-n COUNT` as this primitive's cutoff.
- Add:

  ```text
  wf best prepare SENTENCE (-o LETTERS|-u LETTERS) -g COUNT
      [-m LENGTH] --source seed|best
      [-r DIR|--results-dir DIR]
      [--dfs-count COUNT]
      [--top-count COUNT]
  ```

- Default `--dfs-count` to 1,000,000 and `--top-count` to 1,000. Reject
  negative values. Both match current behaviour: `generate.DFS_LIMIT` is
  already 1,000,000 and `top-segments -n` already defaults to 1000, so neither
  default changes what runs today.
- Implement `prepare` as a normal compound action:
  1. Validate the target, source, result directory, executables, and review
     preflight.
  2. Generate the selected DFS artifact.
  3. Generate `top.segments` from that artifact.
- The preflight in step 1 is the only one. Do not re-check before step 3.
  `wf` is a single-user client and `prepare` holds the shell for the duration
  of the DFS, so no review can open between step 1 and step 3.
- Permit `-f` only with `prepare --source seed`, preserving missing-target
  creation. `prepare` cannot hand one `opts` to both legs — `gen_top_segments`
  rejects `-f`, `-r/--results-dir` and `-n`, all three of which the DFS leg
  needs — so the generation helpers take explicit parameters and the
  flag-validity refusals move to the command layer.
- `--source` is valid only on `gen top.segments`. `Gen.parser()` is shared by
  all four stages (`commands.py:91-104`), so it will be syntactically accepted
  on `dfs.seed`, `dfs.best` and `best.pairs`; refuse it there beside the
  existing `-r`/`-n`/`-f` refusals.
- Update `Gen`'s STAGE help (`commands.py:84-88`). It calls `top.segments`
  "frequent pairs from dfs.seed", which is now either source, and `dfs.best`
  "final DFS results using best.pairs", which is no longer final once the loop
  iterates. `test_gen_help_describes_options_and_positionals` asserts on this
  text.
- `Review`'s empty-`top.segments` refusal (`commands.py:206-208`) hardcodes the
  `dfs.seed` path in `top.segments is empty; regenerate <path>`. Name the
  source recorded in `.top.segments.gen` instead.
- Refactor generation helpers so primitive and compound commands share
  explicit DFS-source, count, and atomic-publication logic.
- `gen dfs.best` and `prepare --source best` fail their preflight when
  `best.pairs` is empty, beside the other input checks in `_dfs_inputs` so the
  refusal lands before anything is created. `dfs-anagrams` given an empty
  `--pairs` is a strictly worse `dfs.seed`; there is no case where running it
  is right. This replaces the warning at `generate.py:209` for the DFS path;
  `build_best_pairs` keeps warning when it writes an empty set.
- `gen top.segments` warns when its output row count comes back below the
  requested `-n`. That means the DFS file is exhausted at this cutoff, and a
  larger `-n` will reproduce the same file — which `stable_mtime` then leaves
  untouched, so nothing downstream moves.
- `state.mark_generated` is `_stamp(path).touch()` (`state.py:344-345`) and
  cannot write contents. Give it a text parameter written via `write_text`,
  which advances mtime whether or not the content changed, so the
  generation-clock property survives. `build_best_pairs` calls the same
  function, and `.best.pairs.gen` stays empty and unparsed.
- Reuse `.top.segments.gen` as both the frontier-source record and generation
  marker:
  - Its contents are exactly `seed\n` or `best\n`, identifying the DFS artifact
    the current `top.segments` was generated from.
  - Its mtime records the last successful `gen top.segments` and advances even
    when stable publication leaves byte-identical `top.segments` untouched.
  - A missing or empty legacy marker means `seed`; when the marker is missing,
    use the `top.segments` mtime as its generation time. Reject any other marker
    contents rather than guessing.
  - Write `top.segments` first and the marker second, which is what
    `gen_top_segments` already does. A crash between them leaves
    `M(top.segments)` behind `mt(top.segments)`, so the row below re-offers the
    generation and the state heals. The residual defect is that `source` names
    the previous DFS file until the retry lands, which is acceptable because
    `source` is informational. Marker-first would be strictly worse: `M` would
    run ahead, the row would not fire, and the state would never be re-offered.
- Generating `top.segments` from either source selects that source as the one
  current frontier and supersedes any pending generation from the other DFS
  artifact. There is no independent per-source history.
- If `gen top.segments` fails after DFS publication, retain the new DFS result
  and previous `top.segments`. Emit a diagnostic with the exact manual
  `gen top.segments --source ...` recovery command; do not resume automatically.

## Review bundle contents

`--yes-pairs` stops being typed by `wf best` and stops appearing in any
message. The plumbing stays in place for a later revival:

- `commands.py:239` — `Review` executing `eval p2`: drop the argument.
- `commands.py:288` — `Notes` executing `notes p2`: drop the argument.
- `state.py:390` — `eval_p2_command`: drop it from the string status prints and
  `complete`/`notes` name in their refusals. Rewrite the docstring of
  `yes_pairs_argv`, which currently justifies itself by saying `best review`
  and `best status` both read it so they cannot drift apart; neither will.
- Everything else stays wired. `wf eval p2 --yes-pairs X` and
  `wf notes p2 --yes-pairs X` remain typeable by hand, `notes.add_yes_pairs`,
  `check_yes_pairs`, `_yes_pairs` and the `yes_pairs` parameter on
  `notes.create` are untouched, and `state.yes_pairs_argv` stays in place with
  no callers.

The flag was what marked already-confirmed pairs in the notes, so with it gone
`wf best review` builds the bundle as `top.segments` less both standing sets:

```text
merge([top.segments]) − classified/no/no.pairs − classified/yes/yes.pairs
```

Under `--source best`, `dfs.best` runs with `--pairs best.pairs`, so the
frontier it produces is weighted toward pairs already confirmed. Without the
diff those arrive in the notes unmarked and are re-reviewed every round.
Re-confirming them buys nothing: `build_best_pairs` intersects against
`classified/yes/yes.pairs` globally, so a pair confirmed in round 1 reaches
`best.pairs` in round 5 whether or not round 5's bundle held it.

That makes an empty bundle ordinary rather than pathological — it is the
convergence signal. `wf best review` exits 0 with a message naming the searches
worth running instead of raising `no review candidates remain after hard-NO
exclusions`. Status does not detect this state: row 6 stays an mtime
comparison, so it keeps reporting `review needed` until the frontier changes.
Rendering the next-search commands in `best review`'s message is what makes
that acceptable — the detection point is the guidance point.

## Review gate and rounds

- Add one shared `_preflight_top_segments(target)` check in BEST command
  orchestration. It refuses when the target has a queued or evaluating review
  and names the bundle.
- Call it before standalone `gen top.segments` and before either `prepare`
  branch performs work.
- Allow standalone `gen dfs.seed` and `gen dfs.best` during a review because
  they do not overwrite the current frontier. This is reachable in one shell:
  the bundle sits in `p2/eval`, the operator is reading notes, the shell is
  free. Status never advertises it — `complete` moves
  `classified/no/no.pairs` and `best.pairs`, so a DFS run during a review is
  out of date the moment the round completes, and status must not recommend
  work it will immediately mark out of date.
- Preserve the existing single target-wide P2 review counter:
  - `wf best review` continues assigning `max(rN) + 1`.
  - Both seed-derived and best-derived frontiers consume the next round when
    submitted.
  - `prepare` and standalone generation do not increment the counter.
  - DFS output filenames receive no generation or review ordinal.
- Read the current frontier source from `.top.segments.gen` when reporting
  review state; the source is informational and does not split review rounds.

## Status precedence

### Structure

`derive_state`'s straight-line early returns become a tuple of rows evaluated
in order, first non-`None` winning. `_missing_artifact` is already this shape.

```python
@dataclass
class Inputs:
    """Everything a row may read, resolved once per target."""
    target: Target

    @cached_property
    def seed(self) -> Path | None: ...

    @cached_property
    def review(self) -> tuple[list[Path], list[Path], list[Path]]: ...

    @cached_property
    def source(self) -> str: ...

    @cached_property
    def seed_search_needed(self) -> list[str]: ...

    @cached_property
    def best_search_needed(self) -> list[str]: ...


ROWS = (
    # G0 — hand-placed inputs
    Row(_letters_missing),
    Row(_seed_missing, provides="seed"),
    # G1 — open review
    Row(_review_queued),
    Row(_review_evaluating),
    # G2 — no frontier; everything below has a top.segments
    Row(_no_frontier, provides="top.segments"),
    # G3 — frontier not yet reviewed
    Row(_review_needed, requires=("top.segments",)),
    # G4 — derived set; everything below has a current best.pairs
    Row(_best_pairs_missing, provides="best.pairs"),
    Row(_best_pairs_out_of_date, requires=("best.pairs", "top.segments")),
    Row(_best_pairs_empty, requires=("best.pairs", "top.segments", "seed")),
    # G5 — a finished search whose frontier was never generated
    Row(_top_segments_behind_dfs, requires=("top.segments", "seed")),
    # G6 — start the next search
    Row(_next_search, requires=("seed",)),
)


def derive_state(target: Target) -> State:
    inputs = Inputs(target)
    for row in ROWS:
        state = row.check(inputs)
        if state is not None:
            return state
    return State("converged")
```

`requires` and `provides` are inert here — a row runs only because every row
above it declined, which is what establishes them. They exist for
`status --all`, below.

Two properties hold this together, and both are load-bearing:

- **Rows are not independent.** Row N assumes rows 1..N-1 did not fire.
  `_review_needed` stats `top.segments` only because `_no_frontier` did not
  fire. In the straight-line version that guard was the preceding line; split
  into functions it becomes invisible. So `Inputs` accessors raise when the
  file is absent rather than returning `None` — a row reaching one out of
  order fails loudly instead of silently comparing against a missing file. The
  group comments in `ROWS` say what each boundary guarantees, and `Row.requires`
  / `Row.provides` say it again as data, for `status --all`.
- **Conditions and command strings live on `Inputs`, not in rows.** Several
  rows print the same command: `_best_pairs_empty` and `_next_search` both
  print `reseed:`, and `_no_frontier` and `_top_segments_behind_dfs` both print
  `gen top.segments --source seed|best`. Duplicating them is the drift
  `eval_p2_command` exists to prevent (`state.py:380-386`). `Inputs` owns
  `seed_search_needed`, `best_search_needed`, and one renderer per command;
  rows only choose which to present.

### Reporting the whole table

`status --all` (`-a`) prints every row under the ordinary report, as a
diagnostic for the precedence itself rather than a second opinion about what to
run next. `derive_state` stays the one authority for the verdict. A row that
fired prints the commands it offers under it, through `render_choices` — the
same renderer `report` uses above the table, so the two cannot drift. Only the
winner's are safe to run: an `also` row's are indented under the row that
offers them rather than standing beside the winner's.

Straight iteration over `ROWS` cannot do this: the rows below the winner read
the files the winner just reported missing, and the `Inputs` accessors raise
rather than return `None` — which is the property above, working as intended.
Catching those exceptions would report "rows that happened not to raise", which
is not the same thing as the rows that were answerable. So each `Row` carries
the group boundary as data:

```python
@dataclass(frozen=True)
class Row:
    check: Callable[[Inputs], State | None]
    requires: tuple[str, ...] = ()   # files its accessors raise without
    provides: str | None = None      # the file it establishes by declining
```

`walk_rows` asks a row only when the rows providing its `requires` declined,
and records it `not asked (needs top.segments)` otherwise. Up to and including
the winner this asks exactly what `derive_state` asked and gets the same
answers, because every row above the winner declined and declining is what
establishes a file. Past the winner the answers are real but partial, so the
table names them apart: `won` for the row `derive_state` returned, `also` for a
row below it that fires. An `also` is a symptom, not an alternative — it reads
the tree as it stands, and the winner's own fix is what moves the files it
dates against, so it is a prediction of the next round at best.

```text
s7/u-vindiesel/m4/g4: best.pairs out of date (hard-NO set changed)
  next: wf best gen s7 -u vindiesel -g 4 best.pairs
  rows:
    no:   _letters_missing
    ...
    won:  _best_pairs_out_of_date   best.pairs out of date (hard-NO set changed)
      next: wf best gen s7 -u vindiesel -g 4 best.pairs
    no:   _best_pairs_empty
    also: _top_segments_behind_dfs  dfs.best generated after top.segments
      next: wf best gen s7 -u vindiesel -g 4 top.segments --source best
    also: _next_search              dfs.seed out of date (hard-NO set changed)
      next: wf best prepare s7 -u vindiesel -g 4 --source seed
```

The row's own function name, not a prose label: the operator reading this is
asking why the table chose what it chose, and the answer is in `state.py` under
that name.

`State` carries labelled alternatives:

```python
@dataclass(frozen=True)
class Choice:
    label: str            # "next", "reseed", "refine", "widen"
    command: str

@dataclass(frozen=True)
class State:
    message: str
    choices: tuple[Choice, ...] = ()
    place: Path | None = None
```

`report()` renders every choice under its own label. A lone `Choice` labelled
`"next"` prints today's `next: COMMAND` line; a lone choice with any other label
prints `LABEL: COMMAND`, so a single `widen:` still says it is a widening rather
than showing a bare `gen` with a bumped `-n`. Two or more print under
`choose next:`, one per line. There is no single-choice special case in
`report()` — the label field is what varies. A multi-choice row still returns
one `State`; first-row-wins is unaffected.

### Derived conditions

```text
seed_search_needed  :=  dfs.seed missing
                     ∨  mt(seed)                     > mt(dfs.seed)
                     ∨  mt(classified/no/no.pairs)   > mt(dfs.seed)

best_search_needed  :=  best.pairs exists ∧ non-empty
                     ∧ ( dfs.best missing
                       ∨ mt(best.pairs)                > mt(dfs.best)
                       ∨ mt(classified/no/no.pairs)    > mt(dfs.best) )

top_segments_behind(X) :=  dfs.X exists
                        ∧  mt(dfs.X) > M(top.segments)
                        ∧  X's search is not needed
```

Both branch conditions carry a reasons list, not a boolean, so the row can
render `dfs.seed out of date (seed changed, hard-NO set changed)`.

`top_segments_behind(X)` requires X's search to be current: a DFS file that is
itself out of date should be re-run, not read. Row 10's `<X>` is the DFS
artifact that is ahead, which is never `Inputs.source` — that names the artifact
the current frontier came from — so the message does not reuse the word
`source`.

### The table

| # | Row | Condition | Message | Choices |
|---|---|---|---|---|
| 1 | `_letters_missing` | `letters` absent | `letters missing` | `place:` the path |
| 2 | `_seed_missing` | no match for `seed.m<N>.*.pairs`; two or more matches raises | `seed missing` | `place:` the glob |
| 3 | `_review_queued` | a bundle matching `review_prefix` in `p2/queued` | `review submitted (NAME)` | `wf eval p2 NAME` |
| 4 | `_review_evaluating` | a directory matching `review_prefix` in `p2/eval` | `review awaiting completion (NAME)` | `best complete` |
| 5 | `_no_frontier` | `top.segments` absent or a dangling symlink | see below | see below |
| 6 | `_review_needed` | `max(mt(archived), default=0) ≤ mt(top.segments)` | `review needed (frontier from <source>)` | `best review` |
| 7 | `_best_pairs_missing` | `best.pairs` absent | `best.pairs missing` | `gen best.pairs` |
| 8 | `_best_pairs_out_of_date` | `mt(top.segments) > M(best.pairs)` ∨ `mt(classified/yes/yes.pairs) > M(best.pairs)` ∨ `mt(classified/no/no.pairs) > M(best.pairs)` | `best.pairs out of date (<reasons>)` | `gen best.pairs` |
| 9 | `_best_pairs_empty` | `line_count(best.pairs) == 0` | see below | see below |
| 10 | `_top_segments_behind_dfs` | `top_segments_behind(seed)` ∨ `top_segments_behind(best)` | `dfs.<X> generated after top.segments` | one or two `gen top.segments --source` |
| 11 | `_next_search` | `seed_search_needed` ∨ `best_search_needed` non-empty | `dfs.<X> out of date (<reasons>)`, or `choose next:` | `reseed:` / `refine:` |
| — | fallthrough | none of the above | `converged` | — |

Rows 1 and 2 fire on absence only. A `letters` or seed path that exists but is
not a regular file keeps raising through `fs.raise_if_not_file`
(`state.py:442-443`), which `Status` reports on stderr with exit 1 — a present
but malformed input is an error, not a state, and reporting it as `missing`
would be worse diagnostically.

Three ordering decisions, each of which the current `derive_state` gets wrong
for the two-search shape:

- **G1 above G2, G5 and G6.** `_preflight_top_segments` refuses `gen
  top.segments` and both `prepare` branches while a bundle is queued or
  evaluating. A row below the review gate offering `prepare` would print a
  command that errors immediately.
- **G4 above G5 and G6.** `prepare --source best` reads `best.pairs` as
  `--pairs`. Rebuilding it costs seconds; running the DFS off a set that is
  behind the classified sets costs hours.
- **G5 above G6.** Generating a frontier from a finished search is seconds; a
  new search is hours. Never offer the hours while the seconds are
  outstanding.

Row 5, `_no_frontier`, by which DFS files exist:

| Present | Message | Choices |
|---|---|---|
| neither | `no search results yet` (plus `(dangling symlink: …)` when one is) | `prepare --source seed`, with `-f` when the target directory is absent |
| `dfs.seed` only | `top.segments missing` | `gen top.segments --source seed` |
| `dfs.best` only | `top.segments missing` | `gen top.segments --source best` |
| both | `top.segments missing` | both, labelled |

The first case is the only bootstrap gate, and it is deliberately narrower than
"`dfs.seed` is missing" — which is also why its message does not name
`dfs.seed`: the state is that no search has run, and the command offered is
`prepare`, not a `dfs.seed` placement. After the first round `dfs.seed` is an
input to the seed search and nothing else, so a cleaned `results/` must not
force a fresh seed DFS while `best.pairs` and `dfs.best` are alive. A missing
`dfs.seed` at that point is one of `seed_search_needed`'s reasons, handled by
row 11.

Row 9, `_best_pairs_empty`. `best.pairs` is present, current, and holds nothing
— no pair in the frontier carries a standing YES verdict. It fires whenever the
set is empty, not only at a dead end, because widening is orders of magnitude
cheaper than either search and the operator should see it before reaching for
hours of DFS.

```text
s2/u-cdef/m4/g4: review confirmed no pairs
  (1000 frontier pairs, 0 in classified/yes)
  choose next:
    widen:  wf best gen s2 -u cdef -g 4 top.segments --source seed -n 2000
    reseed: wf best prepare s2 -u cdef -g 4 --source seed
  or retract NO verdicts in .wf/classified/no/no.pairs
     and run: wf best review s2 -u cdef -g 4
```

- `widen:` renders `-n` as `line_count(top.segments) + 1000`, and takes
  `--source` from the value recorded in `.top.segments.gen`. This is the only
  row that prints a count; every other `prepare` it offers takes the defaults.
- `reseed:` appears only when `seed_search_needed` is non-empty, and comes from
  the same renderer row 11 uses.
- `refine:` never appears — `gen dfs.best` and `prepare --source best` refuse an
  empty `best.pairs`.
- The retraction line is prose because no command retracts a verdict:
  `classify` is union-only (`classify.py:73` → `config.fold_classified`,
  `config.py:210`) and there is no inverse. Note
  that a hand-edit of `classified/no/no.pairs` does not reopen the review —
  row 6 compares archived mtimes against `mt(top.segments)` and neither moves
  — so the operator has to run `wf best review` themselves, which is why the
  message names it. No `-f` is needed: `Review` gates on an in-flight bundle
  and an empty bundle, never on the review clock, so it is already runnable at
  any time.

Row 11, `_next_search`. Both conditions non-empty prints `choose next:` with
`reseed:` and `refine:`; exactly one prints that one line, keeping its
`reseed:` or `refine:` label. Because `complete` folds into
`classified/no/no.pairs`, a round that produces any new NO puts both searches
behind, so the two-way choice is the ordinary steady state and not an edge
case.

The fallthrough is reachable only when a completed round contributed nothing
new to either classified set and `best.pairs` did not change — the search is at
a fixed point, and re-running either side reproduces what is already there.
`converged` says that; `up to date` would read as a lost write.

## Test plan

Rows are individually addressable, so the suite tests them directly rather
than arranging a whole tree to reach row 9.

- **Per row.** Construct an `Inputs`, call one row, assert it fires on exactly
  its condition and returns `None` otherwise. Cover row 5's four sub-cases and
  row 9's three-choice render.
- **Guards.** Assert each row raises when called out of order — `_review_needed`
  with no `top.segments`, `_best_pairs_out_of_date` with no `best.pairs` —
  rather than silently comparing against a missing file.
- **Precedence.** Build states where two rows would both fire and assert the
  earlier wins: an open review against an out-of-date `dfs.seed`; an unreviewed
  frontier against a DFS file newer than `.top.segments.gen`; an out-of-date
  `best.pairs` against both; an empty `best.pairs` against an available reseed.
- **Order.** Assert `ROWS` matches the documented precedence, so the tuple and
  this document cannot drift.
- **Shared renderers.** Assert `_best_pairs_empty` and `_next_search` emit
  byte-identical `reseed:` commands, and that `_no_frontier` and
  `_top_segments_behind_dfs` emit byte-identical `gen top.segments --source`
  commands.
- Verify help, required/invalid `--source`, source mapping, and primitive `-n`,
  including `--source` refused on `gen dfs.seed`, `gen dfs.best` and
  `gen best.pairs`.
- Verify `.best.pairs.gen` is still written empty and never parsed.
- Requiring `--source` invalidates three existing tests, which this change
  updates: `test_gen_top_segments_passes_count_and_preserves_unchanged_mtime`,
  `test_no_op_top_segments_gen_clears_a_reran_dfs_seed`, and
  `test_gen_top_segments_rejects_force_before_running`.
- Verify both `prepare` branches, default and explicit independent counts,
  results-directory forwarding, correct pair inputs, and seed-only `-f`.
- Verify the shared preflight blocks top generation and both compound branches
  for queued/evaluating reviews before subprocess execution; standalone DFS
  remains allowed.
- Verify `gen dfs.best` and `prepare --source best` refuse an empty `best.pairs`
  before creating anything.
- Verify `gen top.segments` warns when its output falls short of `-n`.
- Verify `wf best review` excludes pairs in `classified/yes/yes.pairs` as well
  as `classified/no/no.pairs`, and that an empty bundle exits 0 naming the
  next searches rather than raising.
- Verify no `wf best` command executes or prints `--yes-pairs`, and that
  `wf eval p2 --yes-pairs X` and `wf notes p2 --yes-pairs X` still work.
- Verify a failed `gen top.segments` preserves the old frontier and new DFS
  output and leaves the old source marker unchanged while emitting the manual
  recovery diagnostic.
- Verify `.top.segments.gen` records each selected source, advances on a
  byte-identical regeneration, accepts missing/empty legacy markers as seed,
  and rejects malformed contents.
- Verify review rounds remain one shared sequence across alternating seed and
  best frontiers, while DFS filenames remain unchanged.
- Extend the BEST end-to-end test through one
  `prepare --source best → review → complete` inner-loop round.
