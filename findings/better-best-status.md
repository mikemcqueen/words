# Better `wf best status`: two workflow lanes and maintenance

Date: 2026-08-30

This is a design finding only. It describes how `wf best status` should model
and present the work already represented by the current state rows. It does not
change the BEST workflow, its artifacts, or its commands.

## Problem

There are two main BEST workflow loops for a target:

```text
reseed: prepare --source seed -> review -> complete
refine: prepare --source best -> review -> complete
```

There are also maintenance facts which are true but often do not answer the
operator's main question. For example, `best.pairs` may be out of date because
the global classified YES or NO set changed. That fact is worth reporting, but
it should not hide where the reseed and refine loops stand.

The current status architecture combines both concerns in one precedence
table. `derive_state()` returns the first firing row, and that row owns the
headline and the only ordinary next action. `status --all` evaluates later
answerable rows, but labels their findings `also` and deliberately treats them
as diagnostic rather than operational guidance.

That makes a cheap maintenance fact capable of suppressing the main workflow.

## Revealing live example

The current `s7` status is:

```text
$ ./wf best status s7 --all
s7/u-vindiesel/m4/g4: best.pairs out of date (hard-NO set changed)
  next: wf best gen s7 -u vindiesel -g 4 best.pairs
  rows:
    no:   _letters_missing
    no:   _seed_missing
    no:   _review_queued
    no:   _review_evaluating
    no:   _no_frontier
    no:   _review_needed
    no:   _best_pairs_missing
    won:  _best_pairs_out_of_date   best.pairs out of date (hard-NO set changed)
    no:   _best_pairs_empty
    also: _top_segments_behind_dfs  dfs.best generated after top.segments
    also: _next_search              dfs.seed out of date (hard-NO set changed)
```

The two `also` rows contain the main operational state:

- The refine search has finished. Its next step is to generate the frontier
  from `dfs.best`, then review and complete it.
- The reseed search is out of date. Its next step is `prepare --source seed`,
  then review and complete that frontier.
- The winning `best.pairs` row is maintenance bookkeeping.

For this specific target, a read-only reconstruction of `best.pairs` using the
current formula produced the same 50 lines and the same SHA-256 hash as the
existing file:

```text
02db49172fd6bfa40245daee41df99bd4c8dcd1a68384bed2b04a449098e0dad
```

Because `build_best_pairs()` places byte-identical output with
`stable_mtime=True`, running the winning maintenance command will advance the
generation marker without changing the `best.pairs` content clock. It will not
make `dfs.best` stale. Both main-loop findings will remain after the maintenance
warning is cleared.

Rows below the winner are not always this stable: the winner's action can move
an input clock and change a later result. That means normal status should not
simply print every `also` as an immediately runnable command. It needs to model
the purpose and dependencies of each finding.

## Desired operator view

For the live `s7` state, normal status should say something like:

```text
s7/u-vindiesel/m4/g4:
  choose main loop:
    refine: wf best gen s7 -u vindiesel -g 4 \
              top.segments --source best
    reseed: wf best prepare s7 -u vindiesel -g 4 --source seed
  maintenance:
    best.pairs out of date (hard-NO set changed)
    fix: wf best gen s7 -u vindiesel -g 4 best.pairs
```

The main workflow comes first. Maintenance remains visible and actionable, but
does not replace it.

When maintenance genuinely gates a loop, status should still expose the main
step while distinguishing it from a command recommended right now:

```text
  refine loop:
    after maintenance: wf best prepare ... --source best
  maintenance:
    best.pairs out of date (...)
    fix: wf best gen ... best.pairs
```

The vocabulary should make these distinctions explicit:

- `next` means the command is the recommended runnable step now.
- `after maintenance` names the planned main-loop step whose input first needs
  repair or regeneration.
- `waiting` means the other loop currently owns the shared frontier/review.
- `caught up` means that loop has no useful next iteration under the current
  clocks.

## Model: two lanes, one scheduler

Reseed and refine are two workflow lanes, but they are not concurrent state
machines. They share one `top.segments`, one possible review in flight, one
target-wide `rN` sequence, and one accumulated `best.pairs`.

When no frontier is awaiting review, status may offer reseed and refine as
neutral alternatives. Once the operator chooses one, that source owns the
shared workflow until its frontier has been reviewed and completed:

```text
selected source -> top.segments -> review -> complete -> idle
```

During an active refine round, for example:

```text
  refine loop:
    next: wf best review ...
  reseed loop:
    waiting for refine review
```

After completion, status can again derive both possible next iterations.

No new durable state is needed to answer "what next?":

- `.top.segments.gen` records whether the current frontier came from `seed` or
  `best`.
- queued/evaluating review locations record the open review phase.
- archived review and frontier mtimes say whether the current frontier has
  been completed.
- per-source DFS and input clocks say whether a new search is useful.

Historical questions such as "when was the last refine loop completed?" would
need durable source attribution in archived rounds. That is outside the
next-action status problem.

## Status data shape

The current `Inputs` object remains the right place to resolve paths, cache
facts, evaluate clocks, and render canonical commands. The result derived from
those facts should become a report rather than one winning `State`:

```python
@dataclass(frozen=True)
class LoopState:
    name: str                  # "refine" or "reseed"
    source: str                # "best" or "seed"
    phase: str                 # prepare, frontier, review, complete, ...
    message: str
    action: Choice | None = None
    blocked_by: tuple[str, ...] = ()


@dataclass(frozen=True)
class Notice:
    message: str
    action: Choice | None = None
    affects: tuple[str, ...] = ()


@dataclass(frozen=True)
class StatusReport:
    loops: tuple[LoopState, ...]
    notices: tuple[Notice, ...]
```

`blocked_by` and `affects` are more useful than a generic severity. They let
status say that stale `best.pairs` affects refine without hiding or delaying an
independent reseed step.

Hand-placed prerequisites such as missing `letters` or a missing seed remain
shared target blockers. Malformed paths, duplicate seeds, and multiple review
bundles remain errors rather than ordinary status findings.

## Derivation

Derive the report in four parts.

### 1. Shared target readiness

Evaluate the existing hand-placed inputs first:

- `letters` missing
- seed missing
- malformed or ambiguous target state

These are genuine global blockers because neither loop can be derived safely
without them.

### 2. Shared active round

Determine whether a frontier or review currently owns the target:

- queued review -> queued recovery command
- evaluating review -> `complete`
- unreviewed `top.segments` -> `review`
- otherwise no active round

Read `.top.segments.gen` to assign an active frontier to the refine or reseed
lane. The owning lane gets the next action; the other lane is `waiting`.

### 3. Per-lane next work

When there is no active round, derive each source separately.

For `seed` / reseed:

- If a current `dfs.seed` was generated after the frontier generation clock,
  resume with `gen top.segments --source seed`.
- Otherwise, if `seed_search_needed`, offer `prepare --source seed`.
- Otherwise the lane is caught up.

For `best` / refine:

- If `best.pairs` is absent or empty, refine is unavailable or blocked.
- If a current `dfs.best` was generated after the frontier generation clock,
  resume with `gen top.segments --source best`.
- Otherwise, if `best_search_needed`, offer `prepare --source best`.
- Otherwise the lane is caught up.

If both lanes are available while idle, render them under `choose main loop:`.
They are scheduling alternatives, not instructions to execute both before
reviewing either frontier.

### 4. Maintenance notices

Collect useful truths which do not define the main workflow position:

- `best.pairs` missing, empty, or out of date
- dangling artifact links
- widening suggestions
- other consistency or repair facts

A notice may carry a fix command and name which lane it affects. Notices render
after the loop state.

## Mapping from the current row architecture

This can preserve most of the current status machinery:

1. Keep `Inputs` and its command renderers.
2. Keep row-like checks as fact detectors, but return tagged findings rather
   than letting one global row own the whole report.
3. Split source-aggregating checks so their results can be assigned to lanes:

   ```text
   _top_segments_behind_dfs(source)
   _search_needed(source)
   ```

4. Tag each finding as `shared`, `refine`, `reseed`, or `maintenance`.
5. Select the first meaningful phase within each lane, respecting the existing
   cheap-before-expensive and prerequisite ordering.
6. Render the two loop projections first and maintenance notices afterward.

The existing ordering still contains important safety knowledge. In
particular:

- Resume frontier generation from a finished, current DFS before offering a
  new hours-long search in the same lane.
- Do not recommend a best-derived search with missing, empty, or meaningfully
  stale `best.pairs`.
- Do not offer a frontier-writing command while another source owns an active
  frontier or review.

`status --all` can remain the diagnostic view of the underlying checks, but
`won` and `also` would no longer describe the authority of normal status. More
useful diagnostic labels would be `shared`, `refine`, `reseed`, `maintenance`,
and `not asked`.

## Shared-frontier safety gap

The two lanes require one scheduler rule: an unsubmitted frontier awaiting
review must reserve `top.segments` just as a queued or evaluating review does.

The current `_preflight_top_segments()` refuses replacement only when a review
bundle is in `p2/queued` or `p2/eval`. It does not refuse replacing an
unsubmitted `top.segments` for which status says `review needed`. Consequently,
after choosing the refine action in the `s7` example, an immediate reseed
`prepare` could overwrite the refine frontier before review.

Before presenting this status model as a safe scheduler, frontier writers must
either:

- refuse while the current frontier still needs review, or
- require an explicit operation/flag which discards or supersedes that pending
  frontier.

Normal status should never advertise the waiting lane's write command while
another lane owns an unreviewed frontier.

## Suggested focused tests

The status behavior should cover at least:

1. The live `s7` shape: refine resumes frontier generation, reseed offers a new
   search, and stale `best.pairs` is maintenance.
2. Byte-identical `best.pairs` regeneration does not erase either loop choice.
3. A `best.pairs` regeneration which changes content moves refine from frontier
   recovery to a new `prepare --source best` search.
4. An unreviewed seed frontier makes reseed say `review` and refine say
   `waiting`; the inverse holds for a best frontier.
5. Queued and evaluating reviews remain attached to the recorded source.
6. Completing a round releases the shared frontier and re-derives both lanes.
7. Missing or empty `best.pairs` blocks refine without hiding reseed.
8. A missing target bootstraps only through reseed with the existing `-f`
   behavior.
9. A pending unsubmitted frontier cannot be overwritten by the other lane.
10. `--all` reports the same underlying facts with lane-oriented labels.

## Conclusion

The current `--all` output already detects the important work. Its weakness is
classification and presentation: one global winner treats maintenance,
workflow progress, recovery, and scheduling alternatives as though they were
competing answers to one question.

The better model is:

```text
facts -> shared active round
      -> refine-lane projection
      -> reseed-lane projection
      -> maintenance notices
```

That makes `wf best status` answer the operator's primary question without
discarding any of the useful truths the current precedence table has learned
to detect.
