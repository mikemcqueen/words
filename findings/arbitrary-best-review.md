* workflow: how practical would it be for `wf best review` and `wf best complete`
  to accept an arbitrary input file instead of the implied `top.segments`?
  * the existing machinery should otherwise remain the same: subtract already
    classified YES/NO pairs, split the remaining pairs, and create notes.
  * validate that the input pairs fit within either `letters - used-letters` or
    `only-letters`, as applicable.
  * `wf best complete` would download, merge, and classify the results, then update
    `best.pairs`.

--------------------

# Arbitrary BEST review input assessment

## Assessment

This is practical, but the TODO understates the state-management work.
Accepting a filename is easy; making its results reliably belong to the correct
BEST target is a moderate change.

The recommended interface is to add an optional file to `best review`, but not
to `best complete`:

```text
wf best review SENTENCE [PAIRS-FILE] -u ... -g ...
wf best complete SENTENCE -u ... -g ...
```

With no file, review keeps using `top.segments`. With a file, it copies that
input into a target-owned managed round. Completion should consume that managed
round; requiring the original filename again would create needless mismatch and
deletion/change hazards.

## What already transfers cleanly

The existing review pipeline already:

- Sorts and deduplicates candidates.
- Subtracts both `classified/yes` and `classified/no`.
- Creates an unfiltered P2 bundle and notes.
- Retrieves the notes, classifies both verdicts, and archives the round.

That path is concentrated in `workflow/best/commands.py` and the ordinary P2
completion recipe.

## Issues that must be resolved

### 1. `best.pairs` currently only admits new entries from `top.segments`

~~so build_best_pairs() should change to take a src_segments file, which can
either be top.segments or the supplied file.~~

**Dissolved, not solved** (2026-09-02, `plans/manual-best-pairs.md`).
`build_best_pairs()` is gone. `best.pairs` is no longer derived from anything:
it is an optional hand-edited file that nothing generates, and `gen dfs.best`
unions it with `classified/yes` at run time, filtered down to what the target's
letter bag can spell.

A one-off round's YES verdicts land in `classified/yes` like every other
round's, and reach `--pairs` from there for every target whose bag can spell
them. So there is no per-target set left for a supplied file to feed, and no
`src_segments` parameter to add. The requirement this issue was protecting --
that already-known YES pairs are removed from the review candidates but still
weight the search -- now holds for free.

### 2. The current frontier lock becomes conditionally wrong

`_preflight_top_segments()` says every in-flight review was built from
`top.segments` and blocks rewriting it. That is false for a supplied-file round.

- block frontier replacement only during a `top` review.

## Decisions to lock in

| Decision | Recommendation |
| --- | --- |
| Review interface | Optional positional `PAIRS-FILE`; omission means `top.segments` |
| Complete interface | Keep target-only; no filename |
| External file lifetime | Copy it into managed state; never depend on the original afterward |
| Round identity | Record `top` versus supplied-file provenance in canonical names/state |
| Promotion | Rebuild from durable completed supplied inputs, not only the latest P2 YES output |
| Missing `top.segments` | Allow supplied-file review and completion without it |
| Concurrent frontier generation | Permit it for supplied-file rounds; continue blocking it for top-derived rounds |
| Original basename | Log it if useful, but do not use it as the managed bundle identity |

This is worthwhile and feasible, but it should be treated as adding a second
durable review-source type, not merely replacing one `Path` variable. The
`complete` behavior must change, while its CLI probably should not.

This assessment was based on a read-only source and data-path review. No
production files were changed and no tests were run.


---------------------------------------------------
the following have been designated OUTSIDE OF SCOPE
---------------------------------------------------


### 6. The empty-candidate behavior needs two variants

For ordinary `top.segments`, the current "all already classified;
widen/refine/reseed" guidance makes sense.

For a supplied file:

- Already-known YES pairs still need to enter `best.pairs`.
- Known NO pairs do nothing.
- Search recommendations are unrelated and should not be printed.
- A completely empty input should be rejected distinctly from "all supplied
  pairs already have verdicts."

### 5. "Arbitrary filename" should mean arbitrary path, not arbitrary contents
or bundle name

The external basename should not be inherited: P2 names intentionally allow
only lowercase letters, digits, periods, and hyphens. A target-generated
canonical bundle name avoids collisions and unsafe names.

Contents should be validated before any state moves:

- Exactly two nonempty comma-separated members per nonblank line.
- A canonical spelling policy should be selected--preferably reject rather than
  silently clean punctuation or case.
- The combined letters of both members must be a multiset subset of:
  - the explicitly named bag under `-o`;
  - `letters` minus the named used letters under `-u`.
- Equality with the whole bag is not required; the pair represents only two
  members of a larger solution.

The downstream Nutrimatic loader in
`/home/mike/code/nutrimatic/source/dfs-cli-args.cpp` accepts exactly one comma
but otherwise cleans word spelling aggressively, including case and
punctuation. The workflow boundary should be stricter so the reviewed text,
classified text, and eventual DFS key cannot silently differ.

### 2. Arbitrary rounds cannot masquerade as top-frontier rounds

Status currently decides that `top.segments` has been reviewed by comparing it
with the newest archived target review in `workflow/best/state.py`. If an
arbitrary-file review uses the existing `top.*` identity, completing it could
falsely mark an unrelated `top.segments` as reviewed.

Bundle identity therefore needs a source kind--at least `top` versus
supplied-file--and status must consider only completed `top` rounds when judging
frontier freshness. Notes, completion, and the one-in-flight check should still
recognize both kinds as belonging to the target.

### 4. Large files can strand an opened P2 bundle

Notes support at most 26 parts of 400 rows--10,400 review candidates. That check
currently happens after evaluation has already moved the queue item into an open
bundle.

Supplied-file review should calculate the post-YES/NO candidate count and reject
an oversized batch before submission. Automatically creating several
simultaneous rounds conflicts with the current one-in-flight-per-target
invariant, so explicit rejection is the safer first version.
