# Add one-off input files to BEST review

## Summary

Extend `wf best review` with an optional pairs file:

```text
wf best review SENTENCE [PAIRS-FILE] (-o LETTERS|-u LETTERS) -g COUNT
    [-m LENGTH]
```

Omitting `PAIRS-FILE` preserves the existing `top.segments` workflow.
Supplying it creates an independent one-off review whose only interaction with
the main search loops is contributing confirmed pairs to the target's
accumulated `best.pairs`.

`wf best complete` and `wf best notes` remain target-only. Preserve one
queued/evaluating review total per target so those commands remain unambiguous.

## Interfaces and durable state

- Keep existing top-round names unchanged. Name supplied-file rounds:

  ```text
  oneoff.<sentence>.m<M>.g<G>.<letter-set>.<count>.r<N>.pairs
  ```

  `count` is the sorted, deduplicated full input's line count. Ignore the
  external basename.
- Give one-off rounds their own `rN` sequence, per target, derived the way the
  top sequence already is: the highest ordinal among that kind's archived
  rounds, plus one. Two one-offs run against a single frontier generation are
  then `r1` and `r2` rather than colliding in the archive, and a one-off never
  consumes an ordinal the top sequence would have used. The two kinds cannot
  collide with each other at any ordinal, because the kind leads the name.
- Represent discovered rounds as typed records carrying path, `top|oneoff`
  kind, and ordinal. Review discovery must recognize both kinds, reject
  duplicate ordinals within a kind, and enforce the one-in-flight invariant
  across them.
- Archive the full canonical one-off input as the P2 source. Inside the
  evaluating bundle, place the post-YES/NO candidate set at the ordinary
  `.filtered` path so notes, completion, and the P2 done-set operate on only the
  reviewed subset.
- Refactor the internal P2 evaluator so BEST can supply that prepared
  `.filtered` input after `bundle.begin`; do not add or change public
  `wf eval p2` options.
- Note titles are rendered from the evaluated file's name, so a one-off's
  notes are titled `oneoff.<...>.r<N>.pairs.filtered.aa` where a top round's
  are `top.<...>.r<N>.pairs.aa`. `best notes` on an open one-off reproduces
  those titles exactly, because the bundle still holds `.filtered`.
- `best notes -f` on an *archived* one-off is refused. `p2_archive` deletes the
  `.filtered` derivative, so the subset the round was actually reviewed as
  cannot be rebuilt; recreating from the full archived source would raise notes
  under titles nothing ever fetches -- orphans in the note store rather than
  that round's notes. The refusal names the archived source, which the operator
  can work from by hand.

## Review, completion, and status behavior

- Before moving workflow state, a one-off review must:

  1. Resolve a regular, readable input path.
  2. Sort and deduplicate the full input into its canonical managed form.
  3. Subtract both classified YES and classified NO into the reviewed subset.
  4. Reject without changing state if that subset is empty.
  5. Assign the next one-off round number and preflight queue/evaluation
     collisions.

- Submit the full canonical source, open its P2 bundle, install the filtered
  subset, and create notes through the existing P2 machinery. The original path
  is never read again.
- Top-frontier state remains independent:

  - Only archived `top` rounds satisfy `top.segments` review freshness.
  - A queued/evaluating one-off does not block `gen top.segments` or `prepare`.
  - A queued/evaluating top round continues blocking frontier replacement.
  - A second top or one-off review is refused while either kind is already open.

- Completion classifies the filtered verdicts and archives the full one-off
  source before rebuilding `best.pairs`.
- Rebuild using:

  ```text
  eligible = current top.segments, when present
             union every completed one-off full source *of this target*

  best.pairs =
      prior best.pairs
      union (eligible intersect classified/yes)
      minus classified/no
  ```

  `p2/done/in` is flat and repo-wide -- every bundle of every target archives
  there -- so completed one-offs are found by this target's own one-off prefix,
  `oneoff.<sentence>.m<M>.g<G>.<letter-set>.*.pairs`, and never by a bare
  `oneoff.*`, which would fold another target's one-off into this `best.pairs`.

  This admits already-known YES pairs from a completed one-off, preserves
  sticky/manual entries, and keeps hard NO authoritative. Allow rebuilding
  without `top.segments` when at least one completed one-off source exists;
  preserve the existing missing-frontier failure when neither source type
  exists.
- Store the sorted completed one-off archive filenames in `.best.pairs.gen`,
  one per line, while retaining its mtime as the generation clock. Missing or
  empty legacy markers mean no one-offs were incorporated; malformed names are
  rejected. The manifest is load-bearing rather than merely a crash record:
  `classified/yes` is a `stable_mtime` aggregate, so a one-off whose YES
  verdicts were all already classified moves no clock the mtime rows can see,
  and the pairs a one-off exists to promote come from its *full* source rather
  than from its verdicts.
- Publish `best.pairs` before its marker, so a stop between the two writes
  leaves the generation clock behind the content clock and the state heals by
  re-offering the generation, as `gen top.segments` already does.
- A completed one-off absent from the manifest is a fourth reason inside the
  existing `_best_pairs_out_of_date` row, not a row of its own:

  ```text
  best.pairs out of date (completed one-off review)
    next: wf best gen ... best.pairs
  ```

  That row already reads `best.pairs`, already offers `gen best.pairs`, and
  already renders exactly this message shape from a list of reasons, so the
  condition inherits its position and its `requires` and adds no mechanism. A
  row of its own placed above the frontier rows could not: it would sit above
  `_best_pairs_missing`, whose declining is what *provides* `best.pairs`, so it
  could not declare `requires=("best.pairs",)` without `walk_rows` reporting it
  `n/a` forever, and it would need a second message for the case where
  `best.pairs` does not exist at all. Deferring promotion behind frontier work
  costs nothing, because `best.pairs` is a sticky accumulator and the pairs
  land whenever the rebuild next runs.
- A one-off is not a precedence row. The table answers what to do next in the
  main loop and a one-off is beside that loop by construction, so no row can
  speak for it without lying in some case:

  - `_review_queued` and `_review_evaluating` become `top`-only. A top round in
    flight blocks everything below it, as now; a one-off does not.
  - `_review_needed` is the one row where the one-in-flight invariant bites:
    with a one-off open, `best review` is refused, so the row reports the
    frontier as needing review and offers `wf best complete` in place of
    `wf best review`.
  - `report` prints one footnote line whenever a one-off is in flight and the
    firing row is not already about it, naming the bundle and the command that
    closes it. Every other row keeps its own answer, so `gen top.segments` and
    `prepare` stay both permitted and recommended while a one-off is open --
    which is what makes the independence claimed above visible rather than
    merely true.
  - Otherwise retain the existing main-loop status precedence.

## Tests and acceptance

- Extend focused unit tests for optional CLI help, canonical one-off naming,
  the per-kind ordinal sequences, typed round discovery, duplicate/in-flight
  refusals, and unchanged top naming. Cover two one-offs against one frontier
  generation landing as `r1` and `r2`, and a one-off leaving the top sequence
  untouched.
- Verify a one-off with known YES, known NO, and unknown pairs:

  - evaluates only the unknown subset;
  - archives the complete sorted/deduplicated input;
  - admits known and newly reviewed YES pairs;
  - excludes all hard-NO pairs;
  - succeeds without `top.segments`;
  - remains independent of later changes or deletion of the original file.

- Verify main-loop isolation: completing a one-off cannot mark an older
  `top.segments` reviewed, and an open one-off does not block frontier
  generation.
- Verify the status shape around an open one-off: `_review_queued` and
  `_review_evaluating` decline for it, `_review_needed` offers `best complete`
  rather than `best review`, the frontier and search rows still fire and offer
  their own commands, and the footnote names the open bundle. Verify a
  completed one-off from a *different* target is not folded into this target's
  `best.pairs`.
- Verify multiple completed one-offs accumulate, manual `best.pairs` entries
  remain sticky, and `gen best.pairs` can reconstruct from their archives.
- Exercise the crash-recovery rows for an archived-but-unincorporated one-off
  and a `best.pairs` write whose manifest was not updated.
- Confirm in-flight one-off note recreation uses `.filtered` and reproduces
  the titles creation rendered, while `best notes -f` on an archived one-off is
  refused and names the archived source.
- Preserve the existing top-only end-to-end workflow unchanged, and add a
  one-off lifecycle to the real in-process CLI test with only external note
  operations stubbed.
- Validate with:

  ```text
  python -m unittest discover -s tests -p 'test_workflow_best*.py'
  python -m unittest discover -s tests -p 'test_workflow*.py'
  ```

## Explicitly out of scope

- Pair spelling, comma structure, case, punctuation, and letter-bag validation.
  Inputs are assumed to be acceptable line-oriented pair data.
- Special handling when every supplied pair is already classified. Such an
  input is rejected before submission and does not update `best.pairs`.
- More than one queued/evaluating review per target.
- Adding a pairs-file argument to `best complete`.
- Preserving the exact filtered contents for archived note recreation, and
  recreating an archived one-off's notes at all.
- Preflighting the 26-part note limit; the existing P2 failure behavior
  remains. An input larger than `MAX_PARTS * CHUNK_SIZE` fails in
  `notes.part_paths` after `bundle.begin` has emptied the queue, which leaves
  the bundle open with no notes and no `wf` command that clears it.
- Reconstructing the reviewed subset from the queued artifact. `best review`
  derives it, so a one-off left queued by a failed eval leg and then opened by
  a hand-run `wf eval p2` is reviewed as its full source.
- Any release or version designation; this is a single feature scope.
