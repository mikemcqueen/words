# Explicit YES/NO verdicts for dictionary word review

Supersedes `plans/archive/dict-yes-words.md`.

## Summary

A dictionary word review's notes get two checkboxes, YES and NO, in place of
today's one. Only a ticked box is a verdict. A word with neither box ticked is
undecided: it is recorded as reviewed and nothing else, and comes back in a
later review.

Each round archives its YES verdicts in a new `yes.N` record, and its NO
verdicts, as today, in `removed.N`. `wf gen dict` and every publication also
materialize the standing verdicts:

```text
dict/done/yes.words   union(dict/done/in/*.yes.N)
dict/done/no.words    union(dict/removed/*.removed.N)
```

`wf eval words` filters the candidates on `yes.words` and `no.words`, the
classified words, rather than on `reviewed.words`.

The hand-placed `dict/no.words` already in the live tree is left alone.
Nothing in this plan reads, writes, retires or merges it. The derived file is
`dict/done/no.words`, a different path.

`wf complete words` and `wf remove words` both get two guards:

- **Flipped verdict.** A word in `yes.words` ticked NO, or a word in
  `no.words` ticked YES, fails completion. Nothing overrides it.
- **Late verdict.** A word given a new verdict now, after earlier rounds
  reviewed it without one, prints a warning and fails completion unless
  `--force-verdicts` is given. A verdict is new when the word is not already
  classified that way: re-recording a standing YES as YES, or a standing NO as
  NO, is a no-op and never warns.

  ```text
  WORD was reviewed N times without a verdict
  ```

  N counts the earlier rounds whose `reviewed.N` lists the word and whose
  `yes.N` and `removed.N` don't. A round with no `yes.N` file has no YES
  verdicts, and the same holds for `removed.N`. Records are matched to their
  round by ordinal alone, as `dictionary.records()` returns them.

## Records

| path | meaning | mutable |
|---|---|---|
| `dict/done/in/<stem>.reviewed.N` | every word the round's notes listed | no |
| `dict/done/in/<stem>.yes.N` | the round's words ticked YES; written by every `complete words` round, even when empty, and never by `remove words` | no |
| `dict/removed/<stem>.removed.N` | the round's words ticked NO, removed from the dictionary | yes, for retraction |
| `dict/done/reviewed.words` | union of `*.reviewed.N` | derived |
| `dict/done/yes.words` | union of `*.yes.N` | derived |
| `dict/done/no.words` | union of `*.removed.N` | derived |

`reviewed.words` is kept and still rebuilt, but nothing filters on it any
more: `eval words` uses `yes.words` and `no.words`, and the late-verdict guard
reads the `reviewed.N` records directly. It stays as a list to inspect, and
the staleness check for it in `workflow/best/state.py` stays too.

`reviewed.N` keeps its present meaning: a note listed the word. The verdicts
are kept in their own records, so that each round says on its own which words
it classified and which way, and so that "reviewed" never has to mean
"classified".

A retraction works as it does today: editing `removed.N` and running `wf gen
dict` returns the word to the dictionary and takes it out of `no.words`, so it
is a review candidate again and ticking it YES is not a flip. The edited round
then lists the word in `reviewed.N` with no verdict, so it counts toward that
word's late-verdict warning.

Rounds from before this plan have `removed.N` records, so their removals are
in `no.words` from the first `wf gen dict`. Their kept words stay in
`reviewed.N` only and get no YES verdict: no `yes.N` is backfilled. Those
words are therefore candidates again in `eval words`, and each old round that
listed one counts toward its late-verdict warning.

All sets are normalized bare words, sorted unique under `LC_ALL=C`, as today.
`yes.N` takes its ordinal from the same global sequence as `removed.N` and
`reviewed.N`: `_next_ordinal()` scans all three record kinds, and `records()`
matches `.yes.N` with the existing ordinal pattern.

## Configuration

In `workflow/config.py`, beside `REVIEWED_WORDS_NAME` and `reviewed_words()`:

- `YES_WORDS_NAME = "yes.words"`, `yes_words(root)`;
- `NO_WORDS_NAME = "no.words"`, `no_words(root)`;

both in `dict/done/`. `dictionary.Tree` gains `yes_words` and `no_words`.
`wf init` creates neither; `wf gen dict` does.

## Notes

`wf eval words` and `wf notes words` create notes with
`notes.TWO_CHECKBOXES` in place of `notes.ONE_CHECKBOX`.

## Extraction: `steps/words_extract.py`

Parse each ENEX part for `YES`, for `NO` and for `NONE` with `note
--parse-file --type KIND --lines --two-checkboxes`, as `p2_extract` does, so
that a row counts only when it carries both boxes. `NONE` is the rows with
neither box ticked: the undecided words.

`validate_results()` keeps requiring an exact partition of the evaluated
input, now three ways:

```text
YES ∩ NO = ∅
YES ∪ NO ⊆ evaluated input
YES ∪ NO ∪ UNDECIDED = evaluated input
```

The third rule is what still catches lost rows. Retrieval stops at the first
missing note part, so a deleted or renamed part, or rows deleted from a note,
leave input words in no set at all. Without this rule those words would be
published as undecided and the round would close.

The rule is written once, as `dictionary.verdict_errors(reviewed, yes, no)`,
which returns the two word lists below. `validate_results()` and
`publish_words()` step 1 both call it, so the two checkpoints can't drift
apart. Each checkpoint guards something different. Extraction checks the
notes before any artifact is placed. Publication checks whatever it is given
before any durable record is placed: the bundle artifacts, which may have
been edited by hand since extraction, or the files `remove words` was given.

An undecided row is not a failure. `verdict_errors()` holds only the first
two rules, which publication also checks. The third needs the `NONE` parse,
so it stays in `validate_results()`. The failure message reports three
failures, each with its own word listing (`dictionary.listing()`, at most 20
words):

- `ticked both YES and NO`: `YES ∩ NO`;
- `ticked but not in the evaluated input`: `(YES ∪ NO ∪ UNDECIDED) − input`,
  for example a row whose word was edited in the note application;
- `missing from the notes`: `input − (YES ∪ NO ∪ UNDECIDED)`, for example a
  lost note part or deleted rows.

The count line becomes `input I, YES Y, NO X, undecided U; conflicting C,
extra E, missing M`, and the message still ends by naming `wf -f complete
words NAME`. The bundle
artifacts become `yes`, `no` and `input.words`. The `all` artifact is dropped:
nothing reads it. No `undecided` artifact is written either, because its count
is `input − YES − NO`, and publication reports that count from the sets it has
just normalized (see below).

## Publication: `dictionary.publish_words()`

The signature takes both verdict sets by keyword, so the same-shaped files
can't be swapped:

```python
def publish_words(root: Path, stem: str, reviewed_input: Path, *,
                  yes: Path | None, no: Path, force_verdicts: bool,
                  retry: str, enex: Path | None = None) -> Publication:
```

`yes=None` means the round records no YES verdicts and writes no `yes.N`;
`remove words` passes it. `complete words` always passes its extracted `yes`
artifact, so its round writes `yes.N` even when it is empty, as it already
writes `removed.N` when nothing was ticked NO.

`retry` is the caller's next step, which ends every failure from steps 1, 3
and 4 below. `complete words` passes "correct the notes in the note
application, then run `wf -f complete words NAME`". `remove words` passes
"correct FILE, then rerun `wf remove words`", naming its own file or files.

Before anything durable is placed, inside the staging directory:

1. Normalize `reviewed_input`, `yes` and `no` with `read_words()` (shape
   check, count prefix stripped), sorted unique, into the staged archives.
   Then check `dictionary.verdict_errors()`: `yes ∩ no = ∅` and
   `yes ∪ no ⊆ reviewed_input`. The message names the files as given and
   lists the words. Because publication normalizes its own inputs, `remove
   words` passes the user's files straight through, with no temporary copies.
2. Build the prior YES set and the prior NO set with `_record_union()` (see
   below).
3. **Flip guard.** `no ∩ prior yes` and `yes ∩ prior no` must both be empty.
   Otherwise raise, listing at most 20 words per direction and the total, and
   end with the next step:

   ```text
   cannot record 2 NO words already classified YES:
     apple  dict/done/in/X.yes.3
     pear   dict/done/in/Y.yes.5, dict/done/in/Y.yes.6
   no dictionary records were changed
   correct the notes in the note application, then run `wf -f complete words NAME`
   ```

   Each listed word names the records that hold its standing verdict, not the
   derived `yes.words` or `no.words`. Those may be stale or missing, and the
   records are what a retraction edits. The lookup reads the `yes.N` or
   `removed.N` records again, but only on this failure path. It finds which
   files hold each listed word and builds no second union.

   For YES words that are already classified NO, the message also gives the
   other fix: retract the earlier NO by taking the word out of the
   `removed.N` records listed beside it and running `wf gen dict`, then
   retry. No override exists. The
   message names `-f` only as the re-fetch after the notes are corrected, and
   never suggests `--force-verdicts`.
4. **Late-verdict guard.** For each new verdict, a word of
   `(yes − prior yes) ∪ (no − prior no)`, count the existing rounds whose
   `reviewed.N` lists it and whose `yes.N` and `removed.N` don't.
   Print
   `WORD was reviewed N times without a verdict` for each word with N > 0,
   sorted by word. Without `force_verdicts`, raise after the list:

   ```text
   3 words were reviewed earlier without a verdict; rerun with --force-verdicts to record them
   no dictionary records were changed
   ```

   This count needs each round on its own, so it reads the individual
   `reviewed.N`, `yes.N` and `removed.N` files rather than a union. It is not a
   second build of either union.

   A word already standing NO is still written to this round's `removed.N`
   (step 5), so each round's records say what that round decided. A
   retraction edits every `removed.N` that holds the word. "N new to the
   removal union" and "N newly removed from the dictionary" are counted from
   `no − prior no` only. The report counts (`X NO`) come from the step 1
   sets, so `U = N − Y − X` stays exact. YES is recorded the same way: a
   standing YES re-recorded as YES is written to `yes.N` again.
5. Stage the archives `removed.N` (= `no`), `reviewed.N` and
   `yes.N`, and the
   derived `reviewed.words`, `yes.words`, `no.words`, `words.filtered` and
   `dict.filtered`.

Each verdict union is built once. `_record_union(staging, name, recorded,
new=None)` replaces the two inline `_union()` calls in `publish_words()` and
the one in `gen_dict()`, and serves both record kinds. It returns the union of
`recorded` (`prior`), and with `new` also the union of `recorded` plus `new`
(`prospective`). A publication calls it twice:

- NO: the existing `removed.N` records, with this round's `removed.N`;
- YES: the existing `yes.N` records, with this round's `yes.N` (or no `new`
  when `yes=None`, so that `prospective` is `prior`).

The flip guard reads the two `prior` sets. The two `prospective` sets go to
`_stage_all_derived()`, which stages `yes.words` and `no.words` from them and
derives `words.filtered` and `dict.filtered` from the NO set. No step computes
either union a second time.

Both `publish_words()` and `gen_dict()` stage the derived files through one
shared helper, which replaces their separate calls to
`_stage_reviewed_words()`, `_stage_derived()` and `_stage_small_derived()`:

```python
def _stage_all_derived(tree: Tree, staging: Path, *, no: Path, yes: Path,
                       reviewed: list[Path]) -> Derived:
```

It takes the two verdict unions already built (`prospective` in a
publication, the plain unions in `gen_dict()`) and the `reviewed.N` record
list, including this round's staged archive in a publication. It stages
`reviewed.words`, `yes.words`, `no.words`, `words.filtered` and
`dict.filtered`, all with `stable_mtime=True`. It returns them together, in
placement order, for the batch and for `_report()`. A new derived file is then
added in one place, and the commands can't disagree on what they rebuild.

What the three commands share:

| | `complete words` | `remove words` | `gen dict` |
|---|---|---|---|
| entry point | `publish_words()` | `publish_words()` | `gen_dict()` |
| validation, both guards, archives, placement order | `publish_words()` | `publish_words()` | (none: records no round) |
| YES and NO unions | `_record_union()` | `_record_union()` | `_record_union()` |
| derived files | `_stage_all_derived()` | `_stage_all_derived()` | `_stage_all_derived()` |
| report lines | `_report()` | `_report()` | `_report()` |
| commit, marker | `_commit()`, `_mark_generated()` | same | same |

The only per-command code left is what is actually different: `complete
words` and `remove words` name their inputs and pass `yes` or `yes=None` and
their `retry` text, and `gen_dict()` has no round to record.

### Placement

Staging stays where it is today, in the system temp directory.

Placement order:

```text
1. dict/done/in/<stem>.yes.N               # complete words only
2. dict/removed/<stem>.removed.N
3. dict/done/in/<stem>.reviewed.N
4. dict/done/reviewed.words
5. dict/done/yes.words
6. dict/done/no.words
7. dict/words.filtered
8. dict/dict.filtered
9. dict/done/out/enex/<stem>.reviewed.N/   # complete words only
10. .words.filtered.gen
```

Each placement is one atomic rename. Several renames can't be made atomic
together: staging everything as `.tmp` files beside their destinations and
renaming them at the end still leaves a sequence of renames, with the same
windows as this order. So the order is what makes a partial publication safe,
and it rests on one rule:

**`reviewed.N` is the round's commit record.** It is placed after the round's
verdict records, so a `reviewed.N` on disk means that round's `yes.N` and
`removed.N` are complete. The late-verdict count reads only rounds that have a
`reviewed.N`, so it never mistakes an interrupted round's words for words
reviewed without a verdict.

What an interruption leaves, and how it heals:

- **Before step 3.** A `yes.N` and maybe a `removed.N` exist with no
  `reviewed.N`. Their verdicts are real, since the reviewer made them, and the
  next `wf gen dict` or publication unions them in. The bundle is still
  active, so `wf complete words NAME` reruns publication. `_next_ordinal()`
  sees ordinal N in use and allocates N+1. The rerun records the same
  verdicts, so no flip. The late-verdict guard treats a verdict whose record
  already landed as standing, not new:
  - Interrupted after step 2: both verdict records are on disk, so nothing is
    new, and the rerun needs no `--force-verdicts`.
  - Interrupted after step 1 and before step 2: `yes.N` is on disk but
    `removed.N` is not, so the round's NO words are still new. If the first
    attempt needed `--force-verdicts` for them, the rerun needs it again.

  The orphaned records at N are harmless duplicates.
- **After step 3, before step 10.** The round is fully recorded, and only
  derived files or the marker are behind: `wf gen dict` rebuilds them. A rerun
  of `complete words` records a duplicate round N+1 with the same verdicts.
  Both guards pass, because round N carries its own verdicts. The duplicate
  `reviewed.N+1` lists the round's undecided words a second time, so each of
  them counts one extra time in a later late-verdict warning: a word reviewed
  once without a verdict is reported as reviewed twice. This is accepted. It
  only overstates the count after an interrupted publication was rerun, and
  the warning's purpose (this word was seen before without a verdict) still
  holds.
- **Step 9 fails.** The round is recorded, and the notes stay in the bundle.
  The rerun behaves as in the case above.

`_next_ordinal()` already scans every record kind for this reason: a surviving
half of an interrupted round never has its ordinal reused.

The derived files keep `stable_mtime=True`.

A failed guard leaves the active bundle and its retrieved ENEX in place. After
correcting the notes, `wf -f complete words NAME` re-fetches them and reaches
both guards again; `-f` only re-fetches and accepts nothing.

`_report()` adds lines for `yes.words` and `no.words`, in the same three forms
as the other derived files.

`Publication` gains `yes_dst: Path | None` (None for `remove words`), and
`reviewed`, `yes` and `no` counts taken from the normalized staged archives.
`report_publication()` prints the round's records in placement order under
`recorded as generation N`:

```text
recorded as generation 7 -> dict/done/in/X.yes.7
                         -> dict/removed/X.removed.7
                         -> dict/done/in/X.reviewed.7
```

`words_publish` reports `N words reviewed: Y YES, X NO, U undecided` from those
counts, with `U = N − Y − X`. That is exact because step 1 has already made YES
and NO disjoint subsets of the reviewed set, and the counts come from the
normalized sets, not from the bundle artifacts, which may have been edited by
hand.

## `wf complete words`

`complete words` gains `--force-verdicts`: accept late verdicts. It is
separate from the global `-f`, which keeps meaning "re-fetch the notes and
replace the extraction". `words_publish.run_step()` passes the extracted `yes`
and `no` artifacts, the flag and its `retry` text to `publish_words()`.

## `wf eval words`

- The candidate filter is `yes.words ∪ no.words`, in place of
  `reviewed.words`. `check()` requires both files and names `wf gen dict` when
  either is missing. `filter_reviewed_words()` becomes
  `filter_classified_words(source, classified: list[Path], staged)`, which
  reads both files into one set; `eval words` is its only caller. Both of its
  refusals become `no unclassified words in FILE`: the one on the queued file
  and the one on the `.filtered` file kept from an interrupted eval.
- `eval words` and `notes words` drop `--checked` (`notes.add_checked()`): a
  word review's notes start with no box ticked. `eval p2` and `notes p2` keep
  it.

## `wf remove words`

Direct removal records its removals as `removed.N`. It records no YES
verdicts and writes no `yes.N` (`yes=None`). With `--reviewed`, the reviewed
set is archived as `reviewed.N`; its kept words get no verdict.

`remove words` gains `--force-verdicts` and behaves like `complete words` on
both guards: a word in `yes.words` fails the flip guard, and a word given a
new NO after earlier rounds reviewed it without a verdict prints the
late-verdict warning and fails unless `--force-verdicts` is given.

`remove_words()` no longer filters the submission against `reviewed.words`
(`filter_reviewed_words`). That filter silently dropped a submitted YES word,
so the flip guard could never fire, and it dropped every undecided word, so an
undecided word could never be removed. The guards replace it:

- a submitted word already standing NO is written to the new `removed.N`
  too (publication step 5), and changes nothing downstream;
- a submitted word in `yes.words` fails the flip guard;
- a submitted word that was reviewed earlier without a verdict gets the
  late-verdict warning.

`remove_words()` also drops its own check that the removed words are in
`--reviewed`, and the temporary copies it writes. It passes FILE and the
`--reviewed` file (or FILE again, without `--reviewed`) straight to
`publish_words()`, whose step 1 normalizes them with the same `read_words()`
and makes the subset check once, naming both files. A dictionary edited by
hand, or rebuilt by `wf gen dict`, between the review and this command doesn't
affect that check, which compares only the submitted files. It affects only
the guards, and those read the records at publish time, as they should.

Today's `no unreviewed words in FILE` refusal goes away with the filter. A
submission whose every word is already NO is published like any other. It
reports `0 new to the removal union` and rebuilds the derived files. That is
also what a rerun after an interrupted `remove words` needs: the earlier
attempt's `removed.N` may already be on disk, and the rerun has to go through
to write `reviewed.N` and the derived files.

## `wf gen dict`

Rebuilds `reviewed.words` from its records, builds the YES and NO unions with
`_record_union()`, and stages `yes.words`, `no.words`, `words.filtered` and
`dict.filtered` from them through `_stage_all_derived()`, then advances the
generation marker. On a tree with
no `yes.N` records yet, it writes `yes.words` empty.

## Staleness

`workflow/best/state.py` reports a missing `yes.words` or `no.words` as a
dictionary-staleness reason (`YES words missing`, `NO words missing`), offering
`wf gen dict`, beside the existing `reviewed.words` check.

## History

`workflow/history.py` `_dictionary_event()` adds a YES line for the latest
round, beside its `removals:` line. It finds `<stem>.yes.<ordinal>` in
`tree.reviewed_inputs` with the same stem and ordinal it already uses for
`removed.N`. When the file exists it prints `YES: N words`. When it doesn't,
it prints nothing, because direct removals and rounds from before this plan
have no `yes.N`. `yes.N` can't be edited, so it is not added to the
change-time check that warns `removal records changed since the last
rebuild`.

## Docs and plans

- `docs/workflow-commands.md`: two-checkbox word notes, the classified-word
  filter, the two guards with `--force-verdicts`, and the new records.
- `plans/archive/managed-dict-v2.md` and `plans/archive/dict-review-cycle.md`: point to this
  plan where they describe one-checkbox word notes or the reviewed-words
  filter.

## Tests

Smoke coverage only. Existing tests that assert the one-checkbox partition,
the `remove words` reviewed-word filter, or `no unreviewed words` are updated
to the new behavior or deleted, not kept alongside it.

- A completion with YES, NO and undecided words writes `reviewed.N`, `yes.N`
  and `removed.N`, and `yes.words` and `no.words`. The undecided words appear only in
  `reviewed.N`.
- A flipped verdict fails with and without `-f` and `--force-verdicts`, and
  changes nothing.
- A late verdict fails without `--force-verdicts`, and with only `-f`, and
  publishes with `--force-verdicts`.
- `eval words` drops classified words.
- `remove words` of a YES word fails the flip guard; of a word reviewed
  earlier without a verdict, fails without `--force-verdicts` and publishes
  with it; of an undecided word, removes it.
- Re-recording a standing verdict the same way does not warn.
- A review with undecided words and one word ticked both ways fails, and the
  message has no `missing` count.
- `wf gen dict` rebuilds both verdict files from the records; a word retracted
  from every `removed.N` leaves `no.words`.
- A review whose notes lost rows fails extraction with a `missing` count.
- A standing NO submitted again is written to the new `removed.N`, and is
  not counted as new to the removal union or newly removed.
- A flip-guard failure names the record that holds each listed word.
- History shows the latest review round's YES count.

## TODO, not in this plan

- `eval words -y` / `-n`: include `yes.words` / `no.words` words again, with
  their boxes pre-ticked to match.

- pgui's freqsort word review (nutrimatic) submits through `wf remove words
  --reviewed`, and an unchecked word there means remove. It should move to
  explicit YES/NO verdicts too.
