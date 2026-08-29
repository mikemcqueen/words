# Force-refresh a completed BEST review

## Goal

Support:

```text
wf best complete SENTENCE -o LETTERS -g COUNT [-m LENGTH] -f
wf best complete SENTENCE -u LETTERS -g COUNT [-m LENGTH] -f
```

after the target's latest BEST review has already completed. The command must
download the existing note parts again, extract their current checked and
unchecked pairs, update the durable artifacts for that same review round, fold
new confirmed YES pairs into the standing classified set, and regenerate
`best.pairs`.

This is a refresh of an existing review round. It does not call `note --create`,
does not create an `rN+1` review, and does not move the archived input back
through `p2/queued` or `p2/eval`.

## Current facts that determine the design

For a completed P2 bundle named `BUNDLE`, the current completion recipe leaves:

```text
.wf/p2/done/in/BUNDLE.pairs
.wf/p2/done/out/BUNDLE.p2.yes
.wf/p2/done/out/enex/BUNDLE/*.enex
.wf/p3/queued/BUNDLE.p2.no
```

The first file is enough to recover every note title. Note creation split it
into contiguous parts named `BUNDLE.pairs.aa`, `BUNDLE.pairs.ab`, and so on;
`p2_retrieve._title()` already renders those titles from the source filename
and probes until the first `note not found` response. A refresh therefore does
not need note IDs or a new note-creation pass.

The old per-review YES evidence has not been lost:
`.wf/p2/done/out/BUNDLE.p2.yes` is the exact checked set extracted by the prior
completion. It can be compared with the refreshed YES set before it is
replaced.

There is no `classified/yes/in` provenance directory. The only classified YES
artifact is `.wf/classified/yes/yes.pairs`, a global append-only union containing
both P2 confirmations and explicit `wf classify yes` inputs. Consequently, a
refresh can detect that a pair was unchecked, but it cannot safely remove that
pair from the global aggregate: another review or an explicit classification
may also support it.

Current staging is split between two places:

- P2 retrieval writes each download atomically into the active bundle's
  `enex.part/`, then renames the completed directory to `enex/`.
- P2 extraction writes per-note parser output to fixed names under `/tmp`, then
  merges those files into `BUNDLE.p2.yes` and `BUNDLE.p2.no` inside the bundle.

The refresh path will instead stage all downloads and extracted sets together
in one unique temporary directory. Nothing durable changes until every note
has downloaded and both sets have been extracted successfully.

## Command contract

Keep the current behavior when a review is in flight:

1. A queued review is still an error, including with `-f`; the user must run
   `wf eval p2` before completing it.
2. An evaluating review is completed by the existing P2 recipe. `-f` retains
   its existing lower-level meaning of ignoring `is_done` and overwriting an
   archive collision.
3. With no queued or evaluating review, the command without `-f` retains the
   current `no review awaiting completion` error.
4. With no queued or evaluating review and with `-f`, refresh the target's
   highest completed review round.
5. If the target has no completed review, fail without downloading anything.

Select the highest numeric `rN`, not the lexicographically last filename or the
newest mtime. Add one narrowly scoped parser for names rendered by
`Review.run()`: require
`<target.review_prefix><cutoff>.r<positive-integer>.pairs`, reject malformed
matches, and use the same helper to make a new review `max(rN) + 1`. Continue to
obtain `BUNDLE` with `names.queue_stem("p2", archived_source.name)` and render
the YES/NO artifacts with `names.artifact()`; do not recover those dimensions
by splitting filenames.

## Refresh algorithm

Put the refresh implementation in `workflow/best/refresh.py`; keep
`workflow/best/commands.py` responsible for selecting normal completion versus
refresh and for rebuilding/reporting target state.

### 1. Resolve and preflight the completed round

Given the selected `.wf/p2/done/in/BUNDLE.pairs`, resolve and require:

- the archived source as a regular file;
- `.wf/p2/done/out/BUNDLE.p2.yes` as a regular file;
- `.wf/p2/done/out/enex/BUNDLE` as a directory;
- `.wf/p3/queued/BUNDLE.p2.no` as a regular file.

The P3-NO file must still be queued because refresh will replace the soft-NO
snapshot. If it has already left P3, fail before downloading: silently writing
a second copy would leave downstream P3 state based on the old verdicts.

Do not alter `p2_done.pairs`. The archived source already belongs to that
done-set and refresh does not evaluate a new input.

### 2. Download every existing note into scratch

Create a `tempfile.TemporaryDirectory` under
`.wf/p2/done/out/enex/`. Keeping scratch on the same filesystem permits atomic
renames during publication while still giving the whole attempt a unique,
automatically cleaned directory.

Refactor `workflow/steps/p2_retrieve.py` to expose a primitive that accepts a
source filename and a destination directory. For indices `aa` through `az`:

1. Render the title with the existing `_title(source, index)` rule.
2. Run `note -pf.72 --get TITLE --production`.
3. Write successful stdout to `TITLE.enex.tmp` in scratch and rename it to
   `TITLE.enex` only after the subprocess succeeds.
4. Stop at the first `note not found` response; propagate every other failure.
5. Require at least `.aa`.

The scratch directory is empty at entry, so this downloads every part even
though an older ENEX archive exists. The primitive must not call
`note --create` and must not consult or copy the old ENEX files.

Keep ordinary P2 completion's resumable `enex.part/` behavior by having its
step wrapper call the same primitive with its existing staging directory; only
the refresh caller deliberately starts with empty scratch.

### 3. Extract fresh YES and NO sets into scratch

Refactor `workflow/steps/p2_extract.py` so its parsing primitive accepts:

- the downloaded ENEX paths;
- `YES` or `NO`;
- a caller-supplied parser scratch directory; and
- a caller-supplied final set path.

Write each `note --parse-file ... --type KIND --lines` result under the unique
refresh scratch directory, not to the current fixed `/tmp/<name>.parsed` path.
Merge those per-note files with `setops.merge()` into scratch
`BUNDLE.p2.yes` and `BUNDLE.p2.no`. These are sorted, unique sets and may be
empty.

Before publication, validate that fresh YES and fresh NO are disjoint and that
their union equals the archived source pair set. Treat overlap or missing/extra
pairs as an extraction error and leave every durable artifact untouched. This
prevents a malformed or incomplete note download from being accepted merely
because both parser commands exited successfully.

### 4. Detect checked-to-unchecked changes

Compute in scratch:

```text
added_yes   = fresh_yes - archived_yes
removed_yes = archived_yes - fresh_yes
```

Report the added count. If `removed_yes` is nonempty, emit a warning before any
durable replacement, including the count and up to three sample pairs:

```text
WARNING: refreshed review has N previously confirmed YES pair(s) now unchecked:
PAIR1, PAIR2, PAIR3 (+M more); classified/yes is append-only, so they remain
confirmed YES
```

Proceed after the warning. Fold the complete fresh YES set into
`classified/yes/yes.pairs`; union is idempotent and ensures newly checked pairs
are recorded. Do not subtract `removed_yes` from the classified aggregate.
Because provenance for explicit classifications and other review rounds was
collapsed into that aggregate, automatic retraction would be unsound.

This means forced refresh supports the requested additive correction. It
records the new per-review snapshot and warns about reversals, but a reversal
does not remove a pair from `best.pairs` while that pair remains in the global
confirmed-YES set. A future retraction feature requires classified-YES
provenance and is outside this change.

### 5. Publish the refreshed snapshot

After download, extraction, validation, and warning have all succeeded:

1. Replace `.wf/p2/done/out/enex/BUNDLE` with the staged ENEX directory.
2. Atomically replace `.wf/p2/done/out/BUNDLE.p2.yes` with fresh YES.
3. Atomically replace `.wf/p3/queued/BUNDLE.p2.no` with fresh NO.
4. Fold fresh YES into `.wf/classified/yes/yes.pairs` using
   `config.fold_classified()` so stable-mtime behavior remains centralized.
5. Call `generate.build_best_pairs(target)` and then `report(target)`, exactly
   as normal `wf best complete` does.

Use sibling temporary files plus `Path.replace()` for the two set files. A
nonempty directory cannot be replaced directly with `rename`, so add a small
directory-swap helper for ENEX:

1. require that no stale `BUNDLE.refresh-old` backup exists;
2. rename the old archive directory to that backup;
3. rename the staged directory to the canonical archive path;
4. restore the backup if step 3 fails; and
5. remove the backup only after the canonical directory is in place.

The publication is not a multi-file transaction, but every operation is
overwrite/idempotent and a repeated `-f` refresh converges. In particular,
always fold all of fresh YES rather than only `added_yes`, so a failure after
replacing the archived YES file but before updating the aggregate remains
recoverable by rerunning the command.

## Files to change

- `workflow/best/commands.py`
  - choose normal completion or forced refresh;
  - leave queued-review handling unchanged;
  - rebuild `best.pairs` after either successful path.
- `workflow/best/state.py`
  - represent completed review rounds and select the highest numeric round;
  - use the same numeric-round helper when assigning the next review round.
- `workflow/best/refresh.py` (new)
  - resolve durable round artifacts, stage refresh outputs, compare verdicts,
    warn, and publish.
- `workflow/steps/p2_retrieve.py`
  - expose caller-directed note retrieval while preserving normal completion's
    `enex.part/` resume behavior.
- `workflow/steps/p2_extract.py`
  - expose caller-directed extraction and stop using fixed parser paths in
    `/tmp`.
- `workflow/fs.py`
  - add the narrow recoverable directory replacement helper if it is useful
    outside the BEST wrapper; otherwise keep it private to `best/refresh.py`.
- `tests/test_workflow_p2.py`
  - retain ordinary P2 retrieval/resume/extraction coverage after primitive
    extraction.
- `tests/test_workflow_best.py`
  - cover completed-round selection and command routing.
- `tests/test_workflow_best_e2e.py`
  - exercise a normal review completion followed by a forced refresh.

## Required tests

1. Without `-f`, a completed target still reports no review awaiting
   completion.
2. `-f` does not bypass a queued review and does not call `note --create`.
3. `-f` with no completed review fails before any note subprocess call.
4. Numeric selection chooses `r10` over `r9`; malformed matching review names
   are rejected.
5. Retrieval asks for the exact archived-source titles `.aa`, `.ab`, ... and
   redownloads them despite an existing ENEX archive.
6. A retrieval or parse failure leaves archived ENEX, archived YES, queued NO,
   classified YES, and `best.pairs` byte-for-byte unchanged.
7. A successful refresh replaces the ENEX snapshot, archived per-review YES,
   and queued soft-NO snapshot; stale extra ENEX parts disappear.
8. A newly checked pair is folded into classified YES and appears in
   `best.pairs`.
9. A previously checked pair that is now unchecked produces the warning with
   count/sample, is absent from the refreshed per-review YES file, but remains
   in classified YES and therefore is not automatically retracted from
   `best.pairs`.
10. Fresh YES/NO overlap or failure to partition the archived source aborts
    before publication.
11. Rerunning `-f` with unchanged notes is a successful no-op for set contents
    and does not advance stable mtimes for classified YES or `best.pairs`.
12. Failure during publication is recoverable by rerunning `-f`, including the
    ENEX directory-backup case.

Run the focused workflow suite:

```text
python -m unittest discover -s tests -p 'test_workflow*.py'
git diff --check
```

No production notes should be created or modified by automated tests; fake the
`note --get` and `note --parse-file` subprocesses as the existing P2 tests do.

## Explicitly out of scope

- Creating a new review round; use `wf best review`.
- Recreating or modifying notes during refresh.
- Removing a standing classified YES verdict when a checkbox is unchecked.
- Adding per-source provenance under `classified/yes`.
- Refreshing a review whose soft-NO output has already left `p3/queued`.
- Implementing P3.
