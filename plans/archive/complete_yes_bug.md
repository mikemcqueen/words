# Fix the filtered p2 YES eval/completion lifecycle

## Goal

Make `wf eval yes` and `wf complete yes` agree about the temporary filtered
work file created when previously completed p2 pairs are removed. Preserve the
full source filename by appending `.filtered`, allow completion of review notes
that have already been created with that basename, and remove the temporary
file only after completion succeeds.

No automated tests are required for this change.

## Current bug

`eval_pairs.filter_done_pairs()` correctly appends `.filtered` to the complete
source name:

```text
s8.2-3.4.pairs.p1.90.10.yes
-> s8.2-3.4.pairs.p1.90.10.yes.filtered
```

`wf eval yes` splits that filtered file and consequently creates review notes
whose titles end in `.yes.filtered.aa`, `.yes.filtered.ab`, and so on. The
original `.yes` file remains alongside it in `p2/eval`.

`wf complete yes` accepts the original `.yes` filename, but currently looks for
notes named from that original filename. It therefore searches for
`.yes.aa` instead of `.yes.filtered.aa`. Passing the filtered filename does not
work around this: `_phase2_name()` requires the canonical name to end in
`.yes`. Completion also has no cleanup for the filtered work file.

The same filename construction has already drifted in p1: the producer appends
`.filtered`, while `complete_pairs.py` uses `with_suffix(".filtered")` and looks
for a different name. A shared path helper should remove both opportunities for
future drift.

## Settled naming and ownership

Use two explicit paths throughout p2 processing:

| Role | Example | Ownership and lifetime |
|---|---|---|
| Canonical YES-pairs file | `foo.pairs.p1.90.10.yes` | Moves from `p2/queued` to `p2/eval`, remains the CLI identity, and is archived in `p2/done/in` |
| Review work file | `foo.pairs.p1.90.10.yes.filtered` | Derived sibling containing only pairs not already in `p2_done.pairs`; drives note creation and is deleted after successful completion |

Appending `.filtered` is intentional. Do not use `Path.with_suffix()`: replacing
the last suffix would discard `.pairs` or `.yes`, lose phase/provenance
information, and allow different source artifact types to collapse to the same
name.

The canonical original remains the argument to `wf complete yes`:

```text
wf complete yes foo.pairs.p1.90.10.yes
```

The `.filtered` file is an internal work artifact, not a second valid CLI
identity.

## Implementation

### 1. Centralize filtered-path construction

In `workflow/eval_pairs.py`, add a small public helper such as:

```python
def make_filtered_pairs_path(src_pairs: Path) -> Path:
    return src_pairs.with_name(src_pairs.name + ".filtered")
```

Use it inside `filter_done_pairs()` instead of constructing the name inline.
The helper must append to the full name and must work unchanged for both p1
`.pairs` inputs and p2 `.yes` inputs.

Also update `workflow/complete_pairs.py` to use this helper when finding and
unlinking its p1 filtered sibling. This is the same producer/consumer naming
contract and fixes the already identified p1 cleanup mismatch without changing
the p1 filtering algorithm.

### 2. Make the two p2 paths explicit during eval

Refactor `workflow/eval_yes.py::_eval_yes()` to retain separate variables rather
than overwriting `dst_pairs`:

- `source_pairs`: the canonical `.yes` file moved into `p2/eval`.
- `work_pairs`: either `source_pairs`, when `--no-filter` is used or no
  `p2_done.pairs` exists, or the appended `.yes.filtered` file returned by
  `filter_done_pairs()`.

Split `work_pairs` and continue deriving the split prefix from
`work_pairs.name`. This intentionally preserves the note titles already emitted
by the current code, including `.yes.filtered.aa`. Log both identities when they
differ so the completion operand is unambiguous: the reviewed count/work file
and the canonical `.yes` filename to pass to `wf complete yes`.

Do not rename the filtered file over the canonical YES-pairs file and do not
archive it.

### 3. Resolve the review work file during completion

At the start of `workflow/complete_yes.py::_complete()`:

1. Treat `src_pairs` as the canonical YES-pairs file and validate its existing
   `.yes` naming before performing any note-store operations.
2. Derive `filtered_pairs` with the shared helper.
3. Select `work_pairs = filtered_pairs` when that sibling exists; otherwise use
   `src_pairs`.

Use these roles consistently:

- Pass `work_pairs` to `_retrieve_notes()` so completion looks up the exact note
  titles created by eval.
- Pass `src_pairs.name` to `_phase2_name()` so final p2 YES/NO output names retain
  the canonical p1 probability-band identity and never contain `.filtered`.
- Pass `work_pairs` to `merge_with_done_pairs()` so `p2_done.pairs` records the
  subset actually presented for manual review. Pairs excluded from the work
  file are already present in that aggregate.
- Continue moving only `src_pairs` to `p2/done/in`; it is the canonical submitted
  YES-pairs file.

If the user supplies a filename ending in `.filtered`, reject it before note
retrieval with a clear message directing them to pass the sibling canonical
`.yes` filename. Do not silently strip arbitrary suffixes from CLI input.

This resolution supports both cases:

```text
No p2 history / --no-filter:
  work_pairs == canonical .yes
  note titles: canonical.yes.aa

Existing p2 history:
  work_pairs == canonical .yes.filtered
  note titles: canonical.yes.filtered.aa
```

The second case is backward-compatible with review notes already created by the
current `wf eval yes` implementation.

### 4. Clean up only after successful completion

Keep the filtered work file through note retrieval, parsing, aggregate updates,
and archival. After all of the following have succeeded:

- note parts were retrieved and parsed;
- p2 YES and unchecked/NO output files were written;
- the classified-YES and `p2_done.pairs` aggregates were updated;
- the generated p2 YES output was moved to `p2/done/out`;
- the canonical input was moved to `p2/done/in`;

unlink `filtered_pairs` if it existed. Do not unlink `src_pairs`, and do not
unlink the filtered file on an exception. Retaining it on failure preserves the
exact reviewed subset and permits diagnosis/recovery.

Before the first mutating or external operation, resolve all known destination
paths and apply the existing non-`--force` existence checks where practical.
This reduces late partial failures. Do not broaden this task into a transactional
rewrite of all aggregate updates; retain the current `--force` recovery model.

### 5. Keep phase-routing semantics out of scope

`complete_yes.py` currently treats unchecked note entries as NO and routes them
to `p3/queued`. That conflicts with p3's intended meaning as a second automated
pass over p1 auto-NO pairs, but it is a separate provenance/semantics issue.
This change should preserve the current output routing while referring to those
entries as unchecked/NO rather than claiming they are explicit human rejects.

Do not rename p2 result artifacts, change checkbox parsing, or redesign p3 as
part of this bug fix.

## Files to change

| File | Change |
|---|---|
| `workflow/eval_pairs.py` | Add the shared append-only filtered-path helper and use it in `filter_done_pairs()` |
| `workflow/eval_yes.py` | Track canonical source-pairs and review-work paths separately; improve completion guidance in the success log |
| `workflow/complete_yes.py` | Resolve the filtered sibling, retrieve its notes, derive outputs from the canonical name, merge the reviewed subset, and clean up last |
| `workflow/complete_pairs.py` | Use the shared helper for the existing p1 filtered-file lookup and cleanup |

No test files should be added or changed.

## Validation (no tests)

1. Run `python -m compileall workflow` to catch syntax/import errors.
2. Run `git diff --check` and inspect the diff to confirm every filtered path is
   constructed through the shared append-only helper.
3. Do not create, download, or delete production notes solely for validation.
   On the next user-selected p2 pairs file that has previously completed pairs:
   - confirm `wf eval yes` retains the canonical `.yes` file and creates a
     `.yes.filtered` sibling;
   - confirm its log says to complete the canonical `.yes` filename;
   - after manual review, run `wf complete yes` with that canonical filename;
   - confirm it retrieves `.yes.filtered.a?` notes, writes outputs without a
     `.filtered` component, archives the canonical input, merges only the
     reviewed subset into `p2_done.pairs`, and removes the filtered sibling.
4. For an unfiltered canonical pairs file, confirm completion continues
   retrieving notes directly from the canonical `.yes` filename and performs no
   filtered-file cleanup.

## Existing workspace artifact

Do not bulk-delete existing `.filtered` files during implementation. In
particular, keep the active p2 `s8.2-3.4.pairs.p1.90.10.yes.filtered` work file:
its basename identifies the notes already created for that review. Once the
fixed completion succeeds for its canonical `.yes` sibling, normal cleanup will
remove it.
