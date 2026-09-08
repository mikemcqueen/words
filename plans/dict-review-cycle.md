# Dictionary word review cycle

## Summary

Add a root-global manual review lifecycle for dictionary words:

```text
wf submit words [--as NAME] FILE
wf eval words NAME
wf notes words NAME
wf complete words NAME
```

The lifecycle lives in `.wf/dict/{queued,eval,done}`. It borrows the bundle,
note, retrieval, and resumable-step shape of p2, but it does not pretend that
words are another pair phase. A submitted file is copied into the queue byte
for byte; evaluation filters previously reviewed words by their bare identity
while preserving the current counted lines for display; completion treats a
checked box as "keep" and an unchecked box as "remove"; and the resulting
records flow through the same dictionary publication code as `wf remove
words`.

This is the review-cycle follow-up reserved by `plans/managed-dict-v2.md`.
That plan is implemented and is the source of truth for the dictionary tree,
generation numbering, durable records, derivation, reporting, and accepted
single-process/crash-window model. This plan keeps those semantic contracts but
replaces dictionary-local staging directories with standard-library temporary
storage for `remove words`, `gen dict`, and the shared `complete words`
publication.

The numerical "implausible removal ratio" guard anticipated in that plan is
deliberately dropped. There is no removal-percentage cutoff and no force flag
that bypasses one.

## Confirmed decisions

- `words` is the public lifecycle scope and `dict` is the real layout part.
  Command prose says `words`; filesystem paths say `dict/`.
- `submit words` copies its input unchanged. It does not strip counts, sort,
  deduplicate, remove blank lines, or otherwise normalize the queued copy.
- Without `--as`, the queued name is the submitted path's basename. With
  `--as NAME`, `NAME` becomes the queued filename, eval bundle name, and
  eventual archive stem. The note-title prefix is the evaluated filename:
  `<NAME>` when nothing was filtered and `<NAME>.filtered` when eval removed
  previously reviewed words. The source file is never modified.
- A word queue has one name shape and adds no suffix. For example,
  `--as top.s7.m4.g5.words` produces
  `dict/queued/top.s7.m4.g5.words` and then
  `dict/eval/top.s7.m4.g5.words/`.
- Eval compares bare word identities against `dict/done/reviewed.words`, but
  retains each surviving row's count and position in the submitted ranking.
  Ordinary text I/O normalizes line endings to the platform default; exact
  line-ending preservation is not a contract.
- Eval and `remove words` use the same row parser. It checks for a
  whitespace-only row first and ignores it; otherwise it removes the optional
  `^ *[0-9]+ ` prefix, strips surrounding whitespace, and requires a nonempty
  full `[a-z]+` match. Thus surrounding whitespace is accepted, while a
  nonblank count-only row is malformed. It does not lowercase or otherwise
  clean a malformed word. A failure names the source, line number, and
  offending text; during eval the file remains queued and no bundle or note is
  made.
- Notes have one checkbox. Checked means keep; unchecked means remove.
  `note --parse-file --type YES --lines` reads checked rows and
  `--type NONE --lines` reads unchecked rows. There is no parse-side
  `--checkbox` option and `--two-checkboxes` must not be passed.
- `complete words` extracts normalized sorted-unique `.yes` and `.no` files,
  unions them into `.all`, and requires `.all` to match the normalized
  sorted-unique evaluated input byte for byte. It also requires the YES/NO
  intersection to be empty. These checks live in one `validate_results()`
  function; no separate count-equality check is needed.
- Note lookup ordering is being pursued independently by the operator, with
  descending modified time as the intended order. This plan adds no note
  lookup change or review-round suffix. Partition validation does not prove
  which note round supplied the verdicts when rounds contain identical words.
- The `.removed.N` / `.reviewed.N` ordinal is allocated during completion,
  after note retrieval, extraction, and validation and immediately before the
  prepared dictionary publication. Nothing is reserved at submit or eval.
  Thus a manual `remove words` performed while a review waits simply consumes
  the next ordinal, and completion selects the next one when it actually
  records its verdicts. Only truly concurrent writers can race; concurrency is
  already outside the managed-dictionary contract.
- Completed raw note parts are archived at
  `dict/done/out/enex/<NAME>.reviewed.N/`. No checked/unchecked verdict files
  are archived under `done/out`: removals belong only in `dict/removed`, and
  kept words remain derivable as `reviewed.words` minus the removal union.
- `wf notes words NAME` recreates notes only for an active
  `dict/eval/NAME/` bundle. It does not operate on queued or completed rounds.
  The normalized `done/in` record does not retain the counts or display order
  required to reproduce completed notes exactly.
- Disposable file preparation uses `tempfile.TemporaryDirectory()` for a
  related batch and a standard-library named temporary file for a single
  prepared file. Neither call specifies a directory. No
  `.wf/dict/.staging.*` directory is created. Prepared files are renamed
  directly to their destinations through `setops.place`; no destination-local
  copy is inserted first. Cross-filesystem rename behavior is deliberately not
  checked or handled by this plan.

## What p2 does today, and what words should do

| concern | p2 | dictionary words |
|---|---|---|
| public/layout name | both are `p2` | public `words`, layout `dict` |
| submitted name | adds or preserves a queue suffix from `names.py` | preserves the basename, or uses `--as NAME`, with no suffix |
| queued content | `sort -u` pairs | byte-for-byte copy |
| bundle identity | queue suffix is removed | queued filename itself |
| done filtering | `comm -23` against `p2_done.pairs` | streaming identity subtraction against `dict/done/reviewed.words` |
| filtered presentation | sorted pair set | surviving original lines in original order, with current counts |
| note shape | `--two-checkboxes`, explicit Y and N | `--checkbox`, checked keep and unchecked remove |
| extraction | staged p2 YES and NO pair sets | staged normalized checked and unchecked word sets |
| result validation | rejects the YES/NO intersection | rejects the YES/NO intersection and requires their union to equal evaluated input |
| durable verdicts | folds both explicit pair verdicts into `classified/` | unchecked words become one mutable removal generation; no separate kept store |
| completed-input aggregate | folds evaluated pairs into `p2_done.pairs` | rebuilds `dict/done/reviewed.words` from per-round reviewed inputs |
| completion names | deterministic from the bundle name; no ordinal | allocates the next global dictionary ordinal at completion |
| input archive | moves the original bundle source into `p2/done/in` | writes normalized evaluated words to `dict/done/in/<NAME>.reviewed.N` |
| output archive | p2 YES, p2 NO, and raw enex | raw enex only, under `<NAME>.reviewed.N/` |
| note recreation | active review, plus completed source under `-f` with caveats | active review only |

The reusable boundary is therefore below the phase policy. Keep the generic
bundle directory, note title/splitting, note creation, atomic ENEX retrieval,
and step runner. Keep word queue naming, identity filtering, extraction,
validation, dictionary publication, and completed archive policy specific to
the words lifecycle. Do not add word conditionals throughout p2 steps.

## Layout and scope resolution

Extend `_DICT` in `workflow/config.py`:

```text
dict/
  words.big
  words.filtered
  .words.filtered.gen
  removed/
  queued/
  eval/
    NAME/
  done/
    reviewed.words
    in/
      NAME.reviewed.N
    out/
      enex/
        NAME.reviewed.N/
```

`wf init` grows an existing tree by creating the new directories; it still
does not create either derived dictionary file. Because `show._build_aggregates`
collects every layout part named `queued` or `eval`, the new work appears
automatically in `wf show all queued` and `wf show all eval` under real
`dict/...` paths. On a workflow root initialized before this layout change,
running `wf init` is a hard prerequisite for both aggregate commands: until it
creates `dict/queued` and `dict/eval`, either aggregate view fails rather than
merely omitting dictionary work.

Do not add a `words` alias to the configured layout. The dispatcher consumes
the public `words` scope before invoking the lifecycle action; that action is
configured with canonical layout scope `dict` for every path. Split the
concepts currently combined in `phase` at that command/action boundary:
summaries and errors use public scope `words`, while `Context` and
`config.path()` receive canonical scope `dict`. Layout commands consequently
continue to accept and display `dict`, including `wf show dict queued`; `wf show
words queued` is not added. This also leaves both configuration walkers and the
unrelated partial-match TODO in `dispatch.registry_key` unchanged.

## Queue naming and bundle selection

Represent the suffixless words queue explicitly in `workflow/names.py` rather
than using an empty suffix in `QUEUE_SUFFIXES`. The required operations are:

- `queue_name("words", name) -> name` after the existing safe-name check;
- `queue_stem("words", name) -> name`;
- `queue_names("words", bundle_name) -> (bundle_name,)` for exact source
  lookup inside an eval bundle;
- queue enumeration may use `*`, but source lookup inside a bundle must use the
  exact bundle name so `<NAME>.filtered` is never mistaken for a second source.

Refactor the read-only half of `bundle.begin` into a resolver that returns the
selected queued path before moving it. Exact filename wins; otherwise the
existing unique-prefix behavior applies. For words, the selected filename is
also the canonical bundle name. Let `bundle.begin` accept that already-selected
source so eval does not resolve one file for preflight and then resolve it
again for movement. P1 and p2 retain their current suffix and ambiguity
behavior.

The same exact-source abstraction replaces p1/p2-only wildcard assumptions in
`bundle.source`, `bundle.has_source`, `bundle.evaluated`, and archive callers.
Their observable behavior for p1 and p2 remains unchanged.

## `wf submit words [--as NAME] FILE`

Add a words-specific `Submit` action rather than forcing byte-preserving ranked
text through the pair action's `setops.merge`.

1. Parse `--as NAME` with the command-local argparse parser, so the option may
   appear in the normal argparse positions. Require exactly one `FILE`.
2. Inspect the typed source path without resolving it. Reject a symlink before
   reading or copying anything, then require a readable regular source. Choose
   `NAME` when supplied, otherwise the typed source path's basename, and
   validate it with `names.check_name` plus the dictionary archive-stem guard.
   `--as` is a filename, not a path; empty names, `.` and `..`, separators,
   whitespace, and glob metacharacters are invalid.
3. Resolve `dict/queued/<chosen-name>`. Without `-f`, refuse any existing
   directory entry at that destination.
4. Copy the source bytes to a standard-library named temporary file created
   without a directory override, then rename that file directly into place
   through the ordinary prepared-placement mechanism. With `-f`, replace only
   that exact destination. Do not insert a destination-local copy, use
   `setops.merge`, inspect or transform the contents, or modify the source.
   Cross-filesystem rename failures are outside this plan's contract.
5. Report the actual queued name, which is the operand for `eval words`.

Content validation intentionally belongs to eval's pre-move preparation. A
submission can therefore be staged without changing its presentation, while a
bad file still cannot cross into active review or create external notes.

## `wf eval words NAME`

### Pre-move filtering

Add a dictionary helper with a narrow contract, for example:

```python
def filter_reviewed_words(source: Path, reviewed: Path,
                          staged: Path) -> FilterResult:
    """Write unreviewed original rows in input order; move nothing."""
```

It operates before `bundle.begin`:

1. Require `dict/done/reviewed.words`. If it is absent, diagnose the derived
   input and tell the operator to run `wf gen dict`; do not silently review
   against an empty set.
2. Load its bare words into a set.
3. Stream the selected queued file with ordinary text I/O and pass each row to
   the same parser as `remove words`. Ignore a row that is whitespace-only
   before count-prefix removal. For every other row, remove optional
   `COUNT_PREFIX`, strip surrounding whitespace, then require a nonempty full
   `WORD` (`[a-z]+`) match. On failure raise a `ValueError` naming the queued
   path, line number, and offending remainder.
4. If the identity is already reviewed, omit it. Otherwise write its
   presentation row to a standard-library named temporary file created without
   a directory override, in input order and retaining the current count. Do not
   sort or deduplicate. Ordinary text output uses the platform line ending;
   exact input line-ending preservation is not required.
5. Return word counts and whether any identity was filtered. If zero word rows
   survive, refuse while the source is still queued. Blank rows do not make an
   otherwise empty review nonempty.
6. Select the presentation path that note creation will use: the prepared
   filtered file when filtering occurred, otherwise the unchanged queued
   source. Call `notes.part_count()` on that path and refuse more than
   `notes.MAX_PARTS` (26 parts, or 10,400 physical lines at 400 lines per
   part) while the source is still queued. This check counts physical
   presentation rows, including blanks in an unchanged source, exactly as
   `notes.split()` will.

The helper is intentionally local instead of invoking Nutrimatic's
`dict-filter`: its algorithm is small, the workflow already owns
`COUNT_PREFIX` and `WORD`, and a cross-checkout Python executable would be a
new runtime dependency. The semantics are nevertheless the same important
ones: match the normalized word identity and retain a surviving row's count
and ranking position.

### Opening and making notes

After the complete preflight succeeds:

1. Create `dict/eval/NAME/`. If reviewed words were removed, rename the prepared
   temporary text directly to `dict/eval/NAME/NAME.filtered` through
   `setops.place`.
2. Move the unchanged queued source to `dict/eval/NAME/NAME` last. Its presence
   is the bundle readiness marker: without it, `bundle.evaluated()`, `notes`,
   and `complete` refuse the incomplete bundle rather than falling back to the
   unfiltered source. A retry of eval recognizes a bundle containing only the
   prepared `.filtered` file and finishes this move. When no words were
   filtered, move the source normally and create no derivative.
3. Split `bundle.evaluated()` with the existing note title and chunk-size
   contract, and create each note with `--text --checkbox --production`.
4. Report source, filtered, and ready-for-review word counts without calling
   pair rows words or vice versa.

Do not expose p2's `--yes-pairs` or `--no-filter` options for words. Reopening a
past word for review uses the managed dictionary's manual correction contract:
edit every applicable `done/in/*.reviewed.N`, then run `wf gen dict`.

As with p2, a failure from the external note creation after `bundle.begin`
leaves an active bundle. The queued file is not reconstructed. Note creation
has no automatic partial-batch recovery: the presence of a title cannot prove
that the note belongs to the failed attempt. On failure, report the earlier
note titles whose create calls succeeded, identify the failed title as one to
check and delete if it exists, and direct the operator to delete those notes
manually. After cleanup, `wf notes words NAME` recreates the complete batch.

## Shared note machinery and `wf notes words NAME`

Parameterize `notes.create`/`notes.make` by checkbox mode instead of keeping
`["--text", "--two-checkboxes", "--production"]` hardcoded. Prefer an
explicit mode or option tuple over interacting booleans:

- p2 continues to select `--two-checkboxes` and keeps `--yes-pairs`;
- words selects `--checkbox` and has no `--yes-pairs`.

Note title rendering, `.aa` through `.az`, the 400-line chunks, and the
26-part limit are unchanged. Replace the fixed note-splitting scratch path with
a `tempfile.TemporaryDirectory()` created without a directory override and
kept alive across splitting and creation. P2 and words share that lifetime and
cleanup machinery.

Register `notes words`. It accepts an exact active bundle name, resolves
`bundle.evaluated()`, and recreates the complete set of single-checkbox note
parts after the operator has performed any required manual cleanup. It does
not detect or preserve part of an earlier creation attempt, move anything,
refilter the source, or accept queued/completed records. A queued name directs
the operator to `wf eval words NAME`; a missing or completed name reports that
there is no active words review. Do not fall through to p2's archived-source
`-f` behavior.

Generalize p2's atomic note retrieval into a shared step (or a shared helper
with thin p2/words step wrappers). Its contract is:

- fetch `.aa`, `.ab`, ... until the first missing title or until all 26 names
  through `.az` have been retrieved;
- write each response to a temporary file before placing the `.enex` part;
- resume inside `enex.part/` after a partial retrieval on a plain retry;
- rename `enex.part/` to `enex/` only after the terminal missing title proves
  retrieval complete;
- require at least one fetched part;
- on `-f`, rerun retrieval while retaining the ordinary resume behavior for
  any existing `enex.part/`; when replacing a held `enex/`, keep that complete
  snapshot until the resumed replacement finishes.

`complete -f` means rerun or resume a failed completion, not unconditionally
redownload every note part. In the ordinary post-validation recovery there is
a complete `enex/` and no partial directory, so rerunning retrieval naturally
downloads a replacement. A surviving `enex.part/` may still be reused under
`-f`; changing that edge case is outside this plan.

Remove hardcoded p2 command text from the shared layer. A words diagnostic
names `wf complete words NAME`. Apart from the note-creation and extraction
diagnostics explicitly changed under "Additional fixes" below, p2 diagnostics
remain byte-for-byte p2 prose.

## Word-result extraction and validation

Add a words-specific extraction step. It reads the retrieved `enex/*.enex`
parts twice:

```text
note --parse-file PART --type YES  --lines
note --parse-file PART --type NONE --lines
```

Do not pass `--two-checkboxes`. For the note parser's one-checkbox mode,
`YES` is checked and `NONE` is unchecked. Normalize both result streams with
the same count-prefix removal and `[a-z]+` validation as the evaluated input,
then merge each into a sorted-unique bare-word artifact, `<NAME>.yes` for
checked words and `<NAME>.no` for unchecked words. Normalize
`bundle.evaluated()` to `<NAME>.input.words`, a sorted-unique bare-word artifact
for the durable reviewed input. This is the evaluated subset, not the original
unfiltered submission. Keep it in a `tempfile.TemporaryDirectory()` created
without a directory override until validation succeeds; do not write `done/in`
before publication.

Prepare these artifacts in that temporary directory before renaming any
to their final bundle paths. Call exactly one policy function, which also
prepares the union and intersection there. All set operations use the existing
`LC_ALL=C` contract. Its essential checks are:

```python
def validate_results(reviewed_input: Path,
                     yes: Path, no: Path,
                     all_words: Path, overlap: Path) -> None:
    setops.merge([yes, no], all_words)  # sort -u; staged <NAME>.all
    setops.common(yes, no, overlap)    # staged <NAME>.both
    same_input = filecmp.cmp(all_words, reviewed_input, shallow=False)
    has_overlap = fs.line_count(overlap) != 0
    if not same_input or has_overlap:
        raise ValueError(...)  # diagnostic described below
```

Use `setops.common` for intersection; `setops.diff` is subtraction. On union
mismatch, compute missing words as input minus `.all` and extra words as
`.all` minus input for the diagnostic. On overlap, report words marked both
ways. Report input/YES/NO counts plus the missing, extra, and conflicting word
counts, and identify which checks failed. These diagnostic set differences
also stay in the temporary directory. The diagnostic tells the operator to
correct the checkbox review notes in the note application and directs them to
run `wf -f complete words NAME`. The retrieved ENEX plus active source remain
in place. A plain retry reuses that downloaded snapshot and cannot see edits to
the remote notes; `-f` reruns retrieval and extraction to validate the edited
notes. Validation happens before any dictionary record, derived output, or
archive is written. `-f` does not bypass it.

Together these checks prove that YES and NO partition the evaluated word set:
every input word has exactly one verdict, with no missing or extra identity.
Repeated display rows must have consistent verdicts; duplicate identities in
one verdict set collapse normally. A separate count-equality check is redundant.
Only after validation succeeds place `.yes`, `.no`, `.all`, and `.input.words`
at their final bundle paths. Intersection and diagnostic differences are
disposable temporary artifacts. Completion uses `.no` as the removal set.

The checks cannot distinguish an older note with exactly the same word set
and different but internally consistent verdicts. Note lookup ordering is
independent work, as stated in the confirmed decisions. There is no percentage
or plausibility check: an all-unchecked or all-checked review is valid if the
partition checks pass.

## Dictionary publication and completion

Refactor `workflow/dictionary.py` so direct and reviewed removals share the
path after verdict production. The common operation accepts separately:

- an archive stem (`NAME` for the review cycle);
- normalized reviewed input (every word actually shown in this review);
- normalized removals (the unchecked words);
- optionally, the completed `enex/` directory to archive.

`remove_words()` retains its CLI and report. Its input handling is refactored
to the shared whitespace-stripping row parser described above. It passes the
same normalized submission as both the reviewed input and the removal set,
with no ENEX. `complete words` passes the normalized evaluated input and the
normalized unchecked set. Refactor the common managed-dictionary operation to
prepare each related disposable batch in `tempfile.TemporaryDirectory()`
without a directory override rather than under `dict/`:

1. Enumerate current `.removed.N` and `.reviewed.N` records and allocate
   `max + 1`. This occurs now, during completion. The review consumed no
   ordinal while queued or active.
2. Prepare independent sorted-unique archives in the temporary directory for
   final destinations
   `removed/<NAME>.removed.N` and `done/in/<NAME>.reviewed.N`.
3. In the same temporary directory, prepare the prospective removal union,
   reviewed-input union,
   `done/reviewed.words`, `words.filtered`, delta/effective counts, and report.
4. When completing a review, also preflight the absent archive directory
   `done/out/enex/<NAME>.reviewed.N`. Its ordinal is the same one allocated for
   the two word records. A raw ENEX archive is not a third source scanned by
   the allocator. It can outlive both corresponding word records after a
   supported manual correction, so a later reuse of the same ordinal and name
   must diagnose the destination collision. The operator then relocates or
   removes the orphan explicitly; publication never overwrites it.
5. Complete every content computation and destination-shape check before the
   first durable placement.
6. Commit in recoverable-source order by passing each prepared file directly
   to `setops.place`, whose rename publishes it without an intervening copy:
   removal record, reviewed-input record, `done/reviewed.words`, and
   `words.filtered`. Rename the ENEX directory from the active bundle to its
   archive, then advance `.words.filtered.gen`.
7. Only after successful publication, remove the active bundle's original,
   optional `.filtered`, `.yes`/`.no`/`.all`/`.input.words` artifacts, and any
   `enex.part/` retrieval scratch, then close the empty bundle. Remove
   `enex.part/` with the same
   `shutil.rmtree(..., ignore_errors=True)` mechanism p2 uses. This cleanup is
   not a retryable completion step. If any of it fails, the command still
   succeeds but uses `log.warn` to say that publication completed, identify
   the bundle path, tell the operator not to rerun completion, and print the
   exact shell-quoted recovery command:

   ```text
   rm -rf -- <bundle-path>
   ```

   Emit this diagnostic only after every durable record, derived output, ENEX
   archive, and generation marker has been published successfully. At that
   point the active bundle contains no authoritative state.

The completed reviewed input is the filtered set actually shown to the
reviewer, not the unchanged queued source: rows removed because they were
already present in `reviewed.words` were not part of this round. The per-round
archive is sorted-unique bare words, as required by `managed-dict-v2.md`; its
counted display form remains transient.

No `.yes.N`, `.no.N`, `.checked.N`, or `.unchecked.N` file is archived. The
checked set is recoverable as this round's reviewed input minus this round's
removal set at completion time, while the standing kept set remains globally
derivable from the managed records. Archiving another verdict file would
reintroduce two authoritative copies of a removal.

The success report starts with the review result counts, then reuses the
dictionary report's headline, effective-change, and before/after lines
unchanged. In the review case, "words submitted" means the unchecked words
submitted to the removal publication operation. For example:

```text
400 words reviewed: 125 checked, 275 unchecked
275 words submitted, 260 new to the removal union, 251 newly removed from the dictionary
recorded as generation 8 -> dict/removed/top.s7.m4.g5.words.removed.8
                         -> dict/done/in/top.s7.m4.g5.words.reviewed.8
archived notes          -> dict/done/out/enex/top.s7.m4.g5.words.reviewed.8/
words.filtered 363382 -> 363131
dict/done/reviewed.words 1200 -> 1600
```

Do not promise full transactional completion. The dictionary batch prepares
and preflights all content and expected destination failures but accepts
failure or interruption during its direct final renames. Extending that
sequence by the ENEX rename does not change the concurrency or crash guarantee.
A concurrent `remove words` and `complete words`, transaction markers, locks,
and automatic repair of an orphaned partial round remain out of scope.

## Command registration and help

Extend the root dispatchers without renaming existing targets:

- `submit`: `p1|p2|words`
- `eval`: `p1|p2|p3|words`
- `notes`: `p2|words`
- `complete`: `p1|p2|words`

Use these detailed synopsis shapes:

```text
wf submit words [--as NAME] FILE
wf eval words NAME
wf notes words NAME
wf complete words NAME
```

The global `-d/--dir`, `-f/--force`, and `-h/--help` options continue to be
rendered by the common help machinery. `-f` retains only its existing
operation-specific meanings (queue replacement, refetch/overwrite recovery);
it is not a verdict-ratio override and it never bypasses `validate_results()`.
Every action rejects extra positional arguments through the existing
`usage.invalid_argument` convention.

## Recovery and ordering guarantees

| failure point | durable state and retry |
|---|---|
| bad submitted name or copy failure | no queued destination is published |
| invalid word row during eval preflight | unchanged source remains queued; no eval bundle or notes |
| all words already reviewed | unchanged source remains queued; no empty review is opened |
| note creation fails after promotion | active bundle remains; run `wf notes words NAME` |
| note retrieval stops partway | `enex.part/` retains complete fetched parts; rerun `complete words NAME` |
| forced retrieval after a partial fetch | rerun retrieval using the existing resumable `enex.part/`; retain any complete `enex/` until the resumed replacement is ready |
| result union differs from input or YES/NO overlap | ENEX and active source remain; correct checkbox notes in the note application, then run `wf -f complete words NAME` to download and validate the edits |
| any preparation/preflight failure | no dictionary record, derived output, or ENEX archive is published |
| interruption during final renames | same manually recoverable partial-round window accepted by managed-dict-v2 |
| bundle cleanup fails after successful publication | the round is complete; return success, warn the operator not to rerun completion, and print the exact shell-quoted `rm -rf -- <bundle-path>` command |
| success | dictionary records and ENEX are archived, marker advances, bundle closes |

Completion allocates the ordinal only after retrieval/extraction/validation,
so none of the normal retry rows before publication consumes or reserves a
generation. A manual `remove words` between those attempts is harmless: the
next completion attempt scans the current records and chooses the then-current
next ordinal.

Post-publication bundle cleanup deliberately has a different recovery contract.
It does not rerun the recipe and does not need a publication `is_done`
predicate: the warning directs the operator to remove the now-disposable bundle
manually. Do not add a transaction marker or automatic cleanup recovery for
this case.

## Additional fixes

### P2 recovery, retrieval, note-creation, and cleanup diagnostics and tests

P2 has the same post-publication cleanup boundary even though it cannot
allocate a duplicate dictionary generation. Its `archive` step moves the
source, verdicts, and ENEX into durable locations before deleting scratch files
and `close` removes the bundle. Once those durable moves have all succeeded, a
cleanup failure leaves a disposable but non-retryable active bundle: retrying
starts at retrieval, whose source has already moved.

Keep required P2 archive moves separate from subsequent scratch and bundle
cleanup so the code can distinguish those states. If cleanup fails after all
archive moves succeed, apply the same words policy: return success, use
`log.warn` to identify the bundle, tell the operator not to rerun completion,
and print the exact shell-quoted `rm -rf -- <bundle-path>` command. Never offer
that removal after a partial archive. Add no tests for this rare manual-cleanup
path.

P2 and words use the same deliberately manual response to partial note
creation. If creation of any part fails, do not infer ownership from existing
titles or attempt to resume with only missing parts. The shared creation layer
must fail with a diagnostic that lists every earlier title whose create call
returned successfully, tells the operator to delete each of those notes, and
tells the operator to check and delete the failed title if it exists. After
manual cleanup, the operator recreates the complete batch with `wf notes p2
NAME` or `wf notes words NAME`, respectively. This rare failure path does not
add review identifiers, per-note success markers, automatic cleanup, or
missing-part detection.

- Add shared P2/words coverage that fails creation after the first successful
  part and checks the scope-specific cleanup and recreation diagnostic. Verify
  that the active bundle remains available for full recreation; do not model
  preservation of a partially created remote batch.

P2 already supports forced note retrieval and rejects an extracted pair that
appears in both its YES and NO sets. Its current extraction diagnostic tells
the operator to correct the note and rerun completion, but omits `-f`. A plain
retry skips retrieval when `enex/` exists and parses the rejected snapshot
again.

- Update `workflow/steps/p2_extract.py:_diagnostic()` to direct the operator
  to correct the checkbox notes in the note application, then run
  `wf -f complete p2 NAME` to download and validate those edits. Keep the
  existing overlap validation and forced-retrieval behavior.
- Fix `tests/test_workflow_p2.py:FakeNotes` so a download captures the remote
  verdicts at fetch time in its returned fixture content. The fake parser
  must derive its verdicts from the downloaded file, not from the mutable
  remote `PARTS` mapping. The fixture may use a test-only serialized payload;
  it need not implement an ENEX parser.
- Replace the misleading plain-retry recovery test with a snapshot-aware
  scenario: completion downloads contradictory verdicts and fails; the remote
  fixture is corrected; plain completion still fails without another fetch;
  forced completion fetches the corrected snapshot, re-extracts, validates,
  and completes. Assert that no classification or archive is published on
  either rejected attempt, and that `-f` still rejects an unfixed conflict.
- Use the same snapshot distinction in the words recovery test: editing the
  remote fixture alone must not change what parsing the held ENEX returns.

Do not otherwise change p2 retrieval behavior. In particular, `-f` remains a
retry/resume operation and may reuse a surviving `enex.part/`; it is not a
guarantee that every part is downloaded afresh. P2 already terminates
successfully after retrieving all 26 valid titles through `.az`; sharing that
behavior requires no special termination change or focused regression test.

The diagnostic and snapshot-aware fixture expectations above are explicit
exceptions to retaining existing p2 wording and tests. They do not change
p2's retrieval, verdict, or archive policy.

## Files to change

| file | change |
|---|---|
| `workflow/config.py` | add `dict/queued`, `dict/eval`, and `dict/done/out/enex`; keep `dict` as the only configured layout name |
| `workflow/names.py` | model the suffixless words queue and exact in-bundle source name without changing p1/p2 suffix contracts |
| `workflow/bundle.py` | expose read-only queued selection; let `begin` consume the selected source; use scope-aware exact source lookup |
| `workflow/submit.py` | add `SubmitWords`, `--as`, and atomic byte-preserving queue copy |
| `workflow/eval.py` | add `EvalWords`; run dictionary validation/filtering before `begin`; create single-checkbox notes |
| `workflow/notes.py` | parameterize checkbox mode; add active-only `NotesWords`; replace fixed note-splitting scratch with a standard-library temporary directory while keeping p2 checkbox/title behavior intact |
| `workflow/complete.py` | add a words-specific completion action/recipe; report post-publication P2/words cleanup failures as successful completion with the manual removal warning; keep p1 unchanged |
| `workflow/dictionary.py` | shared whitespace-stripping row parser and streaming identity filter; common direct/reviewed record publication; standard-library temporary preparation with direct `setops.place` renames; remove dictionary-local staging; optional ENEX archive placement |
| `workflow/steps/p2_retrieve.py` and/or a new shared retrieval module | share atomic note retrieval and parameterize command prose while retaining p2's current force/resume and 26-title exhaustion behavior |
| `workflow/steps/p2_extract.py` | correct the post-validation recovery diagnostic to require `-f` after editing remote notes |
| `workflow/steps/p2_archive.py` and `workflow/steps/p2_close.py` | keep required archive moves ahead of scratch/bundle cleanup so only a post-archive failure receives the manual removal warning |
| new word-review step modules under `workflow/steps/` | extract checked/unchecked words, isolate `validate_results`, publish the review, and finalize it with the manual removal warning after publication |
| `workflow/wf.py` | register `words` under submit/eval/notes/complete and update summaries |
| `tests/test_workflow_dictionary.py` | extend common publication coverage to distinct reviewed/removal inputs and ENEX archival |
| `tests/test_workflow_intake.py` | words queue naming, exact copy, `--as`, selection, filtering, and pre-move failures |
| `tests/test_workflow_p2.py` | shared-note/retrieval regression coverage; snapshot-aware fake parsing and corrected forced-retry recovery coverage; keep words lifecycle cases in a new focused file |
| `tests/test_workflow_cli.py` | dispatcher targets, detailed help, missing/extra argument behavior, and `--as` syntax |

Prefer a new `tests/test_workflow_dict_review.py` for the lifecycle recipe and
end-to-end cases, leaving `test_workflow_p2.py` focused on unchanged p2 policy.

## Tests

### Layout, dispatch, and submission

- `wf init` creates `dict/queued`, `dict/eval`, and
  `dict/done/out/enex` on both new and already initialized trees.
- `wf show all queued` and `wf show all eval` include dictionary work under
  `dict/...`; command help says `words`, never a nonexistent `words/...` path.
- All four root dispatchers advertise and resolve `words`; existing p1/p2/p3
  help and invalid-argument behavior is unchanged.
- Submitting without `--as` uses the source basename; `--as
  top.s7.m4.g5.words` uses exactly that name through queue and eval.
- A counted, unsorted file with duplicates, blank lines, and no final newline
  is byte-identical in `dict/queued`; the source is unchanged. The temporary
  prepared copy is renamed directly from standard-library temporary storage.
- An invalid `--as` name and an existing destination without `-f` fail before
  publication. `-f` atomically replaces only the named queued destination.
- A symlink source is refused before any queue or temporary-file mutation;
  `submit words` does not resolve it and never derives provenance from the
  target's basename.

### Eval filtering and notes

- A reviewed bare word is removed even when its newly submitted count differs;
  unreviewed lines retain their counts and order in `<NAME>.filtered`. Ordinary
  text I/O may normalize line endings to the platform default.
- With no reviewed matches, no `.filtered` derivative is created and
  `bundle.evaluated()` returns the unchanged source.
- When every word is reviewed, eval refuses before `bundle.begin`: the queue is
  unchanged, eval is empty, and note creation is not called.
- A presentation of exactly 10,400 physical lines passes the 26-part preflight;
  10,401 lines fails before `bundle.begin`, leaves the source queued, and makes
  no external note call. Cover final lines with and without a newline and both
  filtered and unfiltered presentation branches, including blank source rows
  whose physical presence affects only the unfiltered branch's part count.
- A malformed row (`Foo`, `foo-bar`, a pair, trailing junk, or a nonblank
  count-only row) fails with the
  queued path, line number, and offending identity before any move. The
  optional count prefix and surrounding whitespace are accepted; the count is
  retained in output. Whitespace-only lines are ignored. Exercise the same
  parser cases through direct `remove words`.
- Exact name selection wins and unique-prefix selection resolves to the full
  suffixless queued name; ambiguous prefixes change nothing.
- Words note creation passes `--checkbox` and not `--two-checkboxes` or
  `--yes-pairs`. P2 still passes `--two-checkboxes` and preserves its optional
  `--yes-pairs` behavior.
- Created and retrieved note titles use `<NAME>.aa`, `<NAME>.ab`, ... when
  nothing was filtered and `<NAME>.filtered.aa`,
  `<NAME>.filtered.ab`, ... when filtering produced the evaluated derivative.
- `notes words` recreates from the active evaluated source and does not move or
  refilter it. Queued, completed, ambiguous, and missing names receive the
  scoped diagnostic and make no external call.

### Extraction, validation, and completion

- Retrieval probes exactly the titles eval created. Plain and forced retries
  may resume a partial ENEX retrieval; when no partial cache exists, a forced
  retry over a complete `enex/` downloads and stages its replacement. Either a
  missing next title or all 26 valid titles completes the fetch. Add no focused
  force-cache or 26-part regression requirement.
- Words extraction invokes `--type YES` and `--type NONE` without
  `--two-checkboxes`, strips counts for durable artifacts, and places neither
  result until both parses and validation succeed.
- `validate_results()` accepts an exact partition, including all-checked and
  all-unchecked reviews. It rejects missing words, extra words, equal-count
  substitutions, and YES/NO overlap even when the union equals input. Repeated
  rows with consistent verdicts deduplicate and pass; conflicting verdicts for
  a repeated word fail. Verify the `.yes`, `.no`, `.all`, and `.input.words`
  contents and the content-based comparison with `shallow=False`.
- Validation failures report input/YES/NO and missing/extra/conflicting counts,
  place no extracted bundle results or dictionary state, and cannot be
  bypassed by `-f`. Parse and validation set-operation failures likewise place
  no results.
- After partition validation fails, correcting remote note fixtures leaves a plain
  retry reading the rejected ENEX snapshot. `wf -f complete words NAME`
  downloads the corrected notes, reruns extraction and validation, and
  completes. Assert that the diagnostic gives this exact recovery command.
- A focused test records the remaining limitation: an older note containing
  exactly the input word set with different but consistent verdicts passes
  partition validation. Note-round selection is independent of these checks.
- Completion writes unchecked words only to
  `dict/removed/<NAME>.removed.N`, writes the complete normalized evaluated
  set to `dict/done/in/<NAME>.reviewed.N`, rebuilds both derived files, advances
  the marker, archives ENEX under the approved path, and closes the bundle.
- Words filtered out before review occur in neither per-round completion
  record. Checked words remain in `words.filtered`; unchecked base words leave
  it; unchecked non-base words are still recorded but count as ineffective,
  exactly as under `remove words`.
- Direct `remove words` still passes one set as both reviewed and removed and
  retains its output, no-op, retraction, mtime, and failure behavior. Replace
  the implementation-specific dictionary-local staging test with the system
  temporary/direct-placement expectation below.
- If `remove words` creates generation 7 while a review is queued or active,
  `complete words` allocates generation 8. No ordinal appears in queued/eval
  names and failed retrieval or partition validation consumes none.
- Two completed reviews using the same `--as NAME` receive distinct ordinals
  and distinct input, removal, and ENEX archives.
- A destination-shape or ENEX-archive collision discovered in preflight leaves
  every dictionary record/output unchanged and the bundle recoverable.
- `remove words`, `gen dict`, and `complete words` prepare disposable files
  using standard-library temporary facilities without a directory override,
  leave no `.wf/dict/.staging.*` directory, and pass prepared files directly
  to `setops.place` for their final renames. Tests do not require
  cross-filesystem handling.
- Success output reports reviewed/checked/unchecked counts and the existing
  dictionary delta and before/after semantics without claiming a ratio check.

### Regression and integration

- All current p1 and p2 intake, filtering, notes, retrieval, completion,
  archive, force, and recovery behavior is preserved except for the explicitly
  approved p2 partition diagnostic, snapshot-aware recovery-test changes, and
  untested post-archive cleanup warning under "Additional fixes".
- One CLI-level word lifecycle test uses the real dispatcher from exact-copy
  submission through eval, injected note results, completion, dictionary
  derivation, ENEX archive, and an empty eval slot.
- Run:

  ```bash
  python -m unittest discover -s tests -p 'test_workflow*.py'
  ```

## Operator rollout

1. Update the checkout containing the implemented managed dictionary.
2. Run `wf init` once per workflow root to add the new queue/eval/done-output
   directories.
3. Confirm `words.big`, `words.filtered`, and `done/reviewed.words` are present;
   run `wf gen dict` if either derived file needs creation or repair.
4. Submit the first candidate list, optionally assigning its durable review
   identity:

   ```bash
   wf submit words top.solo-words --as top.s7.m4.g5.words
   wf eval words top.s7.m4.g5.words
   # check every word to keep; leave words to remove unchecked
   wf complete words top.s7.m4.g5.words
   ```

5. Use `wf notes words top.s7.m4.g5.words` only while that eval bundle is
   active if its notes must be recreated.

## Not in scope

- A removal-ratio cutoff, an abandoned-note heuristic, or a force override for
  verdict proportions.
- Preventing or recovering from the internal retrieval-path collision when a
  words review's queued filename and bundle name is exactly `enex` or
  `enex.part`. Those two names remain syntactically valid but unsupported;
  operators must not use them for words reviews. This plan adds no reserved-name
  validation or internal-directory rename.
- Preflighting the replaceable `.words.filtered.gen` marker path with the
  prepared publication batch. `remove words`, `gen dict`, and `complete words`
  continue to write the marker only after publishing their content, so a
  directory or other invalid entry at that path can still fail after durable
  outputs have changed. The operator must keep the marker absent or a regular
  file.
- Changing how `complete -f` handles a surviving `enex.part/`, or treating
  force as an unconditional fresh download. Forced completion reruns or resumes
  the failed recipe and may reuse partial note downloads. The existing
  successful 26-title exhaustion path also receives no special code or test
  work.
- Note lookup ordering changes, which the operator is pursuing independently
  to ensure descending modified time; automatic review-round suffixes are
  not added by this plan.
- Recreating notes from a completed review, or archiving another raw counted
  input solely to make that possible.
- `top.solo-words` generation, BEST status integration for word-review work,
  or automatically submitting a candidate frontier.
- Detecting, preventing, or recovering from cross-filesystem renames out of the
  standard-library-selected temporary location.
- Storing checked/kept or unchecked/removed verdicts a second time under
  `done/out`.
- Changing p2's two-checkbox policy, result validation, archived-note recovery,
  queue suffixes, or pair-set ordering.
- Locks, concurrent dictionary writers, transaction markers, automatic repair
  of a partial final rename sequence, or stronger guarantees than
  `managed-dict-v2.md` already makes.
- Migration of queued or active word reviews; none exist before this feature.
