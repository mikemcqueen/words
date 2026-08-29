# Recreate a review's notes

## Context

`wf best review` submits a target's top segments, opens a P2 bundle, splits the
pairs, and raises one note per part. The notes are the whole point of the round
— they are what the operator checks by hand — but nothing about them is
recorded. Delete them and there is no way back to them short of pushing the
bundle backwards through `p2/queued` and re-running `eval`, which is a state
round-trip taken to reach something that was never stateful.

Note creation is a pure function of two inputs: the pairs file the bundle was
evaluated from, and the optional confirmed-YES set the notes check themselves
against. The split is deterministic (`CHUNK_SIZE = 400`, fixed prefix), and
`eval` writes no manifest. So re-running it is a re-derivation, not a recovery.

The gap is that `Eval.run()` (workflow/eval.py:79) welds three things of
different natures together, and there is no way in to the third without the
first two:

1. `bundle.begin` — a one-way move from `queued/` into `eval/`
2. `bundle.filter_done` — writes a `.filtered` derivative
3. `prepare` — the pure derivation

Outcome: a `notes` primitive that starts from an already-open (or already
archived) bundle and re-runs step 3 alone, plus a `best notes` composite that
resolves the target's current review round and supplies `--yes-pairs`. Step 3
itself moves out of eval.py into a module of its own, since it turns out to be
spelled twice already — see workflow/notes.py below.

## Command contract

```text
wf notes p2 BUNDLE-NAME|SOURCE-FILE [--yes-pairs PATH] [-f]
wf best notes SENTENCE -o|-u LETTERS -g COUNT [-m LENGTH] [-f]
```

The primitive searches three slots. Two can supply a file; one is diagnostic:

| slot | meaning | behavior |
|---|---|---|
| `p2/eval/<bundle>/` | in flight | use it, no `-f` |
| `p2/queued/` | submitted, never opened | error: no notes ever existed; run `wf eval p2 <name>` |
| `p2/done/in/` | completed | `-f` gates use |
| none | — | `bundle not found` |

`-f` here has exactly one meaning — *use the archived one*. None of its other
senses are in play, because `notes` calls neither `bundle.begin` nor
`bundle.filter_done`.

`notes` must not quietly run the eval when it finds a queued file. That would
move state, and the justification for the primitive is that it touches nothing.

This mirrors `plans/force-best-complete.md`'s contract, where a queued review is
an error even with `-f` and `-f` selects the highest completed round.

## Filtering

`notes` makes no filtering decision. It reads whatever `eval` left, via the
existing `bundle.evaluated()` (workflow/bundle.py:106) — the `.filtered`
derivative if present, the source otherwise. Consequences:

- **best review bundle** — `best review` hardcodes `--no-filter`
  (workflow/best/commands.py:234), so no `.filtered` ever existed and in-flight
  and archived give the identical file. Always faithful.
- **ordinary filtered p2 bundle, in flight** — `.filtered` still present.
  Faithful.
- **ordinary filtered p2 bundle, archived** — `p2_archive.run_step` unlinks
  `*.filtered` (workflow/steps/p2_archive.py:55), so only the original
  survives and the notes cover a superset.

Unreachable through `best notes`; the `-f` message names it anyway, since it is
reachable at the primitive.

Do **not** declare `--no-filter` on the notes parser. There is no filtering step
for it to skip, and an inert flag would imply a mode the command does not have.

## Round selection

Per `plans/force-best-complete.md`: select the **highest numeric `rN`**, not the
lexicographically last name or the newest mtime, via one narrowly scoped parser
for names `Review.run()` renders. `Review` then makes a new review `max(rN) + 1`
instead of `len(archived) + 1`, which closes a collision — with a gap in the
archive, counting renders a name that already exists.

One helper, three callers: `Review` (next round), `best notes` (highest round),
and `best complete -f` when that plan lands.

## Changes

### workflow/best/state.py

- Rename `_eval_p2_command` → `eval_p2_command`. It is the shared rendering of
  a displayed `wf eval p2 …` command, already correct in appending
  `yes_pairs_argv`; it just needs to be reachable from commands.py.
- Add `review_rounds(target, archived) -> dict[int, Path]`. Takes the list
  `review_locations` already returned, so no second scan. Matches exactly
  `<review_prefix><cutoff>.r<positive-integer>.pairs`; rejects a non-matching
  sibling rather than guessing. Docstring must say why this narrow exception to
  names.py's "never take a name apart" is admissible: the pattern is what
  `Review` renders, and only the ordinal is recovered.
- Needs `import names` (currently `config, fs`).

Keep it a module function, not a `Target` method — the file's line is that
methods (`command()`, `review_prefix`, `artifact()`) render from fields alone,
while functions taking a Target (`review_locations`, `yes_pairs_argv`) stat the
tree.

### workflow/names.py

- Add `queue_names(phase, bundle_name) -> tuple[str, ...]`: the exact filenames
  one bundle's source can have. `queue_globs` finds *a* phase's queued artifact;
  this names *one bundle's*, which is what a slot holding many side by side has
  to be asked for. Rendered off the same `QUEUE_SUFFIXES` table.

### workflow/bundle.py

- Factor `at_most_one(directory, glob) -> Path | None` out of `one()`; `one()`
  becomes `at_most_one` plus the not-found raise. Keeps the existing
  "multiple … in …" error as the single spelling of that ambiguity.
- Add `source_in(ctx, slot) -> Path | None` — the bundle's source in one of the
  phase's slots, matched by exact name (a prefix would let `…r1.pairs` answer
  for `…r10.pairs`).
- Add `recover(ctx) -> Path` — the three-slot search and its messages.
- Move `_resolve_bundle` here from complete.py as `resolve_name(root, phase,
  positional)`; complete.py calls it. Its rule ("a positional naming a
  directory is used as typed; only a miss falls back to taking the queue suffix
  off") is exactly what `notes` needs, and moving it means one copy, not three.

### workflow/complete.py

- Drop `_resolve_bundle`, call `bundle.resolve_name`. Add `bundle` to imports.

### workflow/notes.py (new)

The note-part naming is already spelled twice, once at each end of the contract
it defines:

```python
def get_split_paths(prefix, n_files, suffix=''):        # eval.py:104
    assert n_files < 27, "got some work to do"
    return [Path(f"{prefix}.a{chr(ord('a') + i)}{suffix}") for i in range(n_files)]

def _title(source: Path, index: int) -> str:            # p2_retrieve.py:47
    return f"{source.name}.a{chr(ord('a') + index)}"
```

`MAX_NOTE_PARTS = 26` in p2_retrieve is the same constant as the `< 27` assert.
That naming is how a completed bundle finds its own notes months later, so
adding a third caller to the creation side is the moment to make it one
spelling. The new module owns it:

- `title(source, index)` and `MAX_PARTS` — the contract, one renderer
- `part_count(path)`, `part_paths(directory, source, count)`, `split(source)` —
  moved from eval.py's `get_split_file_count` / `get_split_paths` /
  `_split_pairs`, with the bare `assert` becoming a real error
- `create(paths, yes_pairs)` — moved from eval.py's `_make_notes`
- `make(pairs, opts)` — split plus create; the whole of what `eval p2` does to
  a bundle beyond opening it
- `add_yes_pairs(parser)` — mirrors the `_add_letter_set` precedent at
  workflow/best/commands.py:18
- `check_yes_pairs(opts)` — the flag's pre-flight, moved with it out of
  `EvalYes.check`; see below. One caller today, two after `Notes` lands, and it
  belongs beside the parser that admits the flag rather than in either command.
- `class Notes(command.Action)` + `P2`

This is also what answers the layering objection: it is not *evaluate* reading
`done/`, it is *notes*, and the module header can say what `extract.py`'s says
— read archived state without disturbing the queue.

Dependencies stay one-way: `eval` → `notes`, `steps/p2_retrieve` → `notes`.

### workflow/eval.py

- Drop `CHUNK_SIZE`, `get_split_paths`, `get_split_file_count`, `_split_pairs`,
  `_make_notes` — all move to notes.py, and none has a caller outside eval.py.
- `EvalYes.parser` calls `notes.add_yes_pairs`; `EvalYes.prepare` becomes one
  call to `notes.make(pairs, opts)`; `EvalYes.check` becomes one call to
  `notes.check_yes_pairs(opts)`.
- Keep the `Eval.check(opts)` hook itself (workflow/eval.py:66). It exists
  because `--yes-pairs` is not read until `note --create`, the last subprocess
  the command runs, while `bundle.begin` has emptied the queue long before —
  and a bad path then fails with no supported recovery, since the retry can
  neither find the queued artifact nor reopen the bundle. The hook runs between
  argument parsing and `bundle.begin`, which is the last moment failing is
  still free.

  `Notes` does not need the hook — it calls neither `begin` nor `filter_done`,
  so there is no state to fail after — but it should still call
  `notes.check_yes_pairs` before splitting: a bad path that reaches
  `note --create` has already made however many parts precede it.
- Update the module header, which currently says p2 "splits its pairs and
  raises notes against them" as though that lives here.

eval.py shrinks by roughly a third and keeps only what opening a bundle means.

### workflow/steps/p2_retrieve.py

- Drop `_title` and `MAX_NOTE_PARTS`; call `notes.title` and `notes.MAX_PARTS`.
  Retrieval then probes for exactly the names creation rendered.

### workflow/wf.py

- Register `"notes": command.Dispatcher("notes    — recreate evaluation notes (p2)", {"p2": notes.P2})`, after `"eval"`.

### workflow/best/commands.py

- Import `eval_p2_command`; `Complete` (commands.py:254) uses it instead of its
  inline `" ".join([...])`, which is a byte-identical second spelling.
- `Review` (commands.py:218) uses `review_rounds` for `max + 1`.
- Add `class Notes(command.Action)`, registered between `review` and `complete`:
  - `_action_target(..., 1)`, `_target_parser()`; accepts `-f` (unlike `Review`
    and `Exclude`, which reject it — here it is the archive gate).
  - queued → raise with `eval_p2_command(target, queued[0].name)`, so the
    message carries `--yes-pairs` exactly as `best status` and `best complete`
    already do.
  - in flight → bundle name is the directory name.
  - else → highest round from `review_rounds`, name via
    `names.queue_stem("p2", path.name)`; empty archive → "no review to recreate
    notes for `<address>`".
  - delegate: `notes.P2.run("notes p2", opts, [bundle_name,
    *yes_pairs_argv(target)])`, letting the primitive own the `-f` gate.
  - then `report(target)` — the convention after a composite, and the printed
    `next:` is genuinely the next step.

`Review` at commands.py:235 keeps calling `yes_pairs_argv` directly: it builds
argv for an invocation, not a displayed string.

## A caveat worth stating in the `-f` message

`yes_pairs_argv` reads `best.pairs` **as it is now**.

- In flight — `build_best_pairs` has not run (it fires in `Complete`,
  commands.py:269), so it is what `best review` passed. Faithful.
- Archived — completing that round rebuilt `best.pairs` from that round's own
  YES verdicts, so recreated notes check themselves against a set containing
  the confirmations the original notes were made to collect.

Not a blocker, but the `-f` message should name it alongside the unfiltered
original.

## Tests

`tests/test_workflow_fixture.py:528` already establishes the mock shape, and
moves with the code: `mock.patch.object(notes, "split", return_value=[])` and
`mock.patch.object(notes, "create")`, since `note --create` and `split.sh` are
external side effects.

Primitive (`tests/test_workflow_p2.py`, which has the bundle lifecycle context):

- in-flight bundle → notes made from `.filtered` when present, source otherwise
- queued → raises, message names `wf eval p2 <name>`
- archived without `-f` → raises; with `-f` → notes made from `done/in`
- absent → `bundle not found`
- `--yes-pairs` reaches `notes.create`
- a missing, non-file, or unreadable `--yes-pairs` raises before any part is
  split or created — the `notes` half of
  `test_eval_p2_checks_yes_pairs_before_it_opens_the_bundle`
  (tests/test_workflow_fixture.py:549), which stays where it is and keeps
  asserting the queue and `p2/eval` are untouched
- `notes.title` and `part_paths` agree, so retrieval probes the names creation
  rendered


Composite (`tests/test_workflow_best.py`):

- `best notes` on an in-flight review re-notes it and appends `--yes-pairs`
  when `best.pairs` exists, omits it when it does not
- queued target → error containing the full `wf eval p2 … --yes-pairs …`
- highest `rN` is selected across a gapped archive (r1, r3 → r3)
- `Review` renders `r4` against that same gapped archive

Regression: `Complete`'s queued error still contains `--yes-pairs` after moving
to `eval_p2_command`.

## Verification

```
source ../.torch/bin/activate && python -m pytest tests/ -q
```

Then end-to-end against the real in-flight bundle, which is a first round
(`best.pairs` absent → `yes_pairs_argv` empty), reproducing exactly what
`best review` ran:

```
cd final && wf best status s7
wf best notes s7 -u vindiesel -g 4
```

Expected: three notes matching the `/tmp/top.s7.m4.g4.u-vindiesel.1000.r1.pairs.a{a,b,c}`
parts, no `--yes-pairs`, and `next: wf best complete s7 -u vindiesel -g 4`.

Confirm the archived path against s6, whose r1 is in `done/in`: without `-f` it
must refuse and name the round; with `-f` it must re-note it.
