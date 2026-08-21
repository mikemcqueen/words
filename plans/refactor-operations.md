# Workflow Operation Refactor

Refactor `~/code/words/workflow` so that every command is a sequence of independently
invokable operations, and one-off commands can be assembled from those operations
without custom logic.

Acceptance test: `extract-p1-yes` — built originally as bespoke logic — becomes a
recipe over shared primitives, holding no resolver or extraction logic of its own, and
produces byte-identical output to the current implementation.

---

## 1. Scope

### In scope

- `src/filter.py` — unify the two filter entry points into one signature.
- `workflow/` — op protocol, `STEPS` recipes, `Context`, name rendering, `eval/<slug>/`
  batch directories.
- `tests/` — a fixture harness (Item 0), updates for the new CLI surface, and the
  `extract-p1-yes` regression test.

### Not in scope

These are decided-out, not deferred:

- **`use_max` semantics.** Left exactly as-is at every call site. It is provably inert
  at the bands in use (`_pmax` bumps `pmin + prange == 1.0` to `1.1`, which makes
  `.any()` and max-over-directions identical), so it changes no output in this refactor.
- **Model / prompt-variant dimension.** Not added to the name record.
- **Global classified sets.** `_process_yes_pairs`'s merge into `classified/yes/yes.pairs`
  is carried over verbatim as one step. `classified/no` and `classified/all` stay unused.
- **p3.** `eval_no` remains a stub; `p3/queued` remains write-only.
- **`plan()` / `--dry-run`.** Not built. `is_done` covers resumability; nothing in scope
  needs to report intent without acting.
- **Nested recipes.** `STEPS` lists are flat. With per-step `is_done` and standalone CLI
  invocation, nesting adds a dispatch level without adding capability.
- **The `_pmax` 1.0 special case.** Unchanged.

### Deferred — TODO, not decided-out

- **`eval`'s note creation is not resumable.** `eval_yes._make_notes` shells out to
  `note --create` per split file, and the splits live in `/tmp`. Nothing local records
  which notes were created, so a crash partway through leaves no way to tell — and a
  re-run creates duplicate notes in the store. The current create/parse/oracle approach
  is kept as-is for this refactor.

  If it is picked up later, the fix needs no new concepts: the note title *is* the split
  file's name, so move the splits into the batch directory and rename each one
  `split/` → `noted/` as its note is created (`is_done` = "`split/` is empty"). Create
  first, rename second — claim-first fails toward a silently missing note, which
  `_retrieve_notes`'s break-on-404 would absorb as a truncated batch that looks like
  success. To close the crash window between create and rename, have `_make_notes` probe
  for the note before creating it, reusing the `"note not found"` check
  `_retrieve_notes` already relies on.

---

## 2. Decisions

### 2.1 Names are rendered, never parsed

A filename is a **content key**: a flat rendering of the dimensions that distinguish
what is in the file. It is not a history of operations.

```
<slug>.<classifier>.<kind>

slug        = <batch>.<slice>          batch-invariant; equals the eval directory name
batch       = stem of the originating p1 result file
slice       = <round(pmin*100)>.<round(prange*100)>        e.g. 90.10
classifier  = p1 | p2 | p3             whose verdict this file represents
kind        = pairs | yes | no | jsonl
```

Example: `foo.90.10.p2.yes` — the 90–100 slice, as judged by p2's manual review.

Rules:

1. **Construction only.** Names are built by rendering; no command builds a name by
   editing another name. `_phase2_name` is deleted. A phase transition does not rename
   an artifact — p2 *derives a new artifact* from p1's, and p1's keeps its name forever.
2. **Never decomposed.** No command recovers a dimension by splitting a filename.
   Where a slug is needed from an existing file, the slug is the *input* and the file is
   found by prefix — `select(slot, "stem:<slug>")` — never the reverse.
3. **Every rendered value is delimiter-free.** This is why the slice renders as `90.10`
   rather than `0.9.0.1`. Any new dimension must render without a `.`.
4. **Invariant dimensions come first**, so the slug is a true prefix of every filename
   in a batch. `filename.startswith(dirname)` holds and is asserted.
5. **Inside a batch, match by kind — never by the directory's name.** Rule 4 makes the
   directory name a *prefix* of what it holds, not an equal: `wf eval p1 a` creates `a/`
   holding `a.pairs`, because `begin` resolves the queued file by `stem:` prefix. A glob
   built from the slug therefore misses the very file it is looking for. The directory is
   already the namespace — everything in it belongs to this batch by construction — so
   scope by suffix (`*.pairs`, `*.p1.yes`, `*.jsonl`) and let `batch.one` enforce
   uniqueness. Rule 4 still holds; it is just not a thing to *glob on*, and it is not
   enforced for the evalpair result, which no `wf` command places.

This reverses today's segment order (`foo.p1.90.10.yes` → `foo.90.10.p1.yes`), because
the current order interleaves the classifier between batch and slice, leaving no
contiguous invariant prefix to hoist into a directory name.

No manifest file. The directory name carries the record; downstream steps append to it.

### 2.2 Directory layout

**Group where you work, flat where you query.**

```
.wf/p2/
    queued/                       flat — one file per pending item
        foo.90.10.p1.yes
    eval/                         one directory per in-flight batch
        foo.90.10/
            foo.90.10.p1.yes      input, moved here by `eval`
            foo.90.10.p2.yes      produced
            foo.90.10.p2.no       produced, awaits `advance`
            enex/                 up to 26 files, contained
    done/
        in/   out/                flat — corpus, globbed in aggregate
        out/enex/foo.90.10/       per-batch, never globbed across batches
        p2_done.pairs
```

- `queued/` and `done/` stay flat: one artifact per item, and aggregate reads
  (`filter_pairs` globs `done/out/*.jsonl` across all history).
- `eval/` is the only slot with many artifacts per item, so it is the only slot that
  gets batch directories.
- The batch directory is **created by `eval`** and **removed by `archive`**. It exists
  iff work is in flight, so `ls eval/` is the in-flight list.
- `enex/` is archived to `done/out/enex/<slug>/` rather than flat, because it is never
  queried across batches — the "flat where you query" rule does not apply to it.

**The directory name is the batch-invariant prefix for that phase, which is not the same
string in p1 and p2.** In p2 it is the slug, as above. In p1 it cannot be: the batch is
the stem of the p1 *result* file, which evalpair has not written yet when `eval` runs,
and the slice is not chosen until `complete` filters into p2. So a p1 batch directory is
named by the submitted pairs filename — `s6.txt.pairs/` holding `s6.txt.pairs`,
`s6.txt.pairs_third_p3_juniper.qwen35.jsonl`, and any `.filtered`. The prefix invariant
holds in both phases, which is the property that actually matters; what varies is how
many dimensions are known at the moment the directory is opened.

Both phases take that directory name as the lifecycle positional (§2.6), so `eval` and
`complete` locate their inputs by prefix and neither has to take a name apart.

### 2.3 Operation protocol

An operation is a **module**. The module object is the identifier; there is no string
registry. This extends the convention `dispatch.py` already uses (`{"pairs": complete_pairs}`).

There are two kinds of operation, and conflating them is the one thing that makes this
protocol unimplementable.

**Primitives** take arguments and return or write what they are told. They are pure
functions over slots and paths: `select`, `merge`, `diff`, `filter`. They have CLI
surfaces (§2.6) and they never appear in a `STEPS` list.

**Steps** take only a `Context` and derive every path they touch from it. They are what
`STEPS` lists contain. A step is typically a thin `ctx`-binding wrapper that calls one
or more primitives.

```python
NAME: str
def inputs(ctx) -> list[Path]       # what this step consumes, derived from ctx alone
def outputs(ctx) -> list[Path]      # what this step produces, derived from ctx alone
def run_step(ctx) -> None           # do it
def is_done(ctx) -> bool            # default: all(p.exists() for p in outputs(ctx))

# plus the existing CLI trio, so every op is standalone-invokable:
def run(command, opts, argv) -> int
def show_help(command, opts, argv) -> int
def help_summary(name) -> str
```

**`is_done` answers "can the runner skip me?", not "is my output complete?"** Three ways
the default gets it wrong, each of which has bitten:

- **A step that moves things** has no output of its own to test, and rendering the
  destination name fails once the source it was rendered from is gone. Its answer is
  *placement*: the things it moves are no longer where it would move them from. That
  stays `False` until every move has landed, which is exactly the resume behaviour a
  multi-move step needs.
- **A step whose input a later step relocates** cannot answer `False` unconditionally
  just because it is idempotent — see §3's note on the folds. Idempotent is not the same
  as always *runnable*.
- **A step writing to a user-named destination** (`-o FILE`) must never treat that file's
  existence as a record. It says nothing about which parameters produced it; skipping on
  it silently keeps a stale file and reports success. Never skip; refuse to clobber
  without `--force`.

**Every move in a multi-move step is independently restartable.** Take whatever is still
there rather than demand to find it — glob the batch, or use `fs.move_into_once` /
`fs.rename_once` for a rendered name — so a retry after a partial failure finishes the
rest instead of dying on what already left. Missing from the source *and* absent at the
destination is a real error and still raises.

**A step never receives a value from the step before it.** There is no scratch dict on
`Context` and no in-memory channel between steps. This is forced by per-step `is_done`
skipping: if a producer is skipped as already-done, any in-memory value it would have
passed forward is simply absent, and the consumer breaks on exactly the resume path the
protocol exists to support. Steps communicate through the filesystem at rendered paths,
or not at all — which is what `inputs(ctx)`/`outputs(ctx)` make explicit.

A recipe is a flat list of modules:

```python
STEPS = [retrieve, extract_yes, extract_no, merge, archive, advance]
```

Runner:

```python
def run_steps(steps, ctx):
    for step in steps:
        if not ctx.force and step.is_done(ctx):
            log.info(f"skip {step.NAME}")
            continue
        step.run_step(ctx)
```

Re-running a recipe after a mid-way failure continues from where it stopped.
`-f/--force` means "ignore `is_done` and overwrite".

Note that `merge` names both a primitive (§2.7 set union over given paths) and a step
(§2.4, fold *this batch's* source pairs into `p{N}_done.pairs`). The step calls the
primitive. Same for `filter`. Where the distinction matters below, "the `merge` step"
and "the `merge` primitive" are written out.

### 2.4 Step ordering

```
extract → merge → archive → advance
```

- **extract** produces into the batch directory. Retryable, phase-private, nothing
  observable outside the phase.
- **merge** folds the source into `p{N}_done.pairs`. Idempotent (`sort -u`) and
  rollback-guarded; runs before anything moves.
- **archive** renames the batch's *inputs* into `done/{in,out}`. Atomic per file, and
  each file independently skippable, so a partial failure resumes rather than restarts.
- **advance** renames produced artifacts into the next phase's `queued/`, then removes
  the batch directory. Last, because publication is the only effect another `wf`
  invocation can observe — and because the directory's existence is what marks the batch
  in flight, so it must outlive every other step.

Today every producing call writes *directly into* the next phase's queue
(`_filter_pairs_to`, `_extract_pairs_to(no)`, `filter_pairs.py`), which fuses production
with publication and is why the order cannot currently be changed. Under this protocol
production always targets the batch directory.

### 2.5 Context

```python
@dataclass(frozen=True)
class Context:
    root: Path            # the .wf directory
    phase: str            # p1 | p2 | p3
    slug: str             # foo.90.10
    force: bool
    selector: str = "all" # passed to select() by steps that read a slot; see §2.3

    @property
    def batch_dir(self) -> Path:            # root/phase/eval/slug
    def artifact(self, classifier, kind) -> Path:
        return self.batch_dir / f"{self.slug}.{classifier}.{kind}"
```

Built once at command entry, and **fully immutable** — no field is written during a
run. `.wf/` remains the durable state across invocations; `Context` is only the binding
within one run.

`selector` is what lets a step call the `select` primitive from `inputs(ctx)` without
needing a previous step to hand it a path list. It carries the CLI's `all` /
`name:<n>` / `stem:<s>` argument verbatim.

`dest`, `pmin` and `prange` were added when Item 5 landed. The four batch coordinates are
enough for a *lifecycle* step, which renders every path it touches from phase and slug —
but `extract-p1-yes` is a corpus query, not a batch operation: it has no slug, and it
writes outside the layout to wherever `-o` points. Those three carry what it needs, and
they are what make `outputs(ctx)` derivable from the context alone, which is the
precondition for `is_done` working at all. The band is a real dimension of the design
(§2.1's slice), not an incidental flag. Steps that operate on a batch ignore all three.

### 2.6 CLI surface

Phase is a **required positional**, replacing the phase-named subcommands:

```
wf submit   p1|p2  FILE          (was: submit pairs | submit yes)
wf eval     p1|p2  SLUG          (was: eval pairs | eval yes)
wf complete p1|p2  SLUG          (was: complete pairs | complete yes)
```

The surface barely moves — `yes` becomes `p2`. What changes is behind it: one
implementation parameterized by phase, instead of two near-identical modules. The
duplication was never in the CLI shape, it was in `submit_pairs.py`/`submit_yes.py` and
`eval_pairs.py`/`eval_yes.py` being separate code. A required value belongs in a
positional, and `dispatch.py`'s registry becomes `{"p1": …, "p2": …}` — a near-mechanical
swap from `{"pairs": …, "yes": …}`.

Lifecycle positionals after the phase are the **slug** (the batch directory name), never
a filename. This is what keeps rule 2 of §2.1 honest: `eval p2 foo.90.10` locates its
input by globbing `queued/foo.90.10.*`, so no command ever has to take a name apart.

Individually invokable operations:

```
wf select   SLOT all|name:NAME|stem:STEM [--glob PAT]
wf filter   JSONL... [--pm PMIN] [--pr PRANGE] [-o FILE]
wf merge    SRC... DST
wf diff     A B
wf retrieve p2      SLUG
wf extract  p1|p2   SLUG
wf archive  p1|p2   SLUG
wf advance  p1|p2   SLUG
```

These four are the primitives of §2.3: they take arguments, not a `Context`.
`select`, `merge` and `diff` take no phase — they operate on slots and paths directly.
`filter` takes none either: it reads archived p1 results and **defaults** its
destination to `p2/queued/<rendered name>`; `-o FILE` overrides that destination.

`filter` takes a path *list*, matching the unified `filter_results` signature of Item 1.
The `-o` override is what lets the same primitive serve both `wf filter` (publish a
re-sliced batch into `p2/queued`) and `extract-p1-yes` (write a corpus-wide extract to an
arbitrary file). Without it the two would need separate implementations, which is the
duplication this refactor removes.

Two commands create new work and are therefore not lifecycle steps: **`submit`**
(external file → `queued/`) and **`filter`** (re-slice an archived p1 result at a new
band → `p2/queued/`). Everything else operates on a batch already in flight.

`submit` is where the sorted-unique set invariant is established — it is the boundary
where an arbitrary external file becomes a repo-managed set.

**`filter` is a second such boundary, and today it leaks.** `workflow/filter_pairs.py`
writes `filter_results` output straight into `p2/queued` with no sort, so that queue
entry is not a set: its lines come out in corpus order. `complete_pairs` sorts its
equivalent output, so the two producers of `p2/queued` disagree. This is live now, and
it is not cosmetic — `eval_yes` runs the queued file through
`eval_pairs.filter_done_pairs`, which is `comm -23`, and `comm` on unsorted input
silently yields a wrong difference rather than an error. Both producers must emit through
the `merge` primitive (which is also where `LC_ALL=C` gets applied), so add
`filter_pairs.py`'s missing sort to the `merge` row of the §2.7 table.

### 2.7 Set operations

A sorted-unique line file *is* a set here; the sorting is load-bearing because
`comm -23` requires it. There is no `normalize` operation — `sort -u` on one input is
the single-input case of union.

| op | implementation | replaces |
|---|---|---|
| `merge` (union) | `cat … \| sort -u` | `merge_pairs`, `_cat_sort_uniq`, both `submit` sorts, `_extract`'s sort |
| `diff` (difference) | `comm -23` | `filter_done_pairs` |

**Every shell-out to `sort` or `comm` runs under `LC_ALL=C`.** Collation is part of
what "sorted" means, and today's call sites inherit the ambient locale — so the same
input can produce a differently-ordered set on two machines with no code change. That
alone would break the §4.2 byte-identical check, which compares output produced here
against output produced on the machine holding the real archive.

The sharper risk is `diff`: `comm -23` assumes its two inputs are ordered under the
*same* collation it is using. A set merged under one locale and diffed under another
yields silently wrong differences, not an error. Pinning `LC_ALL=C` in the `merge` and
`diff` primitives makes the ordering a property of the operation rather than of the
environment it happened to run in.

Producing operations emit sets by construction: they write to a scratch file and place
the result atomically. That is a convention every producing op follows, not a step in
any recipe.

---

## 3. Work items

### Item 0 — Test fixture harness

**New, and first.** `tests/test_workflow_cli.py` asserts help strings and nothing else:
there is no behavioural test of the lifecycle anywhere. This refactor changes the
destination of *every* producing operation, and §4's behavioural checks (resumability,
the prefix invariant, byte-identical extraction) have nothing to run against locally —
the live `.wf` is on another machine.

So build the harness first:

```python
def make_wf(tmp_path, **files) -> Path   # init.init() a .wf under tmp_path, then populate
```

- `workflow.init.ensure_layout` already builds the full tree from `config.CONFIG_LAYOUT`,
  so the harness calls `init.init(opts)` rather than hand-rolling directories. A layout
  change stays a one-line `config.py` edit.
- Synthesise small `.jsonl` results and `.pairs` files — a few dozen rows, hand-chosen
  probabilities that straddle the 0.9 band edge, and at least one multi-direction row so
  `_build_prob_mask`'s `.any(axis=1)` is actually exercised.
- Fake `retrieve`'s network at the seam (Item 6), so recipe tests need no Evernote.

This is what converts §4.1, §4.3 and §4.4 from "run it by hand on the other machine"
into assertions that run in CI. The real-archive runs in §4.2 remain valuable as the
final confirmation, but they stop being the *only* evidence.

Note there is no `pytest` in `../.torch/bin/python`; the suite runs under
`python -m unittest`. Keep it that way or add the dependency deliberately.

### Item 1 — Unify `src/filter.py`

```python
def filter_results(paths: list[Path], yes: bool, out_file,
                   pairs_path: str | None = None,
                   pmin=0.5, prng=1.0, use_max=False)
```

`pairs_path=None` skips the identity mask; otherwise rows must also be members of that
pair set. Fold `filter_pairs` into this and delete it.

Callers:

| caller | paths | pairs_path |
|---|---|---|
| `wf filter` | `[one_jsonl]` | `None` |
| `complete p1` | `[*done_out.glob("*.jsonl"), new_result]` | the batch's source pairs |
| `extract-p1-yes` | all selected | `None` |

The `complete p1` row is correct **only under the new step ordering**. Today
`_complete` moves the result into `done/out` *before* filtering (with a comment
explaining that `filter_pairs` needs all of `done/out`), so the glob alone already
covers the new result. Under §2.4 `archive` runs after `extract`, so at filter time the
new result is still in the batch directory and must be listed explicitly. Listing both
under today's ordering would double-count it.

Carry `prefetch()` and the per-file `try/except`-and-warn from the old `filter_pairs`
onto the merged implementation; the single-file path currently has neither.

This is a hard prerequisite: without a path-list signature, `extract-p1-yes`'s filter
step stays a caller-side loop — the custom logic the refactor exists to remove. It also
unblocks the `# TODO: move complete_pairs.filter_results_to here` comment in
`workflow/filter_pairs.py`, which cannot be done today because the two callers use
functions with different corpus and restriction semantics.

Point `_filter_args` at the unified function (this also removes the `filter_results_dir`
call already fixed in place).

### Item 2 — Extract `select`, `merge`, `diff`

**`select`** is the one genuinely new primitive. It replaces two hand-rolled resolvers:

- `extract_p1_yes._result_paths` — `all` or one named file
- `complete_pairs._resolve_results_path` — stem-match, error on 0 or >1

```python
select(root, slot: list[str], selector: str, glob: str = "*") -> list[Path]
# selector: "all" | "name:<n>" | "stem:<s>"
```

`select` is a **primitive**, not a step (§2.3). Steps reach it from `inputs(ctx)` as
`select(ctx.root, slot, ctx.selector)`.

Item 2 delivers these three as functions only. Their standalone CLI surfaces (§2.6) wait
for Item 6, where the op protocol's `run`/`show_help`/`help_summary` trio is defined and
`dispatch.py`'s registry is rewritten anyway — building half a protocol here and the
other half there would only invite rework. `merge` and `diff` live together in
`workflow/setops.py`, since they share the atomic-place-under-`LC_ALL=C` runner and
neither ever appears in a `STEPS` list.

Absolute selectors bypass the slot, as `workflow/filter_pairs.py` already does.

**`merge`** and **`diff`** are extractions of existing code (`_cat_sort_uniq`,
`filter_done_pairs`), not new behaviour. Replace `plumbum.cmd` with `subprocess`
throughout, and pass `env={**os.environ, "LC_ALL": "C"}` to every `sort` and `comm`
invocation (§2.7). Item 5's `_extract` sort and both `submit` sorts get the same
treatment as they are absorbed into the `merge` primitive.

*(Corrected: an earlier draft claimed `wf` does not start because `plumbum` is missing
from `../.torch/bin/python`. It is installed there and `wf` runs. The swap is still
worth doing — `extract_p1_yes.py` already uses `subprocess`, and one subprocess idiom
beats two — but it is cleanup, not a fix, and nothing is blocked on it.)*

### Item 3 — Name rendering and segment reorder

Add the renderer from §2.1. Delete
`_phase2_name`, `_make_yes_filename`, `_make_pairs_filename`; rewrite `yes_suffix` as
the slice segment only.

Then rename the existing archive to the new segment order (`foo.p1.90.10.yes` →
`foo.90.10.p1.yes`). **Manual step, and deferred** — the live `.wf` is on another
machine and does not exist in this checkout, so there is nothing here to rename. The
renderer lands now; the archive rename happens on the machine that holds the archive,
before `wf` is next run there. Ship a `wf` subcommand or a one-off script for it rather
than a hand-typed `mv` loop, so the rename is reproducible and reviewable.

**The rename must skip anything in flight in `p2/eval`.** Note titles are derived from
the p2 input filename — `eval` splits to `/tmp/<name>.aa…` and titles each note after
its split file, and `_retrieve_notes` reconstructs those same titles to fetch them back.
Renaming a p2/eval file whose notes already exist in Evernote leaves `retrieve` looking
up titles that were never created, and it reports "no note parts found" rather than
anything that points at the cause. Either drain `p2/eval` before renaming, or exclude
in-flight batches from the rename and let them finish under their old names.

### Item 4 — `eval/<slug>/` batch directories

`eval` creates the batch directory and moves the queued file in. `archive` fans the
contents out to `done/{in,out}` and removes the directory.

Delete `_check_already_submitted`: with a batch directory, `is_done` is one stat against
a known path rather than a three-slot name scan. Its lifecycle question ("has this ever
been through p2?") is answered by `done/in`.

Update `show` so `wf show p2 eval` lists batch directories, one line each, rather than
every loose file.

`complete p2` also has to start archiving the *queued* artifact rather than the
`.filtered` derivative `eval` may have produced from it. It archived the derivative and
left the original in `p2/eval` forever — invisible while the slot was flat, but it keeps
the batch directory from ever draining, so `finish` would refuse to remove it. p1 already
had this right: merge the filtered set, archive the original, drop the derivative.

### Item 5 — `extract-p1-yes` as the first recipe

```python
STEPS = [filter]
```

**Corrected from `[select, filter]`.** `select` is a primitive, not a step (§2.3): a
two-step recipe would need `select` to hand its path list to `filter` in memory, which
§2.3 rules out. Instead `filter.inputs(ctx)` calls
`select(ctx.root, ["p1", "done", "out"], ctx.selector, glob="*.jsonl")`, and the
selector arrives on the `Context`.

The acceptance test survives the correction, because the claim it actually tests is not
"two steps compose" but "**this command contains no bespoke logic**". Today
`extract_p1_yes.py` carries its own `_result_paths` resolver and its own `_extract`
(temp file, per-path loop over `filter_results`, `sort -u`, atomic place). After this
item all four are gone: resolution is `select`, filtering is the unified
`filter_results` of Item 1, and set-emission is the standard producing-op convention of
§2.7. What remains is a `STEPS` list and an argument parser.

Pure reads, one output, no state transitions, no network — and a known-good reference
implementation to diff against. It validates the vocabulary on the cheapest possible
case before Item 6 stresses it.

One behaviour changes as a consequence. `_extract` used to refuse to overwrite an
existing output (`fs.raise_if_exists`); under the protocol an existing output means
`is_done`, so a re-run now **skips** the step and reports it rather than failing.
`-f/--force` overwrites. That is the protocol working as specified, not an oversight —
but it turns a hard error into a logged no-op, so the skip must stay visible in the
output.

### Item 6 — `complete` as a recipe

```python
# p1
STEPS = [extract, merge, archive, advance]
# p2
STEPS = [retrieve, extract_yes, extract_no, merge, archive, advance]
```

This is where publication ordering and `is_done` become load-bearing. `retrieve` is the
one non-idempotent operation in the package — it probes note parts `.aa`…`.az` until a
404 and raises on pre-existing output — so it is the step most likely to strain the
protocol.

**`retrieve`'s `outputs(ctx)` cannot be enumerated**, which breaks the default `is_done`.
The part count is only known when the 404 arrives, so given an `enex/` holding
`.aa .ab .ac`, the filesystem cannot distinguish "finished, there were 3 parts" from
"crashed after 3, there are 5 on the server."

Fix it with the §2.7 producing-op convention applied to a *directory* rather than a file:
`retrieve` builds into `enex.part/`, and on the 404 — the moment completion is actually
known — renames it to `enex/`. Directory rename is atomic on one filesystem, so
`enex/` exists ⟺ `retrieve` completed, and `is_done` is one stat. Partial work survives
in `enex.part/`, so re-running skips the parts already there and resumes probing from the
first gap — the "fills the gap instead of raising" behaviour, for free. Write each part
to a temp name and rename it into place as well, so a truncated `.enex` can never pass
as a complete one.

This is deliberately *not* a manifest. Placement already records everything the other
five steps need (`archive` and `advance` are atomic renames; `extract_*` become atomic
under §2.7; the idempotent folds read placement, see below), so a batch manifest would
add a second source of truth that has to be reconciled with placement, and would
relocate rather than remove the crash window it is meant to close. If a manifest is ever
added it should be *advisory provenance only* — model and prompt-variant, timings, row
counts, the things names cannot carry — with a hard rule that `is_done` and every other
control decision never read it.

**An idempotent fold may always run only while its input is in place.** `merge` and
`p2_classify` fold the batch into a set shared across every batch (`p1_done.pairs`,
`classified/yes.pairs`). Union under `sort -u` is idempotent, and the shared set's
existence says nothing about *this* batch, so it is tempting to write `is_done` as a
constant `False` and let the fold always run. That is wrong: idempotent is not the same
as always *runnable*. Both folds sit before `archive`, which relocates their input into
done/, so on any retry after `archive` an always-run fold re-executes and dies on a
missing input — taking out the resume path for `advance`, the one step that still has
work left to do, and leaving the batch permanently un-completable. Placement carries the
answer here too: the input's absence *is* the record that `archive` ran, and therefore
that the fold ran before it. `is_done` reads that (`batch.has_source`), not a manifest.

`extract` no longer writes to `p2/queued` or `p3/queued`; it writes to the batch
directory, and `advance` publishes.

---

## 4. Verification

Split by where it can run. The live `.wf` is on another machine; everything in §4.1 runs
locally against Item 0's fixtures, and §4.2 is confirmation work deferred until the
archive is in reach.

### 4.1 Local, on fixtures — gates each item

1. **`extract-p1-yes` byte-identical.** Snapshot the current implementation's output
   over a fixture `p1/done/out` *before* Item 1; after Item 5, the recipe must produce
   an identical file. Cover both selectors (`all` and a single named result) and at
   least one band whose `pmin + prange != 1.0`, so the `_pmax` special case is not the
   only path tested. This is the acceptance test for the whole refactor.
2. **Resumability.** Run `complete p2` on a fixture batch, stop after the `merge` step,
   re-run without `--force`: it skips to `archive` and completes. Today this raises on
   the already-present enex files. Assert the skip is logged, not just that the exit
   code is 0 — a silently re-run step passes this test for the wrong reason.
3. **Prefix invariant.** Assert `filename.startswith(dirname)` for every artifact
   *file* in every batch directory the fixture run produces. The assertion exempts
   grouping subdirectories (`enex/`, and `enex.part/` while retrieve is mid-flight),
   which are named by role rather than by content key.
4. **CLI tests.** `tests/test_workflow_cli.py` asserts help strings that change with the
   phase-positional surface; update them alongside Item 6. They pass today (14/14 under
   `python -m unittest tests.test_workflow_cli`) — keep them green at every item
   boundary, not just at the end.

### 4.2 Deferred to the machine holding the real `.wf`

5. **Archive rename.** Item 3's segment reorder, applied to the real archive by the
   script Item 3 ships. Run it before any other `wf` command touches that tree.
6. **`extract-p1-yes` byte-identical over the real corpus.** The §4.1.1 check again,
   against a real `p1/done/out` rather than fixtures. This is the confirmation that the
   fixtures were representative; if it disagrees with §4.1.1, the fixtures were wrong,
   not the code.
7. **A real `complete p2` end-to-end,** including `retrieve` against Evernote — the one
   step whose behaviour the fixtures fake rather than exercise.

*(Removed: an earlier draft's "`wf` starts — it does not today (`plumbum`)". It does
start; see Item 2.)*

## 5. Order

Items are sequential. 0 comes first so that every later item lands with a test that can
actually run; 1 unblocks everything else; 3 and 4 are paired (both are pure renaming and
layout refactors of existing behaviour, and Item 4's `is_done` depends on Item 3's
names); 5 proves the protocol before 6 stresses it.

Items 0–6 and §4.1 are all local work and can land in full on this machine. Only §4.2
waits on the other machine, and nothing in Items 0–6 is blocked by that wait.
