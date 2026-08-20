# Workflow Operation Refactor

Refactor `~/code/words/workflow` so that every command is a sequence of independently
invokable operations, and one-off commands can be assembled from those operations
without custom logic.

Acceptance test: `extract-p1-yes` — built originally as bespoke logic — becomes a
two-operation recipe producing byte-identical output to the current implementation.

---

## 1. Scope

### In scope

- `src/filter.py` — unify the two filter entry points into one signature.
- `workflow/` — op protocol, `STEPS` recipes, `Context`, name rendering, `eval/<slug>/`
  batch directories.
- `tests/test_workflow_cli.py` — update for the new CLI surface; add the
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

### 2.3 Operation protocol

An operation is a **module**. The module object is the identifier; there is no string
registry. This extends the convention `dispatch.py` already uses (`{"pairs": complete_pairs}`).

```python
NAME: str
def outputs(ctx) -> list[Path]      # what this step produces
def run_step(ctx) -> None           # do it
def is_done(ctx) -> bool            # default: all(p.exists() for p in outputs(ctx))

# plus the existing CLI trio, so every op is standalone-invokable:
def run(command, opts, argv) -> int
def show_help(command, opts, argv) -> int
def help_summary(name) -> str
```

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

### 2.4 Step ordering

```
extract → merge → archive → advance
```

- **extract** produces into the batch directory. Retryable, phase-private, nothing
  observable outside the phase.
- **merge** folds the source into `p{N}_done.pairs`. Idempotent (`sort -u`) and
  rollback-guarded; runs before anything moves.
- **archive** renames artifacts into `done/{in,out}` and removes the batch directory.
  Atomic per file.
- **advance** renames produced artifacts into the next phase's `queued/`. Last, because
  publication is the only effect another `wf` invocation can observe.

Today every producing call writes *directly into* the next phase's queue
(`_filter_pairs_to`, `_extract_pairs_to(no)`, `filter_pairs.py`), which fuses production
with publication and is why the order cannot currently be changed. Under this protocol
production always targets the batch directory.

### 2.5 Context

```python
@dataclass(frozen=True)
class Context:
    root: Path        # the .wf directory
    phase: str        # p1 | p2 | p3
    slug: str         # foo.90.10
    force: bool

    @property
    def batch_dir(self) -> Path:            # root/phase/eval/slug
    def artifact(self, classifier, kind) -> Path:
        return self.batch_dir / f"{self.slug}.{classifier}.{kind}"
```

Built once at command entry. `.wf/` remains the durable state across invocations;
`Context` is only the binding within one run.

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
wf select   SLOT all|NAME|STEM [--glob PAT]
wf filter   JSONL [--pm PMIN] [--pr PRANGE] [-o FILE]
wf merge    SRC... DST
wf diff     A B
wf retrieve p2      SLUG
wf extract  p1|p2   SLUG
wf archive  p1|p2   SLUG
wf advance  p1|p2   SLUG
```

`select`, `merge` and `diff` take no phase: they operate on slots and paths directly.
`filter` takes none either — it always reads an archived p1 result and writes to
`p2/queued`.

Two commands create new work and are therefore not lifecycle steps: **`submit`**
(external file → `queued/`) and **`filter`** (re-slice an archived p1 result at a new
band → `p2/queued/`). Everything else operates on a batch already in flight.

`submit` is where the sorted-unique set invariant is established — it is the boundary
where an arbitrary external file becomes a repo-managed set.

### 2.7 Set operations

A sorted-unique line file *is* a set here; the sorting is load-bearing because
`comm -23` requires it. There is no `normalize` operation — `sort -u` on one input is
the single-input case of union.

| op | implementation | replaces |
|---|---|---|
| `merge` (union) | `cat … \| sort -u` | `merge_pairs`, `_cat_sort_uniq`, both `submit` sorts, `_extract`'s sort |
| `diff` (difference) | `comm -23` | `filter_done_pairs` |

Producing operations emit sets by construction: they write to a scratch file and place
the result atomically. That is a convention every producing op follows, not a step in
any recipe.

---

## 3. Work items

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

Absolute selectors bypass the slot, as `workflow/filter_pairs.py` already does.

**`merge`** and **`diff`** are extractions of existing code (`_cat_sort_uniq`,
`filter_done_pairs`), not new behaviour. Replace `plumbum.cmd` with `subprocess`
throughout — `plumbum` is imported by `complete_pairs.py` and is not installed in
`../.torch/bin/python`, so `wf` does not currently start.

### Item 3 — Name rendering and segment reorder

Add the renderer from §2.1. Delete
`_phase2_name`, `_make_yes_filename`, `_make_pairs_filename`; rewrite `yes_suffix` as
the slice segment only.

Then rename the existing archive to the new segment order (`foo.p1.90.10.yes` →
`foo.90.10.p1.yes`). **Manual step.** It must land in the same commit as the
renderer so the tree is never in a mixed convention.

### Item 4 — `eval/<slug>/` batch directories

`eval` creates the batch directory and moves the queued file in. `archive` fans the
contents out to `done/{in,out}` and removes the directory.

Delete `_check_already_submitted`: with a batch directory, `is_done` is one stat against
a known path rather than a three-slot name scan. Its lifecycle question ("has this ever
been through p2?") is answered by `done/in`.

Update `show` so `wf show p2 eval` lists batch directories, one line each, rather than
every loose file.

### Item 5 — `extract-p1-yes` as the first recipe

```python
STEPS = [select, filter]
```

Pure reads, one output, no state transitions, no network, no `is_done` needed — and a
known-good reference implementation to diff against. This is the composability proof and
it validates the vocabulary on the cheapest possible case.

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
protocol. Its `is_done` is "the expected enex files exist in the batch directory"; on
partial failure, re-running fills the gap instead of raising.

`extract` no longer writes to `p2/queued` or `p3/queued`; it writes to the batch
directory, and `advance` publishes.

---

## 4. Verification

1. **`extract-p1-yes` byte-identical.** Run the current implementation over
   `p1/done/out` before Item 1; after Item 5, the recipe must produce an identical file.
   This is the acceptance test for the whole refactor.
2. **`wf` starts.** It does not today (`plumbum`). After Item 2 it must, under
   `../.torch/bin/python`.
3. **Resumability.** Interrupt `complete p2` after `merge`; re-run without `--force`;
   it skips through to `archive` and completes. Today this raises on the already-present
   enex files.
4. **Prefix invariant.** Assert `filename.startswith(dirname)` for every artifact in
   every batch directory.
5. **CLI tests.** `tests/test_workflow_cli.py` asserts help strings that change with the
   `-p` surface; update them alongside Item 6.

## 5. Order

Items are sequential. 1 unblocks everything; 3 and 4 are paired (both are pure renaming
and layout refactors of existing behaviour, and Item 4's `is_done` depends on Item 3's
names); 5 proves the protocol before 6 stresses it.
