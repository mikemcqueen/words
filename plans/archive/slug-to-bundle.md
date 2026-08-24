# Replace Workflow Slug and Batch Terminology with Bundle

Replace the AI-introduced `slug` and workflow-specific `batch` vocabulary with
`bundle`, without changing any filenames, paths, command behavior, or lifecycle
semantics.

This is a terminology refactor, not a workflow redesign. A **bundle** is the
collection of related artifacts held together while one evaluation is in
progress.

## Decisions

### Vocabulary

Use one noun for the collection and suffixes that identify its representation:

| Term | Meaning | Type/example |
|---|---|---|
| `bundle` | The conceptual collection of related artifacts | prose or a generic module name |
| `bundle_name` | The bundle's identifying basename | `str`, such as `foo.90.10` |
| `bundle_dir` | The directory containing the in-progress bundle | `Path`, such as `.wf/p2/eval/foo.90.10` |
| `bundle_artifacts` | A collection of paths belonging to a bundle | `list[Path]` when such a variable is needed |

`bundle_dir` is a path, not a basename. Its final component is the bundle name:

```python
bundle_dir = config.path(root, [phase, "eval"]) / bundle_name
assert bundle_dir.name == bundle_name
```

Do not add `active_`: the layout already establishes that `bundle_dir` is under
`eval/`, and there is no competing completed bundle directory.

Concrete artifact variables remain content-specific, for example
`source_pairs_path`, `result_path`, `yes_pairs_path`, and `no_pairs_path`.
`artifact` remains acceptable in generic naming or collection infrastructure that
does not know which kind it is handling.

### Lifecycle

The bundle directory exists only under `<phase>/eval/`:

1. `eval` creates `eval/<bundle_name>/` and moves the queued source into it.
2. Evaluation and completion add related artifacts to that directory.
3. Archive and advance move individual artifacts to `done/{in,out}` or another
   phase's queue and delete scratch files.
4. Completion removes the empty bundle directory.

The directory itself does not move to `done/`. The p2 ENEX archive at
`done/out/enex/<bundle_name>/` is a specific archived-output directory, not the
bundle directory.

### Current and produced bundle names

`Context.bundle_name` identifies the current directory under `eval/`. In p1 that
name comes from the submitted pairs filename because the result name and
probability slice do not exist when `eval` begins.

P1 completion derives a name for the subsequently queued p2/p3 artifacts from the
result stem and probability slice. Call that value `produced_bundle_name` where it
appears beside the current bundle name:

```python
current_bundle_name = ctx.bundle_name
produced_bundle_name = names.bundle_name(result_path.stem, pmin, prange)
```

This preserves the existing distinction instead of pretending the p1 evaluation
directory and its produced artifact prefix always have the same name.

## Scope

### In scope

- Rename the workflow's `slug` identifiers and CLI metavar to `bundle_name` and
  `BUNDLE-NAME`.
- Rename the workflow-specific `batch` module, directory property, helpers, test
  fixtures, class names, messages, and prose to `bundle` equivalents.
- Use specific names for concrete files and paths while touching affected call
  sites.
- Update the operation-refactor plan and its code-review finding so they remain
  usable descriptions of the current implementation.
- Replace older workflow uses of `batch` that actually mean one canonical pairs
  file with the specific pairs-file term instead of mechanically calling the
  file a bundle.

### Out of scope

- Changing any on-disk filename or directory name.
- Moving a whole bundle directory into `done/` or retaining it after completion.
- Changing selector behavior, including `stem:` prefix selection.
- Changing phase routing, step order, filtering, naming segments, probability
  formatting, resumability, or overwrite behavior.
- Adding a bundle object, manifest, or new abstraction beyond the terminology
  already represented by `Context` and the filesystem.
- Replacing unrelated, conventional uses of `batch` for computational batching
  in model, matrix, or corpus-processing code and plans.
- Keeping compatibility aliases such as `names.slug` or `workflow.batch`; these
  are internal Python interfaces and retaining aliases would preserve the
  vocabulary being removed.

No `.wf` migration is required: all rendered string values stay byte-for-byte
identical.

## Rename map

| Current | Replacement |
|---|---|
| `workflow/batch.py` | `workflow/bundle.py` |
| `from workflow import batch` | `from workflow import bundle` |
| `batch.<helper>` | `bundle.<helper>` |
| `Context.slug` | `Context.bundle_name` |
| `Context.batch_dir` | `Context.bundle_dir` |
| `names.slug(result_stem, ...)` | `names.bundle_name(result_stem, ...)` |
| `names.artifact(slug, ...)` parameter | `names.artifact(bundle_name, ...)` |
| `produced_slug()` | `produced_bundle_name()` |
| local `slug` variables | `bundle_name` or `produced_bundle_name`, according to role |
| `SLUG` CLI metavar | `BUNDLE-NAME` |
| test constant `SLUG` | `BUNDLE_NAME` |
| fixture `make_batch()` | `make_bundle()` |
| `BatchDirectoryTests` | `BundleDirectoryTests` |
| `BatchInputSelectionTests` | `BundleInputSelectionTests` |
| `BatchLifecycleTests` | `BundleLifecycleTests` |
| `enex_archive()` | `enex_archive_dir()` |

Within `names.bundle_name()`, name the first parameter `result_stem`, not `batch`:
the current callers pass the originating result JSONL's stem. Keep
`slice_segment()`, `artifact()`, the rendered segment order, and all validation
unchanged.

Rename `batch.INPUT_GLOB` to `bundle.SOURCE_GLOB` because it identifies the source
artifact for each phase rather than every artifact in the bundle.

## Implementation

### 1. Rename the naming and context primitives

Update `workflow/names.py`:

- Rewrite the grammar and comments as
  `<bundle_name>.<classifier>.<kind>`.
- Rename `slug()` to `bundle_name()` and its `batch` parameter to
  `result_stem`.
- Rename `artifact()`'s `slug` parameter to `bundle_name`.
- Preserve every returned string and every validation rule.

Update `workflow/context.py`:

- Rename the dataclass field `slug` to `bundle_name`.
- Rename the `batch_dir` property to `bundle_dir`.
- Update `artifact()` to join the rendered artifact name beneath
  `bundle_dir`.
- Change empty-name diagnostics and comments to bundle terminology.
- Keep the empty default because corpus queries use `Context` without an
  evaluation bundle.

### 2. Rename the generic bundle operations

Rename `workflow/batch.py` to `workflow/bundle.py` and update every import and
call site.

- Rename `INPUT_GLOB` to `SOURCE_GLOB`.
- Use `ctx.bundle_name` and `ctx.bundle_dir` throughout.
- Replace local `directory` with `bundle_dir` where the local represents that
  directory; retain ordinary `directory` where no bundle meaning exists.
- Update docstrings, comments, assertion text, and errors from batch/slug to
  bundle/bundle name.
- Preserve the exact `begin`, `one`, `source`, `evaluated`, `filter_done`, and
  `finish` behavior.

The rename must reach:

- `workflow/eval.py`
- `workflow/complete.py`
- `workflow/filter_pairs.py`
- `workflow/select.py`
- `workflow/steps/merge.py`
- `workflow/steps/p1_{extract,archive,advance}.py`
- `workflow/steps/p2_{retrieve,extract,classify,archive,advance}.py`

Use `bundle_name` for the current context. In `p1_extract.py`, use
`produced_bundle_name` for the result-stem-and-slice value used to render the YES
and NO filenames. Do not introduce generic locals such as `artifact_name` at
concrete call sites.

In `p2_archive.py`, rename `enex_archive()` to `enex_archive_dir()` so the
directory-returning helper follows the same `_dir` convention.

### 3. Update the CLI language without changing parsing

Change the `eval` and `complete` positional metavar from `SLUG` to
`BUNDLE-NAME`. Rename their locals and comments, but leave positional parsing and
selection unchanged.

Examples after the rename:

```text
wf eval     p1|p2 BUNDLE-NAME
wf complete p1|p2 BUNDLE-NAME
```

`BUNDLE-NAME` is still only a string supplied to the existing exact/prefix
resolution. It is not interpreted as a CWD-relative or `-d`-relative path.

### 4. Update tests and fixtures in place

Update the existing tests; do not add tests merely for the vocabulary change.

- Rename imports from `batch` to `bundle`.
- Rename fixture helper `make_batch()` to `make_bundle()`.
- Rename `SLUG`, local `slug`, `batch_dir`, and affected test classes according
  to the rename map.
- Update CLI help assertions to require `BUNDLE-NAME`.
- Rewrite test descriptions and failure messages to use bundle terminology.
- Replace fixture-only names such as `batch.jsonl` with a content-neutral name
  such as `sample.jsonl` when they refer to this workflow concept; keep genuine
  computational-batch terminology outside the workflow lifecycle.
- Preserve all expected paths, filenames, file contents, and lifecycle
  assertions.

Affected test files are:

- `tests/wf_fixture.py`
- `tests/test_workflow_cli.py`
- `tests/test_workflow_primitives.py`
- `tests/test_workflow_steps.py`
- `tests/test_workflow_fixture.py`
- `tests/test_workflow_p2.py`

### 5. Update current workflow documentation

Update `plans/refactor-operations.md` so its naming grammar, directory diagrams,
CLI examples, Context sketch, implementation sections, and verification language use
bundle terminology. Preserve its design decisions and historical implementation
sequence; this pass changes vocabulary, not substance.

Update `findings/cr-operation-refactor.md` to point at `workflow/bundle.py`,
`ctx.bundle_name`, `ctx.bundle_dir`, and `produced_bundle_name`. Preserve the
findings themselves.

In `plans/complete_yes_bug.md`, replace workflow uses where `batch` means a
specific canonical YES-pairs file with concrete terms such as
`canonical_pairs`, `source_pairs`, or “canonical YES-pairs file.” Do not call an
individual file a bundle.

Do not replace unrelated uses such as an inference batch, matrix batch, or a
group of JSONLs processed together merely to obtain a zero-result search.

## Verification

Run these checks after implementation:

1. Compile the renamed workflow modules and affected tests:

   ```sh
   python -m py_compile workflow/*.py workflow/steps/*.py \
       tests/wf_fixture.py tests/test_workflow_*.py
   ```

2. Run the focused workflow suite:

   ```sh
   python -m unittest \
       tests.test_workflow_primitives \
       tests.test_workflow_steps \
       tests.test_workflow_fixture \
       tests.test_workflow_p2 \
       tests.test_workflow_cli
   ```

3. Confirm `wf eval p1 -h`, `wf eval p2 -h`, `wf complete p1 -h`, and
   `wf complete p2 -h` show `BUNDLE-NAME` and no `SLUG` wording.

4. Run a residual terminology search over the affected workflow surface:

   ```sh
   rg -n -i 'slug|batch_dir|workflow[.]batch|from workflow import batch|BatchDirectory|BatchInput|BatchLifecycle|make_batch' \
       workflow tests plans/refactor-operations.md \
       plans/complete_yes_bug.md findings/cr-operation-refactor.md
   ```

   The only allowed `slug` occurrences are migration explanations in this plan
   and Git history. Review remaining `batch` occurrences semantically rather
   than applying an unscoped global replacement.

5. Compare before/after fixture trees and produced filenames for representative
   p1 and p2 lifecycles. They must be identical; only Python identifiers, help
   metavars, diagnostics, comments, tests, and documentation change.

6. Run `git diff --check` and inspect the diff for accidental edits to the
   existing uncommitted changes in `workflow/context.py`, `workflow/extract.py`,
   `workflow/steps/filter.py`, and `tests/test_workflow_fixture.py`.

## Acceptance criteria

- The active workflow code uses `bundle_name` for the identifying string and
  `bundle_dir` for its directory path.
- The workflow-specific `batch.py`, `batch_dir`, `slug`, and `SLUG` vocabulary
  is gone from active code, tests, CLI help, and maintained workflow documents,
  apart from migration references in this plan.
- Concrete file paths retain content-specific names rather than becoming generic
  “artifact name” variables.
- Existing `.wf` paths and rendered filenames do not change.
- The bundle directory is still created only under `eval/`, drained by moving
  individual artifacts, and removed when empty.
- Existing workflow tests pass after identifier and expectation updates, with no
  new behavior-specific tests required.
