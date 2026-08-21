# Code review: operations refactor

Scope: the uncommitted refactor — `git diff HEAD` plus the new untracked
`workflow/batch.py`, `context.py`, `names.py`, `select.py`, `setops.py`, and
`workflow/steps/*`. Effort: medium. Nothing on disk was changed by the review.

**Status: all nine fixed** (2026-08-20), verified against the working tree and the
workflow test modules. Findings 1–2 and 6–7 share one cause and one cure — see §2.3 of
`plans/refactor-operations.md`, amended alongside. The two low-value items at the bottom
were deliberately skipped.

## Resume/retry is broken after `archive`

1. **`workflow/steps/merge.py:26`** — `is_done` returns `False`, but its input is
   `batch.evaluated(ctx)`, which `archive` has already moved out of the batch
   directory. Any failure after `archive` and before `advance` makes every retry
   crash rather than resume, and the batch becomes permanently un-completable.
   Reproduced in both phases: `complete_pairs.STEPS` after `p1_archive.run_step`
   gives `ValueError: no a.pairs in .../p1/eval/a.pairs`; the p2 recipe run
   through `STEPS[:6]` and re-run gives `ValueError: no *.p1.yes in
   .../p2/eval/<slug>`. The comment at merge.py:24 argues `False` is "cheap and
   correct" because the done-set is shared — right about the *output*, but it
   ignores that the *input* is gone by then.

2. **`workflow/steps/p2_classify.py:29`** — same shape: `is_done` always `False`
   while its input `ctx.artifact("p2","yes")` is moved to `p2/done/out` by
   `archive`. On a post-archive retry `setops.merge` shells out to `sort` on a
   nonexistent file and dies with `CalledProcessError`.

## Naming assumptions that don't match `batch.begin`

3. **`workflow/batch.py:69`** — `source()` passes `ctx.slug` to `Path.glob` as a
   literal pattern, so it only matches a file named exactly the slug. But
   `batch.begin` resolves the queued file with `stem:` (prefix) matching, so
   `wf eval p1 a` legitimately creates batch dir `a/` holding `a.pairs`, and every
   later `complete p1 a` step fails with `ValueError: no a in .../p1/eval/a`.
   Reproduced. `workflow/steps/p1_archive.py:23` (`outputs`) shares the
   assumption — it renders the archived name as `done/in/<slug>` while `run_step`
   actually moves `source.name`, so `is_done` is wrong in the same case.

4. **`workflow/steps/p1_advance.py:22`** — `inputs`/`outputs`/`run_step` glob
   `f"{ctx.slug}*.p1.{kind}"`, but `p1_extract.produced_slug` renders the names
   from the `.jsonl` *result* stem, which nothing enforces to be prefixed by the
   batch slug (`batch.begin` only asserts the prefix for the queued input). If
   evalpair writes e.g. `results.jsonl` into the batch, extract produces
   `results.90.10.p1.*`, advance moves nothing, and `batch.finish_ctx` then raises
   `batch <slug> still holds: ...` — after archive has already run, i.e. straight
   into the non-resumable state of finding 1.

## Non-idempotent renames

5. **`workflow/steps/p2_retrieve.py:80`** — `staging.rename(enex_dir(ctx))` fails
   when `enex/` already exists. `-f` skips `is_done` by design, so re-running
   `complete p2` on a batch whose notes were already retrieved raises
   `OSError: [Errno 39] Directory not empty`. Reproduced by calling
   `p2_retrieve.run_step` twice.

6. **`workflow/steps/p1_archive.py:33`** — not restartable mid-step: it moves the
   `.jsonl` first and the source second. If the second `move_into` fails
   (destination already present, or the process dies between the two renames),
   `is_done` stays `False` and the retry raises `ValueError: no *.jsonl in
   <batch dir>` from `batch.one`, because the result has already left.

7. **`workflow/steps/p2_archive.py:35`** — same partial-failure hole, plus a force
   hazard: once the YES artifact has moved but the source has not, the retry's
   first `fs.move_into(ctx.artifact("p2","yes"), ...)` raises `FileNotFoundError`.
   Also `enex.rename(destination)` at line 46 only skips the existence check under
   `--force`; renaming onto an existing non-empty `enex/<slug>` raises
   `OSError: Directory not empty`.

## Silent wrong output

8. **`workflow/steps/filter.py:24`** — `outputs` is `[ctx.dest]` under the default
   "outputs exist ⇒ skip" rule, and `ctx.dest` is a user-supplied `-o FILE`
   outside any batch. `wf extract p1 yes all -o out.pairs --pm 0.5` after an
   earlier `--pm 0.9` run silently skips ("skip filter: already done") and reports
   `YES pairs at out.pairs` while the file still holds the old band. The removed
   `_extract` raised `file already exists` in this case.

9. **`src/filter.py:78`** — the warn-and-continue around `iter_projected_blocks`
   used to apply only to the multi-file `filter_pairs` path; the merged
   `filter_results` now applies it to single-file callers too. `wf filter
   <corrupt>.jsonl` (and `wf extract p1 yes <file>`) now prints
   `WARNING: skipping ...`, writes an empty `<slug>.p1.yes` into `p2/queued`, logs
   `Filtered 0 pairs`, and exits 0, where the old single-path code propagated the
   error.

## Left out as low-value

- `workflow/filter_pairs.py:257` leaks an unclosed file handle in the line count.
- `workflow/steps/p2_extract.py:19` writes parsed output to fixed
  `/tmp/<name>.parsed` paths — pre-existing code, but now reused for both the YES
  and NO kinds.
