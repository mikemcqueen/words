# Plan: Add `wf extract p1 yes`

## Goal

Add a read-only command that extracts evalpair-classified YES pairs from every
archived p1 result file under `.wf/p1/done/out/`.

```text
wf extract p1 yes
wf extract p1 yes --pm 0.8 --pr 0.1
wf extract p1 yes --output p1.80.10.yes
```

The command must not create, replace, or inspect submission state under
`p2/queued`, `p2/eval`, or `p2/done/in`. Publishing a probability band for
manual review remains the responsibility of `wf filter pairs`.

## Semantics

- Discover `*.jsonl` directly under `.wf/p1/done/out/`, in sorted pathname
  order.
- Process the files independently. Do not pass the full path list to
  `compare_native.iter_projected_blocks()`: its multi-file mode aligns parallel
  evaluations of the same pairs rather than concatenating independent batches.
- For each file, call `src.filter.filter_results()` with `yes=True` and write
  its matches to the same aggregate stream.
- Use the same directional rule as `wf filter pairs`: `use_max=False`, so a row
  qualifies when any direction is labeled YES in the requested probability
  band.
- Keep the existing workflow filter defaults, `--pm 0.9 --pr 0.1`. Here,
  "all" means all archived result files, not all probability values. A caller
  can request every YES-labeled probability with `--pm 0 --pr 1`.
- Sort and deduplicate the aggregate before emitting it. If a pair qualifies in
  multiple result files, the output contains that pair once; this gives the
  same union/any-qualifying-result behavior as `complete pairs`.
- Write pairs to stdout by default. `-o/--output FILE` writes the same data to
  an explicit user path.
- Send status and counts to stderr so stdout remains a clean pair stream.

## Command structure

Follow the existing nested dispatch convention:

1. Add `extract` to `workflow.wf.COMMANDS`.
2. Add `workflow/extract.py`, dispatching the `p1` target.
3. Add `workflow/extract_p1.py`, dispatching the `yes` target.
4. Add `workflow/extract_p1_yes.py` as the leaf implementation and help owner.

The leaf parser accepts:

```text
--pm, --prob-min PMIN
--pr, --prob-range PRANGE
-o, --output FILE
```

Its help summary should explicitly say that it extracts from `p1/done/out` and
does not queue the result.

## Filtering and output flow

1. Resolve the archive with `config.path(opts.dir, ["p1", "done", "out"])`.
2. Collect and sort regular `*.jsonl` files. Treat an empty archive as an error
   rather than silently producing an apparently complete empty extract.
3. Create a uniquely named temporary file with `tempfile`; do not use a
   predictable `/tmp/<output-name>` path.
4. Call `filter_results()` once per JSONL, appending every match to that
   temporary file.
5. Run `sort -u` with deterministic byte ordering (`LC_ALL=C`) into a second
   temporary file. For file mode, create this temporary file beside the
   requested destination so the final replacement is atomic.
6. Count the sorted lines, then either copy the completed data to stdout or
   atomically replace the requested destination.
7. Remove temporary artifacts on success or failure.
8. Report the number of source JSONLs and unique emitted pairs to stderr.

Do not silently skip an unreadable or invalid result file. Abort the extraction
and avoid installing an output file; otherwise the result could be mistaken for
a complete historical p1 extract. Staging all unsorted matches before emission
also prevents a failed run from leaving partial data on stdout.

For `--output`, refuse an existing destination unless the global `--force`
option was supplied. Even with `--force`, replace it only after extraction and
sorting complete successfully.

## Reuse and scope

- Reuse `src.filter.filter_results()` as-is. No native-reader change is needed.
- Keep archive traversal in the workflow command because the `.wf` layout is a
  workflow concern.
- Do not reuse `complete_pairs._filter_pairs_to()`: that helper requires a
  source pair-list restriction and also produces p2/p3 workflow artifacts.
- Do not add an `all` mode to `wf filter pairs`; that command retains its
  submission semantics and per-JSONL output identity.
- If probability argument definitions need to be shared to prevent drift,
  extract a small public argument-adder used by both workflow commands; do not
  couple extraction to submission checks or output naming.

## Files to change

| File | Change |
| --- | --- |
| `workflow/wf.py` | Register the new top-level `extract` command. |
| `workflow/extract.py` | Dispatch `extract p1`. |
| `workflow/extract_p1.py` | Dispatch `extract p1 yes`. |
| `workflow/extract_p1_yes.py` | Discover, filter, deduplicate, and emit archived p1 YES pairs. |
| `workflow/filter_pairs.py` | Optional narrow probability-argument reuse only. |
| `tests/test_workflow_cli.py` | Optional additions for help/dispatch behavior; do not create a dedicated test file. |

## Verification

Use an inline temporary workflow fixture; no dedicated test file is required.

1. Put two small independent JSONLs in `p1/done/out`, including one qualifying
   pair duplicated across the files.
2. Run stdout mode and confirm output is sorted, contains the duplicate once,
   and contains no status text.
3. Run a non-default probability band and compare each file's contribution
   with a direct `src.filter.filter_results()` call.
4. Run `--output`, confirm the destination contents match stdout mode, and
   confirm an existing destination is rejected without `--force`.
5. Include an invalid JSONL and confirm the command fails without installing or
   overwriting the requested output.
6. Confirm `git diff --check` and run an import/compile smoke check for the new
   workflow modules.
