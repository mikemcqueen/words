# Brainstorm: Multi-Pass p1 Filtering & Range Tracking

## The Core Problem

`complete pairs` runs `filter_results(..., pmin=0.9)` — the "top 10%" YES pass — then moves the `.jsonl` to `p1/done/out/`. That `.jsonl` still contains valuable signal in the [0.8, 0.9) range (and others), but:

1. **The `.jsonl` is treated as "done"** — semantically finished, even though it's only been mined once.
2. **No record exists of which probability ranges have been extracted** from a given `.jsonl` into p2/queued.

---

## Proposed Design

### 1. Encode pmin + prange in the YES filename

Current: `{pairs_name}.p1.yes`  
Proposed: `{pairs_name}.p1.{pmin_int}.{prange_int}.yes`

Where pmin and prange are stored as **integers in hundredths**:
- 0.9 → `90`, 0.1 → `10`: `s8.1.pairs.4.pairs.p1.90.10.yes`  (range [0.9, 1.0))
- 0.8 → `80`, 0.1 → `10`: `s8.1.pairs.4.pairs.p1.80.10.yes`  (range [0.8, 0.9))

This makes each output file **self-describing** and the set of extracted ranges **queryable by filename glob** — no external database needed.

Parse helper: split on `.p1.`, take next two tokens → reconstruct `pmin=int/100`, `prange=int/100`.

### 2. Where do .jsonl files live?

**Keep them in `p1/done/out/`** — that's the right place. "Done" means the mandatory first filter pass is complete, not that the file can never be mined again. The `.jsonl` is the canonical artifact; it belongs in the archive.

The key insight: the `.jsonl` moves to `p1/done/out/` on `complete pairs`. Subsequent refilter passes read it from there and write new `.p1.{X}.{Y}.yes` files to `p2/queued`.

### 3. New command: `wf filter pairs`

```
wf filter pairs <jsonl-file> [--pm 0.8] [--pr 0.1]
wf filter pairs <jsonl-file> [--prob-min 0.8] [--prob-range 0.1]
```

Mirrors the flags in `src/filter.py` (`--pm`/`--prob-min`, `--pr`/`--prob-range`). Defaults: pmin=0.9, prange=0.1.

1. Accepts a `.jsonl` from `p1/done/out/` (or path-resolved)
2. Computes the encoded filename: `{pairs_stem}.p1.{pmin_int}.{prange_int}.yes`
3. Checks `p2/queued/`, `p2/eval/`, `p2/done/in/` for a file with that encoded name — **refuses if the range was already submitted** (unless `--force`)
4. Calls `filter_results(..., pmin=pmin, prng=prange)` → writes to `p2/queued/`

### 4. Update `complete pairs` to use encoded filename

Change `_filter_results_to` in `complete_pairs.py:74` to pass the encoded filename:
- pmin=0.9, prange=0.1 (implicit, since 0.9+0.1 = 1.0) → `p1.90.10.yes`

This makes the initial pass consistent with subsequent refilter passes.

### 5. Query: "What ranges from this .jsonl have been submitted?"

Either as a `show` subcommand or inline in `filter`:

```
wf show p1 done          # list all .jsonl files in p1/done/out/
wf show p1 done <stem>   # show which ranges have been submitted for a specific .jsonl
```

Implementation: scan `p2/queued/`, `p2/eval/`, `p2/done/in/` for files matching `{stem}.p1.*.*.yes`. Parse pmin/prange from each match. Report gaps (unextracted ranges).

---

## Decisions Made

1. **Filename format**: hundredths integers — `p1.90.10.yes` for [0.9, 1.0), `p1.80.10.yes` for [0.8, 0.9).
2. **NO pass**: `.p1.no` naming unchanged — no probability range to track.
3. **Migration**: existing `.p1.yes` files stay as-is; new naming applies going forward only.
4. **Command name**: `wf filter pairs` with `--pm`/`--prob-min` and `--pr`/`--prob-range` flags (matching `src/filter.py` convention).

---

## Files to Touch

| File | Change |
|------|--------|
| `workflow/complete_pairs.py` | Update YES output filename to include pmin/prange encoding (`p1.90.10.yes`) |
| `workflow/filter_pairs.py` | New file: `filter pairs` subcommand |
| `workflow/wf.py` | Register `filter` command |
| `workflow/config.py` | May need to update layout expectations for new filename suffix pattern |
| `src/filter.py` | No changes needed |
