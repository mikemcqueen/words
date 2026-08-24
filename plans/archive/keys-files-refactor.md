# Unify key/file handling and dispatch in `src/compare.py`

## Context

`compare.py` currently splits 2-way diff/ensemble inputs into `args.files` and n-way ensemble inputs into `args.keys`. That split was an arbitrary early choice, and it now blocks valid `-3` / `-5` combos: e.g. `-3 DIR -k k1,k2` keeps the two keys in `args.keys` (the 2-key fixup is gated on `not args.n_way`), and `load_files_from_keys` only loads those two — so 3-way discovery never sees the rest of the directory.

The working-tree change has loosened validation (n_way>3, removed `n_way > len(keys)` check, added anchor-key passthrough in `run_discovery`), but the dispatch in `main()` and the load step in `load_result_files` still bifurcate by `args.keys` vs `args.files`. The dispatch is also currently inconsistent across n_way values — 2-way and n>2-way use different criteria for "discovery vs explicit."

This refactor:
1. Makes `--keys` a pure UI affordance — all key arguments resolve to file paths up front; `args.files` becomes the single source of truth for "anchor files."
2. Establishes a single uniform dispatch rule that holds for every n_way: **discovery when `len(args.files) < args.n_way`, explicit when `len(args.files) == args.n_way`**.

## Unified dispatch rule

After parse normalization:
- `args.n_way`: always set — `2` (default), `3`, or `5`.
- `args.ensemble`: `None` (diff mode; only legal for `n_way == 2`) or one of `ALL/OR/AND/MAJORITY`.
- `args.files`: list of anchor file path strings, length `0..n_way`.
- `args.discovery_dir`: `Path` when `len(args.files) < n_way`; `None` when fully specified.
- `args.keys`: always `[]` after parse.

Dispatch in `main()`:
```python
if len(args.files) < args.n_way:
    rows = run_discovery(files, expected, args)
elif len(args.files) == args.n_way:
    rows = run_explicit(files, expected, args)
# > n_way is rejected in validation
```

### What each path does

`run_discovery(files, expected, args)`:
- `files` is `discover_files_all(args.discovery_dir)` — the full candidate pool.
- `anchor_keys = frozenset(key_from_path(f) for f in args.files)` — may be empty.
- If `args.n_way == 2 and args.ensemble is None`:
  - `len(anchors) == 0`: `diff.print_2way_diff_all_pairs(files, args)` (all pairs in dir).
  - `len(anchors) == 1`: `diff.print_2way_diff_anchored(anchor_key, files, args)` (FILE-anchored).
- Else (any ensemble — 2-way or n>2):
  - `ensemble.print_discovery_ensemble(args, files, anchor_keys=anchor_keys)`.

`run_explicit(files, expected, args)`:
- `files` is the dict of the `n_way` explicit anchor results, keyed by `key_from_path()`.
- If `args.n_way == 2`:
  - `rule = args.ensemble or 'ALL'`
  - `diff.print_explicit_2way_diff(files, args, ensemble_rule=rule)` — preserves today's behavior for both 2-way diff and 2-way explicit ensemble (the user's `-e RULE` is passed through, 'ALL' is the diff-mode default).
- Else (n_way > 2, ensemble guaranteed non-None by parse normalization):
  - `ensemble.print_discovery_ensemble(args, files, anchor_keys=frozenset(files.keys()))` — anchors are everything; only one combination exists. Replaces today's `run_explicit_nway`.

This collapses today's three runners (`run_explicit_2way`, `run_explicit_nway`, `run_discovery`) into two (`run_discovery`, `run_explicit`), each dispatching internally on `(n_way, ensemble)`.

## Target input combos

For any `-N` (with `N ∈ {2, 3, 5}`):
| Input | Anchors | `discovery_dir` | Path |
|---|---|---|---|
| `DIR` | 0 | `DIR` | discovery |
| `DIR -k k1` | 1 | `DIR` | discovery |
| `DIR -k k1,k2` | 2 | `DIR` | discovery (or explicit if `N==2`) |
| `DIR -k k1,...,kN` | N | None | explicit |
| `FILE_A` | 1 | `FILE_A.parent` | discovery (anchored) when `N==2`; discovery when `N>2` |
| `FILE_A FILE_B` | 2 | `FILE_A.parent` if `N>2`, else None | discovery if `N>2`, explicit if `N==2` |
| `FILE_A -k k1[,...]` | 1 + k count | `FILE_A.parent` | discovery or explicit per len-vs-N |

For 2-way with no ensemble, the existing diff cases (`DIR`, `FILE`, `FILE_A FILE_B`, `DIR -k k1[,k2]`) keep their current behavior because the rule `len(files) < n_way` puts them in discovery (DIR / FILE) or explicit (2 files), and the diff vs ensemble switch is governed by `args.ensemble`.

## Design decisions (confirmed with user)

1. `args.keys` is **cleared** after resolution. Display labels come from `key_from_path()` of the resolved files.
2. All anchor files must **share a parent directory**; otherwise error.
3. Discovery vs explicit is decided by **`len(args.files) vs args.n_way`** — uniform across all n_way values.
4. `-3 FILE_A FILE_B` (no DIR) discovers from `FILE_A.parent` with both files as anchors.
5. **`-2` (explicit) implies ensemble** even with 0 or 1 anchors, matching today's behavior: `DIR` is 2-way diff, `DIR -2` is 2-way ensemble (routed through `ensemble.print_discovery_ensemble`). The implementation tracks `args.nway_explicit` (set when the user passed `-2/-3/-5`) so the default-to-2 case stays in diff while explicit `-2` switches to ensemble.
6. **Scope is validation + branching only.** No edits to `src/diff.py` or `src/ensemble.py`. The OR-only behavior of discovery 2-way diff (`src/diff.py:198,221` hardcode `any_yes_1 | any_yes_2`) is preserved as-is. The refactor reuses every downstream function unchanged.
7. **`enforce_unique=True` for all key resolution** (behavior change). Today `compare.py:102` resolves a single `-k` without uniqueness enforcement; the refactor always enforces. Any prior `-k key1` invocation that matched multiple files will now exit.

## Implementation

### 1. `parse_args` in `src/compare.py` — replace lines 80–122 with the block below

Lines 124–128 (no-pairs early return) are **kept verbatim** — they sit after this block. Lines 130–146 (the old `len(args.files)==2` / "supply 1 or 2 paths" / late `--keys` checks) are **deleted** — their work is subsumed by steps 5–6 below.

```python
# 1. Resolve n_way (always end with 2, 3, or 5; default 2). Track whether the
#    user explicitly chose an n-way mode so we can distinguish `-2` (ensemble)
#    from the implicit 2-way default (diff).
args.nway_explicit = bool(args.two_way or args.three_way or args.five_way)
if args.five_way:    args.n_way = 5
elif args.three_way: args.n_way = 3
else:                args.n_way = 2   # covers -2 and the unset default

args.keys = [k.strip() for k in args.keys.split(',')] if args.keys else []

# 2. --keys requires exactly one positional path (the dir or the seed file)
if args.keys and len(args.files) != 1:
    parser.error('--keys requires exactly one path argument')

# 3. Determine discovery_dir candidate and seed anchor list
first = Path(args.files[0])
if len(args.files) == 1 and first.is_dir():
    discovery_dir = first
    anchor_files: list[Path] = []
else:
    anchor_files = [Path(f) for f in args.files]
    discovery_dir = anchor_files[0].parent

# 4. Resolve --keys against discovery_dir
for key in args.keys:
    anchor_files.append(Path(resolve_key(str(discovery_dir), key, enforce_unique=True)))
args.keys = []

# 5. All anchor files must share the same parent directory
for f in anchor_files:
    if f.parent != discovery_dir:
        parser.error(f'anchor {f} must live in {discovery_dir}')

# 6. Validate anchor count vs n_way
if len(anchor_files) > args.n_way:
    parser.error(f'-{args.n_way} accepts at most {args.n_way} anchor files; got {len(anchor_files)}')

# 7. Finalize args.files and discovery_dir
args.files = [str(f) for f in anchor_files]
args.discovery_dir = discovery_dir if len(anchor_files) < args.n_way else None

# 8. Implied ensemble rule. Any explicit -N flag (-2/-3/-5) or n_way > 2 implies
#    ensemble='ALL' when -e is absent. Plain `DIR` (no -N) stays as diff
#    (ensemble=None) and routes to diff.print_2way_diff_*.
if (args.nway_explicit or args.n_way > 2) and not args.ensemble:
    args.ensemble = 'ALL'
```

Then keep the existing downstream validation for `--pairs`, `--ensemble` MAJORITY parity, `--bad`, `--print-keys`, `--sort`, `--heap-size` — with the rewrites in §1a below for the three checks whose conditions depend on the old `args.files` semantics.

Notes:
- The old 1-key fixup (lines 101–103) and 2-key-without-n_way fixup (lines 105–109) collapse into the single resolution loop above.
- The old "fixup args.n_way for implied -2 cases" (lines 111–118) is no longer needed: n_way always lands at 2/3/5 via step 1, and ensemble inference is handled by step 8 using `args.nway_explicit`.
- `--ensemble MAJORITY` parity check at line 153 still applies.
- `--bad` rules retained: `--top 1`, or explicit 2-way diff, or explicit 2-way ensemble != ALL.
- The old early `--keys` check at line 96 (`len(args.keys) > 1 and not Path(args.files[0]).is_dir()`) is dropped: with the new model, `FILE_A -k k1,k2` is fine when combined with `-3` (3 anchors, explicit) and otherwise errors via step 6 (anchor-count check).

### 1a. Downstream validations to rewrite

These three checks at `compare.py:171,175-184,187-192` encode the old `args.files` semantics. Replace them with the conditions below, which key off `args.discovery_dir` / `args.n_way` / `args.ensemble`:

```python
# --print-keys: discovery only
if args.print_keys and args.discovery_dir is None:
    parser.error('--print-keys requires discovery mode')

# --sort: diff sort keys only for the 2-way diff path; ensemble keys otherwise.
is_2way_diff = (args.n_way == 2 and args.ensemble is None)
valid_sort = set(SORT_DIFF_KEYS) if is_2way_diff else set(SORT_ENSEMBLE_KEYS)
sort_lc = args.sort.lower()
if sort_lc not in valid_sort:
    choices = ', '.join(sorted(valid_sort))
    parser.error(f'--sort: invalid value {args.sort!r}; choices: {choices}')
if args.heap_size <= 0:
    parser.error('--heap-size must be > 0')

# --top vs --heap-size: only the no-anchor 2-way diff path uses the heap.
if (args.discovery_dir is not None
        and not args.files                # no anchors => all-pairs path
        and args.ensemble is None         # 2-way diff
        and args.top and args.top > args.heap_size):
    parser.error('--top cannot exceed --heap-size for directory 2-way diff discovery')
```

### 2. `load_result_files` in `src/compare.py:245`

```python
def load_result_files(expected, args):
    if args.discovery_dir is not None:
        files = discover_files_all(str(args.discovery_dir))
        for f in args.files:
            k = key_from_path(f)
            if k not in files:
                print(f'Error: anchor key {k!r} (from {f}) not found in '
                      f'discovery dir {args.discovery_dir}', file=sys.stderr)
                sys.exit(1)
    else:
        files = {}
        for f in args.files:
            k = key_from_path(f)
            if k in files:
                print(f'Error: duplicate key {k!r}', file=sys.stderr)
                sys.exit(1)
            files[k] = load_eval_results(f)
    if not files:
        print('No files found.', file=sys.stderr)
        sys.exit(1)
    for results in files.values():
        label_eval_results(results, expected, args.method)
        resolve_all_pair_labels(results)
    return files
```

User-input failure modes (missing anchor file, duplicate key) exit with a readable message rather than an `assert` traceback.

`load_files_from_keys` and `load_files_explicit` are deleted.

### 3. Runners in `src/compare.py:264-289`

Replace the three runners with two:

```python
def run_discovery(files, expected, args):
    p = args.discovery_dir
    print(f'\nDirectory: {p}')
    print(f'Found: {len(files)} file(s)\n')
    anchor_keys = frozenset(key_from_path(f) for f in args.files)
    if args.n_way == 2 and args.ensemble is None:
        if not anchor_keys:
            return diff.print_2way_diff_all_pairs(files, args)
        (anchor,) = anchor_keys
        return diff.print_2way_diff_anchored(anchor, files, args)
    return ensemble.print_discovery_ensemble(args, files, anchor_keys=anchor_keys)


def run_explicit(files, expected, args):
    if args.n_way == 2:
        rule = args.ensemble or 'ALL'
        return diff.print_explicit_2way_diff(files, args, ensemble_rule=rule)
    anchor_keys = frozenset(files.keys())
    return ensemble.print_discovery_ensemble(args, files, anchor_keys=anchor_keys)
```

### 4. Dispatch in `main()` (lines 306–312)

```python
if len(args.files) < args.n_way:
    rows = run_discovery(files, expected, args)
else:  # len(args.files) == args.n_way (validation guarantees no >)
    rows = run_explicit(files, expected, args)
```

### 5. `--bad` handling at lines 314–323

Unchanged in spirit; the `top_label` parsing still works because `print_discovery_ensemble` produces the same row format whether called from the discovery or explicit path.

## Critical files

- `src/compare.py` — primary refactor (parse_args normalization, load_result_files, dispatch, runners)
- `src/common.py:332` `resolve_key()` — reused; no changes
- `src/common.py:182` `key_from_path()` — reused; no changes
- `src/common.py:198` `discover_files_all()` — reused; no changes
- `src/ensemble.py:637` `print_discovery_ensemble()` — already accepts `anchor_keys`; no changes
- `src/diff.py` — `print_2way_diff_all_pairs`, `print_2way_diff_anchored`, `print_explicit_2way_diff` reused as-is

## Verification

Smoke matrix (run each, confirm parse succeeds and the output matches the expected path):

```bash
source ../.torch/bin/activate

# 2-way diff (n_way=2 implicit, ensemble=None) — preserves current behavior.
# Routes to diff.print_2way_diff_* (OR-only by construction).
python -m src.compare DIR -p PAIRS                                   # discovery, all-pairs
python -m src.compare FILE_A -p PAIRS                                # discovery, anchored
python -m src.compare FILE_A FILE_B -p PAIRS                         # explicit (rule='ALL' => OR+AND table)
python -m src.compare DIR -k k1 -p PAIRS                             # discovery, anchored
python -m src.compare DIR -k k1,k2 -p PAIRS                          # explicit (n_way=2, no -2 => diff)

# 2-way ensemble (n_way=2 explicit OR -e supplied). Routes to ensemble.print_discovery_ensemble.
# The -2/no-2 distinction is the critical regression test for issue #1.
python -m src.compare DIR -2 -p PAIRS                                # discovery ensemble (vs `DIR` above which is diff)
python -m src.compare DIR -k k1,k2 -2 -p PAIRS                       # explicit ensemble (vs `-k k1,k2` above which is diff)
python -m src.compare FILE_A FILE_B -e OR -p PAIRS                   # explicit ensemble
python -m src.compare FILE_A -2 -p PAIRS                             # discovery ensemble anchored on 1
python -m src.compare DIR -k k1,k2 -e AND -p PAIRS                   # explicit ensemble via keys

# 3-way ensemble — cases this refactor unblocks
python -m src.compare DIR -3 -p PAIRS                                # discovery, 0 anchors
python -m src.compare DIR -3 -k k1 -p PAIRS                          # discovery, 1 anchor
python -m src.compare DIR -3 -k k1,k2 -p PAIRS                       # discovery, 2 anchors  <-- prev broken
python -m src.compare DIR -3 -k k1,k2,k3 -p PAIRS                    # explicit
python -m src.compare FILE_A FILE_B -3 -p PAIRS                      # discovery, 2 file anchors  <-- prev broken
python -m src.compare FILE_A -3 -k k1 -p PAIRS                       # discovery, 1 file + 1 key  <-- prev broken
python -m src.compare FILE_A -3 -k k1,k2 -p PAIRS                    # explicit

# 5-way ensemble
python -m src.compare DIR -5 -p PAIRS                                # discovery, 0 anchors
python -m src.compare DIR -5 -k k1,k2 -p PAIRS                       # discovery, 2 anchors

# Error cases — confirm parse_error message is clean
python -m src.compare /dir1/A.jsonl /dir2/B.jsonl -3 -p PAIRS        # different parents
python -m src.compare FILE_A FILE_B FILE_C -p PAIRS                  # >2 anchors with default n_way=2
```

For each "previously broken" case, confirm `args.discovery_dir` is set, `files` is the full pool from `discover_files_all`, and the printed table includes combinations involving every supplied anchor.
