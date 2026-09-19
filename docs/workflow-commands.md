# Workflow commands

## `wf best gen SENTENCE dfs.seed` / `dfs.best`

```text
wf best gen sN dfs.seed (-o LETTERS | -u LETTERS) -g G [-m M] [-n LIMIT] [-r DIR] [-f] [--dry-run]
wf best gen sN dfs.best (-o LETTERS | -u LETTERS) -g G [-m M] [-n LIMIT] [-r DIR] [--dry-run]
```

Both run one `dfs-anagrams` search (`workflow/best/generate.py`, `_run_dfs`).
Output goes to `results/sN/dfs.…` (or `-r DIR`) and is symlinked into the
target directory as `dfs.seed` or `dfs.best`.

In the paths below, `ROOT` is the workflow root and `TARGET` is
`ROOT/.wf/best/sN/[ou]-LETTERS/mM/gG`.

### Command line

```text
# dfs.seed
dfs-anagrams BAG --wfroot ROOT -t sN -m M -g G -p 10000000 -n LIMIT

# dfs.best
dfs-anagrams BAG --wfroot ROOT -t sN/[ou]-LETTERS/mM/gG -m M -g G -p 10000000 -n LIMIT
```

`BAG` depends on the letter-set form:

- `-o LETTERS`: `LETTERS`
- `-u LETTERS`: `"$(cat ROOT/.wf/best/sN/letters)" -u LETTERS`

`LIMIT` defaults to 1000000. The only difference between the two is `-t`.

### Files

Passed explicitly: none. The only file the workflow reads for the command
line is `ROOT/.wf/best/sN/letters`, and only for `-u`, whose contents become
the positional bag.

Loaded implicitly by `dfs-anagrams` from `--wfroot` and `-t`:

| File | dfs.seed | dfs.best |
|---|---|---|
| `ROOT/.wf/best/idx/wiki-merged.2.index` | yes | yes |
| `ROOT/.wf/dict/words.filtered` | yes | yes |
| `ROOT/.wf/best/sN/seed.mM.*.pairs` (seed tier) | yes | yes |
| `ROOT/.wf/classified/yes/yes.pairs` (YES tier) | yes | yes |
| `ROOT/.wf/classified/no/no.pairs` (excluded) | yes | yes |
| `TARGET/best.pairs` (BEST tier, optional) | no | yes |
| `TARGET/no.pairs` (excluded, optional) | no | yes |

`dfs.best` refuses to run when neither `TARGET/best.pairs` nor
`TARGET/no.pairs` exists, since it would repeat `dfs.seed`.
