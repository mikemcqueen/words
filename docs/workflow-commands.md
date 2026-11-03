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

## Dictionary word review: `wf eval words`, `wf complete words`, `wf remove words`

See `plans/dict-verdicts.md` for the design.

### Notes

`wf eval words NAME` and `wf notes words NAME` create notes with two
checkboxes per word, YES and NO, none ticked. Only a ticked box is a verdict.
A word with neither box ticked is undecided: it is recorded as reviewed and
comes back in a later review.

`wf eval words` drops the words already classified, those in
`dict/done/yes.words` or `dict/done/no.words`, and refuses with `no
unclassified words in FILE` when none are left. Both files must exist; `wf gen
dict` creates them.

### Records

| File | Meaning |
|---|---|
| `dict/done/in/<stem>.reviewed.N` | every word the round's notes listed |
| `dict/done/in/<stem>.yes.N` | the round's YES words; written by every `complete words` round, never by `remove words` |
| `dict/removed/<stem>.removed.N` | the round's new NO words, removed from the dictionary; editable, for retraction |
| `dict/done/reviewed.words` | union of `*.reviewed.N` |
| `dict/done/yes.words` | union of `*.yes.N` |
| `dict/done/no.words` | union of `*.removed.N` |

`wf gen dict` and every publication rebuild the three derived word files and
the two dictionaries. Each round's `removed.N` holds every word that round
ticked NO, including words already NO from earlier rounds, so retracting a NO
means editing every `removed.N` that holds it. The hand-placed `dict/no.words` is a different file and
nothing here reads it.

### Guards

`wf complete words NAME` and `wf remove words FILE` both refuse:

- a **flipped verdict**: a word in `yes.words` ticked NO, or a word in
  `no.words` ticked YES. The message names the records that hold each
  standing verdict. Nothing overrides it. To turn a NO into a YES, take the
  word out of those `removed.N` records and run `wf gen dict` first.
- a **late verdict**: a new verdict for a word earlier rounds reviewed without
  one. Each such word prints `WORD YES was reviewed N times without a
  verdict` (or `NO`, the verdict this round gives it), and the command fails
  unless `--force-verdicts` is given. Re-recording a
  standing verdict the same way never warns.

A failed guard changes no dictionary record. For `complete words`, correct the
notes and run `wf -f complete words NAME`; `-f` only re-fetches the notes and
accepts nothing.
