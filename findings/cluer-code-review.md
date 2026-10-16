# Code review of the uncommitted cluer changes (2026-10-15)

Scope: only the uncommitted changes in `cluer/index.py`,
`cluer/query_index.py`, `findings/cluer-query-index-optimizations.md` and
`findings/tailscale.md`. The 81 unpushed commits on `working` were not
reviewed.

The change makes `-j` treat a hyphen as a word separator, the same as a
space (index format version 3).

## Findings

| # | file | line | finding | state |
|---|---|---|---|---|
| 1 | `tests/test_cluer_index.py` | 71 | `test_exact_clue_queries` fails | Fixed |
| 2 | `cluer/query_index.py` | 22 | `-j` and `-e` now disagree on hyphens; help text misleads | Fixed |
| 3 | `cluer/index.py` | 49 | Number ranges and multi-part compounds count as adjacent | Accepted |
| 4 | `findings/cluer-query-index-optimizations.md` | 180 | Old version 2 size next to new version 3 count | Skipped |
| 5 | `findings/cluer-query-index-optimizations.md` | 190 | Output changes list leaves out hyphen matches | Skipped |
| 6 | `findings/tailscale.md` | 55 | Missing blank line before `**New final/sync.sh:**` | Skipped |
| 7 | `findings/tailscale.md` | 47 | Permissions note leaves out the home directory | Fixed |

### 1. `test_exact_clue_queries` fails

The test still expects the old rule where only a space separates words.
It passes without the change and fails with it:

```
AssertionError: ['New York', 'York New', 'New York!', 'A New York', 'New-York']
             != ['New York', 'York New', 'New York!', 'A New York']
```

Fix: add `'New-York'` to the expected list. There is also no test of the
hyphen rule in `test_adjacent_queries`; add a hyphenated clue such as
`Five-digit code` there.

### 2. `-j` and `-e` disagree on hyphens

`-j "new york"` matches the clue `New-York`, but `-e "new york"` does not,
because `-e` compares `clue.lower()` with `new york` / `york new`. The help
text ("With -e, one or two words use the same separators") reads as if `-e`
accepts hyphens too, so a user would silently miss those clues.

Fix: either make `-e` accept hyphens, or say in the help that `-e` matches
spaces only.

### 3. Any word-hyphen-word counts as adjacent

Examples of pairs `-j` now treats as adjacent:

- `WWI years, 1914-1918` gives `1914,1918`
- `Well-to-do` gives `well,to` and `to,do`
- `Jack-o'-lantern` (apostrophes deleted) gives `jack,o` and `o,lantern`

Pair lists checked with `-f -j` (as in `docs/cluer-pairs.md`) will now
pass pairs like these as backed by a clue.

Accepted: spaces already give pairs like `a,new` from `A New York`, so
hyphens add more of the same kind of noise, and pair lists rarely hold
numbers.

### 4. Stale version 2 size

The bullet after the new version 3 line still gives the version 2 size
(2,242,576 pairs, 17.9MB), a line away from the version 3 count of
2,286,563 pairs. The size and build-time figures describe the space-only
table but read as current.

### 5. Output changes list is incomplete

The "Output changes from the substring test" section lists what was lost
and gained but not the hyphen matches version 3 adds, such as `five,digit`
from `Five-digit code`.

### 6. Missing blank line in tailscale.md

The new "If it still says yes..." paragraph runs straight into
`**New final/sync.sh:**`, so Markdown joins the label onto the end of that
paragraph. The blank line was lost when the Source block was replaced.

### 7. Home directory permissions

With sshd's default `StrictModes`, key login also fails if `~` itself can
be written by group or others. With `~/.ssh` at 700 and `authorized_keys`
at 600 but `/home/mike` at 775, login falls back to a password, and once
`PasswordAuthentication` is off the user is locked out. The note only says
to check `~/.ssh` and `authorized_keys`.
