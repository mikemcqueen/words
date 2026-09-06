# A workflow-managed dictionary

Supersedes `plans/managed-dict.md`. That plan and its review
(`plans/managed-dict-review.md`) stay on disk; the review cites v1 by section
and line, and the last section here says where each of its findings landed.

## Summary

Turn the dictionary into a derived tree at the root of the workflow: a
hand-placed base, a set of hand-editable removal generations, and a derived
dictionary used by workflow-configured invocations of its four consumers. Add
the commands that record a removal and rebuild the derived file, so a word list
can be submitted by hand today, with no review mechanics -- and repoint the one
constant in nutrimatic so
the derived file is what the `*-segments` tools filter by too.

## Context

Some single words in `words.big` are junk, and they pollute every search that
can spell them. Rejecting one today means editing the dictionary the searches
read -- which is also the file `Inputs.dictionary` (`workflow/best/state.py:676`)
dates the frontier against, so a hand-edit is indistinguishable from someone
replacing the base wholesale. It is already happening to the legacy Nutrimatic
dictionary, which has a `.bak` and two `.removed.N` files beside it from
`dict_remove.py` runs.

The rub is that removing a word invalidates the dictionary, which invalidates
the DFS, which costs hours -- and the operator does not want to re-search on
every removal. So the removal has to be cheap to *record* and cheap to *apply to
the next search*, leaving the re-search a thing the operator chooses at the
bottom row of the status table.

This plan is self-contained. `plans/top-solo-words.md` describes the intended
frontier follow-up -- the ranked candidate list (`top.solo-words`) -- but still
uses the v1 dictionary layout and commands and must be revised for v2 before it
can be implemented. A queued/eval/complete cycle for reviewing that list
follows after. Both are anticipated here in the layout and command shape, and
neither is built.
Until they land, the candidate list is produced by hand and the removals are
submitted with §3's command:

```bash
top-segments --solo-words --wfroot ROOT -y TARGET/dfs.seed \
  > TARGET/top.solo-words
```

`TARGET/dfs.best` replaces `TARGET/dfs.seed` when BEST is the selected current
source, and `-n COUNT` may be added for a cutoff. `-y` is intentional: a
confirmed-YES pair is not itself counted, while the other segments on its row
remain eligible. `--wfroot ROOT` is what loads `words.filtered` and the
workflow exclusions. `TARGET/top.solo-words` is the candidate file later passed
to `wf remove words`.

### What changed from v1

Five things, each because a premise turned out to be false.

- **`top-segments` already reads the dictionary**, by a hardcoded name
  (`source/pair-exclusions.cpp:17`). v1's §5 and "Not in scope" both rest on it
  not doing so. Repointing that one constant is now in scope (§6), and it is
  what makes the derived dictionary mean the same thing on both sides.
- **The standing set is gone.** v1 kept both a folded `no.solo-words` and an
  immutable per-submission archive, and the two disagree the moment a word is
  retracted. The generations are now the only source of truth, and they are
  hand-editable by design.
- **The dictionary moves to the root.** A future p1 filter -- dropping
  submitted pairs that contain junk words -- reads it too, so it is not a BEST
  input. Moving it costs nothing extra now, because the nutrimatic constant is
  already being edited for the rename.
- **The tree makes room for a review cycle.** `dict/` is where a
  `queued/eval/done` stage will land, writing the same records by the same
  code as §3's command.
- **What was reviewed is recorded separately from what was removed.** A word
  judged good stays in the dictionary, so only a per-round reviewed input stops
  it returning as a candidate for ever.

---

## Design

### 1. `.wf/dict/`: base, verdicts, derived

| file | what it is |
|---|---|
| `words.big` | hand-placed base at `$WFROOT/.wf/dict/words.big`; it may be a symlink to an operator-chosen source. **Never written by the workflow.** |
| `removed/<input>.removed.N` | one round's removals: sorted-unique bare words. Mutable, and the only record of what is removed |
| `words.filtered` | derived: `words.big` minus the union of `removed/`; workflow-configured dictionary consumers read this |
| `.words.filtered.gen` | generation marker, advanced by every rebuild |
| `done/in/<input>.reviewed.N` | one round's submitted word input: every word it covered, checked or not. Never rewritten by workflow commands; manually correctable |
| `done/reviewed.words` | derived: the union of `done/in/`. What a candidate listing subtracts so nothing is re-reviewed |

The name is `words.filtered` and not `words.dfs` because four tools read it in
their workflow-configured modes, not one: `dfs-anagrams` when passed `--dict`,
and `top-segments`, `first-segments` and `filter-segments` when passed `--wf` or
`--wfroot`. Naming a shared artifact after one of its consumers is how v1's
divergence started.

The base keeps its name and its managed pathname is
`$WFROOT/.wf/dict/words.big`. It may be a symlink, but the symlink target is an
operator choice and is not part of the workflow contract. The operator may
replace the base by hand; the workflow never writes it.

Base replacement has an mtime contract: the followed target must end newer than
`.words.filtered.gen`. Repointing the symlink to older content or preserving an
old timestamp with `cp -p` is unsupported unless the replacement target's mtime
is advanced afterward. Content fingerprinting and backward-mtime replacement
detection are outside this plan.

**Derivation.** `sort -u` over the generations into a temporary file, then
`comm -23` against the base:

```python
with tempfile.TemporaryDirectory() as tmp:
    aggregate = setops.merge(generations, Path(tmp) / "removed")
    setops.diff(base, aggregate, derived, stable_mtime=True)
```

Both through `workflow/setops.py`, never as a raw shell redirect: the module
runs every operation under `LC_ALL=C` (`setops.py:5-10`, without which `comm`
yields a silently wrong difference rather than an error), writes aside and
renames so the derived file is never observed half-written, and does the
content compare `stable_mtime` needs.

`generations` is the directory read in sorted order, filtered by the ordinal
pattern below -- **not** a bare glob. The retraction path is "open a generation
file in your editor", and vim leaves `foo.removed.3~` while emacs leaves
`#foo.removed.3#`; under a bare glob those get unioned back in and the
retraction silently does not take. With no generations the aggregate is an empty
file, so a tree with nothing removed derives a byte-for-byte copy of the base.
`setops.merge` raises on an empty source list, so that case creates the empty
aggregate directly rather than calling it.

Generations may overlap. Editing one file therefore removes only that file's
contribution to the union: a word is restored only after it has been removed
from every generation that contains it. Making records non-overlapping or
providing cross-generation retraction machinery is outside this plan.

`done/reviewed.words` follows the same rule. Its inputs are `done/in/` entries
read in sorted order and filtered by `.*\.reviewed\.([1-9]\d*)`; editor backups
and other stray names do not become reviewed verdicts. With no matching
reviewed inputs, the rebuild creates an empty aggregate directly instead of
calling `setops.merge` with an empty source list.

**Scale.** Each aggregate passes one pathname per selected record to a single
`sort -u` invocation. The supported operating scale is hundreds of generation
files, not tens of thousands; the host's `ARG_MAX` is the accepted hard limit.
This plan adds no batching or compaction mechanism. Rework aggregation only if
actual use approaches that scale.

**`stable_mtime=True` is load-bearing, not tidiness.** `Inputs.dictionary`,
`frontier_outdated`'s `"dictionary changed"` reason, and both DFS freshness
predicates date against the derived file, so a removal that removes nothing --
a word not in the base -- must leave `words.filtered` byte-identical and cost
nothing downstream. The comparison has to be over content, not line count:
`comm -23` is not monotone, and one rebuild that both applies a new removal and
picks up a hand-retraction lands on an identical line count with different
bytes. `setops._place` already does exactly
this and says so (`setops.py:31-37`).

**Prepared batch placement.** Dictionary commands prepare every output before
moving any of them into place. The staging directory is private and lives
under `dict/`, not in the system temporary directory, so every final move stays
on the destination filesystem. A prepared placement records its staged path,
destination, replacement policy, and whether `stable_mtime` found changed
content. Preparing the batch runs every `sort`, `comm`, write, and content
comparison. It does not rename a final destination.

One preflight then checks the complete batch. Each new archive destination must
have no directory entry of any type; an archive is never overwritten. Each
replaceable derived destination must satisfy its existing
absent-or-regular-file contract. A failed check discards the staged batch
before the first final move. Byte-identical derived outputs are also discarded
at this point and omitted from the commit, preserving their content mtimes. The
replaceable marker destination has the same contract. Its temporary file is
written after the other staged files so its mtime dates the completed
preparation, and it is always included even when both derived files are
unchanged. Normal exceptions remove the private staging directory; a staging
directory left by process or system failure matches neither archive input
pattern and is ignored.

The batch commits with same-filesystem atomic renames in this order: the
removal archive, the reviewed-input archive, `done/reviewed.words`,
`words.filtered`, and `.words.filtered.gen`. `wf gen dict` uses the same path
without the two archive placements. The generic `stamp`, `generated`, and
`mark_generated` helpers move from `workflow/best/state.py` to
`workflow/generation.py`. BEST imports the shared API, and the dictionary module
uses `stamp` to include the marker in its prepared batch. Without the marker the
staleness row of §5 is a permanently stuck row: submit a
word that is not in the base, the generation lands with a new mtime,
`stable_mtime` correctly pins the derived file, and "input newer than output"
fires forever with the one write that would clear it suppressed. This is the
same split `top.segments` already uses -- the artifact's own mtime answers
"did the set change?" for readers downstream, the marker answers "has this been
generated since its input moved?"

The preparation and preflight move validation, derivation, cross-filesystem,
and destination-shape failures out of the commit interval. They do not make
multiple renames across `removed/` and `done/in/` atomic. An unexpected rename
failure or interruption can still stop that short sequence; automatic orphan
detection, recovery, and a transaction or round-directory format are outside
this milestone. A crash between placing `words.filtered` and its marker leaves
the marker behind, so the staleness row offers the rebuild again.

**Dictionary paths.** The dictionary is root-global, not a property of one BEST
target, so `config` is the single owner of every path under it. It defines the
base and derived names and exposes `dictionary(root)`, `base_dictionary(root)`,
`removals(root)`, `reviewed_inputs(root)`, and `reviewed_words(root)`.
`Target.dictionary` is removed rather than repointed, and no
`Target.base_dictionary` or `Target.removed_dir` is added. `_dfs_inputs`
(`generate.py:112`) gets the derived path from `config.dictionary(target.root)`
and passes it to `dfs-anagrams --dict` (`generate.py:204`).

The accessors resolve through `config.path`, so a missing `dict/` is a hard
layout error, but returning a file path does not itself require that file to
exist. Read boundaries do that. `Inputs.base_dictionary` and
`Inputs.dictionary` require their respective `config` paths with
`fs.raise_if_not_file`. BEST status has G0 guard rows that touch those accessors;
their decline provides `base_dictionary` and `dictionary` to the later rows'
`requires=` declarations. Missing dictionary inputs therefore get the ordinary
`file not found` diagnostic rather than a migration-specific status message.

Generation is deliberately different. `wf gen dict` requires `words.big`, but
`words.filtered` is its output and is optional on entry. It reads the prior
destination with `fs.optional_file`, so an absent file is the first-build case
while a directory or dangling symlink is still a broken tree. It records
whether the derived path existed for the report, then places the new file there.
The first `wf remove words` uses that same rebuild path, so it also creates
`words.filtered` rather than rejecting its absence.

### 2. Layout, initialization, and the stage that is coming

`dict` moves out of `_BEST["parts"]` (`config.py:106-108`) and becomes a
top-level part in `CONFIG_LAYOUT` (`config.py:115-125`), beside `classified`
and the phases. `best/idx` stays where it is; indexes really are BEST inputs.

```python
_DICT = {
    "description": "The shared Nutrimatic dictionary and its removals",
    "content": True,
    "parts": {
        "removed": {
            "description": "applied word-removal generations"
        },
        "done": {
            "description": "completed removal rounds",
            "content": True,
            "parts": {
                "in": {
                    "description": "per-round submitted word inputs"
                }
            }
        }
    }
}
```

`"content": True` is required on `dict`, not decorative: `LayoutArgs.has_missing` is
`not (parts and (is_leaf or has_content))`, so without it `wf show dict` would
stop listing and demand a subpart. `_PHASE1["done"]` carries it for exactly this
reason -- a part that holds both files and subparts.

`removed` carries no `stable_mtime` flag. Layout parts are directories
(`config.path` calls `fs.raise_if_not_dir` on each consumed part,
`config.py:189-192`; `init.ensure_layout` `ensure_dir`s each one,
`init.py:22-26`), and the flag has nothing to attach to now that there is no
folded standing file. The derived file's `stable_mtime` is passed at the one
call site in the rebuild, which is itself the single chokepoint -- so a caller
that knows only which errand it is running still cannot get it wrong, which was
the property `fold_classified` looked the flag up from the layout to get.

`workflow/init.py` needs no new code: `ensure_layout` creates `dict/`,
`dict/removed/` and `dict/done/in/` from the layout, and `wf init` is idempotent
(`init.py:8-19`), so an existing tree needs a re-run and nothing else.
Neither `words.filtered` nor `done/reviewed.words` is created by `wf init` --
they are derived, and deriving them needs inputs that init has no business
requiring. `done` has `"content": True` because it holds the derived aggregate
as well as the `in/` subpart, matching the phase-one done layout.

**Reserved, not built.** The review cycle lands `queued` and `eval` as further
parts of `_DICT`, and `out` beside `in` under `done` for artifacts produced by
the review. Its `complete` writes exactly what §3's command writes -- the
unchecked words into `removed/`, the submitted word input into `done/in/` -- so
the two paths differ in where the verdict came from and in nothing after that.
Nothing here needs to anticipate it beyond leaving the names free, because
`wf init` grows a tree by re-running.

`show._build_aggregates` walks the whole layout and collects every part named
`queued` or `eval`. Once the future dictionary review adds those parts,
`wf show all queued` and `wf show all eval` intentionally include dictionary
review work beside the phase work. That is the desired meaning of `all`, not a
layout side effect to suppress.

Why the cycle will not simply *be* the verdict store: the generations are
hand-editable by design -- that is the retraction path -- and a stage's
`done/in` is the workflow-immutable record of what a completed round consumed.
Normal commands never rewrite it; a deliberate operator correction may. Same
directory cannot hold both contracts, which is the same reason `classified/`
lives beside the phases rather than inside one
(`config.classified`'s docstring, `config.py:195-202`).

### 3. Command surface

Registered in the root dispatcher (`workflow/wf.py:11-29`), which is verb-first
and scope-second throughout -- `wf submit p1|p2`, `wf eval p1|p2|p3`,
`wf complete p1|p2`, `wf classify yes|no`. The scope names the command's object:
words are removed and reviewed, the dictionary is generated and shown.

**`wf remove words WORDS-FILE`**

The verdict-recording path. Mirrors `workflow/classify.py` in behavior:

1. Inspect the typed path before resolving or reading it. If `WORDS-FILE`
   itself is a symlink, fail with `symlink input not allowed: <path>` before
   allocating or writing an archive. Otherwise read it, stripping a leading
   count prefix (`^ *[0-9]+ `) off each row and discarding blanks. So a slice
   of a `top-segments --solo-words` listing (`  1234 word`) can be pasted in
   unedited, and so can a bare word list.
2. Check the shape: every remaining row must be `[a-z]+`. `words.big` is
   verified all-lowercase (363,389 lines, `sort -cu` clean, zero non-`[a-z]`
   lines), so the guard is well defined, and it is what stops a `--pairs` slice
   pasted by mistake from folding a row with a space in it into a generation
   that then never matches anything. Before any archive is written, resolve the
   dictionary paths through `config` and require `words.big`; do not require
   `words.filtered`, which the rebuild may be creating for the first time.
3. Allocate N as one sequence across both archive namespaces: take one more
   than the maximum matching `.removed.N` in `removed/` or `.reviewed.N` in
   `done/in/`, reusing the directory enumerations required for prospective
   derivation. Prepare two independent sorted-unique archive files under the
   private `dict/` staging directory. They have the same content for this direct
   command but must not be hard links, because the removal record is
   hand-editable independently of the reviewed-input archive.
4. In that staging directory, run the complete prospective derivation with the
   existing selected records plus the prepared sets: the prior and prospective
   removal unions, the prospective reviewed union, and `words.big` minus the
   prospective removal union. This is the same derivation `wf gen dict` runs,
   with the not-yet-archived submission supplied explicitly. It also produces the
   new-to-removal-union delta used by the report. With no prior records, create
   an empty aggregate directly rather than calling `setops.merge` with an empty
   source list.
5. Prepare placements for both archives, both derived outputs, and the marker.
   Preflight every destination as one batch: archives must be absent, while
   replaceable outputs must satisfy their normal file contracts. Do not rerun
   a prospective set operation after this point.
6. Commit the prepared placements in the order specified in §1. Every move is
   an atomic rename on the `dict/` filesystem.

The order is validate, allocate, prepare everything, preflight everything,
archive, publish derived outputs, mark. An ordinary failure therefore leaves no
durable record and no changed output. The residual crash-between-renames window
is explicitly out of scope. Once archives have landed, they precede derived
outputs so a later `wf gen dict` can heal interrupted derived-output
publication; a derived dictionary must never reflect a submission that exists
nowhere in its source records.

**Neither command takes a target, and neither names a next one.** The removal
is global -- `dict/` sits at the root and one removal applies to every target --
so there is no address to scope it by, and `_action_target`
(`commands.py:293-309`) requires all four components of one (SENTENCE, `-a/-l`
LETTERS, `-g` N, `-m` N) besides. Nor does either command suggest what to run
next: it cannot know that a target is involved at all. The operator may have
removed a word to unblock a particular search, in which case `wf best status`
is where they will find the step it opened up, or to feed a future p1 eval
filter, in which case no BEST target is in play. The report says what changed
and stops there.

The report covers what an operator cannot see otherwise:

```
12 words submitted, 9 new to the removal union, 7 newly removed from the dictionary
recorded as generation 3 -> dict/removed/solo-s7-round2.txt.removed.3
words.filtered 363389 -> 363382
```

The third count is `(new to the removal union) ∩ words.big`, not the whole submission
intersected with the immutable base. A previously removed word remains in
`words.big`, so counting all submitted base words would make a resubmission look
effective. Intersecting only the removal-union delta answers whether this
submission logically removed the word from `words.filtered`. The final line
still reports the complete rebuild, including any hand-edits it picks up at the
same time.

The last line renders one of three cases, because there is not always a
before-count to render:

```
words.filtered 363389 -> 363382     the count moved
words.filtered 363389 (new)         no derived file existed
words.filtered 363389 (unchanged)   the rebuild was byte-identical
```

`(new)` is the first `remove words` or `gen dict` after `wf init` -- init
deliberately does not create `words.filtered` (§2), so the state is reachable
and is not an error. `(unchanged)` is what a submission of words that are not
in the base looks like, and rendering it is the report's half of the no-op §4
describes: `stable_mtime` leaves the file alone, so `363389 -> 363389` would
report a write that did not happen.

Which of the three it is cannot come from comparing the counts. A rebuild that
applies a new removal and picks up a hand-retraction in the same pass lands on
an equal count with different bytes (§1) -- a move, and it renders as the
arrow. Byte-identity is what `stable_mtime` already decides inside
`setops._place` (`setops.py:44-52`), which returns `dst` either way and so
does not currently say which branch it took. The rebuild needs that signal, but
the existing `merge`, `fold`, `diff`, and `common` API stays unchanged: callers
already rely on each returning a `Path`. Internally `_place` returns a
`Placement(path, replaced)` value; the existing operations unwrap and return
its `path`, while new `merge_report` and `diff_report` variants return the full
placement only to callers that ask for it. That exposes the comparison already
made without a second `filecmp` in the caller or a repo-wide return-type change.
The counts themselves are `fs.line_count` before and after.
`done/reviewed.words` renders the same three cases on its own line.

Contradiction has no analogue here. There is no positive word verdict for a
removal to collide with, so nothing like `classify.OPPOSITE` is needed.

**`wf gen dict`**

Rebuild `done/reviewed.words` first, then rebuild `words.filtered` from
`words.big` minus `removed/`, advance the dictionary marker, and report the line
counts. `words.big` is a required input;
`words.filtered` is an optional prior destination, so its absence selects the
`(new)` report case instead of failing. Idempotent, seconds, safe to run at any
time. It is what §5 offers, and it is the second half of a retraction: there is
no un-remove command, so retracting a word means editing every generation file
where it occurs, or deleting those files, and then running this.

The command dispatches directly to this rebuild. It does not call
`best.state.derive_state`, `walk_rows`, or `_dictionary_stale`; those are BEST
status machinery. In particular, a missing `done/reviewed.words` is an optional
prior output here: `wf gen dict` recreates it once and never recommends
`wf gen dict` from inside the command.

The dictionary module is also the diagnostic boundary for its set operations.
It catches `subprocess.CalledProcessError` from `sort` or `comm` and raises a
concise `ValueError`, without exception chaining, that names the failed
operation and its relevant inputs. The subprocess's own diagnostic may precede
that message, but no Python traceback reaches the operator. Because
`setops._place` publishes only after the subprocess succeeds, failure preserves
both prior derived outputs and does not advance `.words.filtered.gen`.

Spelled `gen` because `words.filtered` is the same kind of object as
`dfs.seed`, `top.segments` and `dfs.best` (`commands.py:126`) -- derived, placed
with `stable_mtime`, covered by the shared generation-marker contract, and
offered by a staleness row.
Spelled at the root rather than under `best` because the artifact is no longer a
BEST input. `dict` and not `words` because it names the artifact; `wf show dict`
stays the noun view of the same tree.

**Reserved.** `wf submit words`, `wf eval words`, `wf complete words` are the
review cycle. The scope string is `words` while the layout part is `dict`, so
the action's part name has to come apart from the key it is registered under.
That is one attribute, not a lookup table -- `submit.py:37` and `eval.py:46`
pass the phase string straight into `config.path` -- but the same string is also
interpolated into prose (`submit.py:28` renders `{phase}/queued`), so the two
uses have to be split or `wf submit words` will advertise `dict/queued`.
Summaries and errors should say `words`; path strings should say `dict/`,
because a path is a real thing an operator will `cd` into and must not be
aliased. The layout already anticipates the mapping: `config.py:10` carries
`# TODO: "alias": "phase1"`. Landing `words -> dict` there rather than as a
bespoke attribute gives `wf show`, dispatch and the actions one mapping to
read, and gives `dispatch.registry_key`'s partial-match TODO (`dispatch.py:15`)
a defined precedence -- an alias is an exact key, so it resolves before any
prefix match and adding one cannot change what an existing prefix means.

**Notes, when the cycle lands.** The words review uses `--checkbox` (one box per
line) where p2 uses `--two-checkboxes`, so `notes.create`'s hardcoded option
list (`notes.py:84`) becomes a parameter, and the parse side takes the matching
flag the way `p2_extract` does (`p2_extract.py:64-71`). The polarity inverts: a
**checked** word is good and is what survives; the **unchecked** words are the
removals, read with `--type NONE`. This deliberately re-adopts what p2 walked away from -- `p2_close.py:5-8` records
that a NO is now "a reviewer's explicit verdict rather than the absence of a
YES" -- so `complete words` must report the ratio and refuse an implausible one
without a force flag: an abandoned note otherwise removes every word below where
the reviewer stopped. What makes it recoverable rather than merely defensible is
§4: the round lands as one generation file, that file is hand-editable, and
`wf gen dict` is idempotent.

### 4. What is recorded: the generations and the done-set

Two records, answering two questions. `removed/<input>.removed.N` says what was
removed and by which round; `done/in/<input>.reviewed.N` archives the submitted
word input that the round consumed.

#### `removed/<input>.removed.N`

The `.removed.N` suffix and its allocation come from `dict_remove.py`; the stem
and the mutability do not.

- **Name.** The submitted file's own basename with `.removed.N` appended --
  `solo-s7-round2.txt` becomes `solo-s7-round2.txt.removed.3`. Only the suffix
  comes from `dict_remove.py:16`, which appends it to a fixed dictionary name;
  the stem is the input's, because the stem is the provenance. It answers "where
  did this come from" with the name the operator gave it, which no generated
  name can reconstruct. `Path(argv[0]).name` is the stem -- an empty name, `.`,
  or `..` is refused. Before resolving or reading it, a direct symlink input is
  refused with `symlink input not allowed: <path>`; a symlink target's basename
  therefore never becomes archive provenance.
- **Counter.** N is one sequence across both archive namespaces, not per stem:
  allocated by scanning `removed/` for `.*\.removed\.([1-9]\d*)` and
  `done/in/` for `.*\.reviewed\.([1-9]\d*)`, then taking the combined
  `max + 1` (`dict_remove.py:22-30`, widened past its fixed stem and single
  directory). Global is what makes it a *generation* counter
  -- N alone orders every submission and identifies one -- and it keeps two
  submissions of the same filename from colliding, where a per-stem counter
  would give them each their own `1` and lose the ordering. Names that do not
  match are ignored rather than parsed, which is the same filter the derivation
  applies (§1). `[1-9]\d*` is the workflow's own ordinal pattern
  (`state.py:498-500`), so no new convention is introduced. The allocated name
  is unreachable by construction: the scan sees the name of every directory
  entry, regardless of whether it is a regular file, directory, or symlink, so
  anything already named with ordinal N makes the allocator choose at least
  N+1. `fs.raise_if_any_exist` stays as a cheap defensive assertion before
  placement, not as an expected operational branch. A resubmitted archive is
  not a special case:
  `foo.txt.removed.3` submitted again lands as `foo.txt.removed.3.removed.9`,
  which is ugly and correct, and the greedy stem match still reads its
  generation as 9. Reading both namespaces also prevents a surviving half of
  an interrupted round from having its ordinal reused; it does not otherwise
  diagnose or recover that round.
- **Content.** The normalized submission -- stripped, shape-checked,
  sorted-unique -- not a delta against what is already recorded. v1 archived
  deltas so the generations would exactly partition a separate standing set;
  with no standing set there is nothing to partition, and the file can be what
  its stem says it is. Overlapping resubmissions overlap, and `sort -u` absorbs
  them.
- **Mutability.** These are hand-editable, and that is the whole retraction
  path. Deleting a generation removes its contribution; trimming one removes a
  word from that generation. Because overlapping resubmissions are preserved,
  the word is effectively restored only when it has been removed from every
  generation containing it. The next `wf gen dict` is the second half. The cost
  is that an edited generation stops being a record of what was submitted --
  provenance survives at word granularity, history does not -- which is the
  price of one source of truth, and the operator's own file is presumably still
  wherever they made it.
- **A submission that adds nothing still allocates a generation**, unlike
  `dict_remove.py`, which returns before creating the file when nothing was
  removed from the dictionary (`:95-97`). Here the file records what was
  submitted, and a submission of already-removed words is a fact about the
  operator's list worth keeping. What it does not do is move `words.filtered`,
  which `stable_mtime` handles. The report still says the submission was a no-op.
- **Ordering.** Validate, allocate, prepare, preflight, archive, publish, mark.
  The prospective derivation and every destination check must succeed before
  anything durable moves; after that, the durable verdict lands before either
  derived output.

Nothing reads a generation back as an undo log. `dict_remove.py`'s `.removed.N`
files are the only record of what its in-place-edited dictionary used to hold;
here the base is never written and `words.filtered` is regenerable, so a
generation is a verdict, not a diff to replay.

#### `done/in/<input>.reviewed.N`, and `done/reviewed.words`

The generations say what was *removed*. They cannot say what was looked at and
**kept** -- and a word judged good stays in `words.filtered`, so without a
second record it comes back at the top of every candidate listing, round after
round. `done/in/<input>.reviewed.N` is that record: the normalized submitted
input for one round, holding every word the round covered, checked or not, under
the same N and the same stem as the `removed/` file it was decided alongside.
`done/reviewed.words` is their union, derived. The derivation accepts only
names ending `.reviewed.<positive int>`, mirroring the ordinal filter for
`removed/`; backup files and other stray entries in `done/in/` are ignored. If
there are no matching inputs, it writes an empty `done/reviewed.words` without
asking `setops.merge` to merge an empty source list.

**Why this is not the redundancy we just removed.** `removed/` is mutable state
and `done/in/` is an input archive never rewritten by normal workflow commands,
which is exactly the pairing that made v1's `no.solo-words` untenable -- so the
test is whether the two answer the *same* question. They do not. A retraction
changes whether a word is removed; it cannot change whether the word was
reviewed. The two records diverge after a retraction because they are supposed
to, and each stays authoritative for its own question.

A third file kind would fail that test. Storing the round's unchecked words as
`done/out/<input>.no.N` puts the removals in two places, and
retracting one makes the archive and `removed/` disagree about a fact they both
claim -- v1's failure re-created one directory over. The removals live in
`removed/` only; `.no.N` is not written.

- **Content.** The round's whole reviewed input, sorted-unique bare words. For
  §3's command that is the normalized submission itself; for `complete words`
  it is the checked and unchecked words together.
- **Workflow-immutable, manually correctable.** Normal workflow commands never
  rewrite an existing archive, and nothing in the removal-retraction path
  touches one. Re-opening a word for review -- deciding a past verdict should
  not stand at all, rather than reversing it -- is the one case for an operator
  to delete a line and run `wf gen dict`. If the word occurs in overlapping
  reviewed-input archives, it must be deleted from every one before it leaves
  `done/reviewed.words`; automating that cross-record correction is out of
  scope.
- **Always written**, even by a submission that removes nothing. The reviewed
  input is what was looked at, and looking at a word that turned out not to be
  in the base is still having looked at it.

`done/reviewed.words` sits in `done/` beside the per-round `in/` archive,
consistent with `p1/done/p1_done.pairs`: both are the aggregate of what has
completed review. The storage role is identical even though their construction
differs. `p1_done.pairs` is folded in place by each bundle
(`steps/merge.py:33`) and cannot be rebuilt from anything;
`done/reviewed.words` is regenerated from `done/in/`. `bundle.filter_done`
(`bundle.py:237-254`, called at `eval.py:98`) subtracts the phase done-set from
the next round's input so nothing is evaluated twice, and the next plan's
candidate listing subtracts this one the same way.

It is placed with `stable_mtime=True` like its sibling. Nothing dates against it
today, which is why the phase done-sets carry no such flag (`config.py:76-81`),
but it is small enough that the compare is free and the next plan is the reader
that would want it.

This makes a separate store of *good* words unnecessary: the kept words are
`done/reviewed.words` minus the union of `removed/`, and both sides are already
maintained, so `dict/` needs no `kept/`.

### 5. Staleness: `_dictionary_stale`

The staleness row lands immediately above `_next_search`, so the seconds are
offered before the hours. It takes `# G7` and `_next_search` becomes `# G8` --
the groups are a plain sequence and stay one, at the cost of one comment line
edited in a row that does not otherwise change. The numbers are comments:
`ROWS` is an ordered tuple and nothing reads them.

Two required-input guards join G0. `_require_base_dictionary` and
`_require_dictionary` resolve their paths through `config`, call
`fs.raise_if_not_file`, and return `None` when the files exist. Their rows
provide `base_dictionary` and `dictionary`, respectively. They are validation,
not migration states: `words.big` is required, and an absent file or dangling
symlink is a hard error. `_dictionary_stale` is evaluated only after the G0
guard validates it.

```python
# G0 -- required dictionary inputs for BEST status
Row(_require_base_dictionary, provides="base_dictionary"),
Row(_require_dictionary, provides="dictionary"),
```

Below G6 rather than above it, and the ordering is self-correcting. A hand-edit
of a generation leaves the derived dictionary untouched, so its mtime has not
moved, so `_frontier_outdated` declines and this row fires and offers
`wf gen dict`. That rebuild moves the derived file, and `_frontier_outdated`
fires on the next status with a `"dictionary changed"` reason a regen can now
actually answer (§6). Seconds, then minutes, then hours.

```python
# G7 -- the derived dictionary is behind the records it derives from.
# Normally never fires: `remove words` rebuilds. It catches a hand-edit of a
# generation -- which is the retraction path -- and a base replaced under a
# derived file that was not rebuilt.
Row(_dictionary_stale, requires=("base_dictionary", "dictionary")),
```

It fires when any of:

- `done/reviewed.words` is absent. This is a stale state with the reason
  `reviewed words missing`, not a required-input failure. A present directory
  or dangling symlink still fails the normal optional-file contract, because
  generation cannot safely replace a broken destination.
- `max(removed/ mtime, max generation mtime)` is newer than
  `generation.generated(words.filtered)`. Both clocks are needed: adding or deleting a
  generation moves the directory's mtime, editing one in place moves only that
  file's.
- the same compare over `done/in/`, since `done/reviewed.words` is derived from it
  by the same command and dated by the same marker.
- the base is newer than `generation.generated(words.filtered)`. The base may
  be a symlink whose operator-chosen target gets replaced, so this is the
  likeliest real trigger in practice. v1's row claimed to catch it in a comment
  while its condition never looked at the base.

It offers `wf gen dict` only from BEST status, when both required dictionary
files exist and either `done/reviewed.words` is missing or a recorded input is
newer than the marker. A missing base or derived dictionary is stopped by the
G0 guards instead; it is not represented as a state-machine state. `wf gen
dict` remains the command that creates missing derived files, but it does not
walk the status rows and therefore cannot recommend itself.

`_frontier_outdated`'s docstring hedge (`state.py:771-780`) **is removed here**,
not deferred. Its claim is that "top-segments never reads it, so a regen from
the same DFS cannot drop a word the dictionary no longer has", which is false
today and stays false: `top-segments.cpp:98-100` is a word-level reject and §6
points it at the derived file. After this plan a frontier regen genuinely
answers a `"dictionary changed"` reason, which is what the hedge said would have
to wait.

`_review_needed` (`state.py:959-964`) reads the same accessor and needs the same
attention. It dates the frontier against `inputs.dictionary` with a
content-mtime compare, deliberately, while `top.segments` is placed with
`stable_mtime=True` (`generate.py:273`). Any removal that actually removes a
word makes the dictionary newer than `top.segments`; a byte-identical regen does
not move `top.segments`, so the review stays suppressed for that target
indefinitely. Under a plan whose point is making dictionary moves routine, this
is the behavior that changes most, and it should compare against
`generation.generated(top_segments)` the way the `target_no` clause beside it already
does.

The dictionary is also an input to both DFS artifacts. `seed_search_needed`
and `best_search_needed` each compare `words.filtered`'s content mtime with the
applicable existing DFS artifact and append `"dictionary changed"` when it is
newer. The BEST check retains its existing early return when there are no usable
pairs, since in that state there is no BEST search to offer. These comparisons
use the derived file itself, not `.words.filtered.gen`: an effective dictionary
change invalidates each applicable search, while a rebuild whose output is
byte-identical advances only the marker and does not offer hours of redundant
search work.

The row metadata names those reads explicitly. `_review_needed` and
`_frontier_outdated` add `"dictionary"` to their existing `requires=` tuples;
`_dictionary_stale` requires both `"base_dictionary"` and `"dictionary"`.
This matters to `walk_rows`: a row is not asked unless its required-file guard
has declined and established that input. It does not make the derived file a
requirement of `wf gen dict`, which does not walk the BEST rows.
`done/reviewed.words` is deliberately not a `requires=` input: its absence is
the repairable state this row reports, while a present invalid object raises
through `fs.optional_file`.

```python
Row(_review_needed, requires=("top.segments", "dictionary")),
Row(_frontier_outdated, requires=("top.segments", "dictionary")),
Row(_dictionary_stale, requires=("base_dictionary", "dictionary")),
```

### 6. nutrimatic: one constant

`WORKFLOW_DICT_PATH` (`source/pair-exclusions.cpp:17`) becomes
`".wf/dict/words.filtered"`. That one line moves all three `*-segments` tools,
because each formats its help text from the constant and each filters through
the same `all_words_in_dict`:

This path-only integration is the plan's narrow exception to keeping managed
dictionary implementation inside `workflow/`. It does not change any tool's
missing-file behavior or add a migration branch; the constant, its contract
comment, and directly affected fixtures are the complete Nutrimatic scope.
It also does not change the existing empty-dictionary sentinel: a present empty
workflow dictionary leaves the segment tools unfiltered. Empty workflow
dictionaries are unsupported here; separating "not loaded" from "loaded but
empty" in C++ is outside this milestone.

| tool | filter | help |
|---|---|---|
| `top-segments` | `:99` | `:57` |
| `first-segments` | `:103` | `:53` |
| `filter-segments` | `:80` | `:28` |

Plus the contract comment at `pair-exclusions.h:56`, the fixture at
`test-filter-segments.sh:72-73`, and the expected warning text at
`test-first-segments.sh:176`. `test-top-segments.sh` does not touch the
dictionary.

**What the two sides actually do with it**, since v1 got this wrong and it is
the reason the rename matters:

- `dfs-anagrams` filters **generatively, per word, during phase 1**.
  `dfs-class-list.cpp:334-338`: at every word boundary, a word not in the
  dictionary prunes the branch. It never becomes an index entry, never joins a
  class, never reaches phase 2, so no result can contain it.
- `top-segments` filters **post hoc, per output row**. `top-segments.cpp:98-100`
  sets `reject_line` when any segment of a dfs result line holds a word not in
  the dictionary, and `:106` drops the whole line -- so every other segment on
  that row loses its count too.

Same file, two jobs: it constrains what a search can spell, and it discards rows
a frontier would otherwise count. Left pointed at the base, a frontier regen
would filter by a strictly more permissive set than the search that will produce
its next input, and a removed word would keep appearing on `top.segments` until
a re-search.

**The knock-on.** A missing workflow dictionary warns and leaves the set empty
(`pair-exclusions.cpp:341-342`), and `all_words_in_dict` returns true on an
empty set (`:347`) -- so an absent `words.filtered` silently produces an
unfiltered frontier. `words.filtered` is not created by `wf init`, so that state
is reachable. `gen_top_segments` already guards exactly this hazard for the
classified files (`generate.py:246-250`: "a warning would produce an unfiltered
frontier the state machine then believes is filtered") and needs the same check
around `config.dictionary(target.root)` beside them. Unlike the generic
required-file guards, this one must explain how to create the derived input:
`dictionary not generated: <path>; run \`wf gen dict\``. On a tree where the
derived dictionary has not been generated, `_no_frontier` wins before
`_dictionary_stale` and directs the operator to generate `top.segments`; the
command itself therefore has to provide the actionable recovery rather than
ending at a bare missing-file diagnostic. Direct use of the segment tools keeps
their existing warning behavior; workflow generation refuses before launching
them when the required derived dictionary is absent.

A useful consequence: the complete `top-segments --solo-words --wfroot ROOT -y
DFS` invocation above filters by `words.filtered` too, so already-removed words
drop out of `TARGET/top.solo-words` on their own.

---

## Files to change

| file | what |
|---|---|
| `workflow/generation.py` (new) | shared `stamp`, `generated`, and `mark_generated` helpers moved without changing marker names or semantics |
| `workflow/best/state.py` | remove `Target.dictionary`; resolve dictionary paths through `config`; add G0 required-file guards and `requires=` metadata for rows that read them; `_dictionary_stale` + `ROWS` entry at G7, `_next_search` renumbered to G8; remove `_frontier_outdated`'s hedge; use shared generation-clock helpers; `_review_needed` dates against `generation.generated(top_segments)`; both DFS freshness predicates date against the content mtime of `words.filtered` and report `dictionary changed` |
| `workflow/config.py` | `dict` moves from `_BEST["parts"]` to `CONFIG_LAYOUT["parts"]` with `"content": True`, a `removed` part and a `done/in` part; dictionary-name constants; `dictionary()`, `base_dictionary()`, `removals()`, `reviewed_inputs()`, and `reviewed_words()` accessors |
| `workflow/dictionary.py` (new) | require the base; allocate across both archive namespaces; prepare every archive, derived output, and marker under `dict/`; preflight and commit the complete placement batch in order; preserve derived content mtimes; derive the reviewed-input aggregate; strip count prefixes and check word shape; translate set-operation failures into concise `ValueError` diagnostics |
| `workflow/wf.py` | `remove` and `gen` root verbs |
| `workflow/setops.py` | internal prepared-placement support separating writes and comparisons from final renames; `Placement(path, replaced)` plus `merge_report` and `diff_report`; existing operations keep their immediate behavior and return `Path` |
| `workflow/best/generate.py` | resolve the dictionary through `config` in `_dfs_inputs` and `gen_top_segments`, requiring it before invoking either tool |
| `nutrimatic/source/pair-exclusions.cpp` `.h` | narrow external exception: the path constant and its contract comment only |
| `nutrimatic/source/test-filter-segments.sh`, `test-first-segments.sh` | directly affected fixture path and expected warning only |
| `tests/test_workflow_best.py` | fixtures place a base and a derived dictionary (~line 57) |
| `tests/test_workflow_best_rows.py` | same fixture updates (~80, ~397); rename `test_a_dictionary_edit_dates_the_frontier_like_a_classify` to `test_a_dictionary_edit_makes_the_frontier_outdated`, rewrite its obsolete docstring, and add `_dictionary_stale` tests |
| `tests/test_workflow_best_e2e.py` | same (~76); a `remove words` -> `gen dfs` round trip asserting the `--dict` argv |
| `tests/test_workflow_generation.py` (new) | shared marker-path, legacy fallback, marker-content, and byte-identical remark tests moved from BEST state coverage |
| `tests/test_workflow_dictionary.py` (new) | §Tests |

## Tests

- Existing directory and dangling-symlink path checks reuse the established
  `fs.raise_if_not_file` and `fs.optional_file` contracts. One dictionary test
  makes a later destination invalid and verifies that full-batch preflight
  fails before either archive is published.
- A removal recorded via `wf remove words` drops the word from `words.filtered`
  and leaves `words.big` untouched.
- A removal of a word **not** in the base leaves `words.filtered` byte-identical
  and marks no search stale -- the `stable_mtime` property -- while still
  allocating a generation and advancing the marker.
- A removal that changes `words.filtered` makes both an existing `dfs.seed` and
  an applicable existing `dfs.best` stale with the reason `dictionary changed`.
  A byte-identical rebuild makes neither stale; DFS freshness dates the derived
  file's content mtime, not `.words.filtered.gen`.
- Re-submitting an already-removed word is a no-op on `words.filtered`; even
  though the word remains present in `words.big`, the report counts it as zero
  newly removed from the dictionary.
- A submission from `junk.txt` archives as `dict/removed/junk.txt.removed.1`, a
  following one from `other.txt` as `other.txt.removed.2` -- the counter is
  global, not per stem. Two submissions of the same filename get distinct
  generations and neither overwrites the other.
- A submitted path is archived under its basename, not the path the operator
  typed; a basename that is empty, `.`, or `..` is refused.
- Allocation reads one `max + 1` over both namespaces whatever the stems, so
  `removed/` holding `a.txt.removed.1` and `done/in/` holding
  `b.txt.reviewed.3` yields generation 4 and does not reuse 2 or 3. A name not
  ending in the applicable positive-integer suffix is ignored rather than
  parsed, and `foo.removed.3.removed.9` reads as 9.
- A file with `  1234 word` rows and a file of bare words produce the same
  generation; blank lines are discarded; a row that is not `[a-z]+` is refused
  and nothing is written.
- An unsorted submission still yields a sorted generation, and the subsequent
  `comm -23` subtracts every entry. (An already-sorted fixture passes with or
  without the sort, so the unsorted one is the case that proves the line.)
- **Names in `removed/` that do not match the ordinal pattern do not
  contribute to the derivation** -- a `foo.removed.3~` beside `foo.removed.3`
  does not resurrect a retracted word.
- Deleting a generation and running `wf gen dict` restores only words that do
  not remain in another generation; trimming a word restores it only after
  every overlapping occurrence is trimmed. One overlapping fixture proves
  that trimming only one occurrence leaves the word removed.
- A tree with an empty `removed/` derives a `words.filtered` byte-identical to
  `words.big` and an empty `done/reviewed.words`; neither empty aggregate calls
  `setops.merge` with an empty source list.
- The first `wf remove words` and a direct `wf gen dict` both create
  `words.filtered` when it is absent; neither treats that output as a required
  input. Both refuse a missing `words.big` before writing anything.
- `wf remove words` writes `done/in/<input>.reviewed.N` under the same N and
  stem as its `removed/` file, and `done/reviewed.words` is their union. A
  submission of words not in the base still lands in the reviewed-input
  archive. The first submission computes its "new to the removal union" count
  against an explicitly empty prior aggregate rather than calling
  `setops.merge` with no sources.
- **Names in `done/in/` that do not match the reviewed ordinal pattern do not
  contribute to `done/reviewed.words`** -- a `foo.reviewed.3~` beside
  `foo.reviewed.3` is ignored.
- Retracting a uniquely recorded word -- trimming
  `removed/<input>.removed.N` and running `wf gen dict` -- restores it to
  `words.filtered` and leaves `done/reviewed.words` unchanged, so it does not
  return as a candidate.
- Deleting an archived reviewed input removes words not present in another
  reviewed-input archive from `done/reviewed.words` and leaves
  `words.filtered` untouched. An overlapping word remains reviewed until every
  occurrence is removed.
- The kept words are recoverable as `done/reviewed.words` minus the union of
  `removed/`, over a sequence with an overlapping resubmission in it.
- `wf gen dict` places `done/reviewed.words`, then places `words.filtered` and
  advances the dictionary marker; a rebuild that changes neither still clears
  `_dictionary_stale`.
- Deleting `done/reviewed.words` after a successful build makes `wf best status`
  offer `wf gen dict` with `reviewed words missing`. Running `wf gen dict`
  directly recreates the file and does not print a recommendation to run
  itself.
- A failure placing `done/reviewed.words` fails the workflow step before
  `words.filtered` or its marker is touched. A crash after placing
  `words.filtered` but before the prepared marker placement leaves the marker
  behind, so `_dictionary_stale` offers the rebuild again.
- A forced `sort` or `comm` failure reports the dictionary operation and its
  relevant inputs without a Python traceback, preserves both prior derived
  outputs, and does not advance `.words.filtered.gen`.
- An unsorted base and an injected prospective `sort` or `comm` failure occur
  before archival: neither creates a removal generation or reviewed-input
  record, changes a derived output, nor advances the marker.
- The count line renders `(new)` on the first build,
  `(unchanged)` on a submission of words absent from the base, and the arrow
  when the count moved -- and the arrow, not `(unchanged)`, for a rebuild that
  applies one removal and picks up one retraction to land on an equal count.
- `gen dfs.seed` and `gen dfs.best` invoke `dfs-anagrams --dict …/words.filtered`.
- BEST status and DFS generation fail with the ordinary `file not found`
  diagnostic when `words.filtered` is absent. `gen top.segments` instead fails
  with `dictionary not generated: <path>; run \`wf gen dict\``, because when
  the derived file is absent `_no_frontier` can make it the status
  recommendation before `_dictionary_stale` is reached. `wf gen dict` does not
  require the derived file: it is the command's output, while `words.big` is
  its required input.
- `_dictionary_stale` fires on a generation newer than the marker, on an edit in
  place, and on a base newer than the marker; it is cleared by one `wf gen dict`,
  including when that rebuild changes nothing.
- `_dictionary_stale` yields to every row above it: an open review, a missing
  frontier, and an unreviewed frontier all still win.
- The G0 dictionary guards provide `base_dictionary` and `dictionary`.
  `_dictionary_stale` requires both; `_review_needed` and
  `_frontier_outdated` require `dictionary`, in addition to their existing
  requirements.
- `_review_needed` is still offered after a removal that changed
  `words.filtered` but left `top.segments` byte-identical.
- The renamed `test_a_dictionary_edit_makes_the_frontier_outdated` says that
  regenerating the frontier applies the changed derived dictionary to the
  recorded DFS source. Its existing state-machine assertions remain; the
  obsolete claims that `top-segments` never reads the dictionary and that the
  row is only an acknowledged notification are removed.
- `wf init` creates `dict/`, `dict/removed/` and `dict/done/in/` in a new
  workflow layout.
- `python -m unittest discover -s tests -p 'test_workflow*.py'` -- 263 tests
  pass today; expect additions, no removals.
- nutrimatic: `test-filter-segments.sh` and `test-first-segments.sh` pass
  against the new path.
- Live, read-only: `cd final && ./wf best status s7 -a` before and after, and
  confirm `dfs-anagrams` is invoked with `--dict .../words.filtered`
  (`generate.py` `_display_dfs` prints the argv) without running the search.

## Manual operator steps

These are guidance for the operator, not implementation or test requirements:

1. Update the `words` checkout and Nutrimatic binaries by whatever manual
   process the operator chooses.
2. Run `wf init` for the workflow root so the new `dict/`, `dict/removed/`, and
   `dict/done/in/` directories exist.
3. Place `$WFROOT/.wf/dict/words.big`. It may be a symlink, but its target is
   chosen by the operator and is not hardcoded by the workflow.
4. Submit any historical removal lists that should remain effective through
   `wf remove words WORDS-FILE`. Do not copy a `.removed.0` file directly into
   `dict/removed/`, because zero is not a valid managed ordinal.
5. Run `wf gen dict` and ensure subsequent workflow commands resolve
   mutually compatible new-layout binaries.

The operator owns execution and verification of these steps. The workflow adds
no migration-specific behavior or compatibility path.

## Not in scope

- Beyond the manual guidance above: migration support for an existing workflow
  tree, rollout or rollback across the `words` and Nutrimatic repositories,
  legacy-path fallback, a compatibility window, automated verification, or
  migration-specific tests.
- `top.solo-words` generation and the ranked candidate list --
  `plans/top-solo-words.md`.
- The `queued/eval/done` review cycle for words, its `--checkbox` notes and
  `complete words` -- the plan after that. §2 and §3 leave the names and the
  shape free for it; nothing here is built for it.
- Using `done/reviewed.words`. §4 derives it from this plan onward; subtracting it
  from the candidate listing is the next plan's `filter_done`.
- Confirming `note`'s single-checkbox parse. This plan assumes `--type NONE`
  yields the unchecked rows of a `--checkbox` note; establishing that against a
  real enex is separate work, and nothing here depends on it -- §3's command
  reads a word list, not a note.
- `done/out/`. Any enex or other artifact produced by the future review cycle
  belongs to that cycle's plan; §3's command produces no such archive.
- Filtering submitted p1 pairs against the dictionary. The move to the root is
  what makes it possible; nothing here does it.
- `nutrimatic/nutrimatic/dict_remove.py` is a one-off hack: it edits a
  hardcoded `~/code/nutrimatic/idx/words.big` in place. **Do not call it**, and
  do not port the in-place edit. Three of its ideas carry over: the count-prefix
  strip (`^ *[0-9]+ `, `:54-60`); `comm -23` under `LC_ALL=C` (`:78-81`), which
  `workflow/setops.py` already provides; and the `.removed.N` suffix, ported in
  §4 onto a different stem and a different mutability.
- A retraction command. Edit or delete a generation and run `wf gen dict`; §5 is
  what makes that safe to forget halfway through.
- Automatic discovery or correction of a word across overlapping removal or
  reviewed-input records. Manual retraction or reopening must edit every record
  containing the word.
- Changing the segment tools' empty-dictionary sentinel. A present empty
  workflow dictionary remains unfiltered and is an unsupported operating state.
- Fingerprinting `words.big` or detecting a replacement whose followed target
  mtime does not advance. Supported manual replacements must move that mtime
  beyond `.words.filtered.gen`.
- Revalidating the base or existing hand-edited removal and reviewed-input
  records during rebuild. Their sorted-unique lowercase content contracts
  remain operator-maintained.
- Rejecting control characters or otherwise tightening the accepted lexical
  archive basename. Such basenames are unsupported operator inputs.
- Locking or concurrent dictionary-command support. Commands, status reads, and
  direct manual edits use a single-process operator model in this milestone.
- Atomic commit or automatic recovery of a round across the two archive
  directories. Full-batch preparation and preflight narrow the remaining
  failure window to interruption during the final rename sequence; eliminating
  it requires a transaction marker or a different round-directory layout.
- Case or diacritic normalization. `words.big` is one lowercase word per line;
  §3 refuses anything that is not `[a-z]+` rather than folding it.

## Assumptions

- `words.big` remains hand-placed, nonempty, and C-sorted unique. Nothing in
  the workflow writes it, and `wf gen dict` fails loudly rather than repairing
  it. GNU `comm` exits 1 on unsorted input; the dictionary boundary translates
  that `CalledProcessError` into a concise operator-facing diagnostic.
- One removal set serves every target under the root. Per-target word removals
  are not wanted -- that is what target-local `no.pairs` is for.
- Generation files are never renumbered or compacted by the workflow. The
  counter is `max + 1` over what is present, so removing one by hand leaves a
  gap rather than causing a collision.
- Only one dictionary command, status read, or direct edit runs at a time. The
  workflow does not lock `.wf/dict/` or provide concurrent snapshot semantics.
- Dictionary archives remain at the intended scale of hundreds of generation
  files. Aggregation accepts the host's `ARG_MAX` limit and does not batch or
  compact records.

## Where the v1 review's findings landed

| # | finding | disposition |
|---|---|---|
| 1 | `top-segments` does read the dictionary | §6 -- the constant is repointed, and the two filters' semantics are spelled out |
| 2 | `_dictionary_stale` + `stable_mtime` is a stuck row | §1 -- `.words.filtered.gen`, and §5 dates against `generation.generated` |
| 3 | the row's comment does not match its condition | §5 -- base-vs-derived is now a clause, not a claim |
| 4 | the layout cannot hold files | dissolved -- no standing file; `removed/` is a directory |
| 5 | a `done` part makes `dict` non-leaf | §2 -- `"content": True` |
| 6 | `_review_needed` reads the same accessor | §5 -- dates against `generation.generated(top_segments)` |
| 7 | unmigrated trees get a bare error | migration is an operator precondition; absent required files fail at normal read boundaries, while `gen top.segments` names `wf gen dict` for its missing derived input |
| 8 | the diff needs a sorted input | §3 -- normalize before diffing, explicitly |
| 9 | "the target address is optional" is underspecified | §3 -- dissolved: neither command takes a target, and neither names a next command |
| 10 | the before-count does not exist on a first build | §3 -- three rendered cases: `-> `, `(new)`, `(unchanged)` |
| 11 | no shape check on submitted words | §3 step 2 -- `[a-z]+`, at submission |
| 12 | the `dict_remove` no-op citation is off | §4 -- the behavior is inverted and the citation dropped |
| 13 | one listed test may be unconstructible | §4 -- all collisions are unreachable by allocation; the test is dropped and the guard remains defensive |
| 14 | row group numbering collides | §5 -- renumbered: `_dictionary_stale` is G7, `_next_search` becomes G8 |
| 15 | `exclude-words` is not prefix-safe | dissolved -- the verb is `remove`; §3 gives aliases a defined precedence |
| 16 | stale test baseline | §Tests -- 263 |
| new | the generations do not partition a standing set once a word is retracted | dissolved -- there is no standing set |
