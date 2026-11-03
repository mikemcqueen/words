# Add target-local `no.pairs` exclusions

## Summary

Support an optional, user-managed `<target-dir>/no.pairs` file. When present,
it rejects pairs for that target in both DFS searches and `top.segments`
generation, and participates in their freshness checks.

## Context

Target-local `no.pairs` is a search exclusion, not a classification verdict.
Its entries remain eligible for other targets and are never folded into the
global `.wf/classified/no/no.pairs` set.

## Implementation

- Add `optional_file(path)` to `workflow/fs.py`. It:
  - returns `None` when the path is absent and is not a symlink;
  - accepts a regular file or valid symlink;
  - rejects directories and dangling symlinks.
- Use the shared resolver for the existing optional `best.pairs`, the optional
  dictionary used by status, and target-local `no.pairs`. Mandatory dictionary
  checks during generation remain unchanged.
- For both `dfs.seed` and `dfs.best`, retain
  `--exclude-pairs <workflow-root>` for the existing global hard-NO set and
  append `--exclude-pairs <target-dir>/no.pairs` when the target-local file
  exists.
- For `top.segments`, append `-r <target-dir>/no.pairs` when present, in
  addition to the existing `--wfroot … -y` filtering.
- While building the effective `dfs.best` bonus set, use the existing unique
  scratch directory and fixed scratch filenames:

  ```python
  yes_union = setops.merge(
      search_pair_sources(target),
      scratch / "union.yes.pairs",
  )

  no_union = hard_no
  if target_no is not None:
      no_union = setops.merge(
          [hard_no, target_no],
          scratch / "union.no.pairs",
      )

  allowed = setops.diff(
      yes_union,
      no_union,
      scratch / "allowed.pairs",
  )
  ```

  `search_pair_sources()` already returns classified YES followed by optional
  `best.pairs`, so the first merge preserves the current positive-side
  behavior while renaming its scratch output. Merging the two NO sources sorts
  and deduplicates the hand-managed file before `comm -23` reads it. Apply the
  existing bag filter to `allowed.pairs` to produce `dfs.best.pairs`, and count
  `allowed.pairs` for the pre-bag total.
- Change the zero-usable-pair state message to
  `no allowed bonus pair fits this target's letters`, and describe its count
  as pairs remaining after exclusions. When target-local `no.pairs` exists,
  add `or retract target-local exclusions in <target-dir>/no.pairs` to its
  remedies. Make the `gen dfs.best` refusal likewise say that no pair remains
  after exclusions and letter-bag filtering instead of claiming every source
  pair fails the letter bag.
- Add `target_no` to state inputs:
  - a newer file marks `dfs.seed` out of date with
    `target-NO set changed`;
  - when an effective usable bonus set remains, a newer file marks `dfs.best`
    out of date with the same reason;
  - when no usable bonus remains, the existing `_no_usable_pairs` row wins and
    no `dfs.best` search is offered;
  - a file newer than `generated(top.segments)` marks `top.segments` as
    `behind its inputs`;
  - `_review_needed` also compares target-local `no.pairs` with
    `generated(top.segments)`. It declines while the frontier has not been
    regenerated, allowing normal lower-row precedence to apply. After a no-op
    regeneration advances the marker, a genuinely unreviewed frontier becomes
    reviewable.
- Here `generated(top.segments)` means `.top.segments.gen` when that marker is
  present, and the `top.segments` content mtime for a legacy tree without it.
- Preserve the existing dictionary clock behavior: `_review_needed` continues
  to compare the dictionary directly with the `top.segments` content mtime.
  Only target-local `no.pairs` uses `generated(top.segments)` in that row.
- Keep `Row` unchanged. Its `requires` field represents mandatory existence
  gates, not all freshness inputs; optional `no.pairs` is handled by the state
  accessor, like the optional dictionary.
- Use mtime-only optional semantics as selected. Deletion creates no direct
  timestamp invalidation. It may still invalidate `dfs.best` when recomputing
  the usable bonus set produces different `dfs.best.pairs`; deletion is
  otherwise not detected.
- Subtract target-local NO in `wf best review` as well. `Review._top` and
  `Review._oneoff` (`workflow/best/commands.py:386-400,415-430`) diff against
  `classified/no` and `classified/yes` only; give both a shared helper, called
  inside the scratch directory each already opens:

  ```python
  def _union_no_pairs(target, scratch: Path) -> Path:
      """The NO sets a review subtracts: global hard-NO, plus target-local."""
      hard_no = config.classified(target.root, "no")
      fs.raise_if_not_file(hard_no)
      local_no = fs.optional_file(target.artifact("no.pairs"))
      if local_no is None:
          return hard_no
      return setops.merge([hard_no, local_no], scratch / "union.no.pairs")
  ```

  The merge is load-bearing rather than tidiness: `setops.diff` shells out to
  `comm -23`, which under-subtracts silently when its right-hand side is
  unsorted, and `no.pairs` is hand-managed. With no local file the helper
  returns `hard_no` unmerged -- the degenerate union, and today's behavior
  byte for byte. `_top` is only mostly covered by `top-segments -r`: `best
  review` reads `top.segments` without consulting the freshness rows, so a
  frontier generated before the exclusion was written is still reviewable and
  reintroduces every pair in it. `_oneoff` has no cover at all -- its supplied
  file has been through no filter.
- Two review messages name only the classified sets, and would misreport a
  pair the local file removed: `Review._converged`'s "all already classified"
  (`commands.py:453-455`) and `_oneoff`'s "all N pairs are already classified"
  refusal (`:425-428`). Say "already classified or excluded" in both, and name
  `<target-dir>/no.pairs` when it exists -- with `_no_usable_pairs`, this is
  where the feature is discoverable.

## Tests

- Verify absent `no.pairs` preserves existing DFS and `top-segments` command
  lines.
- Verify present `no.pairs` is passed to both DFS variants and `top-segments`.
- Verify local NO entries are removed from the effective `dfs.best.pairs` set.
- Verify a newer `no.pairs` invalidates both DFS artifacts when a usable bonus
  remains, invalidates the frontier, and prevents review of the older
  frontier.
- Verify a no-op frontier regeneration advances its marker and makes a
  genuinely unreviewed frontier reviewable.
- Verify an already reviewed frontier is not reopened after a no-op
  regeneration.
- Verify `_no_usable_pairs` retains precedence when local NO removes the final
  usable bonus, and an open review retains precedence over target-NO
  freshness.
- Verify an older or absent file does not disturb a converged target.
- Verify directories and dangling symlinks named `no.pairs` are rejected.
- Verify a one-off `wf best review` whose supplied file names pairs listed in
  an *unsorted* target-local `no.pairs` excludes them from the review file.
  The fixture has to be unsorted: a sorted one passes with or without the
  merge in `_union_no_pairs`, and `comm -23` reports nothing when it
  under-subtracts, so this is the case that proves the line it exists for.
- Verify direct `wf best review` also subtracts target-local NO from an older
  `top.segments` frontier generated before the exclusion was added.
- Verify `_no_usable_pairs` reports the count as pairs remaining after
  exclusions and names target-local `no.pairs` in its remedies when present.
- Verify `gen dfs.best` refuses an empty effective bonus set with the new
  after-exclusions-and-letter-bag-filtering diagnostic.
- Run the focused BEST suites, then
  `python -m unittest discover -s tests -p 'test_workflow*.py'`.

## Out of scope

- Canonicalizing pair identity across Python set operations and Nutrimatic's
  order-insensitive, normalized pair loader. Textually different or reversed
  spellings retain the existing limitation.
- Detecting a valid symlink being retargeted to an older file.
- Defining or validating an additional hand-managed pair-file format contract.
- Additional valid-symlink, sibling-isolation, `status --all`, or CLI-level
  end-to-end coverage beyond the tests above.
- Changing the dictionary's existing content-mtime comparison.

## Assumptions

- “Seed and best dependencies” means the generated `dfs.seed` and `dfs.best`
  artifacts.
- `no.pairs` remains hand-managed; no workflow command creates, updates, or
  deletes it.
