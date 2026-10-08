# wf classify pairs: one call that records both verdicts

## Why

pgui is getting a review mode. A column's pair list is frozen in a modal and
each pair gets a checkbox: checked means YES, unchecked means NO. SUBMIT
records the verdicts for pgui's current sentence. On success, pgui closes the
modal and bumps a classified version, so its pfilter columns rerun and the new
NO pairs drop out.

`wf classify yes|no -s N FILE` (workflow/classify.py) already exists for
verdicts recorded outside a bundle. Using it from pgui would take two calls,
and the two calls are not atomic. If `classify yes` folds and `classify no`
then fails on a conflict, the YES verdicts are recorded and the NO verdicts
are not. A fixed resubmit is harmless, because folds are a union plus
`sort -u`, but pgui would have to report "partially applied" anyway.

`complete p2` avoids this already. workflow/steps/p2_classify.py:run_step checks
`classify.conflict()` for both kinds before it folds either one. The new
command applies that same rule outside a bundle.

## Command

```
wf classify pairs -s SENTENCE --yes YES-FILE --no NO-FILE
```

- YES-FILE and NO-FILE are plain pair lists, one pair per line, the same
  format as `wf classify yes|no`'s PAIRS-FILE. Both flags are required; either
  file may be empty.
- It first checks the two files against each other with
  `classify.contradictions()`, in either pair order. `conflict()` only compares
  an input with the sets already on disk, so a pair in both files would
  otherwise pass both checks and be folded both ways.
- It then runs `classify.conflict()` for yes and for no. If any check fails,
  it reports the conflict and exits 1 with neither set changed.
- When every check passes, it runs `classify.fold()` for yes, then for no.
  With the global `--dry-run` it runs the same checks, then calls
  `classify.preview()` for each kind instead of folding.
- It accepts `--show-conflicts` the way `wf classify yes|no` does.
- `-s` is optional, as it is in `wf classify yes|no`. pgui always passes it.
- A fold that fails partway (e.g. a disk error after YES has folded) is not
  rolled back. Folds are idempotent unions, so resubmitting is safe.

## The pgui side

- pgui shows every YES pair, so the review list contains pairs that are
  already classified YES, both sentence and global. Left unchecked, one of
  those would be submitted as NO, and a sentence NO may not contradict a
  sentence or global YES. pgui shows those rows checked and locked, since there
  is no un-classify, and leaves them out of both files.
- pgui writes the two files to a temp location, runs `wf` off the UI thread,
  and captures stdout and stderr. On a non-zero exit it shows the output in
  the modal and leaves the checks as they were. It removes the temp files
  afterwards.
- pgui passes the workflow root it reads from `$WFROOT` explicitly, as
  `-d "$WFROOT"`. `wf` takes its root from `-d/--dir`, then `$WFROOT`, then
  the current directory (workflow/wf.py:main).
