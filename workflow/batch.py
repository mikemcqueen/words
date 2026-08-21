# batch.py
#
# One directory per in-flight item, at <phase>/eval/<slug>/.
#
# The directory *is* the record. It is created by `eval` and removed by
# `complete` once the batch has drained into done/, so `ls eval/` is the
# in-flight list and "is this in flight?" is one stat against a known path
# rather than a scan of three slots for a matching name.
#
# Every artifact inside is prefixed by the directory name, which is what lets
# the contents be found by glob without any command taking a name apart.
#
# Every function here takes a Context and nothing else. The batch directory
# itself is `ctx.batch_dir` -- there is no second way to spell it.

from pathlib import Path

from workflow import config, fs, log, select, setops


def in_flight(ctx) -> bool:
    return ctx.batch_dir.is_dir()


def begin(ctx, glob: str = "*") -> Path:
    """Create the batch directory and move the queued artifact into it."""
    src = select.select(ctx.root, [ctx.phase, "queued"], f"stem:{ctx.slug}",
                        glob=glob)[0]

    directory = ctx.batch_dir
    if directory.exists() and not ctx.force:
        raise fs.file_already_exists_error(directory)
    directory.mkdir(parents=True, exist_ok=True)

    dst = directory / src.name
    if not ctx.force:
        fs.raise_if_exists(dst)
    # Invariant dimensions come first, so the slug prefixes everything in here.
    assert dst.name.startswith(ctx.slug), f"{dst.name} is not prefixed by {ctx.slug}"
    src.rename(dst)
    return dst


def one(directory: Path, glob: str) -> Path:
    """The single artifact in a batch directory matching glob."""
    matches = sorted(p for p in directory.glob(glob) if p.is_file())
    if not matches:
        raise ValueError(f"no {glob} in {directory}")
    if len(matches) > 1:
        found = ", ".join(p.name for p in matches)
        raise ValueError(f"multiple {glob} in {directory}: {found}")
    return matches[0]


# The input artifact `eval` moved in, matched by *kind* rather than by name.
# The directory name is only a prefix of what it holds -- `wf eval p1 a` makes
# `a/` holding `a.pairs` -- so matching on the slug misses the very file it is
# looking for. The directory is already the namespace; nothing else in it ends
# in these suffixes.
INPUT_GLOB = {"p1": "*.pairs", "p2": "*.p1.yes"}


def source(ctx) -> Path:
    """The batch's input artifact -- whatever `eval` moved into the directory."""
    return one(ctx.batch_dir, INPUT_GLOB[ctx.phase])


def has_source(ctx) -> bool:
    """Does the batch still hold its input?

    `archive` relocates the input into done/, so its absence is the record that
    archive ran -- and therefore that every step before archive ran too. This
    is what lets an idempotent fold answer `is_done` without a manifest: the
    fold has no output of its own to test, but placement still carries the
    answer.
    """
    return any(p.is_file() for p in ctx.batch_dir.glob(INPUT_GLOB[ctx.phase]))


def filtered(src: Path) -> Path:
    """The derivative `eval` writes when it drops already-done pairs."""
    return src.with_name(src.name + ".filtered")


def evaluated(ctx) -> Path:
    """The input as actually evaluated.

    `eval` writes a `.filtered` derivative when it drops already-done pairs, and
    everything downstream -- note titles, the done-set merge -- follows whichever
    file it chose. The original is what gets archived.
    """
    src = source(ctx)
    candidate = filtered(src)
    return candidate if candidate.exists() else src


def done_pairs(ctx) -> Path:
    """The phase's accumulated done-set, which every batch folds into."""
    return config.path(ctx.root, [ctx.phase, "done"]) / f"{ctx.phase}_done.pairs"


def filter_done(src_pairs: Path, ctx) -> Path:
    """Drop pairs the phase has already evaluated.

    Returns the file to carry forward: the `.filtered` derivative when there is
    a done-set to subtract, otherwise `src_pairs` untouched.
    """
    done = done_pairs(ctx)
    if not done.is_file():
        return src_pairs

    dst = filtered(src_pairs)
    if not ctx.force:
        fs.raise_if_exists(dst)

    setops.diff(src_pairs, done, dst)
    log.info(f"{fs.line_count(dst)} filtered pairs")
    return dst


def finish(ctx) -> None:
    """Remove the batch directory. It must already have drained into done/."""
    directory = ctx.batch_dir
    fs.raise_if_not_dir(directory)
    leftover = sorted(p.name for p in directory.iterdir())
    if leftover:
        raise ValueError(f"batch {ctx.slug} still holds: {', '.join(leftover)}")
    directory.rmdir()
