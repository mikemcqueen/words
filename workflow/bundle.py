# bundle.py
#
# One directory per in-flight bundle, at <phase>/eval/<bundle_name>/.
#
# The directory *is* the record. It is created by `eval` and removed by
# `complete` once the bundle has drained into done/, so `ls eval/` is the
# in-flight list and "is this in flight?" is one stat against a known path
# rather than a scan of three slots for a matching name.
#
# Every artifact inside is prefixed by the directory name, which is what lets
# the contents be found by glob without any command taking a name apart.
#
# Every function here takes a Context and nothing else. The bundle directory
# itself is `ctx.bundle_dir` -- there is no second way to spell it.

from pathlib import Path

from workflow import config, fs, log, names, select, setops


def in_flight(ctx) -> bool:
    return ctx.bundle_dir.is_dir()


def begin(ctx) -> Path:
    """Create the bundle directory and move the queued artifact into it."""
    src = select.select(ctx.root, [ctx.phase, "queued"],
                        f"stem:{ctx.bundle_name}",
                        glob=names.queue_globs(ctx.phase))[0]

    bundle_dir = ctx.bundle_dir
    if bundle_dir.exists() and not ctx.force:
        raise fs.file_already_exists_error(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    dst = bundle_dir / src.name
    if not ctx.force:
        fs.raise_if_exists(dst)
    # Invariant dimensions come first, so the bundle name prefixes everything.
    assert dst.name.startswith(ctx.bundle_name), (
        f"{dst.name} is not prefixed by {ctx.bundle_name}")
    src.rename(dst)
    return dst


def one(bundle_dir: Path, glob) -> Path:
    """The single artifact in a bundle directory matching glob."""
    matches = fs.globs(bundle_dir, glob)
    if not matches:
        raise ValueError(f"no {fs.spell(glob)} in {bundle_dir}")
    if len(matches) > 1:
        found = ", ".join(p.name for p in matches)
        raise ValueError(f"multiple {fs.spell(glob)} in {bundle_dir}: {found}")
    return matches[0]


# The input artifact `eval` moved in is matched by the phase's queue contract
# rather than by name. The directory name is only a prefix of what it holds --
# `wf eval p1 a` makes `a/` holding `a.pairs` -- so matching on the bundle name
# misses the very file it is looking for. The directory is already the
# namespace; nothing else in it ends in these suffixes.
def source_globs(ctx) -> tuple[str, ...]:
    return names.queue_globs(ctx.phase)


def source(ctx) -> Path:
    """The bundle's source artifact -- whatever `eval` moved into it."""
    return one(ctx.bundle_dir, source_globs(ctx))


def has_source(ctx) -> bool:
    """Does the bundle still hold its source artifact?

    `archive` relocates the input into done/, so its absence is the record that
    archive ran -- and therefore that every step before archive ran too. This
    is what lets an idempotent fold answer `is_done` without a manifest: the
    fold has no output of its own to test, but placement still carries the
    answer.
    """
    return bool(fs.globs(ctx.bundle_dir, source_globs(ctx)))


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
    """The phase's accumulated done-set, which every bundle folds into."""
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
    """Remove the bundle directory. It must already have drained into done/."""
    bundle_dir = ctx.bundle_dir
    fs.raise_if_not_dir(bundle_dir)
    leftover = sorted(p.name for p in bundle_dir.iterdir())
    if leftover:
        raise ValueError(
            f"bundle {ctx.bundle_name} still holds: {', '.join(leftover)}")
    bundle_dir.rmdir()
