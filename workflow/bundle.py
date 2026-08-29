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
# itself is `ctx.bundle_dir` -- there is no second way to spell it. The one
# exception is `resolve_source`, which renders the name a Context is built
# from and so necessarily runs before there is one.

from pathlib import Path

from workflow import config, fs, log, names, select, setops


def in_flight(ctx) -> bool:
    return ctx.bundle_dir.is_dir()


def begin(ctx) -> Path:
    """Create the bundle directory and move the queued artifact into it."""
    # The bundle name as a prefix is the base rule. A Context that carries a
    # selector of its own -- `eval` builds one when the user named the queued
    # file exactly -- overrides it, which is what makes a prefix shared by two
    # queue shapes openable at all. `all` is the field's unset default and must
    # not be honoured here: it would open whichever bundle sorted first.
    selector = (f"stem:{ctx.bundle_name}"
                if ctx.selector == select.ALL else ctx.selector)
    src = select.select(ctx.root, [ctx.phase, "queued"], selector,
                        glob=names.queue_globs(ctx.phase))[0]

    bundle_dir = ctx.bundle_dir
    if bundle_dir.exists():
        # A bundle holds exactly one source artifact -- `source` resolves it by
        # the queue contract, not by name, so a second one makes the bundle
        # unresolvable and there is no command to un-open it. `-f` reopens a
        # directory an earlier run left behind; it does not stack a source onto
        # one already in flight. p2 makes that a live spelling rather than a
        # hypothetical: its two queue shapes collapse to one bundle name, so
        # naming the second file is the natural next thing to try.
        held = fs.globs(bundle_dir, source_globs(ctx))
        if held:
            raise ValueError(f"bundle {ctx.bundle_name} is already open on "
                             f"{', '.join(p.name for p in held)}")
        if not ctx.force:
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


def at_most_one(directory: Path, glob) -> Path | None:
    """The single file in directory matching glob, or None if there is none.

    For the caller whose question is "is it here?" rather than "give it to
    me". Ambiguity is still an error either way, so both callers spell it
    once.
    """
    matches = fs.globs(directory, glob)
    if len(matches) > 1:
        found = ", ".join(p.name for p in matches)
        raise ValueError(f"multiple {fs.spell(glob)} in {directory}: {found}")
    return matches[0] if matches else None


def one(bundle_dir: Path, glob) -> Path:
    """The single artifact in a bundle directory matching glob."""
    match = at_most_one(bundle_dir, glob)
    if match is None:
        raise ValueError(f"no {fs.spell(glob)} in {bundle_dir}")
    return match


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


def resolve_source(root: Path, phase: str,
                   positional: str) -> tuple[str, Path | None]:
    """The bundle a command's positional names, and the file it named.

    The directory is canonical and its name is not ambiguous, so this is only
    an escape hatch: a command takes either the bundle name or the name of the
    source file itself, and the string that opened a bundle has to be able to
    reach it again.

    Returns the bundle name -- the eval directory, and the prefix of every
    artifact under it -- and, when the positional named a file exactly, that
    file. Carrying the path is the point rather than a convenience: p2's queue
    admits two shapes, so `foo.pairs` and `foo.p1.yes` share one bundle name,
    and a caller that only got the name back would have to re-derive the file
    and find both. Naming one exactly is how the user says which, and that
    answer has to survive the trip. This is `eval._resolve_queued`'s bargain,
    widened from the queue to every slot that can hold a source.
    """
    # Only a bare filename can name something in a slot; a path with separators
    # in it is reported as the miss it is.
    if Path(positional).name != positional:
        return positional, None
    evals = config.path(root, [phase, "eval"])
    if (evals / positional).is_dir():
        return positional, None
    try:
        stem = names.queue_stem(phase, positional)
    except ValueError:
        # Not a queue shape -- or a phase with no queue contract at all. Either
        # way there is nothing to strip, so the miss is reported as typed.
        return positional, None
    if (evals / stem).is_dir():
        return stem, None
    # The slots that hold a source outside a bundle, in the order `recover`
    # reads them. An exact hit is a file, not a pattern, so this is the one
    # lookup here that a name could not have been globbed into.
    for slot in (["queued"], ["done", "in"]):
        found = config.path(root, [phase, *slot]) / positional
        if found.is_file():
            return stem, found
    return positional, None


def source_in(ctx, slot: list[str], exact: Path | None = None) -> Path | None:
    """This bundle's source artifact in one of the phase's slots, if it is there.

    Matched by the exact names the queue contract admits for this bundle name,
    not by prefix: the slots that hold many bundles side by side would
    otherwise let `...r1.pairs` answer for `...r10.pairs`.

    `exact` is a source the caller already resolved by name. It routes itself
    to its own slot by where it sits, so the caller passes it to every slot and
    the three questions `recover` asks stay three questions. It also settles
    the ambiguity the name alone cannot: a bundle whose queue holds both shapes
    matches two names here, and the user who typed one of them has answered
    that already.
    """
    directory = config.path(ctx.root, [ctx.phase, *slot])
    if exact is not None:
        return exact if exact.parent == directory else None
    return at_most_one(directory,
                       names.queue_names(ctx.phase, ctx.bundle_name))


def recover(ctx, source: Path | None = None) -> Path:
    """The bundle's evaluated source, wherever the phase is holding it now.

    Three slots, of which two can answer and one is only diagnostic. In flight
    is the plain case. Queued means the bundle was never opened, so there is
    nothing to recover -- the derivation the caller wants is exactly what
    `eval` does, and running it here would move state a read has no business
    moving. Archived is a completed round, behind `-f`.

    `-f` carries one meaning here and only one -- use the archived source --
    because this touches neither `begin` nor `filter_done`, the two places its
    other senses live.
    """
    if in_flight(ctx):
        # An open bundle answers for itself. `source` is ignored rather than
        # preferred: it can only have come from the queue or the archive, and
        # neither is what this round is working on.
        return evaluated(ctx)

    queued = source_in(ctx, ["queued"], source)
    if queued is not None:
        raise ValueError(
            f"{ctx.bundle_name} is queued and has never been opened, so it "
            f"has no notes to recreate; run wf eval p2 {queued.name}")

    archived = source_in(ctx, ["done", "in"], source)
    if archived is None:
        raise ValueError(f"bundle not found: {ctx.bundle_name}")
    if not ctx.force:
        raise ValueError(
            f"{ctx.bundle_name} is completed; use -f to work from the "
            f"archived {archived.name}. Two caveats: it is the unfiltered "
            f"original, because the .filtered derivative does not survive "
            f"archiving, and a --yes-pairs set is read as it stands now, so "
            f"it may already hold that round's own confirmations.")
    return archived


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
