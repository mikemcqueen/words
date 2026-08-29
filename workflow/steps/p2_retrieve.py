# steps/p2_retrieve.py
#
# Fetch the manually-classified note parts back out of the note store.
#
# This is the one step whose output set is not knowable in advance: parts are
# probed .aa, .ab, ... until one is missing, so given an enex/ holding three
# files the filesystem cannot say whether that is "finished at three" or
# "crashed after three". The fix is to make the *directory* the atomic unit:
# parts land in enex.part/, and only the 404 -- the moment completion is
# actually known -- renames it to enex/. So enex/ exists iff retrieve finished,
# is_done is one stat, and a partial run resumes into the parts already there
# instead of raising. Each part is placed atomically too, so a truncated .enex
# can never pass for a complete one.

import shutil
import subprocess

from pathlib import Path

from workflow import bundle, fs, log, notes


NAME = "retrieve"


def enex_dir(ctx) -> Path:
    return ctx.bundle_dir / "enex"


def partial_dir(ctx) -> Path:
    return ctx.bundle_dir / "enex.part"


def inputs(ctx) -> list[Path]:
    return [bundle.evaluated(ctx)]


def outputs(ctx) -> list[Path]:
    return [enex_dir(ctx)]


def is_done(ctx) -> bool:
    return enex_dir(ctx).is_dir()


def run_step(ctx) -> None:
    source = bundle.evaluated(ctx)
    staging = partial_dir(ctx)
    staging.mkdir(exist_ok=True)

    fetched = 0
    # The titles creation rendered, probed in the order it made them.
    for index in range(notes.MAX_PARTS):
        title = notes.title(source, index)
        part = staging / f"{title}.enex"
        if part.exists():
            fetched += 1
            continue

        result = subprocess.run(["note", "-pf.72", "--get", title, "--production"],
                                capture_output=True, text=True)
        if result.returncode != 0:
            if "note not found" in result.stderr:
                break
            raise RuntimeError(f"failed to retrieve note {title}:\n{result.stderr}")

        tmp = staging / f"{title}.enex.tmp"
        tmp.write_text(result.stdout)
        tmp.replace(part)
        fetched += 1

    if not fetched:
        raise RuntimeError(f"no note parts found for {source.name} (expected at least .aa)")

    log.info(f"retrieved {fetched} note part(s)")

    # `-f` skips is_done, so this runs against an enex/ a previous run already
    # completed -- and on the one step that caches state living outside the
    # workflow, that is exactly the request: the held copy is stale, fetch it
    # again. Placement can say "we hold a copy", never "we hold the current
    # one", so this is the only sense -f could carry here.
    #
    # Dropped now rather than on entry, because the replacement is fetched and
    # staged by this point: clearing first and then losing the network would
    # trade a good copy for a partial one, which is what enex.part/ exists to
    # prevent. A crash in the gap leaves no enex/ and a complete enex.part/,
    # so the next plain run resumes and renames it into place.
    #
    # Here rather than in `fs.rename_once`: its other caller is p2_archive,
    # renaming enex/ into done/out, and a generic "force removes the
    # destination" would silently delete a previous completion's archived
    # notes. POSIX rename replaces a directory target only when it is empty, so
    # without this the forced rename raises ENOTEMPTY and strands enex.part/ in
    # the bundle.
    if ctx.force:
        shutil.rmtree(enex_dir(ctx), ignore_errors=True)
    fs.rename_once(staging, enex_dir(ctx), ctx.force)
