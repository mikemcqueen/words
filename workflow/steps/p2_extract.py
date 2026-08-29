# steps/p2_extract.py
#
# Parse the retrieved enex into the bundle's YES and NO sets.
#
# One step for both kinds, because the two are only valid together. The notes
# carry two checkboxes and nothing in a note stops a reviewer ticking Y and N
# on one row; `note --parse-file` answers each --type independently, so such a
# row comes back in both sets. Each set is well-formed on its own -- the
# contradiction exists only in the pair of them -- so no per-kind step can see
# it, and by the time a later step could, both are already placed: `classify`
# folds YES into the standing classified verdicts and `advance` publishes NO
# into p3, in opposite directions, neither undoable by any command here.
#
# So the kinds are merged in /tmp, checked against each other there, and placed
# into the bundle only once they agree. That is `_place`'s discipline one level
# up, and `retrieve`'s enex.part/ discipline applied to a pair of files rather
# than a directory: nothing half-good is ever observable at a final path, and a
# rejected review leaves nothing in the bundle to delete or to be resumed past.
#
# Every /tmp name carries the kind. The two staged sets exist at once, to be
# intersected; the per-part parses used to share one name across kinds and were
# safe only by running one after the other.

import subprocess

from pathlib import Path

from workflow import fs, log, setops
from workflow.steps import p2_retrieve


NAME = "extract"

KINDS = ("yes", "no")

# How many contradictions the diagnostic spells out before it stops counting.
# The reviewer has to go find these rows in the note; a wall of them is no more
# help than a handful and a total.
LISTED = 20


def inputs(ctx) -> list[Path]:
    return sorted(p2_retrieve.enex_dir(ctx).glob("*.enex"))


def outputs(ctx) -> list[Path]:
    return [ctx.artifact("p2", kind) for kind in KINDS]


def _scratch(ctx, name: str) -> Path:
    """A /tmp path for one of this bundle's intermediate sets.

    Keyed on the bundle name, which every artifact in a bundle is prefixed by,
    so two bundles cannot land on one path.
    """
    return Path(f"/tmp/{ctx.bundle_name}.p2.{name}")


def _parse_note_files(paths: list[Path], kind: str) -> list[Path]:
    parsed = []
    for path in paths:
        out_path = Path(f"/tmp/{path.name}.{kind}.parsed")
        with out_path.open("w") as f:
            subprocess.run(["note", "--parse-file", str(path), "--type", kind,
                            "--lines"], stdout=f, check=True)
        parsed.append(out_path)
    return parsed


def _stage(ctx, enex: list[Path], kind: str) -> Path:
    """The merged set for one kind, in /tmp, not yet the bundle's."""
    return setops.merge(_parse_note_files(enex, kind),
                        _scratch(ctx, f"{kind}.staged"))


def _contradictions(ctx, staged: dict) -> list[str]:
    """The pairs marked both ways: the intersection of the two staged sets.

    Both are `sort -u` output, so `comm -12` is reading them as what they are.
    """
    both = setops.common(staged["yes"], staged["no"], _scratch(ctx, "both"))
    return both.read_text().splitlines()


def _diagnostic(ctx, both: list[str]) -> str:
    shown = "\n  ".join(both[:LISTED])
    rest = len(both) - LISTED
    more = f"\n  ... and {rest:,} more" if rest > 0 else ""
    return (f"{len(both):,} pair(s) marked both YES and NO in "
            f"{ctx.bundle_name} -- a row cannot be both:\n"
            f"  {shown}{more}\n"
            f"Correct those rows in the note and re-run "
            f"`wf complete p2 {ctx.bundle_name}`.")


def run_step(ctx) -> None:
    fs.raise_if_not_dir(p2_retrieve.enex_dir(ctx))
    enex = inputs(ctx)
    staged = {kind: _stage(ctx, enex, kind) for kind in KINDS}

    both = _contradictions(ctx, staged)
    if both:
        # Nothing has been placed, so there is nothing to take back: the bundle
        # is exactly as `retrieve` left it and a re-run redoes this step whole.
        raise ValueError(_diagnostic(ctx, both))

    for kind in KINDS:
        placed = setops.merge([staged[kind]], ctx.artifact("p2", kind))
        # The count is of the placed artifact, so it is the deduped set the
        # bundle carries forward rather than the sum of what the parts held.
        log.info(f"extracted {fs.line_count(placed):,} {kind.upper()} "
                 f"pairs from {len(enex)} note part(s)")
