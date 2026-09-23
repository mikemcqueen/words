# classify.py
#
# Record a standing verdict about pairs, outside any bundle.
#
# `complete p2` already folds a review batch's ticked YES and ticked NO rows
# into classified/yes and classified/no; this is the same fold for a verdict
# that does not come from a batch at all -- a pair judged by hand, outside any
# note, with no bundle to complete. Both paths write the same aggregates, so a
# manual call and a reviewed batch are indistinguishable once folded, which is
# the point.
#
# The two kinds are not symmetric in what they mean downstream. Review
# filtering (review_filter) drops pairs of either kind, so neither is offered
# for review again. Beyond that, a YES is ignored when counting segments, while
# a NO rejects outright: `top-segments --wfroot` and `dfs-anagrams
# --exclude-pairs` drop any line holding a NO pair.
#
# With `-s N` the verdict goes to classified/sN instead and answers for that
# sentence alone. Only `eval p2 -s N` reads it back: that sentence's reviews
# drop its pairs, along with the globally classified ones. Nothing else does:
# best reviews and generation, including the segment counting and NO
# rejection above, read the global sets only.
#
# The input is normalized on the way in -- `sort -u` over the union -- because
# the aggregate is later handed to `comm` and to a tool that assumes a set.

import argparse
from pathlib import Path

from workflow import command, config, fs, log, usage


# The verdict each kind contradicts. Recording both for one pair is a mistake
# the workflow cannot resolve on the user's behalf: it may be a slip, or it may
# be a reversal, and there is no un-classify to undo the earlier call with.
OPPOSITE = {"yes": "no", "no": "yes"}

# A sentence's verdicts sit beside the global ones and may disagree in one
# direction only. A sentence YES may stand against a global NO -- a pair that
# is wrong in general can still be right for one sentence. `eval p2 -s N`
# never offers a globally classified pair, so that YES is recorded here, with
# `wf classify yes -s N`, rather than through a review. A sentence NO may
# not be recorded against a global YES. A global verdict is never checked
# against the sentences. Within one scope, YES and NO contradict as they
# always have.

SAMPLE = 3


def contradictions(src: Path, opposing: Path) -> list[str]:
    """Return src pairs that oppose a pair in opposing, in either order.

    Pair order is presentation, not identity, for a standing verdict. Build the
    small lookup in memory so both ``first,second`` and ``second,first`` are
    forbidden without relying on the line identity and ordering required by
    ``comm``.
    """
    if not opposing.exists():
        return []

    prohibited = set()
    for pair in opposing.read_text().splitlines():
        first, separator, second = pair.partition(",")
        prohibited.add(pair)
        if separator:
            prohibited.add(f"{second},{first}")

    return sorted(set(src.read_text().splitlines()) & prohibited)


def contradiction_message(kind: str, pairs: list[str],
                          sentence: str | None = None) -> str:
    shown = ", ".join(pairs[:SAMPLE])
    more = (f" (+{len(pairs) - SAMPLE} more)"
            if len(pairs) > SAMPLE else "")
    scope = f" in {sentence}" if sentence else ""
    return (f"Cannot classify {kind.upper()}: "
            f"{len(pairs)} input pair(s) already classified "
            f"{OPPOSITE[kind].upper()}{scope}: {shown}{more}")


def opposing(root: Path, kind: str,
             sentence: str | None = None) -> list[tuple[Path, str | None]]:
    """The sets a kind may not overlap, each with the sentence it is scoped to.

    Global verdicts oppose only each other. Sentence YES opposes that
    sentence's NO; sentence NO opposes that sentence's YES and global YES.
    """
    other = OPPOSITE[kind]
    if sentence is None:
        return [(config.classified(root, other), None)]
    sets = [(config.classified(root, other, sentence), sentence)]
    if kind == "no":
        sets.append((config.classified(root, other), None))
    return sets


def conflict(root: Path, kind: str, src: Path,
             sentence: str | None = None) -> str | None:
    """Why src cannot be classified kind in this scope, or None if it can."""
    for other, scope in opposing(root, kind, sentence):
        pairs = contradictions(src, other)
        if pairs:
            return contradiction_message(kind, pairs, scope)
    return None


def shown(root: Path, dst: Path) -> str:
    """dst as reported: relative to classified/, e.g. s8/yes/yes.pairs."""
    return dst.relative_to(config.path(root, ["classified"])).as_posix()


def fold(root: Path, kind: str, src: Path,
         sentence: str | None = None) -> Path:
    """Fold src into a classified set and report its new and total counts."""
    dst = config.classified(root, kind, sentence)
    before = fs.line_count(dst) if dst.exists() else 0
    config.fold_classified(root, kind, src, sentence)
    total = fs.line_count(dst)
    log.success(f"Classified {kind.upper()}: {total - before} new, "
                f"{total} total → {shown(root, dst)}")
    return dst


class Classify(command.Action):
    def __init__(self, kind: str, label: str):
        super().__init__(
            summary=f"{kind.ljust(8)}— union {label} pairs into classified/{kind}",
            positional="PAIRS-FILE",
        )
        self.kind = kind

    def parser(self):
        p = argparse.ArgumentParser(add_help=False)
        p.add_argument(
            "-s", "--sentence", type=int, metavar="N",
            help=f"record the verdict for sentence N only, in "
                 f"classified/sN/{self.kind}")
        return p

    def run(self, command, opts, argv) -> int:
        argv = self.parse(opts, argv)
        if not argv:
            return usage.missing_argument(self.format_help(command))
        if len(argv) > 1:
            return usage.invalid_argument(argv[1], self.format_help(command))

        src = Path(argv[0]).resolve()
        fs.raise_if_not_file(src)
        sentence = (None if opts.sentence is None
                    else config.sentence_name(opts.sentence))

        message = conflict(opts.dir, self.kind, src, sentence)
        if message:
            log.error(message)
            return 1

        dst = config.classified(opts.dir, self.kind, sentence)
        before = fs.line_count(dst) if dst.exists() else 0
        if opts.dry_run:
            current = set(dst.read_text().splitlines()) if dst.exists() else set()
            total = len(current | set(src.read_text().splitlines()))
            log.success(f"Would classify {self.kind.upper()}: "
                        f"{total - before} new, {total} total → "
                        f"{shown(opts.dir, dst)}")
            return 0

        fold(opts.dir, self.kind, src, sentence)
        return 0

YES = Classify("yes", "confirmed-YES")
NO = Classify("no", "hard-NO")
