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
# `pcomm`, which checks inputs against it here, needs neither.

import argparse
import subprocess
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

    Pair order is presentation, not identity, for a standing verdict. `pcomm`
    treats ``first,second`` and ``second,first`` as one pair, needs no sorted
    input, and holds only the smaller file in memory -- the input, once the
    classified sets have grown -- and prints a shared pair in src's spelling.
    It refuses a line that is not ``word,word``, naming the file and line.
    """
    if not opposing.exists():
        return []

    result = subprocess.run(["pcomm", "-12", str(src), str(opposing)],
                            stdout=subprocess.PIPE, text=True, check=True)
    return sorted(set(result.stdout.splitlines()))


def contradiction_message(kind: str, pairs: list[str],
                          sentence: str | None = None,
                          show_all: bool = False) -> str:
    """Name the first SAMPLE pairs inline, or with show_all every pair, one
    per line."""
    scope = f" in {sentence}" if sentence else ""
    head = (f"Cannot classify {kind.upper()}: "
            f"{len(pairs)} input pair(s) already classified "
            f"{OPPOSITE[kind].upper()}{scope}")
    return listing(head, pairs, show_all)


def listing(head: str, pairs: list[str], show_all: bool = False) -> str:
    """head, then the first SAMPLE pairs inline, or with show_all every pair,
    one per line."""
    if show_all:
        return "\n".join([f"{head}:", *pairs])
    shown = ", ".join(pairs[:SAMPLE])
    more = (f" (+{len(pairs) - SAMPLE} more)"
            if len(pairs) > SAMPLE else "")
    return f"{head}: {shown}{more}"


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
             sentence: str | None = None,
             show_all: bool = False) -> str | None:
    """Why src cannot be classified kind in this scope, or None if it can."""
    for other, scope in opposing(root, kind, sentence):
        pairs = contradictions(src, other)
        if pairs:
            return contradiction_message(kind, pairs, scope, show_all)
    return None


def add_show_conflicts(p: argparse.ArgumentParser) -> None:
    """The --show-conflicts flag, shared by `wf classify` and `complete p2`."""
    p.add_argument(
        "--show-conflicts", action="store_true",
        help="on a conflict, list every conflicting pair, not just "
             f"the first {SAMPLE}")


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


def preview(root: Path, kind: str, src: Path,
            sentence: str | None = None) -> None:
    """Report the counts fold would report, without folding."""
    dst = config.classified(root, kind, sentence)
    before = fs.line_count(dst) if dst.exists() else 0
    current = set(dst.read_text().splitlines()) if dst.exists() else set()
    total = len(current | set(src.read_text().splitlines()))
    log.success(f"Would classify {kind.upper()}: {total - before} new, "
                f"{total} total → {shown(root, dst)}")


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
        add_show_conflicts(p)
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

        message = conflict(opts.dir, self.kind, src, sentence,
                           opts.show_conflicts)
        if message:
            log.error(message)
            return 1

        if opts.dry_run:
            preview(opts.dir, self.kind, src, sentence)
            return 0

        fold(opts.dir, self.kind, src, sentence)
        return 0

YES = Classify("yes", "confirmed-YES")
NO = Classify("no", "hard-NO")


class Pairs(command.Action):
    """Both verdicts from one review in one call: neither set changes unless
    both can.

    `complete p2` checks both kinds before folding either; this is the same
    rule for a review that has no bundle, such as pgui's. Two `wf classify
    yes|no` calls could leave the YES folded and the NO refused.
    """

    def __init__(self):
        super().__init__(
            summary="pairs   — union YES and NO pairs into classified/, "
                    "both or neither",
        )

    def parser(self):
        p = argparse.ArgumentParser(add_help=False)
        p.add_argument(
            "-s", "--sentence", type=int, metavar="N",
            help="record the verdicts for sentence N only, in classified/sN")
        p.add_argument("--yes", dest="yes_file", metavar="YES-FILE",
                       help="pairs to classify YES (may be empty)")
        p.add_argument("--no", dest="no_file", metavar="NO-FILE",
                       help="pairs to classify NO (may be empty)")
        add_show_conflicts(p)
        return p

    def run(self, command, opts, argv) -> int:
        argv = self.parse(opts, argv)
        if argv:
            return usage.invalid_argument(argv[0], self.format_help(command))
        if opts.yes_file is None or opts.no_file is None:
            return usage.missing_argument(self.format_help(command))

        sources = {"yes": Path(opts.yes_file).resolve(),
                   "no": Path(opts.no_file).resolve()}
        for src in sources.values():
            fs.raise_if_not_file(src)
        sentence = (None if opts.sentence is None
                    else config.sentence_name(opts.sentence))

        # The opposing sets on disk do not include the other input, so a pair
        # in both files would pass both checks and be folded both ways.
        both = contradictions(sources["yes"], sources["no"])
        if both:
            log.error(listing(
                f"Cannot classify: {len(both)} pair(s) in both --yes and --no",
                both, opts.show_conflicts))
            return 1

        for kind, src in sources.items():
            message = conflict(opts.dir, kind, src, sentence,
                               opts.show_conflicts)
            if message:
                log.error(message)
                return 1

        for kind, src in sources.items():
            if opts.dry_run:
                preview(opts.dir, kind, src, sentence)
            else:
                fold(opts.dir, kind, src, sentence)
        return 0

PAIRS = Pairs()
