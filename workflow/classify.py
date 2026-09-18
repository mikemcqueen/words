# classify.py
#
# Record a standing verdict about pairs, outside any bundle.
#
# `complete p2` already folds a review batch's confirmed YES into
# classified/yes; this is the same fold for a verdict that does not come from a
# batch at all -- a pair judged by hand, outside any note, with no bundle to
# complete.
#
# The two kinds are not symmetric in what they mean downstream. A confirmed YES
# joins the same aggregate p2 writes to, so a manual call and a reviewed batch
# are indistinguishable once folded, which is the point. A hard NO is a policy
# rather than a review outcome: it says the pair must not appear in a result at
# all, which is stronger than p2's unchecked soft NOs -- those mean only "not
# confirmed" and stay in the phase done-set. Nothing in this repo consumes the
# NO aggregate; `dfs-anagrams --exclude-pairs` is the reader it is written for,
# and until that lands a hard NO is recorded but not enforced.
#
# The input is normalized on the way in -- `sort -u` over the union -- because
# the aggregate is later handed to `comm` and to a tool that assumes a set.

from pathlib import Path

from workflow import command, config, fs, log, usage


# The verdict each kind contradicts. Recording both for one pair is a mistake
# the workflow cannot resolve on the user's behalf: it may be a slip, or it may
# be a reversal, and there is no un-classify to undo the earlier call with.
OPPOSITE = {"yes": "no", "no": "yes"}

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


def contradiction_message(kind: str, pairs: list[str]) -> str:
    shown = ", ".join(pairs[:SAMPLE])
    more = (f" (+{len(pairs) - SAMPLE} more)"
            if len(pairs) > SAMPLE else "")
    return (f"Cannot classify {kind.upper()}: "
            f"{len(pairs)} input pair(s) already classified "
            f"{OPPOSITE[kind].upper()}: {shown}{more}")


def fold(root: Path, kind: str, src: Path) -> Path:
    """Fold src into a classified set and report its new and total counts."""
    dst = config.classified(root, kind)
    before = fs.line_count(dst) if dst.exists() else 0
    config.fold_classified(root, kind, src)
    total = fs.line_count(dst)
    log.success(f"Classified {kind.upper()}: {total - before} new, "
                f"{total} total → {dst.name}")
    return dst


class Classify(command.Action):
    def __init__(self, kind: str, label: str):
        super().__init__(
            summary=f"{kind.ljust(8)}— union {label} pairs into classified/{kind}",
            positional="PAIRS-FILE",
        )
        self.kind = kind

    def _contradictions(self, opts, src: Path) -> list[str]:
        other = config.classified(opts.dir, OPPOSITE[self.kind])
        return contradictions(src, other)

    def run(self, command, opts, argv) -> int:
        if not argv:
            return usage.missing_argument(self.format_help(command))
        if len(argv) > 1:
            return usage.invalid_argument(argv[1], self.format_help(command))

        src = Path(argv[0]).resolve()
        fs.raise_if_not_file(src)

        contradictions = self._contradictions(opts, src)
        if contradictions:
            log.error(contradiction_message(self.kind, contradictions))
            return 1

        dst = config.classified(opts.dir, self.kind)
        before = fs.line_count(dst) if dst.exists() else 0
        if opts.dry_run:
            current = set(dst.read_text().splitlines()) if dst.exists() else set()
            total = len(current | set(src.read_text().splitlines()))
            log.success(f"Would classify {self.kind.upper()}: "
                        f"{total - before} new, {total} total → {dst.name}")
            return 0

        fold(opts.dir, self.kind, src)
        return 0


YES = Classify("yes", "confirmed-YES")
NO = Classify("no", "hard-NO")
