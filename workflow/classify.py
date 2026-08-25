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

import tempfile

from pathlib import Path

from workflow import command, config, fs, log, setops, usage


# The verdict each kind contradicts. Recording both for one pair is a mistake
# the workflow cannot resolve on the user's behalf: it may be a slip, or it may
# be a reversal, and there is no un-classify to undo the earlier call with. So
# this warns and proceeds rather than refusing -- the last write is what the
# user just asked for, and the collision is what they need to be told about.
OPPOSITE = {"yes": "no", "no": "yes"}

SAMPLE = 3


class Classify(command.Action):
    def __init__(self, kind: str, label: str):
        super().__init__(
            summary=f"{kind.ljust(8)}— union {label} pairs into classified/{kind}",
            positional="PAIRS-FILE",
        )
        self.kind = kind

    def _warn_if_contradicted(self, opts, dst: Path) -> None:
        other = config.classified(opts.dir, OPPOSITE[self.kind])
        if not other.exists():
            return
        with tempfile.NamedTemporaryFile(prefix="wf-classify-",
                                         suffix=".pairs") as scratch:
            overlap = setops.common(dst, other, Path(scratch.name))
            pairs = overlap.read_text().split()
        if not pairs:
            return
        shown = ", ".join(pairs[:SAMPLE])
        more = f" (+{len(pairs) - SAMPLE} more)" if len(pairs) > SAMPLE else ""
        log.warn(f"{len(pairs)} pair(s) now classified both {self.kind.upper()} "
                 f"and {OPPOSITE[self.kind].upper()}: {shown}{more}")

    def run(self, command, opts, argv) -> int:
        if not argv:
            return usage.missing_argument(self.format_help(command))
        if len(argv) > 1:
            return usage.invalid_argument(argv[1], self.format_help(command))

        src = Path(argv[0]).resolve()
        fs.raise_if_not_file(src)

        dst = config.classified(opts.dir, self.kind)
        before = fs.line_count(dst) if dst.exists() else 0
        config.fold_classified(opts.dir, self.kind, src)
        total = fs.line_count(dst)

        log.success(f"Classified {self.kind.upper()}: {total - before} new, "
                    f"{total} total → {dst.name}")
        self._warn_if_contradicted(opts, dst)
        return 0


YES = Classify("yes", "confirmed-YES")
NO = Classify("no", "hard-NO")
