# eval.py
#
# Open a bundle: move a queued artifact into <phase>/eval/<bundle_name>/ and
# prepare it
# for whatever does the actual evaluating.
#
# Unlike submit and complete, the phases here are not the same command with
# different constants -- p2 raises notes against its pairs, and p1 does nothing
# of the kind. So the shared run() carries the shape both phases agree on and
# hands the difference to prepare(). What p2's difference *is* belongs to
# notes.py, which owns that derivation and offers it as a command of its own;
# what is left here is what opening a bundle means.

import argparse

from pathlib import Path

from workflow import (
    bundle, command, config, context, fs, log, names, notes, setops, usage,
)


def _resolve_queued(root: Path, phase: str, positional: str) -> tuple[str, str]:
    """Read eval's positional, which may name either a bundle or a queued file.

    Returns the bundle name -- the eval directory, and the prefix of every
    artifact derived in it -- along with the selector that finds the queued
    artifact to open.

    Exact match first. A positional that is the name of a file in the phase's
    queue is taken as that file, and the bundle name is what is left when the
    queue suffix comes off. Everything else is a bundle name, found by prefix,
    which stays the form to reach for.

    The exact form exists because a prefix can be ambiguous by construction:
    p2's queue admits two shapes, so `s6.90.10.pairs` submitted and
    `s6.90.10.p1.yes` advanced can share the slot and the prefix `s6.90.10`
    names both. Naming the file is how the user says which -- and it costs
    nothing, because the bundle name is still derived from the contract rather
    than from whatever the user typed.
    """
    # Only a bare filename can name something in the slot; a path with
    # separators in it is not a queued name and falls through to the prefix
    # form, which reports it as the miss it is.
    if Path(positional).name == positional:
        queued = config.path(root, [phase, "queued"]) / positional
        if queued.is_file():
            return names.queue_stem(phase, positional), f"name:{positional}"
    return positional, f"stem:{positional}"


class Eval(command.Action):
    def __init__(self, phase: str, summary: str,
                 source_noun: str = "pairs", ready_for: str = "evalpairs"):
        super().__init__(summary=summary, positional="BUNDLE-NAME|QUEUED-FILE")
        self.phase = phase
        self.source_noun = source_noun
        self.ready_for = ready_for

    def parser(self):
        p = argparse.ArgumentParser(add_help=False)
        p.add_argument("--no-filter", action="store_true",
                       help="skip filtering already-evaluated pairs")
        return p

    def check(self, opts) -> None:
        """Whatever this phase must know is good before the bundle is opened.

        `bundle.begin` moves the queued source out of the queue and nothing
        moves it back: a failure after that point leaves a retry with no
        queued artifact to find and an open bundle it refuses to reopen. So
        anything answerable from the arguments alone is answered here, while
        failing is still free.
        """

    def prepare(self, pairs: Path, ctx, opts) -> None:
        """What this phase does with its pairs once the bundle is open."""

    def _run(self, command, opts, argv, prepared: Path | None = None) -> int:
        rest = self.parse(opts, argv)
        if not rest:
            return usage.missing_argument(self.format_help(command))

        self.check(opts)
        if prepared is not None:
            fs.raise_if_not_readable(prepared)

        bundle_name, selector = _resolve_queued(opts.dir, self.phase, rest[0])
        ctx = context.Context(root=opts.dir, phase=self.phase,
                              force=opts.force, bundle_name=bundle_name,
                              selector=selector)
        pairs = bundle.begin(ctx)
        log.info(f"{fs.line_count(pairs)} source {self.source_noun}")
        if prepared is not None:
            pairs = setops.merge([prepared], bundle.filtered(pairs))
            log.info(f"{fs.line_count(pairs)} filtered pairs")
        elif not opts.no_filter:
            pairs = bundle.filter_done(pairs, ctx)

        self.prepare(pairs, ctx, opts)

        # TODO: (optionally?) copy file to somewhere specified by user

        log.success(f"{fs.line_count(pairs)} pairs ready for {self.ready_for}: "
                    f"{pairs.name}")
        return 0

    def run(self, command, opts, argv) -> int:
        return self._run(command, opts, argv)

    def run_prepared(self, command, opts, argv, prepared: Path) -> int:
        """Open a bundle and install a caller-prepared evaluated subset.

        This is an internal composite-command seam, not a CLI mode. The queued
        source remains the full artifact that completion archives, while the
        ordinary `.filtered` derivative drives note titles and every downstream
        P2 step.
        """
        return self._run(command, opts, argv, prepared)


class EvalYes(Eval):
    def __init__(self):
        # Both queue shapes are opened the same way. An advanced `*.p1.yes`
        # arrives carrying a p1 verdict and a submitted `*.pairs` does not, but
        # that difference is provenance, not procedure: manual review reads the
        # pairs either way.
        super().__init__(phase="p2", summary="p2      — evaluate pairs for manual review",
                         source_noun="pairs", ready_for="manual filtering")

    def parser(self):
        p = super().parser()
        notes.add_yes_pairs(p)
        return p

    def check(self, opts) -> None:
        notes.check_yes_pairs(opts)

    def prepare(self, pairs: Path, ctx, opts) -> None:
        notes.make(pairs, opts)


class EvalNo(command.Action):
    def __init__(self):
        super().__init__(summary="p3      — evaluate no pairs")

    def show_help(self, command, opts, argv) -> int:
        return usage.default_help(self.summary, argv, "usage: wf eval no [options]")

    def run(self, command, opts, argv) -> int:
        # TODO: implement no pairs evaluation
        return 0


P1 = Eval("p1", "p1      — evaluate pairs")
P2 = EvalYes()
P3 = EvalNo()
