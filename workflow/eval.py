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
import tempfile

from pathlib import Path

from workflow import (
    bundle, command, config, context, dictionary, fs, log, notes, setops, usage,
)


class Eval(command.Action):
    def __init__(self, phase: str, summary: str,
                 source_noun: str = "pairs", ready_for: str = "evalpairs"):
        super().__init__(summary=summary, positional="BUNDLE-NAME|QUEUED-FILE")
        self.phase = phase
        self.source_noun = source_noun
        self.ready_for = ready_for

    def parser(self):
        return argparse.ArgumentParser(add_help=False)

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

    def filter(self, pairs: Path, ctx, opts) -> Path:
        """Apply this phase's review-history filter."""
        return bundle.filter_done(pairs, ctx)

    def _run(self, command, opts, argv, prepared: Path | None = None) -> int:
        rest = self.parse(opts, argv)
        if not rest:
            return usage.missing_argument(self.format_help(command))
        if len(rest) > 1:
            return usage.invalid_argument(rest[1], self.format_help(command))

        self.check(opts)
        if prepared is not None:
            fs.raise_if_not_readable(prepared)

        bundle_name, selected = bundle.resolve_queued(opts.dir, self.phase,
                                                      rest[0])
        ctx = context.Context(root=opts.dir, phase=self.phase,
                              force=opts.force, bundle_name=bundle_name)
        pairs = bundle.begin(ctx, selected)
        log.info(f"{fs.line_count(pairs)} source {self.source_noun}")
        if prepared is not None:
            # The caller's subset can turn out to be the whole source -- a
            # one-off whose pairs were all unreviewed -- and then there is
            # nothing for a derivative to claim.
            merged = setops.merge([prepared], bundle.filtered(pairs))
            pairs = bundle.keep_if_changed(pairs, merged)
        else:
            pairs = self.filter(pairs, ctx, opts)

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
        p.add_argument(
            "--filter-completed", action="store_true",
            help="also filter pairs already present in p2_done.pairs")
        notes.add_yes_pairs(p)
        return p

    def check(self, opts) -> None:
        notes.check_yes_pairs(opts)

    def filter(self, pairs: Path, ctx, opts) -> Path:
        return bundle.filter_done(
            pairs, ctx, filter_completed=opts.filter_completed)

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


class EvalWords(command.Action):
    def __init__(self):
        super().__init__(summary="words   — evaluate dictionary words for manual review",
                         positional="NAME")

    def run(self, command_text, opts, argv) -> int:
        if not argv:
            return usage.missing_argument(self.format_help(command_text))
        if len(argv) > 1:
            return usage.invalid_argument(argv[1],
                                          self.format_help(command_text))

        bundle_name, selected = bundle.resolve_queued(opts.dir, "dict", argv[0])
        reviewed = config.reviewed_words(opts.dir)
        if not reviewed.is_file():
            raise ValueError(f"dictionary reviewed words not generated: {reviewed}; "
                             f"run `wf gen dict`")
        ctx = context.Context(root=opts.dir, phase="dict", force=opts.force,
                              bundle_name=bundle_name)

        # An interrupted eval leaves its prepared derivative in the bundle with
        # the source still queued, and the retry finishes the move rather than
        # recomputing. That retained file is what `bundle.evaluated()` and
        # `notes.make` go on to use, so it is also what the part-count preflight
        # and the reported counts have to describe -- resolve it before either.
        retained_path = ctx.bundle_dir / f"{bundle_name}.filtered"
        if retained_path.exists():
            fs.raise_if_not_file(retained_path)
            retained: Path | None = retained_path
        else:
            retained = None

        prepared = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="w", prefix="wf-eval-words-", delete=False) as tmp:
                prepared = Path(tmp.name)
            result = dictionary.filter_reviewed_words(selected, reviewed,
                                                      prepared)
            if result.surviving_words == 0:
                raise ValueError(f"no unreviewed words in {selected}")
            if retained is None:
                presentation = prepared if result.filtered else selected
                surviving, filtering = result.surviving_words, result.filtered
            else:
                surviving, filtering = len(dictionary.read_words(retained)), True
                if surviving == 0:
                    raise ValueError(f"no unreviewed words in {retained}")
                presentation = retained
            logical_presentation = (Path(f"{bundle_name}.filtered")
                                    if filtering else selected)
            notes.part_paths(Path("."), logical_presentation,
                             notes.part_count(presentation))

            ctx.bundle_dir.mkdir(parents=True, exist_ok=True)
            if retained is not None:
                log.info(f"reusing prepared {retained.name} from an "
                         f"interrupted eval")
                prepared.unlink(missing_ok=True)
                prepared = None
            elif result.filtered:
                setops.place(setops.Staged(
                    prepared, ctx.bundle_dir / f"{bundle_name}.filtered", True))
                prepared = None
            source = bundle.begin(ctx, selected)
            presentation = bundle.evaluated(ctx)
            notes.make(presentation, opts, notes.ONE_CHECKBOX,
                       f"wf notes words {bundle_name}")
        finally:
            if prepared is not None:
                prepared.unlink(missing_ok=True)

        log.info(f"{result.source_words} source words")
        if filtering:
            log.info(f"{surviving} filtered words")
        log.success(f"{surviving} words ready for manual review: "
                    f"{presentation.name}")
        return 0


WORDS = EvalWords()
