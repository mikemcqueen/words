import argparse
import tempfile

from pathlib import Path

from workflow import (
    classify, command, config, fs, log, names, notes, setops, submit, usage,
    complete as complete_phase, eval as evaluate,
)
from workflow.best import generate
from workflow.best.state import (
    SOURCES, Choice, Inputs, check_letter_set, eval_p2_command, one_target,
    render_choices, report, review_locations, review_rounds, target_no_pairs,
    targets, top_segments_source,
)


def _add_letter_set(parser: argparse.ArgumentParser) -> None:
    """The two spellings of one working bag, of which exactly one is required.

    Both are ordinary optional flags, checked by hand beside the checks -g and
    -m already get. Argparse's required mutually exclusive group would report
    the failure against a bare parser carrying no prog -- naming wf.py, without
    the subcommand or the positionals -- and exit around log.error, where every
    other required argument here gets the command's own help.
    """
    parser.add_argument("-o", metavar="LETTERS",
                        help="letter set: use only these letters")
    parser.add_argument("-u", metavar="LETTERS",
                        help="letter set: the sentence less these letters")


def _letter_set(opts) -> str | None:
    """The letter-set directory name -o/-u selects, or None if neither did."""
    if opts.o is not None and opts.u is not None:
        raise ValueError("-o and -u name one letter set; give exactly one")
    if opts.o is not None:
        return f"o-{opts.o}"
    if opts.u is not None:
        return f"u-{opts.u}"
    return None


def _add_source(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", metavar="SOURCE",
                        help="DFS results to work from: seed or best")


def _check_source(source: str) -> None:
    if source not in SOURCES:
        raise ValueError(
            f"--source names one DFS artifact; expected "
            f"{' or '.join(SOURCES)}, got {source!r}")


def _check_target_dirs(target, may_create: bool, force: bool) -> None:
    """The three levels below the sentence: only a seed search creates them."""
    missing = next(
        (path for path in (target.letter_set_dir, target.universe_dir,
                           target.target_dir) if not path.exists()), None)
    if missing is None:
        return
    if not may_create:
        fs.raise_if_not_dir(missing)
    elif not force:
        raise FileNotFoundError(
            f"directory does not exist: {missing}; use -f to force creation")


def _preflight_top_segments(target) -> None:
    """Refuse to rewrite the frontier a top review is still reading.

    top.segments is the bundle in flight -- the notes were derived from it and
    `complete` folds its verdicts back against it -- so regenerating it under
    an open review would leave the round answering a question nothing asked.
    One check, called by every command that writes the frontier, so what
    status refuses and what the command refuses are the same condition.
    """
    queued, evaluating, _ = review_locations(target)
    in_flight = [round_ for round_ in (*queued, *evaluating)
                 if round_.kind == "top"]
    if not in_flight:
        return
    location = in_flight[0]
    raise ValueError(
        f"review bundle in flight: {location.name} in {location.parent}; "
        f"top.segments is what it was built from")


class Status(command.Action):
    def __init__(self):
        super().__init__(summary="status   — report BEST PAIRS target state",
                         positional="[ADDRESS]")

    def parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "-a", "--all", action="store_true", dest="all_rows",
            help="report every precedence row, not just the one that fired")
        return parser

    def run(self, command_text, opts, argv) -> int:
        argv = self.parse(opts, argv)
        if len(argv) > 1:
            return usage.invalid_argument(argv[1],
                                          self.format_help(command_text))
        selected = targets(opts.dir, argv[0] if argv else None)
        if not selected:
            print("no BEST PAIRS targets")
            return 0
        failed = False
        for target in selected:
            # A malformed sibling -- two seeds in one universe, two review
            # bundles for one target -- must not cost the operator the rest
            # of the listing. status is the command they run to find out.
            try:
                check_letter_set(target)
                report(target, rows=opts.all_rows)
            except (OSError, ValueError) as e:
                log.error(f"{target.address}: {e}")
                failed = True
        return 1 if failed else 0


class Gen(command.Action):
    STAGES = ("dfs.seed", "top.segments", "dfs.best")

    def __init__(self):
        super().__init__(summary="gen      — generate one BEST PAIRS artifact",
                         positional="SENTENCE STAGE",
                         positional_help=(
                             ("SENTENCE", "sentence identifier under .wf/best "
                              "(for example, s2)"),
                             ("STAGE", "artifact to generate: dfs.seed "
                              "(provisional DFS results), top.segments "
                              "(frequent pairs from the DFS results named by "
                              "--source), or dfs.best (DFS results weighted "
                              "by the confirmed-YES pairs this target's "
                              "letters can spell)"),
                         ))

    def parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        _add_letter_set(parser)
        parser.add_argument("-g", type=int, metavar="COUNT",
                            help="number of segments")
        parser.add_argument("-m", type=int, default=4, metavar="LENGTH",
                            help="min word length")
        parser.add_argument("-r", "--results-dir", type=Path, metavar="DIR",
                            help="DFS output directory (default: results)")
        parser.add_argument(
            "-n", dest="count", type=int, metavar="COUNT",
            help="maximum results to output (for dfs-anagrams and "
                 "top-segments)")
        _add_source(parser)
        return parser

    def _stage(self, target, stage: str, opts) -> None:
        """Run one stage, having refused the flags it does not take.

        The refusals live here rather than in generate because the generation
        helpers now take explicit parameters -- `prepare` drives two of them
        with disagreeing flags and cannot hand one opts to both.
        """
        if stage in ("dfs.seed", "dfs.best"):
            if stage == "dfs.best" and opts.force:
                raise ValueError("-f/--force is only valid for gen dfs.seed")
            generate.gen_dfs(target, final=stage == "dfs.best",
                             force=opts.force, results_dir=opts.results_dir,
                             count=opts.count)
            return
        if opts.force:
            raise ValueError("-f/--force is only valid for gen dfs.seed")
        if opts.results_dir is not None:
            raise ValueError("-r/--results-dir is only valid for DFS stages")
        _preflight_top_segments(target)
        generate.gen_top_segments(target, source=opts.source,
                                  count=opts.count)

    def run(self, command_text, opts, argv) -> int:
        rest = self.parse(opts, argv)
        letter_set = _letter_set(opts)
        if len(rest) < 2 or opts.g is None or letter_set is None:
            return usage.missing_argument(self.format_help(command_text))
        if len(rest) > 2:
            return usage.invalid_argument(rest[2],
                                          self.format_help(command_text))
        sentence, stage = rest
        if stage not in self.STAGES:
            return usage.invalid_argument(stage, self.format_help(command_text))
        if opts.m < 1 or opts.g < 1:
            raise ValueError("-m and -g require positive integers")
        if opts.count is not None and opts.count < 0:
            raise ValueError("-n requires a non-negative integer")
        # One parser serves all four stages, so --source is syntactically
        # accepted everywhere and refused here beside the other stage flags.
        if stage != "top.segments":
            if opts.source is not None:
                raise ValueError("--source is only valid for gen top.segments")
        elif opts.source is None:
            return usage.missing_argument(self.format_help(command_text))
        else:
            _check_source(opts.source)

        target = one_target(opts.dir, sentence, letter_set, opts.m, opts.g)
        if stage == "dfs.seed":
            # Before anything is created, and gated on the same condition
            # status gates it on, so the label status refuses is the label gen
            # refuses.
            check_letter_set(target)
        _check_target_dirs(target, may_create=stage == "dfs.seed",
                           force=opts.force)
        self._stage(target, stage, opts)
        report(target)
        return 0


class Prepare(command.Action):
    """One search and the frontier generated from it, in one command.

    The pair is what an inner-loop round actually is, and splitting it across
    two commands left the frontier a stage the operator could forget after
    hours of DFS had already landed.
    """

    def __init__(self):
        super().__init__(
            summary="prepare  — run a DFS search and generate the frontier",
            positional="SENTENCE")

    def parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        _add_letter_set(parser)
        parser.add_argument("-g", type=int, metavar="COUNT",
                            help="number of segments")
        parser.add_argument("-m", type=int, default=4, metavar="LENGTH",
                            help="min word length")
        _add_source(parser)
        parser.add_argument("-r", "--results-dir", type=Path, metavar="DIR",
                            help="DFS output directory (default: results)")
        parser.add_argument("--dfs-count", type=int, metavar="COUNT",
                            default=generate.DFS_LIMIT,
                            help="maximum dfs-anagrams results "
                                 f"(default: {generate.DFS_LIMIT})")
        parser.add_argument("--top-count", type=int, metavar="COUNT",
                            default=generate.TOP_LIMIT,
                            help="maximum top.segments pairs "
                                 f"(default: {generate.TOP_LIMIT})")
        return parser

    def run(self, command_text, opts, argv) -> int:
        rest = self.parse(opts, argv)
        letter_set = _letter_set(opts)
        if (not rest or opts.g is None or letter_set is None
                or opts.source is None):
            return usage.missing_argument(self.format_help(command_text))
        if len(rest) > 1:
            return usage.invalid_argument(rest[1],
                                          self.format_help(command_text))
        if opts.m < 1 or opts.g < 1:
            raise ValueError("-m and -g require positive integers")
        _check_source(opts.source)
        if opts.dfs_count < 0 or opts.top_count < 0:
            raise ValueError(
                "--dfs-count and --top-count require non-negative integers")
        seed_search = opts.source == "seed"
        if opts.force and not seed_search:
            raise ValueError(
                "-f/--force is only valid for prepare --source seed")

        target = one_target(opts.dir, rest[0], letter_set, opts.m, opts.g)
        if seed_search:
            check_letter_set(target)
        _check_target_dirs(target, may_create=seed_search, force=opts.force)
        # The only preflight: wf is a single-user client and this holds the
        # shell for the whole DFS, so no review can open before the frontier
        # is generated.
        _preflight_top_segments(target)
        generate.prepare(target, source=opts.source, force=opts.force,
                         results_dir=opts.results_dir,
                         dfs_count=opts.dfs_count, top_count=opts.top_count)
        report(target)
        return 0


def _target_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    _add_letter_set(parser)
    parser.add_argument("-g", type=int, metavar="N")
    parser.add_argument("-m", type=int, default=4, metavar="N")
    return parser


def _action_target(action, command_text, opts, argv, positionals: int,
                   maximum: int | None = None):
    rest = action.parse(opts, argv)
    letter_set = _letter_set(opts)
    if len(rest) < positionals or opts.g is None or letter_set is None:
        return usage.missing_argument(action.format_help(command_text))
    maximum = positionals if maximum is None else maximum
    if len(rest) > maximum:
        return usage.invalid_argument(
            rest[maximum], action.format_help(command_text))
    if opts.m < 1 or opts.g < 1:
        raise ValueError("-m and -g require positive integers")
    target = one_target(opts.dir, rest[0], letter_set, opts.m, opts.g)
    fs.raise_if_not_dir(target.target_dir)
    return target, rest


class Exclude(command.Action):
    def __init__(self):
        super().__init__(summary="exclude  — classify hard-NO pairs for a target",
                         positional="SENTENCE PAIRS-FILE")

    def parser(self):
        return _target_parser()

    def run(self, command_text, opts, argv) -> int:
        parsed = _action_target(self, command_text, opts, argv, 2)
        if isinstance(parsed, int):
            return parsed
        target, rest = parsed
        if opts.force:
            raise ValueError("-f/--force is not valid for best exclude")
        code = classify.NO.run("classify no", opts, [rest[1]])
        if code == 0:
            report(target)
        return code


def _names_local_no(target) -> str:
    """Where a refusal says which target-local file did some of the excluding.

    Only when there is one: an operator who has never written a no.pairs is
    not sent looking for it. With `_no_usable_pairs` this is where a message
    that says "or excluded" says what did the excluding.
    """
    local_no = target_no_pairs(target)
    if local_no is None:
        return ""
    return f"; target-local exclusions: {local_no}"


class Review(command.Action):
    def __init__(self):
        super().__init__(summary="review   — submit a target for P2 review",
                         positional="SENTENCE [PAIRS-FILE]",
                         positional_help=(
                             ("SENTENCE", "sentence identifier under .wf/best"),
                             ("PAIRS-FILE", "optional one-off pairs file; "
                              "omit it to review top.segments", "?"),
                         ))

    def parser(self):
        return _target_parser()

    def run(self, command_text, opts, argv) -> int:
        parsed = _action_target(self, command_text, opts, argv, 1, maximum=2)
        if isinstance(parsed, int):
            return parsed
        target, rest = parsed
        if opts.force:
            raise ValueError("-f/--force is not valid for best review")

        supplied = Path(rest[1]).resolve() if len(rest) == 2 else None
        if supplied is not None:
            fs.raise_if_not_readable(supplied)
            return self._oneoff(target, supplied, opts)
        return self._top(target, opts)

    @staticmethod
    def _union_no_pairs(target, scratch: Path) -> Path:
        """The NO sets a review subtracts: global hard-NO, plus target-local.

        The merge is load-bearing rather than tidiness: setops.diff shells out
        to `comm -23`, which under-subtracts in silence when its right-hand
        side is unsorted, and no.pairs is hand-managed. With no local file the
        helper returns hard_no unmerged -- the degenerate union, and the
        behaviour before this existed, byte for byte.
        """
        hard_no = config.classified(target.root, "no")
        fs.raise_if_not_file(hard_no)
        local_no = target_no_pairs(target)
        if local_no is None:
            return hard_no
        return setops.merge([hard_no, local_no], scratch / "union.no.pairs")

    @staticmethod
    def _in_flight(target, queued, evaluating) -> None:
        in_flight = [*queued, *evaluating]
        if in_flight:
            location = in_flight[0]
            raise ValueError(
                f"review bundle already in flight: {location.name} in "
                f"{location.parent}")

    def _top(self, target, opts) -> int:
        queued, evaluating, archived = review_locations(target)
        self._in_flight(target, queued, evaluating)

        top_segments = target.artifact("top.segments")
        fs.raise_if_not_file(top_segments)
        cutoff = fs.line_count(top_segments)
        if cutoff == 0:
            source = top_segments_source(target)
            raise ValueError(
                f"top.segments is empty; regenerate "
                f"{target.artifact(f'dfs.{source}')}")

        # max + 1, not count + 1: an archive with a gap in it -- a round
        # deleted, or never archived -- would otherwise render a name that
        # already exists.
        round_number = max(review_rounds(target, archived, "top"), default=0) + 1
        review_name = (f"{target.review_prefix('top')}{cutoff}."
                       f"r{round_number}.pairs")
        # The hard-NO set is checked by _union_no_pairs, which is the one
        # that reads it now.
        confirmed_yes = config.classified(target.root, "yes")
        fs.raise_if_not_file(confirmed_yes)
        with tempfile.TemporaryDirectory(prefix="wf-best-review-") as tmp:
            scratch = Path(tmp)
            collated = setops.merge([top_segments], scratch / "top.pairs")
            # Both standing sets come out, not just the hard-NO one. A YES
            # verdict is global and reaches --pairs directly from
            # classified/yes, so re-asking about a pair that already has one
            # buys nothing. It is a no-op against a -y-filtered frontier and
            # still load-bearing for _oneoff, whose supplied file has been
            # through no filter at all.
            #
            # The target-local exclusions come out here too, and this is only
            # mostly covered by the -r the generation passes: a frontier made
            # before the exclusion was written is still reviewable, and would
            # otherwise re-ask every pair in it.
            remaining = setops.diff(
                collated, self._union_no_pairs(target, scratch),
                scratch / "remaining.pairs")
            review_file = setops.diff(
                remaining, confirmed_yes, scratch / review_name)
            if fs.line_count(review_file) == 0:
                return self._converged(target, cutoff)
            code = submit.P2.run("submit p2", opts, [str(review_file)])
        if code != 0:
            return code
        code = evaluate.P2.run(
            "eval p2", opts, ["--no-filter", review_name])
        if code == 0:
            report(target)
        return code

    def _oneoff(self, target, supplied: Path, opts) -> int:
        confirmed_yes = config.classified(target.root, "yes")
        fs.raise_if_not_file(confirmed_yes)

        with tempfile.TemporaryDirectory(prefix="wf-best-oneoff-") as tmp:
            scratch = Path(tmp)
            canonical = setops.merge([supplied], scratch / "canonical.pairs")
            cutoff = fs.line_count(canonical)
            # The target-local exclusions have no cover at all here: nothing
            # filtered the supplied file, so without this a one-off keeps
            # asking about pairs the operator excluded.
            remaining = setops.diff(
                canonical, self._union_no_pairs(target, scratch),
                scratch / "remaining.pairs")
            reviewed = setops.diff(
                remaining, confirmed_yes, scratch / "reviewed.pairs")
            if fs.line_count(reviewed) == 0:
                raise ValueError(
                    f"one-off review has no candidates: all {cutoff} pairs "
                    f"are already classified or excluded"
                    f"{_names_local_no(target)}")

            queued, evaluating, archived = review_locations(target)
            self._in_flight(target, queued, evaluating)
            round_number = max(
                review_rounds(target, archived, "oneoff"), default=0) + 1
            review_name = (f"{target.review_prefix('oneoff')}{cutoff}."
                           f"r{round_number}.pairs")
            managed = canonical.with_name(review_name)
            canonical.rename(managed)
            code = submit.P2.run("submit p2", opts, [str(managed)])
            if code != 0:
                return code
            code = evaluate.P2.run_prepared(
                "eval p2", opts, [review_name], reviewed)

        if code == 0:
            report(target)
        return code

    @staticmethod
    def _converged(target, cutoff: int) -> int:
        """Every frontier pair already has a verdict: ordinary, not an error.

        `_frontier_outdated` reports this in status too -- the classify
        that produced those verdicts moved a classified set past the
        frontier's marker -- but this is reached in the window between a
        classify and the next frontier regen, so the detection point is still
        a guidance point. The cheap regeneration goes ahead of the searches,
        under the same condition and the same renderers the row uses.
        """
        print(f"{target.address}: no review candidates remain "
              f"({cutoff} frontier pairs, all already classified or "
              f"excluded){_names_local_no(target)}")
        inputs = Inputs(target)
        choices = inputs.search_choices()
        if inputs.frontier_outdated:
            regen = inputs.gen_top_command(inputs.source)
            choices = (Choice("refresh", regen), *choices)
        for line in render_choices(choices):
            print(line)
        if not choices:
            print("  converged: both searches are up to date")
        return 0


class Notes(command.Action):
    """Re-raise the notes of the target's current review round.

    The notes are what the operator actually works in, and nothing records
    them -- so deleting them used to mean pushing the bundle backwards through
    the queue to reach a derivation that was never stateful. This resolves
    which round is current and hands the primitive the same two inputs
    `best review` gave it.
    """

    def __init__(self):
        super().__init__(summary="notes    — recreate a target's review notes",
                         positional="SENTENCE")

    def parser(self):
        return _target_parser()

    def run(self, command_text, opts, argv) -> int:
        parsed = _action_target(self, command_text, opts, argv, 1)
        if isinstance(parsed, int):
            return parsed
        target, _ = parsed
        queued, evaluating, archived = review_locations(target)
        if queued:
            # Notes for this round have never existed, and making them is the
            # eval this bundle is still waiting for -- which moves state, so it
            # is named rather than run.
            raise ValueError(
                f"review bundle is queued: {queued[0].name}; "
                f"run {eval_p2_command(target, queued[0].name)}")
        if evaluating:
            bundle_name = evaluating[0].name
        else:
            if not archived:
                raise ValueError(
                    f"no review to recreate notes for {target.address}")
            # Preserve top-only selection by highest ordinal. Across separate
            # kind sequences there is no comparable ordinal, so archive time
            # identifies the last completed round instead.
            kinds = {round_.kind for round_ in archived}
            key = ((lambda round_: round_.ordinal) if len(kinds) == 1 else
                   (lambda round_: (round_.path.stat().st_mtime_ns,
                                    round_.ordinal, round_.kind)))
            latest = max(archived, key=key)
            if latest.kind == "oneoff" and opts.force:
                raise ValueError(
                    "cannot recreate notes for archived one-off source: "
                    f"{latest.path}")
            bundle_name = names.queue_stem("p2", latest.name)
        code = notes.P2.run("notes p2", opts, [bundle_name])
        if code == 0:
            report(target)
        return code


class Complete(command.Action):
    def __init__(self):
        super().__init__(summary="complete — complete target P2 review",
                         positional="SENTENCE")

    def parser(self):
        return _target_parser()

    def run(self, command_text, opts, argv) -> int:
        parsed = _action_target(self, command_text, opts, argv, 1)
        if isinstance(parsed, int):
            return parsed
        target, _ = parsed
        queued, evaluating, _ = review_locations(target)
        if queued:
            raise ValueError(
                f"review bundle is queued: {queued[0].name}; "
                f"run {eval_p2_command(target, queued[0].name)}")
        if not evaluating:
            raise ValueError(f"no review awaiting completion for {target.address}")

        code = complete_phase.P2.run(
            "complete p2", opts, [evaluating[0].name])
        if code != 0:
            return code
        report(target)
        return 0


COMMAND = command.Dispatcher(
    "best     — manage BEST PAIRS workflow state",
    {
        "status": Status(),
        "gen": Gen(),
        "prepare": Prepare(),
        "exclude": Exclude(),
        "review": Review(),
        "notes": Notes(),
        "complete": Complete(),
    },
)
