import argparse
import tempfile

from pathlib import Path

from workflow import (
    classify, command, config, fs, log, names, notes, setops, submit, usage,
    complete as complete_phase, eval as evaluate,
)
from workflow.best import generate
from workflow.best.state import (
    check_letter_set, eval_p2_command, one_target, report, review_locations,
    review_rounds, targets, yes_pairs_argv,
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


class Status(command.Action):
    def __init__(self):
        super().__init__(summary="status   — report BEST PAIRS target state",
                         positional="[ADDRESS]")

    def run(self, command_text, opts, argv) -> int:
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
                report(target)
            except (OSError, ValueError) as e:
                log.error(f"{target.address}: {e}")
                failed = True
        return 1 if failed else 0


class Gen(command.Action):
    STAGES = {
        "dfs.seed": generate.gen_dfs_seed,
        "top.segments": generate.gen_top_segments,
        "best.pairs": generate.gen_best_pairs,
        "dfs.best": generate.gen_dfs_best,
    }

    def __init__(self):
        super().__init__(summary="gen      — generate one BEST PAIRS artifact",
                         positional="SENTENCE STAGE",
                         positional_help=(
                             ("SENTENCE", "sentence identifier under .wf/best "
                              "(for example, s2)"),
                             ("STAGE", "artifact to generate: dfs.seed "
                              "(provisional DFS results), top.segments "
                              "(frequent pairs from dfs.seed), best.pairs "
                              "(confirmed reviewed pairs), or dfs.best "
                              "(final DFS results using best.pairs)"),
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
        return parser

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

        target = one_target(opts.dir, sentence, letter_set, opts.m, opts.g)
        if stage == "dfs.seed":
            # Before anything is created, and gated on the same condition
            # status gates it on, so the label status refuses is the label gen
            # refuses.
            check_letter_set(target)
        missing = next(
            (path for path in (target.letter_set_dir, target.universe_dir,
                               target.target_dir) if not path.exists()), None)
        if missing is not None:
            if stage != "dfs.seed":
                fs.raise_if_not_dir(missing)
            elif not opts.force:
                raise FileNotFoundError(
                    f"directory does not exist: {missing}; "
                    "use -f to force creation")
        self.STAGES[stage](target, opts)
        report(target)
        return 0


def _target_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    _add_letter_set(parser)
    parser.add_argument("-g", type=int, metavar="N")
    parser.add_argument("-m", type=int, default=4, metavar="N")
    return parser


def _action_target(action, command_text, opts, argv, positionals: int):
    rest = action.parse(opts, argv)
    letter_set = _letter_set(opts)
    if len(rest) < positionals or opts.g is None or letter_set is None:
        return usage.missing_argument(action.format_help(command_text))
    if len(rest) > positionals:
        return usage.invalid_argument(
            rest[positionals], action.format_help(command_text))
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


class Review(command.Action):
    def __init__(self):
        super().__init__(summary="review   — submit a target for P2 review",
                         positional="SENTENCE")

    def parser(self):
        return _target_parser()

    def run(self, command_text, opts, argv) -> int:
        parsed = _action_target(self, command_text, opts, argv, 1)
        if isinstance(parsed, int):
            return parsed
        target, _ = parsed
        if opts.force:
            raise ValueError("-f/--force is not valid for best review")

        top_segments = target.artifact("top.segments")
        fs.raise_if_not_file(top_segments)
        cutoff = fs.line_count(top_segments)
        if cutoff == 0:
            raise ValueError(
                f"top.segments is empty; regenerate {target.artifact('dfs.seed')}")

        queued, evaluating, archived = review_locations(target)
        in_flight = [*queued, *evaluating]
        if in_flight:
            location = in_flight[0]
            raise ValueError(
                f"review bundle already in flight: {location.name} in "
                f"{location.parent}")

        # max + 1, not count + 1: an archive with a gap in it -- a round
        # deleted, or never archived -- would otherwise render a name that
        # already exists.
        round_number = max(review_rounds(target, archived), default=0) + 1
        review_name = (f"{target.review_prefix}{cutoff}."
                       f"r{round_number}.pairs")
        hard_no = config.classified(target.root, "no")
        fs.raise_if_not_file(hard_no)
        with tempfile.TemporaryDirectory(prefix="wf-best-review-") as tmp:
            scratch = Path(tmp)
            collated = setops.merge([top_segments], scratch / "top.pairs")
            review_file = setops.diff(
                collated, hard_no, scratch / review_name)
            if fs.line_count(review_file) == 0:
                raise ValueError(
                    "no review candidates remain after hard-NO exclusions")
            code = submit.P2.run("submit p2", opts, [str(review_file)])
        if code != 0:
            return code
        code = evaluate.P2.run(
            "eval p2", opts,
            ["--no-filter", review_name, *yes_pairs_argv(target)])
        if code == 0:
            report(target)
        return code


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
            rounds = review_rounds(target, archived)
            if not rounds:
                raise ValueError(
                    f"no review to recreate notes for {target.address}")
            # -f is the primitive's gate, not this one's: it is what says
            # "work from the archived round", and the primitive owns both the
            # refusal and the caveats that come with it.
            bundle_name = names.queue_stem(
                "p2", rounds[max(rounds)].name)
        code = notes.P2.run("notes p2", opts,
                            [bundle_name, *yes_pairs_argv(target)])
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
        generate.build_best_pairs(target)
        report(target)
        return 0


COMMAND = command.Dispatcher(
    "best     — manage BEST PAIRS workflow state",
    {
        "status": Status(),
        "gen": Gen(),
        "exclude": Exclude(),
        "review": Review(),
        "notes": Notes(),
        "complete": Complete(),
    },
)
