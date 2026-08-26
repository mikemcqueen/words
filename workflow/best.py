# best.py
#
# A BEST PAIRS target is one sentence, minimum word length, and exact segment
# count: best/s2/m4/g4. The dynamic body is intentionally absent from
# CONFIG_LAYOUT; this module validates its shapes and derives all state from the
# files already on disk.

import argparse
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

from dataclasses import dataclass
from pathlib import Path

from workflow import (
    classify, command, config, fs, log, setops, submit, usage,
    complete as complete_phase, eval as evaluate,
)


SHAPES = (
    (re.compile(r"s[1-9]"), "s[1-9]"),
    (re.compile(r"m\d+"), "m<N>"),
    (re.compile(r"g\d+"), "g<N>"),
)

DFS_LIMIT = 1_000_000
INDEX_NAME = "wiki-merged.2.index"
DICTIONARY_NAME = "words.big"


@dataclass(frozen=True)
class Target:
    root: Path
    best_dir: Path
    sentence: str
    universe: str
    segments: str

    @property
    def min_words(self) -> int:
        return int(self.universe[1:])

    @property
    def segment_count(self) -> int:
        return int(self.segments[1:])

    @property
    def address(self) -> str:
        return f"{self.sentence}/{self.universe}/{self.segments}"

    @property
    def sentence_dir(self) -> Path:
        return self.best_dir / self.sentence

    @property
    def universe_dir(self) -> Path:
        return self.sentence_dir / self.universe

    @property
    def target_dir(self) -> Path:
        return self.universe_dir / self.segments

    @property
    def letters(self) -> Path:
        return self.sentence_dir / "letters"

    def artifact(self, name: str) -> Path:
        return self.target_dir / name

    def command(self, verb: str, stage: str | None = None) -> str:
        argv = ["wf", "best", verb, self.sentence,
                "-g", str(self.segment_count)]
        if self.min_words != 4:
            argv.extend(["-m", str(self.min_words)])
        if stage is not None:
            argv.append(stage)
        return " ".join(argv)

    @property
    def review_prefix(self) -> str:
        # The final dot prevents g4 from matching g45.
        return (f"top.{self.sentence}.m{self.min_words}."
                f"g{self.segment_count}.")

    def seed(self) -> Path | None:
        matches = sorted(self.universe_dir.glob("seed*.pairs"))
        if not matches:
            return None
        if len(matches) > 1:
            found = ", ".join(path.name for path in matches)
            raise ValueError(f"multiple seeds in {self.universe_dir}: {found}")
        fs.raise_if_not_file(matches[0])
        return matches[0]


def _parse_address(address: str) -> list[str]:
    parts = address.split("/")
    if len(parts) > len(SHAPES):
        raise ValueError(f"invalid BEST PAIRS address: {address!r}")
    for depth, part in enumerate(parts):
        pattern, spelling = SHAPES[depth]
        if pattern.fullmatch(part) is None:
            raise ValueError(
                f"invalid BEST PAIRS address component {part!r}; "
                f"expected {spelling}")
    return parts


def _children(parent: Path, depth: int) -> list[Path]:
    pattern, _ = SHAPES[depth]
    return sorted(path for path in parent.iterdir()
                  if path.is_dir() and pattern.fullmatch(path.name))


def targets(root: Path, address: str | None = None) -> list[Target]:
    """Every fully-qualified target under an optional dynamic-tree prefix."""
    best_dir = config.path(root, ["best"])
    parts = _parse_address(address) if address is not None else []
    start = best_dir.joinpath(*parts)
    if parts:
        fs.raise_if_not_dir(start)

    found: list[Target] = []

    def walk(path: Path, names: list[str]) -> None:
        if len(names) == len(SHAPES):
            found.append(Target(root, best_dir, *names))
            return
        for child in _children(path, len(names)):
            walk(child, [*names, child.name])

    walk(start, parts)
    return found


def one_target(
    root: Path, sentence: str, min_words: int, segment_count: int
) -> Target:
    """Resolve one command target, requiring its hand-placed parent levels."""
    address = f"{sentence}/m{min_words}/g{segment_count}"
    parts = _parse_address(address)
    best_dir = config.path(root, ["best"])
    target = Target(root, best_dir, *parts)
    fs.raise_if_not_dir(target.sentence_dir)
    fs.raise_if_not_dir(target.universe_dir)
    if target.target_dir.exists():
        fs.raise_if_not_dir(target.target_dir)
    return target


@dataclass(frozen=True)
class State:
    message: str
    next_command: str | None = None
    place: Path | None = None


def _missing_artifact(target: Target, name: str) -> State | None:
    path = target.artifact(name)
    if path.exists():
        fs.raise_if_not_file(path)
        return None
    detail = ""
    if path.is_symlink():
        link = path.readlink()
        missing = link if link.is_absolute() else path.parent / link
        detail = f" (dangling symlink: {missing.resolve()})"
    return State(f"{name} missing{detail}",
                 next_command=target.command("gen", name))


def _newer(path: Path, reference: Path) -> bool:
    fs.raise_if_not_file(path)
    return path.stat().st_mtime_ns > reference.stat().st_mtime_ns


def _out_of_date(name: str, reasons: list[str], command_text: str) -> State:
    return State(f"{name} out of date ({', '.join(reasons)})",
                 next_command=command_text)


def _review_state(target: Target, top_segments: Path) -> State | None:
    queued, evaluating, archived = _review_locations(target)
    if queued:
        return State(f"review submitted ({queued[0].name})",
                     next_command=f"wf eval p2 {queued[0].name}")
    if evaluating:
        return State(f"review awaiting completion ({evaluating[0].name})",
                     next_command=target.command("complete"))

    newest = max((path.stat().st_mtime_ns for path in archived), default=0)
    if newest > top_segments.stat().st_mtime_ns:
        return None
    return State("review needed", next_command=target.command("review"))


def _review_locations(target: Target) -> tuple[list[Path], list[Path], list[Path]]:
    prefix = target.review_prefix
    queued_dir = config.path(target.root, ["p2", "queued"])
    queued = fs.globs(queued_dir, f"{prefix}*.pairs")
    eval_dir = config.path(target.root, ["p2", "eval"])
    evaluating = sorted(path for path in eval_dir.glob(f"{prefix}*")
                        if path.is_dir())

    in_flight = [*queued, *evaluating]
    if len(in_flight) > 1:
        found = ", ".join(path.name for path in in_flight)
        raise ValueError(f"multiple review bundles for {target.address}: {found}")
    done_dir = config.path(target.root, ["p2", "done", "in"])
    archived = fs.globs(done_dir, f"{prefix}*.pairs")
    return queued, evaluating, archived


def derive_state(target: Target) -> State:
    """Return the first missing, stale, or human-gated stage for target."""
    if not target.letters.exists():
        return State("letters missing", place=target.letters)
    fs.raise_if_not_file(target.letters)

    seed = target.seed()
    if seed is None:
        return State("seed missing", place=target.universe_dir / "seed*.pairs")

    missing = _missing_artifact(target, "dfs.seed")
    if missing is not None:
        return missing
    dfs_seed = target.artifact("dfs.seed")
    hard_no = config.classified(target.root, "no")
    reasons = []
    if _newer(seed, dfs_seed):
        reasons.append("seed changed")
    if _newer(hard_no, dfs_seed):
        reasons.append("hard-NO set changed")
    if reasons:
        return _out_of_date(
            "dfs.seed", reasons, target.command("gen", "dfs.seed"))

    missing = _missing_artifact(target, "top.segments")
    if missing is not None:
        return missing
    top_segments = target.artifact("top.segments")
    if _newer(dfs_seed, top_segments):
        return _out_of_date(
            "top.segments", ["dfs.seed changed"],
            target.command("gen", "top.segments"))

    review = _review_state(target, top_segments)
    if review is not None:
        return review

    missing = _missing_artifact(target, "best.pairs")
    if missing is not None:
        return missing
    best_pairs = target.artifact("best.pairs")
    confirmed_yes = config.classified(target.root, "yes")
    reasons = []
    if _newer(top_segments, best_pairs):
        reasons.append("top.segments changed")
    if _newer(confirmed_yes, best_pairs):
        reasons.append("confirmed-YES set changed")
    if _newer(hard_no, best_pairs):
        reasons.append("hard-NO set changed")
    if reasons:
        return _out_of_date(
            "best.pairs", reasons, target.command("gen", "best.pairs"))

    missing = _missing_artifact(target, "dfs.best")
    if missing is not None:
        return missing
    dfs_best = target.artifact("dfs.best")
    reasons = []
    if _newer(best_pairs, dfs_best):
        reasons.append("best.pairs changed")
    if _newer(hard_no, dfs_best):
        reasons.append("hard-NO set changed")
    if reasons:
        return _out_of_date(
            "dfs.best", reasons, target.command("gen", "dfs.best"))
    return State("up to date")


def report(target: Target) -> State:
    state = derive_state(target)
    print(f"{target.address}: {state.message}")
    if state.place is not None:
        print(f"  place: {state.place}")
    if state.next_command is not None:
        print(f"  next: {state.next_command}")
    return state


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
        for target in selected:
            report(target)
        return 0


def _command_not_found(name: str) -> FileNotFoundError:
    return FileNotFoundError(f"command not found: {name}")


def _require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise _command_not_found(name)


def _format_duration(elapsed: float) -> str:
    seconds = max(0, round(elapsed))
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes}m{seconds:02d}s" if minutes else f"{seconds}s"


def _seed_annotation(seed: Path) -> str:
    return seed.name[len("seed"):-len(".pairs")]


def _dfs_name(target: Target, seed: Path, limit: int) -> str:
    return (f"dfs.{target.sentence}{_seed_annotation(seed)}."
            f"m{target.min_words}.x2.g{target.segment_count}.{limit}")


def _display_dfs(argv: list[str], letters: Path) -> None:
    displayed = [shlex.quote(arg) for arg in argv]
    displayed[2] = f'"$(cat {shlex.quote(str(letters))})"'
    print("Running dfs-anagrams:", file=sys.stderr)
    print(f"  {' '.join(displayed)}", file=sys.stderr)


def _display_command(argv: list[str]) -> None:
    print(f"Running {argv[0]}:", file=sys.stderr)
    print(f"  {shlex.join(argv)}", file=sys.stderr)


def _same_link_target(link: Path, destination: Path) -> bool:
    if not link.is_symlink():
        return False
    spelled = link.readlink()
    target = spelled if spelled.is_absolute() else link.parent / spelled
    return target.resolve() == destination


def _publish_link(link: Path, destination: Path) -> None:
    if _same_link_target(link, destination):
        return
    tmp = link.with_name(link.name + ".tmp")
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(destination)
    tmp.replace(link)


def _dfs_inputs(target: Target, results_dir: Path) -> tuple[Path, Path, Path]:
    _require_command("dfs-anagrams")
    index = target.best_dir / "idx" / INDEX_NAME
    dictionary = target.best_dir / "dict" / DICTIONARY_NAME
    fs.raise_if_not_file(index)
    fs.raise_if_not_file(dictionary)
    fs.raise_if_not_file(target.letters)
    seed = target.seed()
    if seed is None:
        raise FileNotFoundError(
            f"seed missing: {target.universe_dir / 'seed*.pairs'}")
    fs.raise_if_not_file(config.classified(target.root, "no"))
    fs.raise_if_not_dir(results_dir)
    return index, dictionary, seed


def _gen_dfs_seed(target: Target, opts) -> None:
    results_dir = Path(opts.results_dir or "results").resolve()
    index, dictionary, seed = _dfs_inputs(target, results_dir)
    sentence_results = results_dir / target.sentence
    if sentence_results.exists():
        fs.raise_if_not_dir(sentence_results)

    if not target.target_dir.exists():
        if not opts.force:
            fs.raise_if_not_dir(target.target_dir)
        target.target_dir.mkdir()

    sentence_results.mkdir(exist_ok=True)
    fs.raise_if_not_dir(sentence_results)

    limit = DFS_LIMIT if opts.count is None else opts.count
    rendered = sentence_results / _dfs_name(target, seed, limit)
    scratch = rendered.with_name(rendered.name + ".tmp")
    letters = target.letters.read_text().rstrip("\r\n")
    argv = [
        "dfs-anagrams", str(index), letters,
        "-m", str(target.min_words),
        "-S", "20",
        "-p", "10000000",
        "-n", str(limit),
        "--word-bonus", "1",
        "--dict", str(dictionary),
        "--pairs", str(seed),
        "--exclude-pairs", str(target.root),
        "-x", "2",
        "-g", str(target.segment_count),
    ]
    _display_dfs(argv, target.letters)
    started = time.monotonic()
    with scratch.open("w") as output:
        subprocess.run(argv, stdout=output, check=True)
    scratch.replace(rendered)
    _publish_link(target.artifact("dfs.seed"), rendered)
    elapsed = _format_duration(time.monotonic() - started)
    log.success(f"Generated {fs.line_count(rendered)} results in {elapsed} "
                f"→ {rendered}")


def _gen_top_segments(target: Target, opts) -> None:
    if opts.force:
        raise ValueError("-f/--force is only valid for gen dfs.seed")
    if opts.results_dir is not None:
        raise ValueError("-r/--results-dir is only valid for DFS stages")
    _require_command("top-segments")
    dfs_seed = target.artifact("dfs.seed")
    fs.raise_if_not_file(dfs_seed)
    argv = ["top-segments", "--pairs"]
    if opts.count is not None:
        argv.extend(["-n", str(opts.count)])
    argv.append(str(dfs_seed))
    _display_command(argv)
    setops._place(argv, target.artifact("top.segments"), stable_mtime=True)
    count = fs.line_count(target.artifact("top.segments"))
    log.success(f"Generated {count} top segments → "
                f"{target.address}/top.segments")


def _build_best_pairs(target: Target) -> None:
    top_segments = target.artifact("top.segments")
    confirmed_yes = config.classified(target.root, "yes")
    hard_no = config.classified(target.root, "no")
    fs.raise_if_not_file(top_segments)
    fs.raise_if_not_file(confirmed_yes)
    fs.raise_if_not_file(hard_no)

    destination = target.artifact("best.pairs")
    before = set(destination.read_text().splitlines()) \
        if destination.exists() else set()
    with tempfile.TemporaryDirectory(prefix="wf-best-pairs-") as tmp:
        scratch = Path(tmp)
        collated = setops.merge([top_segments], scratch / "top.pairs")
        candidates = setops.merge(
            [collated, destination] if destination.exists() else [collated],
            scratch / "candidates.pairs")
        confirmed = setops.common(
            candidates, confirmed_yes, scratch / "confirmed.pairs")
        setops.diff(confirmed, hard_no, destination, stable_mtime=True)

    after = set(destination.read_text().splitlines())
    log.success(f"Generated {len(after)} best pairs "
                f"({len(after - before)} added, {len(before - after)} dropped) "
                f"→ {target.address}/best.pairs")
    if not after:
        log.warn("BEST PAIRS is empty; dfs.best would run without pair bonuses")


def _gen_best_pairs(target: Target, opts) -> None:
    if opts.force:
        raise ValueError("-f/--force is only valid for gen dfs.seed")
    if opts.results_dir is not None:
        raise ValueError("-r/--results-dir is only valid for DFS stages")
    if opts.count is not None:
        raise ValueError("-n is not valid for gen best.pairs")
    _build_best_pairs(target)


class Gen(command.Action):
    STAGES = {
        "dfs.seed": _gen_dfs_seed,
        "top.segments": _gen_top_segments,
        "best.pairs": _gen_best_pairs,
    }

    def __init__(self):
        super().__init__(summary="gen      — generate one BEST PAIRS artifact",
                         positional="SENTENCE STAGE")

    def parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("-g", type=int, metavar="N")
        parser.add_argument("-m", type=int, default=4, metavar="N")
        parser.add_argument("-r", "--results-dir", type=Path, metavar="DIR")
        parser.add_argument("-n", dest="count", type=int, metavar="N")
        return parser

    def run(self, command_text, opts, argv) -> int:
        rest = self.parse(opts, argv)
        if len(rest) < 2 or opts.g is None:
            return usage.missing_argument(self.format_help(command_text))
        if len(rest) > 2:
            return usage.invalid_argument(rest[2],
                                          self.format_help(command_text))
        sentence, stage = rest
        if stage not in self.STAGES:
            return usage.invalid_argument(stage, self.format_help(command_text))
        if opts.m < 0 or opts.g < 0 or (opts.count is not None and opts.count < 0):
            raise ValueError("-m, -g, and -n require non-negative integers")

        target = one_target(opts.dir, sentence, opts.m, opts.g)
        if not target.target_dir.exists() and not (
            stage == "dfs.seed" and opts.force
        ):
            fs.raise_if_not_dir(target.target_dir)
        self.STAGES[stage](target, opts)
        report(target)
        return 0


def _target_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-g", type=int, metavar="N")
    parser.add_argument("-m", type=int, default=4, metavar="N")
    return parser


def _action_target(action, command_text, opts, argv, positionals: int):
    rest = action.parse(opts, argv)
    if len(rest) < positionals or opts.g is None:
        return usage.missing_argument(action.format_help(command_text))
    if len(rest) > positionals:
        return usage.invalid_argument(
            rest[positionals], action.format_help(command_text))
    if opts.m < 0 or opts.g < 0:
        raise ValueError("-m and -g require non-negative integers")
    target = one_target(opts.dir, rest[0], opts.m, opts.g)
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

        queued, evaluating, archived = _review_locations(target)
        in_flight = [*queued, *evaluating]
        if in_flight:
            location = in_flight[0]
            raise ValueError(
                f"review bundle already in flight: {location.name} in "
                f"{location.parent}")

        round_number = len(archived) + 1
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
            "eval p2", opts, ["--no-filter", review_name])
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
        queued, evaluating, _ = _review_locations(target)
        if queued:
            raise ValueError(
                f"review bundle is queued: {queued[0].name}; "
                f"run wf eval p2 {queued[0].name}")
        if not evaluating:
            raise ValueError(f"no review awaiting completion for {target.address}")

        code = complete_phase.P2.run(
            "complete p2", opts, [evaluating[0].name])
        if code != 0:
            return code
        _build_best_pairs(target)
        report(target)
        return 0


COMMAND = command.Dispatcher(
    "best     — manage BEST PAIRS workflow state",
    {
        "status": Status(),
        "gen": Gen(),
        "exclude": Exclude(),
        "review": Review(),
        "complete": Complete(),
    },
)
