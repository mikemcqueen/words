# best.py
#
# A BEST PAIRS target is one sentence, minimum word length, and exact segment
# count: best/s2/m4/g4. The dynamic body is intentionally absent from
# CONFIG_LAYOUT; this module validates its shapes and derives all state from the
# files already on disk.

import re

from dataclasses import dataclass
from pathlib import Path

from workflow import command, config, fs, usage


SHAPES = (
    (re.compile(r"s[1-9]"), "s[1-9]"),
    (re.compile(r"m\d+"), "m<N>"),
    (re.compile(r"g\d+"), "g<N>"),
)


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
    if queued:
        return State(f"review submitted ({queued[0].name})",
                     next_command=f"wf eval p2 {queued[0].name}")
    if evaluating:
        return State(f"review awaiting completion ({evaluating[0].name})",
                     next_command=target.command("complete"))

    done_dir = config.path(target.root, ["p2", "done", "in"])
    archived = fs.globs(done_dir, f"{prefix}*.pairs")
    newest = max((path.stat().st_mtime_ns for path in archived), default=0)
    if newest > top_segments.stat().st_mtime_ns:
        return None
    return State("review needed", next_command=target.command("review"))


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


COMMAND = command.Dispatcher(
    "best     — manage BEST PAIRS workflow state",
    {"status": Status()},
)
