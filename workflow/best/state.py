import re

from dataclasses import dataclass
from pathlib import Path

from workflow import config, fs


SHAPES = (
    (re.compile(r"s[1-9]"), "s[1-9]"),
    (re.compile(r"m[1-9]\d*"), "m<N>"),
    (re.compile(r"g[1-9]\d*"), "g<N>"),
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


def _stamp(path: Path) -> Path:
    """The generation marker beside an artifact, touched by every gen."""
    return path.with_name(f".{path.name}.gen")


def _generated(path: Path) -> Path:
    """The path whose mtime dates an artifact against its own inputs.

    An artifact placed with stable_mtime keeps its mtime through a
    byte-identical regeneration, which is what stops a no-op from cascading
    into an hours-long DFS downstream. That same mtime cannot also answer
    "has this been generated since its input moved?", because the answer it
    gives never changes: an input that moves and yields identical content
    reports stale forever, and the gen offered to clear it is the one write
    stable_mtime suppresses. The marker answers that question -- it advances
    on every gen, no-op or not -- and the artifact's own mtime goes on
    answering the first for whatever reads it downstream.

    Absent the marker the artifact dates itself, which is what a tree built
    before the marker existed, or an artifact placed by hand, will do.
    """
    stamp = _stamp(path)
    return stamp if stamp.exists() else path


def mark_generated(path: Path) -> None:
    _stamp(path).touch()


def _out_of_date(name: str, reasons: list[str], command_text: str) -> State:
    return State(f"{name} out of date ({', '.join(reasons)})",
                 next_command=command_text)


def review_locations(target: Target) -> tuple[list[Path], list[Path], list[Path]]:
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


def _review_state(target: Target, top_segments: Path) -> State | None:
    queued, evaluating, archived = review_locations(target)
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
    if _newer(dfs_seed, _generated(top_segments)):
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
    generated = _generated(best_pairs)
    reasons = []
    if _newer(top_segments, generated):
        reasons.append("top.segments changed")
    if _newer(confirmed_yes, generated):
        reasons.append("confirmed-YES set changed")
    if _newer(hard_no, generated):
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
