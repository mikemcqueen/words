import re

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from workflow import config, fs


SHAPES = (
    (re.compile(r"s[1-9]"), "s[1-9]"),
    (re.compile(r"[ou]-[a-z]+"), "[ou]-<letters>"),
    (re.compile(r"m[1-9]\d*"), "m<N>"),
    (re.compile(r"g[1-9]\d*"), "g<N>"),
)

# The two levels named by depth rather than by walking to them: the letter set,
# whose absence under a sentence is the pre-letter-set tree, and the universe,
# which is what such a sentence holds there instead.
LETTER_SET_DEPTH = 1
UNIVERSE_DEPTH = 2


@dataclass(frozen=True)
class Target:
    root: Path
    best_dir: Path
    sentence: str
    letter_set: str
    universe: str
    segments: str

    @property
    def min_words(self) -> int:
        return int(self.universe[1:])

    @property
    def segment_count(self) -> int:
        return int(self.segments[1:])

    @property
    def letter_form(self) -> str:
        """Which dfs-anagrams bag form the label names: only, or used."""
        return self.letter_set[0]

    @property
    def named_letters(self) -> str:
        """The letters the label names, passed to dfs-anagrams verbatim."""
        return self.letter_set[2:]

    @property
    def address(self) -> str:
        return (f"{self.sentence}/{self.letter_set}/"
                f"{self.universe}/{self.segments}")

    @property
    def sentence_dir(self) -> Path:
        return self.best_dir / self.sentence

    @property
    def letter_set_dir(self) -> Path:
        return self.sentence_dir / self.letter_set

    @property
    def universe_dir(self) -> Path:
        return self.letter_set_dir / self.universe

    @property
    def target_dir(self) -> Path:
        return self.universe_dir / self.segments

    @property
    def letters(self) -> Path:
        return self.sentence_dir / "letters"

    def artifact(self, name: str) -> Path:
        return self.target_dir / name

    def command(self, verb: str, stage: str | None = None,
                force: bool = False) -> str:
        argv = ["wf", "best", verb, self.sentence,
                f"-{self.letter_form}", self.named_letters,
                "-g", str(self.segment_count)]
        if self.min_words != 4:
            argv.extend(["-m", str(self.min_words)])
        if stage is not None:
            argv.append(stage)
        if force:
            argv.append("-f")
        return " ".join(argv)

    @property
    def review_prefix(self) -> str:
        # The letter set sits before the cutoff and the round ordinal, so the
        # bundle name stays a true prefix of everything derived under it. The
        # final dot prevents g4 from matching g45, and u-that from matching
        # u-thatandmore.
        return (f"top.{self.sentence}.m{self.min_words}."
                f"g{self.segment_count}.{self.letter_set}.")

    @property
    def seed_stem(self) -> str:
        """A seed is a property of the sentence and -m, and of nothing below.

        A restricted bag's candidates are a subset of the full bag's, so one
        P1 cycle -- the expensive stage -- serves every letter set at that -m.
        The level that keys on -m now sits below the letter set, so the seed
        moves up to the sentence and takes m<N> into its name.
        """
        return f"seed.m{self.min_words}"

    @property
    def seed_glob(self) -> Path:
        """How to spell the seed the tool looks for, in a diagnostic."""
        return self.sentence_dir / f"{self.seed_stem}.*.pairs"

    def seed(self) -> Path | None:
        matches = sorted(self.sentence_dir.glob(self.seed_glob.name))
        if not matches:
            return None
        if len(matches) > 1:
            found = ", ".join(path.name for path in matches)
            raise ValueError(f"multiple seeds in {self.sentence_dir}: {found}")
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


def _raise_if_pre_letter_set(sentence_dir: Path) -> None:
    """Diagnose a sentence still in the pre-letter-set shape.

    Every address now carries a letter set in second position, so nothing
    reaches s2/m4/g4 and no milestone restores it. Left undiagnosed the walk
    would simply find no children and status would print "no BEST PAIRS
    targets" -- work not yet done, about work that exists. This is the same
    diagnostic an address component gets, at the one level where an absent
    match is ambiguous.
    """
    pattern, _ = SHAPES[UNIVERSE_DEPTH]
    stale = sorted(path.name for path in sentence_dir.iterdir()
                   if path.is_dir() and pattern.fullmatch(path.name))
    if not stale:
        return
    name = sentence_dir.name
    raise ValueError(
        f"{name} predates the letter set: {', '.join(stale)} sits where a "
        f"letter set belongs. Move each under {name}/<letter-set>/ and "
        f"rename the seed to {name}/seed.m<N>.*.pairs")


def targets(root: Path, address: str | None = None) -> list[Target]:
    """Every fully-qualified target under an optional dynamic-tree prefix."""
    best_dir = config.path(root, ["best"])
    parts = _parse_address(address) if address is not None else []
    start = best_dir.joinpath(*parts)
    if len(parts) == len(SHAPES) and not start.exists():
        # Only a complete address names one target, and one target that does
        # not exist is answered rather than rejected: status synthesizes it and
        # reports the gen -f that would create it. Synthesis starts below the
        # sentence, because s<N>/ is the level the tool never creates and the
        # one holding both hand-placed files, so an absent one is a typo.
        target = Target(root, best_dir, *parts)
        fs.raise_if_not_dir(target.sentence_dir)
        return [target]
    if parts:
        fs.raise_if_not_dir(start)

    found: list[Target] = []

    def walk(path: Path, names: list[str]) -> None:
        if len(names) == len(SHAPES):
            found.append(Target(root, best_dir, *names))
            return
        children = _children(path, len(names))
        if not children and len(names) == LETTER_SET_DEPTH:
            _raise_if_pre_letter_set(path)
        for child in children:
            walk(child, [*names, child.name])

    walk(start, parts)
    return found


def one_target(root: Path, sentence: str, letter_set: str, min_words: int,
               segment_count: int) -> Target:
    """Resolve one command target, requiring its hand-placed parent level."""
    address = f"{sentence}/{letter_set}/m{min_words}/g{segment_count}"
    parts = _parse_address(address)
    best_dir = config.path(root, ["best"])
    target = Target(root, best_dir, *parts)
    fs.raise_if_not_dir(target.sentence_dir)
    # Every command calls this and only gen dfs.seed may create, so the three
    # levels below the sentence must be directories if they are there at all,
    # and are otherwise left to the caller.
    for path in (target.letter_set_dir, target.universe_dir,
                 target.target_dir):
        if path.exists():
            fs.raise_if_not_dir(path)
    return target


def _bag(text: str) -> str:
    """The multiset of characters of text, whitespace dropped, sorted.

    dfs-anagrams drops whitespace itself in clean_letters, so the reduction
    drops it too. Only `letters` carries any -- a label is [a-z]+ by the
    grammar -- but one function applies to both sides.
    """
    return "".join(sorted(re.sub(r"\s", "", text)))


def _without(bag: str, named: str) -> str | None:
    """bag less named, or None when named is not a multiset subset of bag."""
    remaining = Counter(bag)
    remaining.subtract(named)
    if any(count < 0 for count in remaining.values()):
        return None
    return "".join(sorted(remaining.elements()))


def _working_bag(letter_set: str, sentence: str) -> str | None:
    """The bag a letter set searches, or None if it names no proper subset."""
    named = _bag(letter_set[2:])
    remaining = _without(sentence, named)
    if remaining is None or not remaining:
        return None
    return named if letter_set[0] == "o" else remaining


def check_letter_set(target: Target) -> None:
    """Validate a letter set about to be created, against its own siblings.

    The label is the value: it is passed to dfs-anagrams verbatim, with no
    lookup and no stored mapping, so it is not canonical and two labels can
    name one working bag. The bag is derived here and never written down.

    Creation is the one condition, tested in one place -- the directory is
    absent -- and that gate is also what stops the check firing on itself: an
    existing letter set is one of its own siblings, and its bag necessarily
    equals its bag. It catches an anagram typo (u-thisandtaht) as a duplicate;
    a typo that changes the bag passes every check that can be written, which
    is what the -f refusal is for and why no smarter check replaces the flag.
    """
    if target.letter_set_dir.exists():
        return
    if not target.letters.exists():
        # There is nothing to validate a letter set against without the bag,
        # and both callers name the missing file a moment later.
        return
    fs.raise_if_not_file(target.letters)
    sentence = _bag(target.letters.read_text())
    named = _bag(target.named_letters)
    remaining = _without(sentence, named)
    if remaining is None:
        raise ValueError(f"{target.letter_set}: not a subset of "
                         f"{target.sentence}/letters")
    if not remaining:
        # Proper is what rejects a label naming every letter, on each side:
        # under u- the working bag is empty and dfs-anagrams would refuse it
        # only at run time, hours of tree later; under o- it is the full bag,
        # the unrestricted search the letter set exists to avoid.
        raise ValueError(f"{target.letter_set}: names every letter of "
                         f"{target.sentence}/letters, not a proper subset")
    working = named if target.letter_form == "o" else remaining
    for sibling in _children(target.sentence_dir, LETTER_SET_DEPTH):
        if _working_bag(sibling.name, sentence) == working:
            raise ValueError(f"{target.letter_set} searches the same letters "
                             f"as {sibling.name}")


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
    # dfs.seed is the first artifact derive_state reaches, so it is the only
    # stage reportable while the target directory is absent -- and the only
    # one -f is valid on. The offered command is therefore the one that works.
    return State(f"{name} missing{detail}",
                 next_command=target.command(
                     "gen", name, force=not target.target_dir.exists()))


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
        return State("seed missing", place=target.seed_glob)

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
