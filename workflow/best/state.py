import filecmp
import re
import tempfile

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from workflow import config, fs, setops


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

    def command(self, verb: str, *args: str, force: bool = False) -> str:
        """How to spell one of this target's own commands, for the operator.

        The trailing args are whatever the verb takes past the address -- a
        stage, a --source, an -n -- and are rendered in the order given, so
        one renderer serves gen, prepare and the bare verbs alike.
        """
        argv = ["wf", "best", verb, self.sentence,
                f"-{self.letter_form}", self.named_letters,
                "-g", str(self.segment_count)]
        if self.min_words != 4:
            argv.extend(["-m", str(self.min_words)])
        argv.extend(args)
        if force:
            argv.append("-f")
        return " ".join(argv)

    def review_prefix(self, kind: str) -> str:
        # The letter set sits before the cutoff and the round ordinal, so the
        # bundle name stays a true prefix of everything derived under it. The
        # final dot prevents g4 from matching g45, and u-that from matching
        # u-thatandmore.
        if kind not in REVIEW_KINDS:
            raise ValueError(f"unknown review kind: {kind!r}")
        return (f"{kind}.{self.sentence}.m{self.min_words}."
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


# ------------------------------------------------------------- search pairs


def search_bag(target: Target) -> str:
    """The multiset of letters this target's search may spend, sorted."""
    fs.raise_if_not_file(target.letters)
    working = _working_bag(target.letter_set,
                           _bag(target.letters.read_text()))
    if working is None:
        # check_letter_set guarantees this at creation time, but it returns
        # early for a letter set that already exists -- so an established
        # target whose letters file was edited under it reaches here.
        raise ValueError(
            f"{target.address}: {target.letter_set} names no proper subset "
            f"of {target.sentence}/letters")
    return working


def search_pair_sources(target: Target) -> list[Path]:
    """The pair files unioned into --pairs, in display order."""
    confirmed_yes = config.classified(target.root, "yes")
    fs.raise_if_not_file(confirmed_yes)
    sources = [confirmed_yes]
    best_pairs = target.artifact("best.pairs")
    # Optional, so absence is not an error -- but a directory or a dangling
    # symlink under that name is, rather than a silent omission, which is why
    # the second half of the gate is not redundant.
    if best_pairs.exists() or best_pairs.is_symlink():
        fs.raise_if_not_file(best_pairs)
        sources.append(best_pairs)
    return sources


def build_search_pairs(target: Target, scratch: Path) -> tuple[Path, int]:
    """The pairs dfs.best may use, and how many stood before the bag filter.

    The merge is what normalises a hand-edited best.pairs: it may be unsorted
    and may hold duplicates, and setops.diff shells out to comm, which
    requires LC_ALL=C order on both sides.

    Subtracting the hard-NO set is redundant against dfs-anagrams itself --
    emit tests exclude_pairs and returns before it reaches the pair flag --
    but it is what makes the returned count honest, and it keeps the file
    meaning what its name says.

    The bag filter is the load-bearing step, and it is safe because
    enumeration is bounded by the letter bag: a pair the bag cannot spell is
    never emitted, never looked up, and cannot affect a score whether it is
    in the set or out of it. The input is already sorted-unique and a filter
    preserves order, so the result is still a set comm and filecmp can read.
    """
    union = setops.merge(search_pair_sources(target), scratch / "union.pairs")
    standing = setops.diff(union, config.classified(target.root, "no"),
                           scratch / "standing.pairs")
    bag = search_bag(target)
    filtered = scratch / "dfs.best.pairs"
    with standing.open() as source, filtered.open("w") as out:
        for line in source:
            # Every line in the classified sets and in a best.pairs matches
            # ^[a-z]*,[a-z]*$, so the comma is the only non-letter to strip;
            # _bag drops whitespace for anything hand-typed.
            pair = line.strip()
            if not pair:
                continue
            if _without(bag, _bag(pair.replace(",", ""))) is not None:
                out.write(f"{pair}\n")
    return filtered, fs.line_count(standing)


# ----------------------------------------------------------- generation clock

SOURCES = ("seed", "best")

# How much a widen adds to the frontier cutoff it already has. One step, not a
# multiplier: the operator reads the rendered -n and edits it if they want
# more.
WIDEN_STEP = 1000


def _stamp(path: Path) -> Path:
    """The generation marker beside an artifact, written by every gen."""
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


def mark_generated(path: Path, text: str = "") -> None:
    """Record a successful generation, optionally with what it was made from.

    write_text rather than touch: a marker that also carries contents has to
    advance its mtime whether or not those contents changed, or the generation
    clock would stall exactly where stable_mtime stalls the content clock.
    """
    _stamp(path).write_text(text)


def top_segments_source(target: Target) -> str:
    """Which DFS artifact the current top.segments was generated from.

    A missing or empty marker means seed: that is what every tree built before
    the frontier had a choice of source was generated from. Anything else is
    rejected rather than guessed at -- the value selects an input to hours of
    search, and a wrong guess spends them on the wrong one.
    """
    marker = _stamp(target.artifact("top.segments"))
    if not marker.exists():
        return "seed"
    recorded = marker.read_text().strip()
    if not recorded:
        return "seed"
    if recorded not in SOURCES:
        raise ValueError(
            f"{marker}: unrecognised top.segments source {recorded!r}; "
            f"expected {' or '.join(SOURCES)}")
    return recorded


def _newer(path: Path, reference: Path) -> bool:
    fs.raise_if_not_file(path)
    return path.stat().st_mtime_ns > reference.stat().st_mtime_ns


def _dangling(path: Path) -> str:
    """How to name a symlink with nothing under it, or "" for anything else."""
    if path.exists() or not path.is_symlink():
        return ""
    link = path.readlink()
    missing = link if link.is_absolute() else path.parent / link
    return f" (dangling symlink: {missing.resolve()})"


# ------------------------------------------------------------------- review


REVIEW_KINDS = ("top", "oneoff")


@dataclass(frozen=True)
class ReviewRound:
    path: Path
    kind: str
    ordinal: int

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def parent(self) -> Path:
        return self.path.parent


def _review_round(target: Target, path: Path, kind: str,
                  evaluating: bool = False) -> ReviewRound:
    suffix = "" if evaluating else r"\.pairs"
    pattern = re.compile(
        re.escape(target.review_prefix(kind))
        + r"[1-9]\d*\.r([1-9]\d*)" + suffix)
    match = pattern.fullmatch(path.name)
    if match is None:
        raise ValueError(
            f"{path.name} is not a {kind} review round of {target.address}")
    return ReviewRound(path, kind, int(match.group(1)))


def _rounds_in(directory: Path, target: Target, *, files: bool) \
        -> list[ReviewRound]:
    rounds = []
    for kind in REVIEW_KINDS:
        prefix = target.review_prefix(kind)
        if files:
            paths = fs.globs(directory, f"{prefix}*.pairs")
        else:
            paths = sorted(path for path in directory.glob(f"{prefix}*")
                           if path.is_dir())
        rounds.extend(_review_round(target, path, kind, not files)
                      for path in paths)
    return sorted(rounds, key=lambda round_: round_.name)


def _check_round_ordinals(target: Target, rounds: list[ReviewRound]) -> None:
    found: dict[tuple[str, int], ReviewRound] = {}
    for round_ in rounds:
        key = (round_.kind, round_.ordinal)
        if key in found:
            earlier = found[key]
            raise ValueError(
                f"two {round_.kind} review rounds numbered r{round_.ordinal} "
                f"for {target.address}: {earlier.name}, {round_.name}")
        found[key] = round_


def review_locations(target: Target) \
        -> tuple[list[ReviewRound], list[ReviewRound], list[ReviewRound]]:
    """Queued, evaluating, and archived rounds belonging to one target."""
    queued_dir = config.path(target.root, ["p2", "queued"])
    queued = _rounds_in(queued_dir, target, files=True)
    eval_dir = config.path(target.root, ["p2", "eval"])
    evaluating = _rounds_in(eval_dir, target, files=False)

    in_flight = [*queued, *evaluating]
    if len(in_flight) > 1:
        found = ", ".join(path.name for path in in_flight)
        raise ValueError(f"multiple review bundles for {target.address}: {found}")
    done_dir = config.path(target.root, ["p2", "done", "in"])
    archived = _rounds_in(done_dir, target, files=True)
    _check_round_ordinals(target, [*in_flight, *archived])
    return queued, evaluating, archived


def eval_p2_command(target: Target, queued_name: str) -> str:
    """The `wf eval p2 ...` a queued review has to be run through.

    Every place that names it -- what `status` prints, and the refusals
    `complete` and `notes` raise -- renders it here, so the flags cannot drift
    between the message the operator reads and the command they then type.
    """
    return " ".join(["wf", "eval", "p2", queued_name])


def review_rounds(target: Target, discovered: list[ReviewRound],
                  kind: str) -> dict[int, ReviewRound]:
    """One kind of the target's completed review rounds, by ordinal.

    Takes the list `review_locations` already returned, so no second scan.

    This is the one place a name is taken apart, against names.py's rule, and
    it is admissible because the pattern is exactly what `Review` renders and
    the only thing recovered is the ordinal -- the dimension `Review` has to
    read back to render the next one. Selection is by that integer and not by
    sort order or mtime, because `r10` sorts before `r2` and an archive can be
    touched by anything. A sibling the pattern does not admit is rejected
    rather than guessed at: it is either a name nothing here rendered, or a
    prefix collision, and both are worth stopping for.

    Top-frontier and supplied-file rounds have separate sequences. Within the
    top sequence, which DFS artifact supplied the frontier remains
    informational: seed and best are two rounds of one top review kind.
    """
    if kind not in REVIEW_KINDS:
        raise ValueError(f"unknown review kind: {kind!r}")
    rounds: dict[int, ReviewRound] = {}
    for round_ in discovered:
        if round_.kind != kind:
            continue
        if round_.ordinal in rounds:
            raise ValueError(
                f"two {kind} review rounds numbered r{round_.ordinal} for "
                f"{target.address}: {rounds[round_.ordinal].name}, "
                f"{round_.name}")
        rounds[round_.ordinal] = round_
    return rounds


# -------------------------------------------------------------------- state


@dataclass(frozen=True)
class Choice:
    """One command the operator may run next, under the label it answers to."""
    label: str
    command: str


@dataclass(frozen=True)
class State:
    message: str
    choices: tuple[Choice, ...] = ()
    place: Path | None = None
    # A parenthetical under the message, for a row whose numbers are what make
    # the message actionable, and prose after the choices, for a way forward
    # that is not a command anything renders.
    detail: str | None = None
    note: tuple[str, ...] = ()


def render_choices(choices: tuple[Choice, ...]) -> list[str]:
    """The lines presenting one or several alternatives, indented for report.

    One choice is a line of its own under its label -- `next:` for the ordinary
    single way forward, `widen:` or `refine:` where the label is what says what
    kind of step it is. Several go under `choose next:`, with the labels padded
    so the commands line up and can be read down.
    """
    if not choices:
        return []
    if len(choices) == 1:
        return [f"  {choices[0].label}: {choices[0].command}"]
    width = max(len(choice.label) for choice in choices) + 1
    return ["  choose next:",
            *(f"    {(choice.label + ':').ljust(width)} {choice.command}"
              for choice in choices)]


@dataclass
class Inputs:
    """Everything a row may read, resolved once per target.

    Rows are not independent: row N runs only because rows 1..N-1 returned
    None, and reads files their absence would have reported. Split into
    functions, that guard stops being the preceding line and becomes invisible,
    so the accessors raise on an absent file rather than dating against one --
    a row reaching one out of order fails loudly instead of quietly.

    The conditions and the command strings live here rather than in the rows
    because several rows share them: `_no_usable_pairs` and `_next_search`
    both offer a reseed, `_no_frontier` and `_top_segments_behind_dfs` both
    offer a top.segments generation, and `_frontier_behind_classified` and
    `Review._converged` share one condition. Two renderings of one command is
    the drift `eval_p2_command` exists to prevent.
    """

    target: Target

    # ---------------------------------------------------------- resolved paths

    def _present(self, name: str) -> Path:
        path = self.target.artifact(name)
        fs.raise_if_not_file(path)
        return path

    def dfs(self, source: str) -> Path:
        return self.target.artifact(f"dfs.{source}")

    @cached_property
    def hard_no(self) -> Path:
        return config.classified(self.target.root, "no")

    @cached_property
    def confirmed_yes(self) -> Path:
        return config.classified(self.target.root, "yes")

    @cached_property
    def seed(self) -> Path | None:
        return self.target.seed()

    @cached_property
    def top_segments(self) -> Path:
        return self._present("top.segments")

    @cached_property
    def review(self) -> tuple[list[ReviewRound], list[ReviewRound],
                              list[ReviewRound]]:
        return review_locations(self.target)

    @cached_property
    def oneoff_in_flight(self) -> ReviewRound | None:
        queued, evaluating, _ = self.review
        return next((round_ for round_ in (*queued, *evaluating)
                     if round_.kind == "oneoff"), None)

    @cached_property
    def source(self) -> str:
        return top_segments_source(self.target)

    @cached_property
    def dfs_present(self) -> tuple[str, ...]:
        return tuple(source for source in SOURCES
                     if self.dfs(source).exists())

    # ------------------------------------------------------------- conditions

    @cached_property
    def seed_search_needed(self) -> list[str]:
        if self.seed is None:
            raise FileNotFoundError(f"seed missing: {self.target.seed_glob}")
        dfs_seed = self.dfs("seed")
        if not dfs_seed.exists():
            return ["missing"]
        reasons = []
        if _newer(self.seed, dfs_seed):
            reasons.append("seed changed")
        if _newer(self.hard_no, dfs_seed):
            reasons.append("hard-NO set changed")
        return reasons

    @cached_property
    def usable_pairs(self) -> tuple[int, int, bool]:
        """(standing pairs, how many this bag spells, whether dfs.best used them).

        status never writes the list: it is recomputed into a temp directory
        and compared against the one gen_dfs published. gen_dfs is the only
        writer, and only on a run that finished.

        filecmp caches by (path, size, mtime) on both sides, which is what
        setops._place has to clear because it reuses one temp path. Here the
        directory name is unique per call, so no two comparisons can share a
        key and there is nothing to clear.
        """
        with tempfile.TemporaryDirectory(prefix="wf-usable-pairs-") as tmp:
            pairs, standing = build_search_pairs(self.target, Path(tmp))
            stored = self.target.artifact("dfs.best.pairs")
            current = (stored.is_file()
                       and filecmp.cmp(pairs, stored, shallow=False))
            return standing, fs.line_count(pairs), current

    @cached_property
    def frontier_behind_classified(self) -> list[str]:
        """Classified sets written since the frontier was last generated."""
        generated = _generated(self.top_segments)
        reasons = []
        if _newer(self.confirmed_yes, generated):
            reasons.append("confirmed-YES set changed")
        if _newer(self.hard_no, generated):
            reasons.append("hard-NO set changed")
        return reasons

    @cached_property
    def best_search_needed(self) -> list[str]:
        # No pair this bag can spell makes dfs.best a strictly worse dfs.seed,
        # so there is no such search to offer and no reason to date one.
        _, usable, current = self.usable_pairs
        if usable == 0:
            return []
        dfs_best = self.dfs("best")
        if not dfs_best.exists():
            return ["missing"]
        reasons = []
        # A content comparison, not a clock comparison, and that is the point:
        # classified/yes is one file shared by every target, so a YES recorded
        # for one bag must not mark every other target's dfs.best stale and
        # offer hours that would reproduce the same file byte for byte. An
        # absent dfs.best.pairs reads as changed -- there is no record of what
        # the run used, and re-running is the only way to get one.
        if not current:
            reasons.append("usable pair set changed")
        if _newer(self.hard_no, dfs_best):
            reasons.append("hard-NO set changed")
        return reasons

    def search_needed(self, source: str) -> list[str]:
        return (self.seed_search_needed if source == "seed"
                else self.best_search_needed)

    def top_segments_behind(self, source: str) -> bool:
        """Has a finished search landed that the frontier was never made from?

        The search has to be current: a DFS file that is itself out of date
        wants re-running, not reading, and offering to read it would spend the
        frontier on results the next row is about to call stale.
        """
        dfs = self.dfs(source)
        if not dfs.exists() or self.search_needed(source):
            return False
        return _newer(dfs, _generated(self.top_segments))

    # -------------------------------------------------------------- renderers

    def prepare_command(self, source: str) -> str:
        # -f is the seed search's alone: it is what creates the levels below
        # the sentence, and only a seed search may create them.
        force = source == "seed" and not self.target.target_dir.exists()
        return self.target.command("prepare", "--source", source, force=force)

    def gen_top_command(self, source: str, count: int | None = None) -> str:
        argv = ["top.segments", "--source", source]
        if count is not None:
            argv.extend(["-n", str(count)])
        return self.target.command("gen", *argv)

    def review_command(self) -> str:
        return self.target.command("review")

    def search_choices(self) -> tuple[Choice, ...]:
        """The searches worth running now, cheapest first."""
        choices = []
        if self.seed_search_needed:
            choices.append(Choice("reseed", self.prepare_command("seed")))
        if self.best_search_needed:
            choices.append(Choice("refine", self.prepare_command("best")))
        return tuple(choices)


def _search_message(name: str, reasons: list[str]) -> str:
    if reasons == ["missing"]:
        return f"{name} missing"
    return f"{name} out of date ({', '.join(reasons)})"


# ---------------------------------------------------------------------- rows
#
# Each returns a State when its condition holds and None otherwise. ROWS is
# evaluated in order and the first State wins, so a row may assume every row
# above it declined -- which is what lets it read files their absence would
# have reported.


def _letters_missing(inputs: Inputs) -> State | None:
    letters = inputs.target.letters
    if letters.exists():
        # Present but not a regular file is an error, not a state: reporting
        # it as missing would send the operator to place a file that is there.
        fs.raise_if_not_file(letters)
        return None
    return State("letters missing", place=letters)


def _seed_missing(inputs: Inputs) -> State | None:
    if inputs.seed is not None:
        return None
    return State("seed missing", place=inputs.target.seed_glob)


def _review_queued(inputs: Inputs) -> State | None:
    queued, _, _ = inputs.review
    queued = [round_ for round_ in queued if round_.kind == "top"]
    if not queued:
        return None
    command_text = eval_p2_command(inputs.target, queued[0].name)
    return State(f"review submitted ({queued[0].name})",
                 choices=(Choice("next", command_text),))


def _review_evaluating(inputs: Inputs) -> State | None:
    _, evaluating, _ = inputs.review
    evaluating = [round_ for round_ in evaluating if round_.kind == "top"]
    if not evaluating:
        return None
    return State(f"review awaiting completion ({evaluating[0].name})",
                 choices=(Choice("next", inputs.target.command("complete")),))


def _top_segments_choices(inputs: Inputs,
                          sources: tuple[str, ...]) -> tuple[Choice, ...]:
    if len(sources) == 1:
        return (Choice("next", inputs.gen_top_command(sources[0])),)
    return tuple(Choice(source, inputs.gen_top_command(source))
                 for source in sources)


def _no_frontier(inputs: Inputs) -> State | None:
    top_segments = inputs.target.artifact("top.segments")
    if top_segments.exists():
        fs.raise_if_not_file(top_segments)
        return None
    detail = _dangling(top_segments)
    present = inputs.dfs_present
    if not present:
        # The one bootstrap gate, and deliberately narrower than "dfs.seed is
        # missing": after the first round dfs.seed is an input to the seed
        # search and nothing else, so a cleaned results/ must not force a fresh
        # seed DFS while dfs.best is alive.
        detail += "".join(_dangling(inputs.dfs(source)) for source in SOURCES)
        return State(f"no search results yet{detail}",
                     choices=(Choice("next", inputs.prepare_command("seed")),))
    return State(f"top.segments missing{detail}",
                 choices=_top_segments_choices(inputs, present))


def _review_needed(inputs: Inputs) -> State | None:
    _, _, archived = inputs.review
    top_rounds = [round_ for round_ in archived if round_.kind == "top"]
    newest = max((round_.path.stat().st_mtime_ns for round_ in top_rounds),
                 default=0)
    if newest > inputs.top_segments.stat().st_mtime_ns:
        return None
    command = (inputs.target.command("complete")
               if inputs.oneoff_in_flight is not None
               else inputs.review_command())
    return State(f"review needed (frontier from {inputs.source})",
                 choices=(Choice("next", command),))


def _no_usable_pairs(inputs: Inputs) -> State | None:
    """Nothing in the standing YES union fits this target's letters.

    The union goes green as soon as anyone anywhere says YES, so the dead-end
    test is whether any standing pair is spellable from this bag. Fires
    whenever it is not, not only at a dead end: widening the frontier is
    orders of magnitude cheaper than either search, and a wider frontier means
    more review candidates, more YES verdicts, and more union entries -- so
    the operator should see it before reaching for hours of DFS.
    """
    # The frontier is read before the early return, not after it, so the
    # accessor that reports an absent top.segments is the preceding line
    # rather than something a declining row would skip past.
    frontier = fs.line_count(inputs.top_segments)
    standing, usable, _ = inputs.usable_pairs
    if usable != 0:
        return None
    choices = [Choice("widen", inputs.gen_top_command(
        inputs.source, frontier + WIDEN_STEP))]
    if inputs.seed_search_needed:
        choices.append(Choice("reseed", inputs.prepare_command("seed")))
    # No refine: gen dfs.best and prepare --source best both refuse a pair set
    # this bag cannot spell. And no command retracts a verdict -- classify is
    # union-only -- so the way back is prose. A hand-edit of the hard-NO set
    # does not reopen the review either, which is why the review is named with
    # it; hand-adding to best.pairs needs no review at all.
    return State(
        "no confirmed pair fits this target's letters",
        detail=f"({standing} confirmed pairs, none spellable here)",
        choices=tuple(choices),
        note=(f"or retract NO verdicts in {inputs.hard_no}",
              f"   and run: {inputs.review_command()}",
              f"or add pairs by hand to "
              f"{inputs.target.artifact('best.pairs')}"))


def _top_segments_behind_dfs(inputs: Inputs) -> State | None:
    behind = tuple(source for source in SOURCES
                   if inputs.top_segments_behind(source))
    if not behind:
        return None
    names = " and ".join(f"dfs.{source}" for source in behind)
    return State(f"{names} generated after top.segments",
                 choices=_top_segments_choices(inputs, behind))


def _frontier_behind_classified(inputs: Inputs) -> State | None:
    """A classify landed after the frontier was last generated.

    Dated against the generation marker, not top.segments' own mtime:
    setops._place leaves the content mtime behind on a no-op regen, so a
    content-clock comparison would report stale forever. The marker clock is
    also what terminates the loop -- a no-op regen bumps it, this row
    declines, and _review_needed above it still declines because the content
    mtime did not move.
    """
    reasons = inputs.frontier_behind_classified
    if not reasons:
        return None
    return State(
        f"top.segments behind the classified sets ({', '.join(reasons)})",
        choices=_top_segments_choices(inputs, (inputs.source,)))


def _next_search(inputs: Inputs) -> State | None:
    messages = []
    if inputs.seed_search_needed:
        messages.append(_search_message("dfs.seed", inputs.seed_search_needed))
    if inputs.best_search_needed:
        messages.append(_search_message("dfs.best", inputs.best_search_needed))
    choices = inputs.search_choices()
    if not choices:
        return None
    return State("; ".join(messages), choices=choices)


@dataclass(frozen=True)
class Row:
    """One entry in the precedence table, with what it reads and what it settles.

    `requires` names the files a row's `Inputs` accessors raise without, and
    `provides` the one a row establishes by declining: `_no_frontier` returning
    None is what says there is a top.segments to read. In `derive_state` both
    are inert -- a row runs only because every row above it declined, which is
    exactly what establishes them, and the group comments below said so in
    prose. `walk` reports every row rather than stopping at the first, so it
    needs the same statement as data: which rows the winner leaves
    unanswerable, and must not call.
    """

    check: Callable[["Inputs"], State | None]
    requires: tuple[str, ...] = ()
    provides: str | None = None

    @property
    def name(self) -> str:
        return self.check.__name__


ROWS = (
    # G0 -- hand-placed inputs
    Row(_letters_missing),
    Row(_seed_missing, provides="seed"),
    # G1 -- open review: the frontier is being read and must not be rewritten
    Row(_review_queued),
    Row(_review_evaluating),
    # G2 -- no frontier; everything below has a top.segments
    Row(_no_frontier, provides="top.segments"),
    # G3 -- frontier not yet reviewed
    Row(_review_needed, requires=("top.segments",)),
    # G4 -- the dead end: no standing pair this bag can spell
    Row(_no_usable_pairs, requires=("top.segments", "seed")),
    # G5 -- a finished search whose frontier was never generated. Above G6
    # because generating from the newer DFS satisfies both conditions at once,
    # where a regen from the recorded source would bump the marker past the
    # finished search and lose it.
    Row(_top_segments_behind_dfs, requires=("top.segments", "seed")),
    # G6 -- a classify the frontier has not been rebuilt against. Below
    # _review_needed so a freshly generated frontier gets reviewed rather than
    # immediately regenerated, and above the searches so the seconds are
    # offered before the hours.
    Row(_frontier_behind_classified, requires=("top.segments",)),
    # G7 -- start the next search
    Row(_next_search, requires=("seed",)),
)


def derive_state(target: Target) -> State:
    """Return the first missing, stale, or human-gated stage for target."""
    inputs = Inputs(target)
    for row in ROWS:
        state = row.check(inputs)
        if state is not None:
            return state
    # Not "up to date", which would read as a lost write: the frontier has
    # been rebuilt against the classified sets as they stand, and no pair the
    # last dfs.best could use has changed, so re-running either search
    # reproduces what is already there.
    return State("converged")


# ------------------------------------------------------------- every row
#
# What `status --all` reports: the whole table rather than its first firing
# row. This is a diagnostic for the precedence itself -- which rows were
# asked, what each answered, and which the winner left unanswerable -- and
# not a second opinion about what to run next. derive_state stays the one
# authority for that, which is why report calls it either way.


@dataclass(frozen=True)
class Verdict:
    """One row's answer: fired, declined, or never asked."""

    row: Row
    state: State | None = None
    # The files the row would have read that no row above it established. Set
    # only when the row was skipped, and a skipped row has no state.
    unmet: tuple[str, ...] = ()

    @property
    def fired(self) -> bool:
        return self.state is not None


def walk_rows(target: Target) -> list[Verdict]:
    """Ask every row, skipping the ones the answers above it made unreadable.

    A row is asked only when the rows providing its `requires` declined, so
    the accessors keep raising on an absent file rather than being taught to
    return None for one. Up to and including the row derive_state returns
    this asks exactly what derive_state asked and gets the same answers: every
    row above the winner declined, and declining is what establishes a file.
    Past the winner the answers are real but partial -- a row reads the tree
    as it stands, and the winner's own fix is what moves the files the rows
    below it date against, so an `also` there is a prediction of the next
    round at best.
    """
    inputs = Inputs(target)
    established: set[str] = set()
    verdicts = []
    for row in ROWS:
        unmet = tuple(name for name in row.requires if name not in established)
        if unmet:
            verdicts.append(Verdict(row, unmet=unmet))
            continue
        state = row.check(inputs)
        if state is None and row.provides is not None:
            established.add(row.provides)
        verdicts.append(Verdict(row, state=state))
    return verdicts


def _row_labels(verdicts: list[Verdict]) -> list[str]:
    """What each verdict is called in the table.

    The first row to fire is the one derive_state returned, and every row
    below it that fires is reading a tree the winner's own fix will move --
    so the two are named apart. `also` is a symptom, not an alternative: the
    command that clears it is the winner's.
    """
    labels = []
    won = False
    for verdict in verdicts:
        if verdict.unmet:
            labels.append("n/a")
        elif not verdict.fired:
            labels.append("no")
        else:
            labels.append("also" if won else "won")
            won = True
    return labels


def render_rows(verdicts: list[Verdict]) -> list[str]:
    """The table under the report, one line per row, in precedence order.

    The row's own function name, not a prose label: the operator reading this
    is asking why the table chose what it chose, and the answer is in
    state.py under that name.

    Under a row that fired, the commands it offers, through the same renderer
    report uses so the table cannot drift from the report above it. Only the
    winner's are safe to run: an `also` row reads the tree as it stands, and
    the winner's own fix is what moves the files it dates against, so its
    commands are indented under the row that offers them rather than left to
    read as a second recommendation.
    """
    labels = _row_labels(verdicts)
    label_width = max(len(label) for label in labels) + 1
    name_width = max(len(verdict.row.name) for verdict in verdicts)
    lines = ["  rows:"]
    for verdict, label in zip(verdicts, labels):
        label = (label + ":").ljust(label_width)
        if verdict.unmet:
            trailer = f"not asked (needs {', '.join(verdict.unmet)})"
        elif verdict.state is None:
            trailer = ""
        else:
            trailer = verdict.state.message
        name = verdict.row.name.ljust(name_width)
        lines.append(f"    {label} {name}  {trailer}".rstrip())
        if verdict.state is not None:
            lines.extend(f"    {line}"
                         for line in render_choices(verdict.state.choices))
    return lines


def report(target: Target, rows: bool = False) -> State:
    state = derive_state(target)
    print(f"{target.address}: {state.message}")
    if state.detail is not None:
        print(f"  {state.detail}")
    if state.place is not None:
        print(f"  place: {state.place}")
    for line in render_choices(state.choices):
        print(line)
    for line in state.note:
        print(f"  {line}")
    oneoff = Inputs(target).oneoff_in_flight
    if oneoff is not None:
        print(f"  one-off review in flight ({oneoff.name}); close with: "
              f"{target.command('complete')}")
    if rows:
        for line in render_rows(walk_rows(target)):
            print(line)
    return state
