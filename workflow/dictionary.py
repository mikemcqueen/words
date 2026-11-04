# dictionary.py
#
# The workflow-managed dictionary: a hand-placed base, a set of hand-editable
# removal generations, and the derived file every workflow-configured consumer
# reads.
#
# Removing a word invalidates the dictionary, which invalidates the DFS, which
# costs hours. So a removal has to be cheap to *record* and cheap to apply to
# the next search, leaving the re-search something the operator chooses from the
# status table. That is the whole shape here: `remove words` archives a round
# and rebuilds seconds' worth of derived files, and `gen dict` rebuilds them
# again from whatever the records say now -- which is also the second half of a
# retraction, there being no un-remove command. Editing or deleting a generation
# is the first half.
#
# Three records per round. `removed/<input>.removed.N` holds the round's NO
# verdicts, the words removed, and is mutable by design so a removal can be
# retracted. `done/in/<input>.yes.N` holds its YES verdicts. `done/in/
# <input>.reviewed.N` archives every word the round listed, with or without a
# verdict, and is placed last: it is the round's commit record. A word listed
# with no verdict is undecided and comes back in a later review; see
# plans/dict-verdicts.md for the guards that keep verdicts from flipping.
#
# Nothing here calls nutrimatic's dict_remove.py, which edits a hardcoded
# dictionary in place. Three of its ideas carry over: the count-prefix strip,
# `comm -23` under LC_ALL=C (which setops already provides), and the .removed.N
# suffix -- ported onto a different stem and a different mutability.

import argparse
import re
import subprocess
import tempfile

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from workflow import command, config, fs, generation, log, setops, usage


# What a `top-segments --solo-words` row looks like, so a slice of one can be
# pasted in unedited beside a bare word list.
COUNT_PREFIX = re.compile(r"^ *[0-9]+ ")

# words.big is verified all-lowercase, so the guard is well defined -- and it is
# what stops a --pairs slice pasted by mistake from folding a row with a space
# in it into a generation that then never matches anything.
WORD = re.compile(r"[a-z]+")
LETTER = re.compile(r"[a-z]")

REMOVED = "removed"
REVIEWED = "reviewed"
YES = "yes"

# The workflow's own ordinal pattern (see best/state.py's review rounds), so no
# new convention is introduced. Names that do not match are ignored rather than
# parsed: vim leaves foo.removed.3~ and emacs leaves #foo.removed.3#, and under
# a bare glob those get unioned back in and a hand-retraction silently does not
# take.
ORDINAL = r"[1-9]\d*"

# ------------------------------------------------------------------- the tree


@dataclass(frozen=True)
class Tree:
    """Every path a dictionary command touches, resolved once through config.

    The accessors resolve through `config.path`, so a missing `dict/` is a hard
    layout error here. Returning a file path does not assert the file exists --
    that is each read boundary's own call, and the two boundaries disagree on
    purpose: `words.big` and `dict` are required, while `words.filtered` and
    `dict.filtered` are this module's outputs and are absent on a tree that has
    never rebuilt.
    """

    dir: Path
    base: Path
    derived: Path
    small_base: Path
    small_derived: Path
    removals: Path
    reviewed_inputs: Path
    reviewed_words: Path
    yes_words: Path
    no_words: Path
    enex_archives: Path

    @property
    def marker(self) -> Path:
        return generation.stamp(self.derived)


def tree(root: Path) -> Tree:
    return Tree(dir=config.path(root, ["dict"]),
                base=config.base_dictionary(root),
                derived=config.dictionary(root),
                small_base=config.small_base_dictionary(root),
                small_derived=config.small_dictionary(root),
                removals=config.removals(root),
                reviewed_inputs=config.reviewed_inputs(root),
                reviewed_words=config.reviewed_words(root),
                yes_words=config.yes_words(root),
                no_words=config.no_words(root),
                enex_archives=config.path(root, ["dict", "done", "out",
                                                "enex"]))


def _relative(root: Path, path: Path) -> str:
    """How to name one of these files to the operator: dict/removed/x.

    A path is a real thing they will `cd` into, so it is spelled from the
    workflow root rather than abbreviated.
    """
    return str(path.relative_to(root / config.CONFIG_ROOT))


# --------------------------------------------------------------- the records


def _pattern(kind: str) -> re.Pattern:
    return re.compile(rf".*\.{kind}\.({ORDINAL})")


def records(directory: Path, kind: str) -> list[tuple[int, Path]]:
    """The ordinal-named entries in one archive directory, in sorted order.

    One scan serves both readers: the derivation, which unions these, and the
    allocator, which takes one more than the largest ordinal it sees. Reading
    every directory entry rather than only the regular files is deliberate on
    the allocator's side -- anything already named with ordinal N, of any type,
    makes the next allocation at least N+1, so the allocated name is unreachable
    by construction. The greedy stem match reads foo.removed.3.removed.9 as 9,
    which is ugly and correct.
    """
    pattern = _pattern(kind)
    found = []
    for entry in sorted(directory.iterdir()):
        match = pattern.fullmatch(entry.name)
        if match is not None:
            found.append((int(match.group(1)), entry))
    return found


def _next_ordinal(*record_lists) -> int:
    """One sequence across every archive namespace, not one per stem.

    Global is what makes it a generation counter -- N alone orders every
    submission and identifies one -- and it keeps two submissions of the same
    filename from colliding, where a per-stem counter would give them each their
    own 1 and lose the ordering. Reading both namespaces also stops a surviving
    half of an interrupted round from having its ordinal reused.
    """
    ordinals = [ordinal for found in record_lists for ordinal, _ in found]
    return max(ordinals, default=0) + 1


# ------------------------------------------------------------ the submission


def read_words(path: Path) -> list[str]:
    """The submitted word list, normalized and shape-checked.

    The typed path is inspected before it is resolved or read: a symlink is
    refused here, so a symlink target's basename never becomes archive
    provenance.
    """
    if path.is_symlink():
        raise ValueError(f"symlink input not allowed: {path}")
    fs.raise_if_not_file(path)
    words = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        word = parse_word_row(path, number, line)
        if word is not None:
            words.append(word)
    return words


def parse_word_row(path: Path, number: int, line: str) -> str | None:
    """Return one normalized identity, ignoring only whitespace-only rows."""
    if not line.strip():
        return None
    word = COUNT_PREFIX.sub("", line).strip()
    if not word or WORD.fullmatch(word) is None:
        raise ValueError(f"{path}:{number}: not a word: {word!r}; "
                         f"expected one lowercase word per line")
    return word


def word_text(words) -> str:
    """Words as file text: one per line, each line ending in a newline."""
    return "".join(f"{word}\n" for word in words)


def write_words(path: Path, words) -> None:
    path.write_text(word_text(words))


def listing(words: list[str], limit: int = 20) -> str:
    """The first `limit` words, one per indented line, then how many more."""
    listed = "\n  ".join(words[:limit])
    remainder = len(words) - limit
    more = f"\n  ... and {remainder:,} more" if remainder > 0 else ""
    return f"  {listed}{more}"


@dataclass(frozen=True)
class FilterResult:
    source_words: int
    surviving_words: int
    filtered_words: int
    # Of filtered_words, those dropped only for being in a YES pair.
    yes_pair_words: int = 0

    @property
    def filtered(self) -> bool:
        return self.filtered_words != 0


# ------------------------------------------------- words in classified pairs
#
# The dictionary is global, so a word removed from it is gone for every
# sentence. A word in any standing YES pair -- global or any sentence's --
# must therefore never be removed, and a pair holding a removed word must
# never be classified YES. Word review drops such words before they are shown,
# publication refuses them as NO, and pair classification refuses the pairs.
# Nothing is classified YES on the pair's behalf.


def is_csv_pair(line: str) -> bool:
    """Whether line is one `word,word` pair: exactly two comma-separated
    fields, each nonempty, with no whitespace at either end and at least one
    a-z letter. A word may hold other characters, such as a space (`nathan
    hale`), an apostrophe (`ain't`) or an ampersand (`usc&gs`)."""
    fields = line.split(",")
    return len(fields) == 2 and all(
        field and field == field.strip() and LETTER.search(field)
        for field in fields)


def pair_words(path: Path, number: int, line: str) -> list[str]:
    """The two words of line `number` of path; none for a blank line.

    Refuses a line that is not a pair, naming the file and line.
    """
    if not line.strip():
        return []
    if not is_csv_pair(line):
        raise ValueError(
            f"invalid pair in {path} at line {number}: expected exactly two "
            f"nonempty comma-separated words, each with an a-z letter and no "
            f"surrounding whitespace; got {line!r}")
    return line.split(",")


def yes_pairs_by_word(root: Path) -> dict[str, list[tuple[str, Path]]]:
    """Each word in a standing YES pair, with every (pair, file) holding it."""
    held: dict[str, list[tuple[str, Path]]] = {}
    for path in config.yes_pair_files(root):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            for word in pair_words(path, number, line):
                held.setdefault(word, []).append((line, path))
    return held


def removed_records(root: Path, words: set[str]) -> dict[str, list[Path]]:
    """Each of words with the removed.N records holding it, so a refusal can
    name the file to edit."""
    held: dict[str, list[Path]] = {}
    for _, path in records(config.removals(root), REMOVED):
        for word in sorted(word_set(path) & words):
            held.setdefault(word, []).append(path)
    return held


def removed_word_pairs(root: Path, src: Path) -> list[str]:
    """src's pairs that hold a word classified NO, each naming the word and
    its removed.N records.

    Checked against dict/done/no.words, the same NO words `eval words` skips,
    so a retraction counts once `wf gen dict` has rebuilt it. A missing
    no.words means no word has been removed yet.
    """
    no_words = config.no_words(root)
    removed = word_set(no_words) if no_words.is_file() else set()
    hits_by_line = []
    for number, line in enumerate(src.read_text().splitlines(), 1):
        hits = [word for word in pair_words(src, number, line)
                if word in removed]
        if hits:
            hits_by_line.append((line, hits))
    if not hits_by_line:
        return []
    held = removed_records(
        root, {word for _, hits in hits_by_line for word in hits})
    found = []
    for line, hits in hits_by_line:
        where = "; ".join(
            f"{word}: " + ", ".join(_relative(root, path)
                                    for path in held.get(word, [no_words]))
            for word in hits)
        found.append(f"{line} ({where})")
    return sorted(set(found))


def classified_files(root: Path) -> list[Path]:
    """The YES and NO verdict unions a submitted word list is filtered by.

    Both are derived files; a workflow that has never run `wf gen dict` has
    neither, and filtering against a missing one would silently pass every
    word through.
    """
    files = [config.yes_words(root), config.no_words(root)]
    for classified in files:
        if not classified.is_file():
            raise ValueError(f"dictionary verdicts not generated: "
                             f"{classified}; run `wf gen dict`")
    return files


def classified_set(classified: list[Path]) -> set[str]:
    """Every word in the given verdict files."""
    words: set[str] = set()
    for path in classified:
        fs.raise_if_not_file(path)
        words.update(read_words(path))
    return words


def unclassified_rows(source: Path, classified: set[str]):
    """Each row of source, unchanged, with whether eval's filter keeps it.

    The one rule for which rows survive, shared by `eval words` and
    `submit words --separate`, so the parts submit cuts are the parts eval
    sees. Rows keep their own line endings; blank rows and classified words
    are not kept.
    """
    with source.open("r", newline="") as incoming:
        for number, line in enumerate(incoming, start=1):
            identity = parse_word_row(source, number, line.rstrip("\r\n"))
            yield line, identity, (identity is not None
                                   and identity not in classified)


def unreviewable_words(root: Path) -> tuple[set[str], set[str]]:
    """(words with a verdict, words in a YES pair): what word review skips.

    Shared by `eval words` and `submit words --separate`, so the parts submit
    cuts are the parts eval sees.
    """
    return (classified_set(classified_files(root)),
            set(yes_pairs_by_word(root)))


def filter_classified_words(source: Path, verdicts: set[str],
                            in_yes_pairs: set[str],
                            staged: Path) -> FilterResult:
    """Write unclassified presentation rows in order without moving source."""
    known = verdicts | in_yes_pairs
    source_words = surviving = filtered_count = yes_pair_count = 0
    with staged.open("w") as outgoing:
        for line, identity, kept in unclassified_rows(source, known):
            if identity is None:
                continue
            source_words += 1
            if not kept:
                filtered_count += 1
                if identity not in verdicts:
                    yes_pair_count += 1
                continue
            outgoing.write(line.rstrip("\r\n") + "\n")
            surviving += 1
    return FilterResult(source_words, surviving, filtered_count,
                        yes_pair_count)


def verdict_errors(reviewed, yes, no) -> tuple[list[str], list[str]]:
    """The words that break a round's verdicts, as two sorted lists.

    The first is the words ticked both YES and NO; the second the verdict words
    that are not in the reviewed set. Extraction and publication both check
    this, so the two checkpoints cannot drift apart.
    """
    yes, no = set(yes), set(no)
    return sorted(yes & no), sorted((yes | no) - set(reviewed))


def archive_name(name: str) -> str:
    """Validate a filename that will be interpolated into archive records."""
    if name in ("", ".", "..") or Path(name).name != name:
        raise ValueError(f"cannot name an archive after {name!r}")
    return name


def archive_stem(path: Path) -> str:
    """The submitted file's own basename, which is the provenance.

    It answers "where did this come from" with the name the operator gave it,
    which no generated name can reconstruct.
    """
    return archive_name(path.name)


# ------------------------------------------------------------- set operations


@contextmanager
def _operation(what: str, *inputs: Path):
    """Report a failed `sort` or `comm` as a diagnostic, not a traceback.

    The subprocess's own message may precede this one; what it cannot say is
    which of the dictionary's several set operations it was running. No
    chaining: the operator is being told what failed and over what, and a
    Python traceback is not part of that. Because setops publishes only after
    the subprocess succeeds, a failure preserves both prior derived outputs and
    does not advance the marker.
    """
    try:
        yield
    except subprocess.CalledProcessError:
        named = ", ".join(str(path) for path in inputs)
        raise ValueError(f"dictionary {what} failed over {named}") from None


def _union(srcs: list[Path], dst: Path) -> Path:
    """sort -u over srcs, where no source at all is the empty set.

    setops.merge raises on an empty source list, so a tree with nothing recorded
    writes the empty aggregate directly rather than asking for a union of
    nothing.
    """
    if not srcs:
        dst.write_text("")
        return dst
    return setops.merge(srcs, dst)


# ------------------------------------------------------- the prepared batch


@dataclass(frozen=True)
class Prepared:
    """One staged output and what its destination is allowed to be.

    An archive is never overwritten, so its destination must have no directory
    entry of any type. A derived output or a marker is replaceable and has only
    to satisfy the ordinary absent-or-regular-file contract.
    """

    staged: setops.Staged
    replace: bool = True

    @property
    def dst(self) -> Path:
        return self.staged.dst


def _preflight(prepared: list[Prepared]) -> None:
    """Check the complete batch before the first final move.

    Validation, derivation, and destination-shape failures happen before this
    point, so an ordinary preparation failure leaves no durable record and no
    changed output. Cross-filesystem and unexpected rename failures remain in
    the deliberately accepted publication window.
    """
    for placement in prepared:
        dst = placement.dst
        if placement.replace:
            fs.optional_file(dst)
        elif dst.exists() or dst.is_symlink():
            raise fs.file_already_exists_error(dst)


def _commit(prepared: list[Prepared]) -> None:
    """Publish the batch in order through the prepared-placement primitive.

    Byte-identical derived outputs are dropped rather than renamed, which is
    what preserves their content mtimes -- and their content mtime is what dates
    every search downstream.
    """
    for placement in prepared:
        if placement.staged.replaced:
            setops.place(placement.staged)


@contextmanager
def _staging(tree: Tree):
    with tempfile.TemporaryDirectory(prefix="wf-dictionary-") as tmp:
        yield Path(tmp)


# ---------------------------------------------------------------- derivation


def _record_union(staging: Path, name: str, recorded: list[Path],
                  new: Path | None = None) -> tuple[Path, Path]:
    """The union of one verdict kind's records, built once.

    Returns (prior, prospective): the union of `recorded`, and with `new` the
    union of `recorded` plus `new`. Without `new`, prospective is prior.
    """
    prior = staging / f"prior.{name}"
    with _operation(f"{name} union", *recorded):
        _union(recorded, prior)
    if new is None:
        return prior, prior
    with _operation(f"{name} union", prior, new):
        prospective = _union([prior, new], staging / f"next.{name}")
    return prior, prospective


@dataclass(frozen=True)
class Derived:
    """Every derived file a rebuild stages, in placement order."""

    reviewed_words: setops.Staged
    yes_words: setops.Staged
    no_words: setops.Staged
    dictionary: setops.Staged
    small_dictionary: setops.Staged

    def placements(self) -> list["Prepared"]:
        return [Prepared(self.reviewed_words), Prepared(self.yes_words),
                Prepared(self.no_words), Prepared(self.dictionary),
                Prepared(self.small_dictionary)]


def _stage_all_derived(tree: Tree, staging: Path, *, no: Path, yes: Path,
                       reviewed: list[Path]) -> Derived:
    """Stage reviewed.words, yes.words, no.words and both dictionaries.

    `no` and `yes` are the verdict unions already built; `reviewed` is every
    reviewed.N record, including a publication's own staged archive.

    stable_mtime is load-bearing and not tidiness. Inputs.dictionary, the
    frontier's "dictionary changed" reason, and both DFS freshness predicates
    date against words.filtered, so a removal that removes nothing -- a word
    not in the base -- must leave it byte-identical and cost nothing
    downstream.
    """
    dst, path = tree.reviewed_words, staging / config.REVIEWED_WORDS_NAME
    if reviewed:
        with _operation("reviewed-word union", *reviewed):
            reviewed_words = setops.stage_merge(reviewed, dst, path,
                                                stable_mtime=True)
    else:
        reviewed_words = setops.stage_text("", dst, path, stable_mtime=True)
    yes_words = setops.stage_text(yes.read_text(), tree.yes_words,
                                  staging / config.YES_WORDS_NAME,
                                  stable_mtime=True)
    no_words = setops.stage_text(no.read_text(), tree.no_words,
                                 staging / config.NO_WORDS_NAME,
                                 stable_mtime=True)
    with _operation("dictionary derivation", tree.base, no):
        derived = setops.stage_diff(tree.base, no, tree.derived,
                                    staging / config.BIG_DICTIONARY_NAME,
                                    stable_mtime=True)
    with _operation("small dictionary derivation", tree.small_base, no):
        small = setops.stage_diff(tree.small_base, no, tree.small_derived,
                                  staging / config.SMALL_DICTIONARY_NAME,
                                  stable_mtime=True)
    return Derived(reviewed_words, yes_words, no_words, derived, small)


def _mark_generated(tree: Tree) -> None:
    """Advance the generation clock, once the derived files have landed.

    Written even where both derived files came out byte-identical and
    stable_mtime left them alone. Without that the staleness row is a
    permanently stuck row: submit a word that is not in the base, the
    generation lands, stable_mtime correctly pins the derived file, and "input
    newer than output" fires for ever with the one write that would clear it
    suppressed.

    After the commit rather than inside it, and not staged: a crash between the
    derived file and the marker leaves the generation clock behind the content
    clock, so the row re-offers the rebuild and the state heals. A marker
    published from a batch carries its staging mtime through the rename, which
    dates the clock before the files it is meant to date -- the ordering
    gen_top_segments spells out for its own marker.
    """
    generation.mark_generated(tree.derived)


# ------------------------------------------------------------------ reporting


def _before(path: Path) -> int | None:
    """The destination's line count, or None where nothing is there yet."""
    return fs.line_count(path) if path.is_file() else None


def _count_line(label: str, staged: setops.Staged) -> str:
    """One of three cases, because there is not always a before-count.

    Which it is cannot come from comparing the counts: a rebuild that applies a
    new removal and picks up a hand-retraction in the same pass lands on an
    equal count with different bytes, and that is a move. Byte-identity is what
    stable_mtime already decided during staging.

    Read before the commit, while the staged file is still where it was
    written: the rename that publishes it is what takes it away.
    """
    before = _before(staged.dst)
    after = fs.line_count(staged.path)
    if before is None:
        return f"{label} {after} (new)"
    if not staged.replaced:
        return f"{label} {before} (unchanged)"
    if after == before:
        return f"{label} {before} -> {after}"
    return f"{label} {before} -> {after} ({after - before:+})"


def _report(root: Path, tree: Tree, derived: Derived) -> list[str]:
    """The count lines, composed before anything is published."""
    return [_count_line(config.BIG_DICTIONARY_NAME, derived.dictionary),
            _count_line(config.SMALL_DICTIONARY_NAME, derived.small_dictionary),
            _count_line(_relative(root, tree.reviewed_words),
                        derived.reviewed_words),
            _count_line(_relative(root, tree.yes_words), derived.yes_words),
            _count_line(_relative(root, tree.no_words), derived.no_words)]


# ------------------------------------------------------------------ commands


@dataclass(frozen=True)
class Publication:
    ordinal: int
    yes_dst: Path | None
    removal_dst: Path
    reviewed_dst: Path
    enex_dst: Path | None
    reviewed: int
    yes: int
    no: int
    new_to_union: int
    effective: int
    count_lines: tuple[str, ...]


def word_set(path: Path) -> set[str]:
    """A record's words, read as the unions read it: one word per line."""
    return set(path.read_text().split())


def _closing(retry: str) -> str:
    return f"no dictionary records were changed\n{retry}"


def _check_verdicts(reviewed_input: Path, yes: Path | None, no: Path,
                    reviewed: set[str], yes_set: set[str], no_set: set[str],
                    retry: str) -> None:
    """Publication step 1: YES and NO disjoint, and both within reviewed."""
    conflicting, extra = verdict_errors(reviewed, yes_set, no_set)
    if not conflicting and not extra:
        return
    lines = []
    if conflicting:
        lines.append(f"{len(conflicting):,} words are in both {yes} and "
                     f"{no}:\n{listing(conflicting)}")
    if extra:
        named = f"{no}" if yes is None else f"{yes} or {no}"
        lines.append(f"{len(extra):,} words in {named} are not in "
                     f"{reviewed_input}:\n{listing(extra)}")
    raise ValueError("\n".join(lines) + "\n" + _closing(retry))


def _holders(root: Path, words: list[str],
             recorded: list[Path]) -> dict[str, list[str]]:
    """Which records hold each word: only read on the flip-guard failure."""
    wanted = set(words)
    held: dict[str, list[str]] = {word: [] for word in words}
    for path in recorded:
        for word in sorted(word_set(path) & wanted):
            held[word].append(_relative(root, path))
    return held


def _flip_listing(root: Path, words: list[str], recorded: list[Path],
                  limit: int = 20) -> str:
    shown = words[:limit]
    held = _holders(root, shown, recorded)
    width = max(len(word) for word in shown)
    lines = [f"  {word.ljust(width)}  {', '.join(held[word])}"
             for word in shown]
    if len(words) > limit:
        lines.append(f"  ... and {len(words) - limit:,} more")
    return "\n".join(lines)


def _flip_guard(root: Path, yes_set: set[str], no_set: set[str],
                prior_yes: set[str], prior_no: set[str],
                yes_records: list[Path], removal_records: list[Path],
                retry: str) -> None:
    """Publication step 3: a standing verdict is never recorded the other way."""
    no_flipped = sorted(no_set & prior_yes)
    yes_flipped = sorted(yes_set & prior_no)
    if not no_flipped and not yes_flipped:
        return
    sections = []
    if no_flipped:
        sections.append(
            f"cannot record {len(no_flipped):,} NO words already classified "
            f"YES:\n{_flip_listing(root, no_flipped, yes_records)}")
    if yes_flipped:
        sections.append(
            f"cannot record {len(yes_flipped):,} YES words already classified "
            f"NO:\n{_flip_listing(root, yes_flipped, removal_records)}\n"
            f"to retract an earlier NO instead, take the word out of the "
            f"removed.N records listed beside it, run `wf gen dict`, then "
            f"retry")
    raise ValueError("\n".join(sections) + "\n" + _closing(retry))


def _yes_pair_guard(root: Path, no_set: set[str], retry: str) -> None:
    """Publication step 1b: a word in a standing YES pair is never removed.

    Checked against the pairs as they stand now, not as they stood when the
    review was opened, so a pair classified YES in between still counts.
    """
    held = yes_pairs_by_word(root)
    words = sorted(no_set & held.keys())
    if not words:
        return
    rows = [(word, pair, _relative(root, path))
            for word in words for pair, path in held[word]]
    limit = 20
    shown = rows[:limit]
    word_width = max(len(row[0]) for row in shown)
    pair_width = max(len(row[1]) for row in shown)
    lines = [f"  {word.ljust(word_width)}  {pair.ljust(pair_width)}  {path}"
             for word, pair, path in shown]
    if len(rows) > limit:
        lines.append(f"  ... and {len(rows) - limit:,} more")
    raise ValueError(
        f"cannot record {len(words):,} NO words that are in classified YES "
        f"pairs:\n" + "\n".join(lines) + "\n"
        f"to remove such a word, take its pairs out of the yes.pairs files "
        f"listed beside it first\n" + _closing(retry))


def _by_ordinal(found: list[tuple[int, Path]]) -> dict[int, list[Path]]:
    grouped: dict[int, list[Path]] = {}
    for ordinal, path in found:
        grouped.setdefault(ordinal, []).append(path)
    return grouped


def _words_in(paths: list[Path], wanted: set[str]) -> set[str]:
    found: set[str] = set()
    for path in paths:
        found.update(word_set(path) & wanted)
    return found


def _late_verdicts(new: set[str], reviewed_records, yes_records,
                   removal_records) -> dict[str, int]:
    """How many earlier rounds listed each newly classified word unclassified.

    A round counts only once its reviewed.N exists -- that record is placed
    last -- and its records are matched to it by ordinal alone.
    """
    counts: dict[str, int] = {}
    if not new:
        return counts
    yes_by = _by_ordinal(yes_records)
    removed_by = _by_ordinal(removal_records)
    for ordinal, paths in _by_ordinal(reviewed_records).items():
        listed = _words_in(paths, new)
        if not listed:
            continue
        classified = (_words_in(yes_by.get(ordinal, []), listed)
                      | _words_in(removed_by.get(ordinal, []), listed))
        for word in listed - classified:
            counts[word] = counts.get(word, 0) + 1
    return counts


def _late_guard(counts: dict[str, int], yes_set: set[str],
                force_verdicts: bool, retry: str, accept: str) -> None:
    """Publication step 4: warn on each late verdict; fail without force.

    Each warning names the verdict this round gives the word; a word not in
    `yes_set` was ticked NO. `accept` is the command that records them.
    """
    if not counts:
        return
    for word in sorted(counts):
        verdict = "YES" if word in yes_set else "NO"
        times = "time" if counts[word] == 1 else "times"
        log.warn(f"{word} {verdict} was reviewed {counts[word]} {times} "
                 f"without a verdict")
    if force_verdicts:
        return
    raise ValueError(f"{len(counts):,} words were reviewed earlier without a "
                     f"verdict\nno dictionary records were changed\n"
                     f"Run `{accept}` to accept these verdicts\n"
                     f"or {retry}")


def publish_words(root: Path, stem: str, reviewed_input: Path, *,
                  yes: Path | None, no: Path, force_verdicts: bool,
                  retry: str, accept: str,
                  enex: Path | None = None) -> Publication:
    """Publish one round's verdicts and rebuild the derived files.

    `yes=None` records no YES verdicts and writes no yes.N. `retry` is the
    caller's next step, which ends every validation and guard failure;
    `accept` is the caller's command with --force-verdicts, offered first when
    the late-verdict guard fails.
    Validate, guard, stage and preflight everything, then place the round's
    records -- reviewed.N last of them, as its commit record -- then the
    derived files, the archived notes and the marker. An ordinary failure
    leaves no durable record and no changed output.
    """
    archive_name(stem)
    reviewed_set = set(read_words(reviewed_input))
    yes_set = set(read_words(yes)) if yes is not None else set()
    no_set = set(read_words(no))
    _check_verdicts(reviewed_input, yes, no, reviewed_set, yes_set, no_set,
                    retry)
    _yes_pair_guard(root, no_set, retry)

    dictionary = tree(root)
    fs.raise_if_not_file(dictionary.base)
    fs.raise_if_not_file(dictionary.small_base)

    removal_records = records(dictionary.removals, REMOVED)
    reviewed_records = records(dictionary.reviewed_inputs, REVIEWED)
    yes_records = records(dictionary.reviewed_inputs, YES)
    ordinal = _next_ordinal(removal_records, reviewed_records, yes_records)
    yes_dst = (dictionary.reviewed_inputs / f"{stem}.{YES}.{ordinal}"
               if yes is not None else None)
    removal_dst = dictionary.removals / f"{stem}.{REMOVED}.{ordinal}"
    reviewed_dst = dictionary.reviewed_inputs / f"{stem}.{REVIEWED}.{ordinal}"
    enex_dst = (dictionary.enex_archives / f"{stem}.{REVIEWED}.{ordinal}"
                if enex is not None else None)

    if enex is not None:
        fs.raise_if_not_dir(enex)
        fs.raise_if_not_dir(dictionary.enex_archives)
        assert enex_dst is not None
        if enex_dst.exists() or enex_dst.is_symlink():
            raise fs.file_already_exists_error(enex_dst)

    removal_paths = [path for _, path in removal_records]
    yes_paths = [path for _, path in yes_records]
    with _staging(dictionary) as staging:
        # Step 2: each verdict union, built once. removed.N holds every word
        # this round ticked NO, standing NOs included, so each round's records
        # say what that round decided.
        removal = setops.stage_text(word_text(sorted(no_set)),
                                    removal_dst, staging / "removal.archive")
        no_prior, no_union = _record_union(staging, "removed", removal_paths,
                                           removal.path)
        prior_no = word_set(no_prior)
        removed_words = sorted(no_set - prior_no)
        new_removals = staging / "new.removals"
        write_words(new_removals, removed_words)
        yes_archive = None
        if yes_dst is not None:
            yes_archive = setops.stage_text(
                word_text(sorted(yes_set)), yes_dst,
                staging / "yes.archive")
        yes_prior, yes_union = _record_union(
            staging, "yes", yes_paths,
            yes_archive.path if yes_archive is not None else None)
        prior_yes = word_set(yes_prior)

        _flip_guard(root, yes_set, no_set, prior_yes, prior_no, yes_paths,
                    removal_paths, retry)
        _late_guard(_late_verdicts((yes_set - prior_yes) | (no_set - prior_no),
                                   reviewed_records, yes_records,
                                   removal_records),
                    yes_set, force_verdicts, retry, accept)

        reviewed = setops.stage_text(
            word_text(sorted(reviewed_set)), reviewed_dst,
            staging / "reviewed.archive")
        with _operation("removed-word intersection", new_removals,
                        dictionary.base):
            effective = setops.common(new_removals, dictionary.base,
                                      staging / "effective")
        derived = _stage_all_derived(
            dictionary, staging, no=no_union, yes=yes_union,
            reviewed=[path for _, path in reviewed_records] + [reviewed.path])

        archives = [Prepared(removal, replace=False),
                    Prepared(reviewed, replace=False)]
        if yes_archive is not None:
            archives.insert(0, Prepared(yes_archive, replace=False))
        batch = archives + derived.placements()
        _preflight(batch)
        if enex_dst is not None and (enex_dst.exists() or enex_dst.is_symlink()):
            raise fs.file_already_exists_error(enex_dst)

        result = Publication(
            ordinal=ordinal,
            yes_dst=yes_dst,
            removal_dst=removal_dst,
            reviewed_dst=reviewed_dst,
            enex_dst=enex_dst,
            reviewed=len(reviewed_set),
            yes=len(yes_set),
            no=len(no_set),
            new_to_union=len(removed_words),
            effective=fs.line_count(effective),
            count_lines=tuple(_report(root, dictionary, derived)),
        )
        _commit(batch)
        if enex is not None:
            assert enex_dst is not None
            enex.rename(enex_dst)
        _mark_generated(dictionary)
        return result


def report_publication(root: Path, result: Publication) -> None:
    log.success(f"{result.no} NO words submitted, "
                f"{result.new_to_union} new to the removal union, "
                f"{result.effective} newly removed from the dictionary")
    lead = f"recorded as generation {result.ordinal} "
    recorded = [result.yes_dst, result.removal_dst, result.reviewed_dst]
    for index, path in enumerate(p for p in recorded if p is not None):
        prefix = lead if index == 0 else " " * len(lead)
        print(f"{prefix}-> {_relative(root, path)}")
    if result.enex_dst is not None:
        print(f"archived notes          -> {_relative(root, result.enex_dst)}/")
    for line in result.count_lines:
        print(line)


def remove_words(root: Path, src: Path, reviewed_src: Path | None = None,
                 force_verdicts: bool = False) -> None:
    """Record one round of word removals and rebuild the derived files.

    With reviewed_src, that file is the round's whole reviewed set -- kept and
    removed words alike -- and is archived as reviewed.N; its kept words get no
    verdict. Without it, the removals are the whole reviewed set. Direct
    removal records no YES verdicts, so it writes no yes.N.

    The submitted files go straight to publish_words, which normalizes them,
    checks every removed word is in the reviewed set, and applies both verdict
    guards: a word standing YES cannot be removed, and a word reviewed earlier
    without a verdict needs --force-verdicts. A word already standing NO is
    written to the new removed.N too, but changes nothing downstream.

    Neither this nor `gen dict` takes a target -- dict/ sits at the root and one
    removal applies to every target -- and neither names a next command. The
    report says what changed and stops there.
    """
    stem = archive_stem(src)
    reviewed_input = reviewed_src if reviewed_src is not None else src
    named = src if reviewed_src is None else f"{src} or {reviewed_src}"
    result = publish_words(
        root, stem, reviewed_input, yes=None, no=src,
        force_verdicts=force_verdicts,
        retry=f"correct {named}, then rerun `wf remove words`",
        accept=_remove_command(src, reviewed_src) + " --force-verdicts")
    report_publication(root, result)


def _remove_command(src: Path, reviewed_src: Path | None) -> str:
    """The `wf remove words` command line that submitted these files."""
    command_text = f"wf remove words {src}"
    if reviewed_src is not None:
        command_text += f" --reviewed {reviewed_src}"
    return command_text


def gen_dict(root: Path) -> None:
    """Rebuild the derived files from whatever the records say now.

    Idempotent, seconds, safe to run at any time. It is what the staleness row
    offers, and it is the second half of a retraction: there is no un-remove
    command, so retracting a word means editing every removed.N file where it
    occurs, or deleting those files, and then running this.

    `words.big` and `dict` are required inputs; the derived files are optional
    prior destinations, so their absence is the first-build case rather than a
    failure. Nothing here walks the BEST status rows, so the command cannot --
    and does not -- recommend itself.
    """
    dictionary = tree(root)
    fs.raise_if_not_file(dictionary.base)
    fs.raise_if_not_file(dictionary.small_base)
    recorded = [path for _, path in records(dictionary.removals, REMOVED)]
    yes_recorded = [path for _, path in
                    records(dictionary.reviewed_inputs, YES)]
    reviewed = [path for _, path in
                records(dictionary.reviewed_inputs, REVIEWED)]

    with _staging(dictionary) as staging:
        no_union, _ = _record_union(staging, "removed", recorded)
        yes_union, _ = _record_union(staging, "yes", yes_recorded)
        derived = _stage_all_derived(dictionary, staging, no=no_union,
                                     yes=yes_union, reviewed=reviewed)
        batch = derived.placements()
        _preflight(batch)

        counts = _report(root, dictionary, derived)
        _commit(batch)
        _mark_generated(dictionary)

        log.success(f"Rebuilt the dictionaries from {len(recorded)} removal "
                    f"generation(s), {len(yes_recorded)} YES record(s) and "
                    f"{len(reviewed)} reviewed input(s)")
        for line in counts:
            print(line)


class _Once(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest) is not None:
            raise ValueError(f"{option_string} is given more than once")
        setattr(namespace, self.dest, values)


class RemoveWords(command.Action):
    def __init__(self):
        super().__init__(
            summary="words   — record a round of dictionary word removals",
            positional="WORDS-FILE",
            positional_help=(
                ("WORDS-FILE", "one word per line, or a slice of a "
                 "`top-segments --solo-words` listing with its counts left on"),
            ))

    def parser(self):
        p = argparse.ArgumentParser(add_help=False)
        p.add_argument("--reviewed", dest="reviewed_src", metavar="WORDS",
                       type=Path, action=_Once,
                       help="every word reviewed this round, kept or removed; "
                            "folded into reviewed.words")
        p.add_argument("--force-verdicts", action="store_true",
                       help="record NO verdicts for words reviewed earlier "
                            "without a verdict")
        return p

    def run(self, command_text, opts, argv) -> int:
        argv = self.parse(opts, argv)
        if not argv:
            return usage.missing_argument(self.format_help(command_text))
        if len(argv) > 1:
            return usage.invalid_argument(argv[1],
                                          self.format_help(command_text))
        remove_words(opts.dir, Path(argv[0]), opts.reviewed_src,
                     opts.force_verdicts)
        return 0


class GenDict(command.Action):
    def __init__(self):
        super().__init__(
            summary="dict    — rebuild the derived dictionaries and done-set")

    def run(self, command_text, opts, argv) -> int:
        if argv:
            return usage.invalid_argument(argv[0],
                                          self.format_help(command_text))
        gen_dict(opts.dir)
        return 0


REMOVE_WORDS = RemoveWords()
GEN_DICT = GenDict()
