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
# Two records, answering two questions. `removed/<input>.removed.N` says what
# was removed and by which round, and is mutable by design. `done/in/
# <input>.reviewed.N` archives the round's whole submitted input -- every word
# it covered, kept or removed -- so a word judged good does not come back at the
# top of every candidate listing. A retraction changes the first and cannot
# change the second, which is why they are not the redundancy a single folded
# standing set would be.
#
# Nothing here calls nutrimatic's dict_remove.py, which edits a hardcoded
# dictionary in place. Three of its ideas carry over: the count-prefix strip,
# `comm -23` under LC_ALL=C (which setops already provides), and the .removed.N
# suffix -- ported onto a different stem and a different mutability.

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

REMOVED = "removed"
REVIEWED = "reviewed"

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
    purpose: `words.big` is required, while `words.filtered` is this module's
    output and is absent on a tree that has never rebuilt.
    """

    dir: Path
    base: Path
    derived: Path
    removals: Path
    reviewed_inputs: Path
    reviewed_words: Path
    enex_archives: Path

    @property
    def marker(self) -> Path:
        return generation.stamp(self.derived)


def tree(root: Path) -> Tree:
    return Tree(dir=config.path(root, ["dict"]),
                base=config.base_dictionary(root),
                derived=config.dictionary(root),
                removals=config.removals(root),
                reviewed_inputs=config.reviewed_inputs(root),
                reviewed_words=config.reviewed_words(root),
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


def _next_ordinal(removal_records, reviewed_records) -> int:
    """One sequence across both archive namespaces, not one per stem.

    Global is what makes it a generation counter -- N alone orders every
    submission and identifies one -- and it keeps two submissions of the same
    filename from colliding, where a per-stem counter would give them each their
    own 1 and lose the ordering. Reading both namespaces also stops a surviving
    half of an interrupted round from having its ordinal reused.
    """
    ordinals = [ordinal for ordinal, _ in (*removal_records, *reviewed_records)]
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


@dataclass(frozen=True)
class FilterResult:
    source_words: int
    surviving_words: int
    filtered_words: int

    @property
    def filtered(self) -> bool:
        return self.filtered_words != 0


def filter_reviewed_words(source: Path, reviewed: Path,
                          staged: Path) -> FilterResult:
    """Write unreviewed presentation rows in order without moving source."""
    fs.raise_if_not_file(reviewed)
    reviewed_set = set(read_words(reviewed))
    source_words = surviving = filtered_count = 0
    with source.open("r") as incoming, staged.open("w") as outgoing:
        for number, line in enumerate(incoming, start=1):
            identity = parse_word_row(source, number, line.rstrip("\r\n"))
            if identity is None:
                continue
            source_words += 1
            if identity in reviewed_set:
                filtered_count += 1
                continue
            outgoing.write(line.rstrip("\r\n") + "\n")
            surviving += 1
    return FilterResult(source_words, surviving, filtered_count)


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


def _stage_reviewed_words(tree: Tree, staging: Path,
                          srcs: list[Path]) -> setops.Staged:
    dst, path = tree.reviewed_words, staging / "reviewed.words"
    if not srcs:
        return setops.stage_text("", dst, path, stable_mtime=True)
    with _operation("reviewed-word union", *srcs):
        return setops.stage_merge(srcs, dst, path, stable_mtime=True)


def _stage_derived(tree: Tree, staging: Path,
                   removed: Path) -> setops.Staged:
    """words.filtered: the base minus the union of every removal generation.

    stable_mtime is load-bearing and not tidiness. Inputs.dictionary, the
    frontier's "dictionary changed" reason, and both DFS freshness predicates
    date against this file, so a removal that removes nothing -- a word not in
    the base -- must leave it byte-identical and cost nothing downstream.
    """
    with _operation("dictionary derivation", tree.base, removed):
        return setops.stage_diff(tree.base, removed, tree.derived,
                                 staging / config.DICTIONARY_NAME,
                                 stable_mtime=True)


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
    return f"{label} {before} -> {after}"


def _report(root: Path, tree: Tree, derived: setops.Staged,
            reviewed_words: setops.Staged) -> list[str]:
    """The two count lines, composed before anything is published."""
    return [_count_line(config.DICTIONARY_NAME, derived),
            _count_line(_relative(root, tree.reviewed_words), reviewed_words)]


# ------------------------------------------------------------------ commands


@dataclass(frozen=True)
class Publication:
    ordinal: int
    removal_dst: Path
    reviewed_dst: Path
    enex_dst: Path | None
    submitted: int
    new_to_union: int
    effective: int
    count_lines: tuple[str, ...]


def publish_words(root: Path, stem: str, reviewed_input: Path,
                  removals: Path, enex: Path | None = None) -> Publication:
    """Publish one normalized reviewed/removal round and rebuild dictionary."""
    archive_name(stem)
    dictionary = tree(root)
    fs.raise_if_not_file(dictionary.base)
    fs.raise_if_not_file(reviewed_input)
    fs.raise_if_not_file(removals)

    removal_records = records(dictionary.removals, REMOVED)
    reviewed_records = records(dictionary.reviewed_inputs, REVIEWED)
    ordinal = _next_ordinal(removal_records, reviewed_records)
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

    with _staging(dictionary) as staging:
        with _operation("submission normalization", reviewed_input, removals):
            removal = setops.stage_merge([removals], removal_dst,
                                         staging / "removal.archive")
            reviewed = setops.stage_merge([reviewed_input], reviewed_dst,
                                          staging / "reviewed.archive")

        recorded = [path for _, path in removal_records]
        prior = staging / "prior.removed"
        prospective = staging / "next.removed"
        with _operation("removal union", *recorded):
            _union(recorded, prior)
            _union([*recorded, removal.path], prospective)
        with _operation("removal delta", prior, prospective):
            delta = setops.diff(prospective, prior, staging / "delta")
        with _operation("removed-word intersection", delta, dictionary.base):
            effective = setops.common(delta, dictionary.base,
                                      staging / "effective")

        derived = _stage_derived(dictionary, staging, prospective)
        reviewed_words = _stage_reviewed_words(
            dictionary, staging,
            [path for _, path in reviewed_records] + [reviewed.path])
        batch = [Prepared(removal, replace=False),
                 Prepared(reviewed, replace=False),
                 Prepared(reviewed_words), Prepared(derived)]
        _preflight(batch)
        fs.raise_if_any_exist([removal_dst, reviewed_dst])
        if enex_dst is not None and (enex_dst.exists() or enex_dst.is_symlink()):
            raise fs.file_already_exists_error(enex_dst)

        result = Publication(
            ordinal=ordinal,
            removal_dst=removal_dst,
            reviewed_dst=reviewed_dst,
            enex_dst=enex_dst,
            submitted=fs.line_count(removal.path),
            new_to_union=fs.line_count(delta),
            effective=fs.line_count(effective),
            count_lines=tuple(_report(root, dictionary, derived,
                                      reviewed_words)),
        )
        _commit(batch)
        if enex is not None:
            assert enex_dst is not None
            enex.rename(enex_dst)
        _mark_generated(dictionary)
        return result


def report_publication(root: Path, result: Publication) -> None:
    log.success(f"{result.submitted} words submitted, "
                f"{result.new_to_union} new to the removal union, "
                f"{result.effective} newly removed from the dictionary")
    lead = f"recorded as generation {result.ordinal} "
    print(f"{lead}-> {_relative(root, result.removal_dst)}")
    print(f"{' ' * len(lead)}-> {_relative(root, result.reviewed_dst)}")
    if result.enex_dst is not None:
        print(f"archived notes          -> {_relative(root, result.enex_dst)}/")
    for line in result.count_lines:
        print(line)


def remove_words(root: Path, src: Path) -> None:
    """Record one round of word removals and rebuild the derived files.

    Validate, allocate, prepare everything, preflight everything, archive,
    publish derived outputs, mark. An ordinary failure therefore leaves no
    durable record and no changed output. Once archives have landed they precede
    the derived outputs, so a later `wf gen dict` can heal an interrupted
    publication: a derived dictionary must never reflect a submission that
    exists nowhere in its source records.

    Neither this nor `gen dict` takes a target -- dict/ sits at the root and one
    removal applies to every target -- and neither names a next command. It
    cannot know whether a target is involved at all: the operator may have
    removed a word to unblock one search, or to feed a filter no BEST target is
    in play for. The report says what changed and stops there.
    """
    words = read_words(src)
    stem = archive_stem(src)
    reviewed = fs.optional_file(config.reviewed_words(root))
    with tempfile.TemporaryDirectory(prefix="wf-remove-words-") as tmp:
        submission = Path(tmp) / "submission"
        submission.write_text("".join(f"{word}\n" for word in words))
        if reviewed is not None:
            unreviewed = Path(tmp) / "unreviewed"
            filtered = filter_reviewed_words(submission, reviewed, unreviewed)
            if filtered.surviving_words == 0:
                raise ValueError(f"no unreviewed words in {src}")
            submission = unreviewed
        result = publish_words(root, stem, submission, submission)
    report_publication(root, result)


def gen_dict(root: Path) -> None:
    """Rebuild the derived files from whatever the records say now.

    Idempotent, seconds, safe to run at any time. It is what the staleness row
    offers, and it is the second half of a retraction: there is no un-remove
    command, so retracting a word means editing every generation file where it
    occurs, or deleting those files, and then running this.

    `words.big` is a required input; `words.filtered` is an optional prior
    destination, so its absence is the first-build case rather than a failure.
    Nothing here walks the BEST status rows, so the command cannot -- and does
    not -- recommend itself.
    """
    dictionary = tree(root)
    fs.raise_if_not_file(dictionary.base)
    recorded = [path for _, path in records(dictionary.removals, REMOVED)]
    reviewed = [path for _, path in
                records(dictionary.reviewed_inputs, REVIEWED)]

    with _staging(dictionary) as staging:
        prospective = staging / "removed"
        with _operation("removal union", *recorded):
            _union(recorded, prospective)
        reviewed_words = _stage_reviewed_words(dictionary, staging, reviewed)
        derived = _stage_derived(dictionary, staging, prospective)

        batch = [Prepared(reviewed_words), Prepared(derived)]
        _preflight(batch)

        counts = _report(root, dictionary, derived, reviewed_words)
        _commit(batch)
        _mark_generated(dictionary)

        log.success(f"Rebuilt the dictionary from {len(recorded)} removal "
                    f"generation(s) and {len(reviewed)} reviewed input(s)")
        for line in counts:
            print(line)


class RemoveWords(command.Action):
    def __init__(self):
        super().__init__(
            summary="words   — record a round of dictionary word removals",
            positional="WORDS-FILE",
            positional_help=(
                ("WORDS-FILE", "one word per line, or a slice of a "
                 "`top-segments --solo-words` listing with its counts left on"),
            ))

    def run(self, command_text, opts, argv) -> int:
        if not argv:
            return usage.missing_argument(self.format_help(command_text))
        if len(argv) > 1:
            return usage.invalid_argument(argv[1],
                                          self.format_help(command_text))
        remove_words(opts.dir, Path(argv[0]))
        return 0


class GenDict(command.Action):
    def __init__(self):
        super().__init__(
            summary="dict    — rebuild the derived dictionary and done-set")

    def run(self, command_text, opts, argv) -> int:
        if argv:
            return usage.invalid_argument(argv[0],
                                          self.format_help(command_text))
        gen_dict(opts.dir)
        return 0


REMOVE_WORDS = RemoveWords()
GEN_DICT = GenDict()
