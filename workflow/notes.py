# notes.py
#
# The note parts an evaluated pairs file becomes, and the one rendering of
# their names.
#
# Creation splits the file into contiguous chunks and raises one note per
# chunk; retrieval, a phase later, finds those notes again by rendering the
# same titles and probing until one is missing. Both ends of that contract used
# to spell the naming rule for themselves -- `get_split_paths` in eval.py and
# `_title` in steps/p2_retrieve.py, with a bare `assert n_files < 27` and
# `MAX_NOTE_PARTS = 26` as two spellings of one bound. One module owns it now,
# so a completed bundle looks for exactly the notes creation made.
#
# The `notes` command is that derivation reached on its own. Note creation is a
# pure function of two inputs -- the evaluated pairs file and the optional
# confirmed-YES set the notes check themselves against -- because the split is
# deterministic and `eval` writes no manifest. So recreating deleted notes is a
# re-derivation, not a recovery. What made it unreachable is that `eval` welds
# it to `bundle.begin`, a one-way move out of the queue. This command calls
# neither that nor `filter_done`: like extract.py, it reads state the phase has
# already placed -- open bundle or archive -- and disturbs the queue not at all.

import argparse
import subprocess

from pathlib import Path

from workflow import bundle, command, context, fs, log, usage


CHUNK_SIZE = 400

# The bound is the title scheme's, not the note store's: a part suffix is one
# letter, .aa through .az.
MAX_PARTS = 26

# Where a split lands. The parts are scratch -- `note --create` reads them and
# nothing in the workflow refers to them again.
STAGING = Path("/tmp")


def title(source: Path, index: int) -> str:
    """The note title of one part of source: `<name>.aa`, `<name>.ab`, ..."""
    return f"{source.name}.a{chr(ord('a') + index)}"


def part_count(path: Path, chunk_size: int = CHUNK_SIZE) -> int:
    """How many parts a pairs file splits into, at the size `split` uses."""
    n_lines = fs.line_count(path)
    n_files = n_lines // chunk_size
    return n_files + (1 if n_files * chunk_size < n_lines else 0)


def part_paths(directory: Path, source: Path, count: int) -> list[Path]:
    """Where the parts of source land under directory, in order.

    The bound is checked here rather than after the split, so a file too large
    to name is refused before anything is written.
    """
    if count > MAX_PARTS:
        last = chr(ord('a') + MAX_PARTS - 1)
        raise ValueError(
            f"{source.name} splits into {count} parts; the note title scheme "
            f"names at most {MAX_PARTS} (.aa through .a{last})")
    return [directory / title(source, index) for index in range(count)]


def split(source: Path) -> list[Path]:
    """Split source into note-sized parts under STAGING and return them."""
    paths = part_paths(STAGING, source, part_count(source))
    # check=True raises on non-zero return code
    subprocess.run(["split.sh", f"{source}", f"{CHUNK_SIZE}",
                    f"{STAGING / source.name}"],
                   stdout=subprocess.DEVNULL, check=True)
    fs.raise_if_any_not_file(paths)
    return paths


def create(paths: list[Path], yes_pairs: Path | None = None) -> None:
    log.info(f"Creating {len(paths)} notes...")
    # One argument list for both shapes: a review that has a confirmed-YES set
    # to check itself against differs from one that does not by two arguments,
    # not by a second call.
    options = ["--text", "--two-checkboxes", "--production"]
    if yes_pairs is not None:
        options += ["--yes-pairs", str(yes_pairs)]
    for path in paths:
        subprocess.run(["note", "-pf.72", "--create", f"{path}", *options],
                       stdout=subprocess.DEVNULL, check=True)


def add_yes_pairs(parser: argparse.ArgumentParser) -> None:
    """The flag both commands that raise notes admit."""
    parser.add_argument("--yes-pairs", metavar="PATH",
                        help="confirmed-YES pairs the notes check themselves "
                             "against")


def check_yes_pairs(opts) -> None:
    """The flag's pre-flight, beside the parser that admits it.

    The path reaches a file only in `note`'s argument list, in the last
    subprocess either command runs. By then `eval` has emptied the queue, and
    `notes` has already raised however many parts precede the failure. Neither
    is worth a bad path, and both can answer this from the arguments alone.
    """
    if opts.yes_pairs:
        fs.raise_if_not_readable(Path(opts.yes_pairs))


def _yes_pairs(opts) -> Path | None:
    return Path(opts.yes_pairs) if opts.yes_pairs else None


def make(pairs: Path, opts) -> list[Path]:
    """Split a bundle's evaluated pairs and raise one note per part.

    The whole of what `eval p2` does to a bundle beyond opening it.
    """
    paths = split(pairs)
    create(paths, _yes_pairs(opts))
    return paths


class Notes(command.Action):
    """Re-raise the notes of a bundle that already has them, or had them."""

    def __init__(self, phase: str, summary: str):
        super().__init__(summary=summary,
                         positional="BUNDLE-NAME|SOURCE-FILE")
        self.phase = phase

    def parser(self):
        # Deliberately no --no-filter: there is no filtering step here to skip
        # -- `notes` reads whatever `eval` left -- and an inert flag would
        # imply a mode the command does not have.
        p = argparse.ArgumentParser(add_help=False)
        add_yes_pairs(p)
        return p

    def run(self, command_text, opts, argv) -> int:
        rest = self.parse(opts, argv)
        if not rest:
            return usage.missing_argument(self.format_help(command_text))

        check_yes_pairs(opts)
        bundle_name, named = bundle.resolve_source(opts.dir, self.phase,
                                                   rest[0])
        ctx = context.Context(root=opts.dir, phase=self.phase,
                              force=opts.force, bundle_name=bundle_name)
        source = bundle.recover(ctx, named)
        paths = make(source, opts)
        log.success(f"{len(paths)} note(s) recreated from {source.name}")
        return 0


P2 = Notes("p2", "p2      — recreate a manual review's notes")
