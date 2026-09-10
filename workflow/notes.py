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
import tempfile

from pathlib import Path

from workflow import bundle, command, config, context, fs, log, names, usage


CHUNK_SIZE = 400

# The bound is the title scheme's, not the note store's: a part suffix is one
# letter, .aa through .az.
MAX_PARTS = 26

TWO_CHECKBOXES = "two-checkboxes"
ONE_CHECKBOX = "checkbox"


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


class SplitPaths(list[Path]):
    """Part paths that retain ownership of their temporary directory."""

    def __init__(self, paths: list[Path], temporary) -> None:
        super().__init__(paths)
        self.temporary = temporary

    def cleanup(self) -> None:
        self.temporary.cleanup()


def split(source: Path) -> list[Path]:
    """Split source into note-sized parts under temporary storage."""
    temporary = tempfile.TemporaryDirectory(prefix="wf-notes-")
    directory = Path(temporary.name)
    try:
        paths = part_paths(directory, source, part_count(source))
        # check=True raises on non-zero return code
        subprocess.run(["split.sh", f"{source}", f"{CHUNK_SIZE}",
                        f"{directory / source.name}"],
                       stdout=subprocess.DEVNULL, check=True)
        fs.raise_if_any_not_file(paths)
    except BaseException:
        temporary.cleanup()
        raise
    return SplitPaths(paths, temporary)


def create(paths: list[Path], yes_pairs: Path | None = None,
           checkbox_mode: str = TWO_CHECKBOXES,
           retry_command: str | None = None,
           checked: str | None = None) -> None:
    log.info(f"Creating {len(paths)} notes...")
    # One argument list for both shapes: a review that has a confirmed-YES set
    # to check itself against differs from one that does not by two arguments,
    # not by a second call.
    checkbox = {TWO_CHECKBOXES: "--two-checkboxes",
                ONE_CHECKBOX: "--checkbox"}.get(checkbox_mode)
    if checkbox is None:
        raise ValueError(f"unknown note checkbox mode: {checkbox_mode}")
    options = ["--text", checkbox, "--production"]
    if yes_pairs is not None:
        options += ["--yes-pairs", str(yes_pairs)]
    if checked is not None:
        options += ["--checked", checked]
    created: list[Path] = []
    retry_command = retry_command or "wf notes p2 NAME"
    for path in paths:
        try:
            subprocess.run(
                ["note", "-pf.72", "--create", f"{path}", *options],
                stdout=subprocess.DEVNULL, check=True)
        except (OSError, subprocess.CalledProcessError) as error:
            succeeded = "\n".join(f"  {p.name}" for p in created)
            earlier = (f"Delete the notes already created:\n{succeeded}\n"
                       if created else "No earlier note creation was confirmed.\n")
            raise ValueError(
                f"note creation failed at {path.name}.\n{earlier}"
                f"Check whether {path.name} was created, and delete it if it "
                f"exists. After manual cleanup, recreate the complete batch "
                f"with `{retry_command}`.") from error
        created.append(path)


def add_checked(parser: argparse.ArgumentParser) -> None:
    """Add the initial checked-type option used by P2 note creation."""
    parser.add_argument(
        "--checked", type=str.upper, choices=("YES", "NO"), metavar="TYPE",
        help="initial checkbox type to check (YES or NO)")


def add_yes_pairs(parser: argparse.ArgumentParser) -> None:
    """Add the confirmed-YES input option used by P2 note creation."""
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
    value = getattr(opts, "yes_pairs", None)
    return Path(value) if value else None


def make(pairs: Path, opts, checkbox_mode: str = TWO_CHECKBOXES,
         retry_command: str | None = None) -> list[str]:
    """Split a bundle's evaluated pairs and raise one note per part.

    The whole of what `eval p2` does to a bundle beyond opening it.

    Returns the titles of the notes created, not the parts they were made
    from: the parts are scratch this deletes on the way out, and their names
    are the titles -- the same ones retrieval re-derives to find these notes
    again.
    """
    if retry_command is None:
        if pairs.parent.name == "in":
            try:
                named = names.queue_stem("p2", pairs.name)
            except ValueError:
                named = "NAME"
        else:
            named = pairs.parent.name
        retry_command = f"wf notes p2 {named}"
    paths = split(pairs)
    try:
        create(paths, _yes_pairs(opts), checkbox_mode, retry_command,
               getattr(opts, "checked", None))
        return [path.name for path in paths]
    finally:
        if isinstance(paths, SplitPaths):
            paths.cleanup()


class Notes(command.Action):
    """Re-raise the notes of a bundle that already has them, or had them."""

    def __init__(self, phase: str, summary: str):
        super().__init__(summary=summary,
                         positional="BUNDLE-NAME|SOURCE-FILE")
        self.phase = phase

    def parser(self):
        # Deliberately no review-filter flags: there is no filtering step here
        # -- `notes` reads whatever `eval` left -- and an inert flag would imply
        # a mode the command does not have.
        p = argparse.ArgumentParser(add_help=False)
        add_checked(p)
        add_yes_pairs(p)
        return p

    def run(self, command_text, opts, argv) -> int:
        rest = self.parse(opts, argv)
        if not rest:
            return usage.missing_argument(self.format_help(command_text))
        if len(rest) > 1:
            return usage.invalid_argument(rest[1],
                                          self.format_help(command_text))

        check_yes_pairs(opts)
        bundle_name, named = bundle.resolve_source(opts.dir, self.phase,
                                                   rest[0])
        ctx = context.Context(root=opts.dir, phase=self.phase,
                              force=opts.force, bundle_name=bundle_name)
        source = bundle.recover(ctx, named)
        titles = make(source, opts)
        log.success(f"{len(titles)} note(s) recreated from {source.name}")
        return 0


P2 = Notes("p2", "p2      — recreate a manual review's notes")


class NotesWords(command.Action):
    """Recreate a complete note batch for one active dictionary review."""

    def __init__(self):
        super().__init__(summary="words   — recreate a dictionary review's notes",
                         positional="NAME")

    def run(self, command_text, opts, argv) -> int:
        if not argv:
            return usage.missing_argument(self.format_help(command_text))
        if len(argv) > 1:
            return usage.invalid_argument(argv[1],
                                          self.format_help(command_text))
        name = argv[0]
        bundle_name = names.check_name(name, "bundle name")
        ctx = context.Context(root=opts.dir, phase="dict", force=opts.force,
                              bundle_name=bundle_name)
        if not ctx.bundle_dir.is_dir():
            queued = config.path(opts.dir, ["dict", "queued"]) / bundle_name
            if queued.is_file():
                raise ValueError(
                    f"{bundle_name} is queued and has no active words review; "
                    f"run `wf eval words {bundle_name}`")
            raise ValueError(f"no active words review: {bundle_name}")
        source = bundle.evaluated(ctx)
        titles = make(source, opts, ONE_CHECKBOX,
                      f"wf notes words {bundle_name}")
        log.success(f"{len(titles)} note(s) recreated from {source.name}")
        return 0


WORDS = NotesWords()
