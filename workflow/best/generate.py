import shlex
import shutil
import subprocess
import sys
import time

from pathlib import Path

from workflow import config, fs, generation, log, setops
from workflow.best.state import (
    INDEX_NAME, Target, target_no_pairs,
)


DFS_LIMIT = 1_000_000
# What top-segments itself defaults -n to. Named here because `prepare` passes
# its own cutoff explicitly and has to pass the one the primitive would have.
TOP_LIMIT = 1000
def _command_not_found(name: str) -> FileNotFoundError:
    return FileNotFoundError(f"command not found: {name}")


def _require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise _command_not_found(name)


def _format_duration(elapsed: float) -> str:
    seconds = max(0, round(elapsed))
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes}m{seconds:02d}s" if minutes else f"{seconds}s"


def _seed_annotation(target: Target, seed: Path) -> str:
    # The seed carries m<N> in its name now that it sits above the level that
    # keys on -m, so the chop takes the constructed prefix rather than a
    # literal "seed". What is left is still opaque and never taken apart.
    return seed.name[len(target.seed_stem):-len(".pairs")]


def _dfs_name(target: Target, seed: Path, limit: int, final: bool) -> str:
    stage = ".best" if final else ""
    # The letter set trails: these names are read by eye in results/, nothing
    # matches them by prefix, and last is where the key the operator scans for
    # stands out. Without it two letter sets at one s/m/g render one path and
    # each gen replaces the other's hours of results.
    return (f"dfs.{target.sentence}{_seed_annotation(target, seed)}."
            f"m{target.min_words}.x2.g{target.segment_count}{stage}."
            f"{limit}.{target.letter_set}")


def _display_dfs(argv: list[str], target: Target, bag_position: int) -> None:
    displayed = [shlex.quote(arg) for arg in argv]
    if target.letter_form == "u":
        # Under u- the positional is the frozen bag, a hundred characters that
        # would bury the rest. Under o- it is the label's own letters, which
        # are short and are shown literally.
        displayed[bag_position] = (
            f'"$(cat {shlex.quote(str(target.letters))})"')
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


def _dfs_inputs(target: Target, results_dir: Path,
                final: bool) -> Path:
    """Everything a DFS run reads, checked before anything is created."""
    _require_command("dfs-anagrams")
    index = target.best_dir / "idx" / INDEX_NAME
    # Root-global and derived: the searches read what `wf gen dict` produced,
    # not the hand-placed base, so a word removed from the dictionary is a word
    # the search cannot spell.
    dictionary = config.dictionary(target.root)
    fs.raise_if_not_file(index)
    fs.raise_if_not_file(dictionary)
    fs.raise_if_not_file(target.letters)
    seed = target.seed()
    if seed is None:
        raise FileNotFoundError(f"seed missing: {target.seed_glob}")
    fs.raise_if_not_file(seed)
    fs.raise_if_not_file(config.classified(target.root, "yes"))
    fs.raise_if_not_file(config.classified(target.root, "no"))
    fs.raise_if_not_dir(results_dir)
    if final:
        # Both are optional hand-placed inputs. Resolve them here so a
        # directory or dangling symlink fails before a target or result is
        # created, even though dfs-anagrams will discover the paths itself.
        best_pairs = fs.optional_file(target.artifact("best.pairs"))
        no_pairs = target_no_pairs(target)
        if best_pairs is None and no_pairs is None:
            raise ValueError(
                f"dfs.best would repeat dfs.seed for {target.address}; add "
                f"{target.artifact('best.pairs')} or "
                f"{target.artifact('no.pairs')} to make it differ")
    return seed


def gen_dfs(target: Target, *, final: bool, force: bool,
            results_dir: Path | None, count: int | None,
            dry_run: bool = False) -> None:
    """Run one DFS search and publish its results as dfs.seed or dfs.best.

    Takes what it needs by name rather than an opts namespace, because
    `prepare` runs this and `gen_top_segments` back to back and the two legs
    disagree about every flag: -f, -r and -n belong to the DFS leg and are
    refused by the other. Which flags a command accepts is the command
    layer's to say.

    Under --dry-run every input is still resolved and every refusal still
    fires -- the printed command is only worth having if it is the one the
    real run would have issued -- and the run stops at the print, before
    anything is created.
    """
    results_dir = Path(results_dir or "results").resolve()
    _run_dfs(target, final=final, force=force, results_dir=results_dir,
             count=count, dry_run=dry_run)


def _run_dfs(target: Target, *, final: bool, force: bool,
             results_dir: Path, count: int | None, dry_run: bool) -> None:
    seed = _dfs_inputs(target, results_dir, final)
    sentence_results = results_dir / target.sentence
    if sentence_results.exists():
        fs.raise_if_not_dir(sentence_results)

    if not target.target_dir.exists() and not force:
        fs.raise_if_not_dir(target.target_dir)

    limit = DFS_LIMIT if count is None else count
    rendered = sentence_results / _dfs_name(target, seed, limit, final)
    scratch = rendered.with_name(rendered.name + ".tmp")
    if target.letter_form == "o":
        bag = [target.named_letters]
    else:
        letters = target.letters.read_text().rstrip("\r\n")
        bag = [letters, "-u", target.named_letters]
    selected_target = target.address if final else target.sentence
    argv = [
        "dfs-anagrams", *bag,
        "--wfroot", str(target.root),
        "-t", selected_target,
        "-m", str(target.min_words),
        "-g", str(target.segment_count),
        "-p", "10000000",
        "-n", str(limit),
    ]
    _display_dfs(argv, target, 1)
    if dry_run:
        return
    # The directories are created here, after the print rather than with the
    # checks above: -f still creates any missing part of
    # <letter-set>/m<N>/g<N> only once every input has validated, and a dry
    # run reaches the command it would have run without leaving a tree.
    if not target.target_dir.exists():
        target.target_dir.mkdir(parents=True)
    sentence_results.mkdir(exist_ok=True)
    fs.raise_if_not_dir(sentence_results)
    started = time.monotonic()
    with scratch.open("w") as output:
        subprocess.run(argv, stdout=output, check=True)
    scratch.replace(rendered)
    artifact = "dfs.best" if final else "dfs.seed"
    _publish_link(target.artifact(artifact), rendered)
    if final:
        # Last in the publication sequence: a crash before this point leaves
        # the marker missing or old, so status offers the search once more.
        generation.mark_generated(target.artifact("dfs.best"))
    elapsed = _format_duration(time.monotonic() - started)
    log.success(f"Generated {fs.line_count(rendered)} results in {elapsed} "
                f"→ {rendered}")


def gen_top_segments(target: Target, *, source: str,
                     count: int | None) -> None:
    """Generate the review frontier from one of the two DFS artifacts.

    The source is recorded beside top.segments, in the same marker whose mtime
    dates the generation, because there is one frontier and generating it from
    either side supersedes the other -- so what it came from is a property of
    the frontier rather than a history to keep.
    """
    _require_command("top-segments")
    dfs = target.artifact(f"dfs.{source}")
    fs.raise_if_not_file(dfs)
    # pair-exclusions only warns on a missing classified file, and a warning
    # would produce an unfiltered frontier the state machine then believes is
    # filtered. All three are checked here so it cannot -- the dictionary is
    # the same hazard: an absent one warns, all_words_in_dict returns true on
    # an empty set, and the frontier comes out unfiltered.
    fs.raise_if_not_file(config.classified(target.root, "yes"))
    fs.raise_if_not_file(config.classified(target.root, "no"))
    # Unlike the generic required-file guards, this one says how to create its
    # input. On a tree that has never rebuilt, _no_frontier wins before
    # _dictionary_stale and sends the operator here, so the command itself has
    # to carry the actionable recovery.
    dictionary = config.dictionary(target.root)
    if fs.optional_file(dictionary) is None:
        raise ValueError(f"dictionary not generated: {dictionary}; "
                         f"run `wf gen dict`")
    argv = ["top-segments", "--pairs"]
    if count is not None:
        argv.extend(["-n", str(count)])
    # The two flags are not symmetric and both are wanted. -y ignores: a YES
    # pair is not counted, but its line still contributes every other segment
    # on it. --wfroot rejects: a line holding a NO pair is dropped whole, so
    # it stops contributing counts for its other segments -- the same
    # semantics --exclude-pairs applies at search time, so a regenerated
    # frontier stays consistent with a re-run search.
    argv.extend(["--wfroot", str(target.root), "-y"])
    # --wfroot is itself an -r of the root's hard-NO set, so the target-local
    # file joins it as a second one and gets the same whole-row rejection.
    target_no = target_no_pairs(target)
    if target_no is not None:
        argv.extend(["-r", str(target_no)])
    argv.append(str(dfs))
    _display_command(argv)
    top_segments = target.artifact("top.segments")
    # top.segments first and the marker second: a crash between them leaves the
    # generation clock behind the content clock, so status re-offers the
    # generation and the state heals. Marker-first would run the clock ahead,
    # the row would not fire, and the state would never be re-offered.
    setops._place(argv, top_segments, stable_mtime=True)
    generation.mark_generated(top_segments, f"{source}\n")
    rows = fs.line_count(top_segments)
    log.success(f"Generated {rows} top segments → "
                f"{target.address}/top.segments")
    if count is not None and rows < count:
        # The DFS file is exhausted at this cutoff, so a larger -n reproduces
        # the same file, which stable_mtime then leaves untouched -- nothing
        # downstream would move. The frontier grows by searching, not asking.
        log.warn(f"top.segments holds {rows} of the {count} requested; "
                 f"dfs.{source} is exhausted at this cutoff and a larger -n "
                 f"would reproduce the same file")


def prepare(target: Target, *, source: str, force: bool,
            results_dir: Path | None, dfs_count: int | None,
            top_count: int | None, dry_run: bool = False) -> None:
    """Run one search and generate the frontier from it, in one shell.

    The two legs take independent cutoffs: -n means results out of
    dfs-anagrams to one and frontier rows out of top-segments to the other,
    and one flag for both would silently mean whichever the caller last
    thought about.
    """
    # Checked before the search rather than between the legs, so an absent
    # top-segments costs nothing instead of the hours the DFS just spent.
    _require_command("top-segments")
    gen_dfs(target, final=source == "best", force=force,
            results_dir=results_dir, count=dfs_count, dry_run=dry_run)
    if dry_run:
        # The frontier leg reads the DFS results the search would have
        # written, so there is no second command to render from a tree the
        # dry run deliberately did not touch.
        return
    try:
        gen_top_segments(target, source=source, count=top_count)
    except BaseException:
        # The DFS results are published and worth keeping; the frontier is
        # whatever it already was. Naming the one command that finishes the
        # job beats resuming, because whatever stopped it is still there.
        recovery = target.command(
            "gen", "top.segments", "--source", source, "-n", str(top_count))
        log.error(f"dfs.{source} is in place but top.segments was not "
                  f"generated; rerun: {recovery}")
        raise
