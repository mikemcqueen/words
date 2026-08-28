import shlex
import shutil
import subprocess
import sys
import tempfile
import time

from pathlib import Path

from workflow import config, fs, log, setops
from workflow.best.state import Target, mark_generated


DFS_LIMIT = 1_000_000
INDEX_NAME = "wiki-merged.2.index"
DICTIONARY_NAME = "words.big"


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


def _display_dfs(argv: list[str], target: Target) -> None:
    displayed = [shlex.quote(arg) for arg in argv]
    if target.letter_form == "u":
        # Under u- the positional is the frozen bag, a hundred characters that
        # would bury the rest. Under o- it is the label's own letters, which
        # are short and are shown literally.
        displayed[2] = f'"$(cat {shlex.quote(str(target.letters))})"'
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
        raise FileNotFoundError(f"seed missing: {target.seed_glob}")
    fs.raise_if_not_file(config.classified(target.root, "no"))
    fs.raise_if_not_dir(results_dir)
    return index, dictionary, seed


def _gen_dfs(target: Target, opts, final: bool) -> None:
    if final and opts.force:
        raise ValueError("-f/--force is only valid for gen dfs.seed")
    results_dir = Path(opts.results_dir or "results").resolve()
    index, dictionary, seed = _dfs_inputs(target, results_dir)
    pairs = target.artifact("best.pairs") if final else seed
    fs.raise_if_not_file(pairs)
    sentence_results = results_dir / target.sentence
    if sentence_results.exists():
        fs.raise_if_not_dir(sentence_results)

    if not target.target_dir.exists():
        if not opts.force:
            fs.raise_if_not_dir(target.target_dir)
        # -f now creates any missing part of <letter-set>/m<N>/g<N>, having
        # validated every input first: a run that cannot start leaves no tree.
        target.target_dir.mkdir(parents=True)

    sentence_results.mkdir(exist_ok=True)
    fs.raise_if_not_dir(sentence_results)

    limit = DFS_LIMIT if opts.count is None else opts.count
    rendered = sentence_results / _dfs_name(target, seed, limit, final)
    scratch = rendered.with_name(rendered.name + ".tmp")
    if target.letter_form == "o":
        bag = [target.named_letters]
    else:
        letters = target.letters.read_text().rstrip("\r\n")
        bag = [letters, "-u", target.named_letters]
    argv = [
        "dfs-anagrams", str(index), *bag,
        "-m", str(target.min_words),
        "-S", "20",
        "-p", "10000000",
        "-n", str(limit),
        "--word-bonus", "1",
        "--dict", str(dictionary),
        "--pairs", str(pairs),
        "--exclude-pairs", str(target.root),
        "-x", "2",
        "-g", str(target.segment_count),
    ]
    _display_dfs(argv, target)
    started = time.monotonic()
    with scratch.open("w") as output:
        subprocess.run(argv, stdout=output, check=True)
    scratch.replace(rendered)
    artifact = "dfs.best" if final else "dfs.seed"
    _publish_link(target.artifact(artifact), rendered)
    elapsed = _format_duration(time.monotonic() - started)
    log.success(f"Generated {fs.line_count(rendered)} results in {elapsed} "
                f"→ {rendered}")


def gen_dfs_seed(target: Target, opts) -> None:
    _gen_dfs(target, opts, final=False)


def gen_dfs_best(target: Target, opts) -> None:
    _gen_dfs(target, opts, final=True)


def gen_top_segments(target: Target, opts) -> None:
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
    mark_generated(target.artifact("top.segments"))
    count = fs.line_count(target.artifact("top.segments"))
    log.success(f"Generated {count} top segments → "
                f"{target.address}/top.segments")


def build_best_pairs(target: Target) -> None:
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
    mark_generated(destination)

    after = set(destination.read_text().splitlines())
    log.success(f"Generated {len(after)} best pairs "
                f"({len(after - before)} added, {len(before - after)} dropped) "
                f"→ {target.address}/best.pairs")
    if not after:
        log.warn("BEST PAIRS is empty; dfs.best would run without pair bonuses")


def gen_best_pairs(target: Target, opts) -> None:
    if opts.force:
        raise ValueError("-f/--force is only valid for gen dfs.seed")
    if opts.results_dir is not None:
        raise ValueError("-r/--results-dir is only valid for DFS stages")
    if opts.count is not None:
        raise ValueError("-n is not valid for gen best.pairs")
    build_best_pairs(target)
