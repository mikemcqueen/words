# complete_pairs.py

from pathlib import Path

from src.filter import filter_results

from workflow import log, fs, config


def help_summary(name):
    return "pairs   — complete a running pairs file"


def help(command, opts, argv):
    print(help_summary(command))
    print("usage: wf complete pairs [pair-file]")
    return 0


def _resolve_pair_file(src_dir: Path, argv) -> Path:
    # if filename was supplied, use it
    if argv:
        name = argv[0]
        src_path = src_dir / Path(name).name
        fs.raise_if_not_file(src_path)
        return src_path

    # no filename was supplied. use pairs file in src_dir if there is exactly one.
    paths: list[Path] = []
    for p in src_dir.iterdir():
        if p.is_file() and p.suffix == ".pairs":
            if len(paths):
                raise ValueError("Missing FILE parameter. Use `wf show pairs` to see files.")
            paths.append(p)

    if not paths:
        raise ValueError(f"No .pairs files found in {src_dir}")
    return paths[0]


def _complete(src_pairs: Path, opts) -> int:
    log.info(src_pairs)

    src_results = src_pairs.parent / (src_pairs.name + ".jsonl")
    fs.raise_if_not_file(src_results)

    yes_dir = config.path(opts.dir, ["p2", "queued"])
    yes_results = yes_dir / (src_pairs.name + ".yes")
    fs.raise_if_exists(yes_results)
    with yes_results.open("w") as f:
        filter_results(str(src_results), True, f.write)

    no_dir = config.path(opts.dir, ["p3", "queued"])
    no_results = no_dir / (src_pairs.name + ".no")
    fs.raise_if_exists(no_results)
    with no_results.open("w") as f:
        filter_results(str(src_results), False, f.write)

    # TODO 

    return 0


def run(command, opts, argv):
    src_dir = config.path(opts.dir, ["p1", "running"])
    src_path = _resolve_pair_file(src_dir, argv)
    return _complete(src_path, opts)
