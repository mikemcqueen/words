# complete_pairs.py

import shutil

from pathlib import Path
from plumbum.cmd import cat, sort
from workflow import log, fs, config, usage
from src.filter import filter_results


def _usage_text():
    return "usage: wf complete pairs PAIRS-FILE"


def help_summary(name):
    return "pairs   — complete a running pairs file"


def show_help(command, opts, argv):
    return print(_usage_text())


def _resolve_pairs_path(src_dir: Path, argv) -> Path:
    pairs_fn = argv[0]
    pairs_path = src_dir / Path(pairs_fn).name
    fs.raise_if_not_file(pairs_path)
    return pairs_path


def _resolve_results_path(src_dir: Path, pairs_path: Path) -> Path:
    results_path = None
    count = 0
    for p in src_dir.iterdir():
        if p.suffix == ".jsonl" and p.name.startswith(pairs_path.stem) and p.is_file():
            results_path = p
            count += 1

    if count == 0:
        raise ValueError("result file not found for pairs file: {pairs_path.name}")
    if count > 1:
        raise ValueError("multiple result files found for pairs file: {pairs_path.name}")
    assert count == 1
    return results_path


def _merge_with_done_pairs(src_pairs: Path, done_pairs: Path):
    done_pairs_old = done_pairs.parent / "p1_done.old"
    done_pairs.rename(done_pairs_old)
    # preserve done_pairs across unexpected failures
    try:
        # done_pairs <- old_done_pairs ∪ src_pairs
        ((cat[str(done_pairs_old), str(src_pairs)] | sort["-u"]) > str(done_pairs))()
    except Exception as e:
        done_pairs_old.rename(done_pairs)
        raise e

# Workflow 1.2
def _complete(src_pairs: Path, src_results: Path, opts) -> int:
    log.info(f"found: {src_pairs.name}, {src_results.name}")

    # 1.2.a.i
    yes_dir = config.path(opts.dir, ["p2", "queued"])
    yes_results = yes_dir / (src_pairs.name + ".yes")
    if not opts.force:
        fs.raise_if_exists(yes_results)
    with yes_results.open("w") as f:
        filter_results(str(src_results), True, f)

    # 1.2.a.ii.
    no_dir = config.path(opts.dir, ["p3", "queued"])
    no_results = no_dir / (src_pairs.name + ".no")
    if not opts.force:
        fs.raise_if_exists(no_results)
    with no_results.open("w") as f:
        filter_results(str(src_results), False, f)

    # 1.2.b
    done_dir = config.path(opts.dir, ["p1", "done"])
    done_pairs = done_dir / "p1_done"
    if done_pairs.exists():
        # done <- done ∪ src 
        _merge_with_done_pairs(src_pairs, done_pairs)
    else:
        # done <- src
        done_pairs.write_bytes(src_pairs.read_bytes())

    # 1.2.c
    dst_pairs = config.path(opts.dir, ["p1", "done", "pairs"]) / src_pairs.name
    src_pairs.rename(dst_pairs)    
    dst_results = config.path(opts.dir, ["p1", "done", "results"]) / src_results.name
    src_results.rename(dst_results)

    log.success(f"Completed pairs {src_pairs.name}")
    return 0


def run(command, opts, argv):
    if not argv:
        details = _usage_text()
        return usage.missing_argument(details)

    src_dir = config.path(opts.dir, ["p1", "running"])
    pairs_path = _resolve_pairs_path(src_dir, argv)
    results_path = _resolve_results_path(src_dir, pairs_path)
    return _complete(pairs_path, results_path, opts)
