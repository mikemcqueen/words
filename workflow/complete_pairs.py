# complete_pairs.py

import shutil

from pathlib import Path
from plumbum.cmd import cat, sort
from workflow import log, fs, config, usage
from src.filter import filter_results


def _usage_text():
    return "usage: wf complete pairs PAIRS-FILE"


def help_summary(name):
    return "pairs   — complete a pairs file evaluation"


def show_help(command, opts, argv):
    return print(_usage_text())


def _resolve_results_path(src_dir: Path, pairs_path: Path) -> Path:
    results_path = None
    count = 0
    for p in src_dir.iterdir():
        if p.suffix == ".jsonl" and p.name.startswith(pairs_path.stem) and p.is_file():
            results_path = p
            count += 1

    if count == 0:
        raise ValueError(f"result file not found for pairs file: {pairs_path.name}")
    assert results_path
    if count > 1:
        raise ValueError(f"multiple result files found for pairs file: {pairs_path.name}")
    assert count == 1
    return results_path


def _cat_sort_uniq(src: Path, dst: Path):
    dst_old = dst.parent / f"{dst.name}.old"
    dst.rename(dst_old)
    # preserve dst across unexpected failures
    try:
        ((cat[str(dst_old), str(src)] | sort["-u"]) > str(dst))()
    except Exception as e:
        dst_old.rename(dst)
        raise e


def merge_pairs(src_pairs: Path, dst_pairs) -> None:
    if dst_pairs.exists():
        _cat_sort_uniq(src_pairs, dst_pairs)
    else:
        dst_pairs.write_bytes(src_pairs.read_bytes())


def merge_with_done_pairs(phase: str, src_pairs: Path, opts) -> None:
    done_pairs = config.path(opts.dir, [phase, "done"]) / f"{phase}_done.pairs"
    merge_pairs(src_pairs, done_pairs)


def move_to_done(phase: str, src_in: Path, src_out: Path, opts) -> None:
    dst_in = config.path(opts.dir, [phase, "done", "in"]) / src_in.name
    src_in.rename(dst_in)
    dst_out = config.path(opts.dir, [phase, "done", "out"]) / src_out.name
    src_out.rename(dst_out)


# Workflow 1.2
def _complete(src_pairs: Path, src_results: Path, opts) -> int:
    log.info(f"found: {src_pairs.name}, {src_results.name}")

    # 1.2.a.i. YES go to "need manual review" queue.
    yes_dir = config.path(opts.dir, ["p2", "queued"])
    yes_results = yes_dir / (src_pairs.name + ".yes")
    if not opts.force:
        fs.raise_if_exists(yes_results)
    with yes_results.open("w") as f:
        filter_results(str(src_results), True, f)

    # 1.2.a.ii. NO go to the "need another automated pass" queue.
    no_dir = config.path(opts.dir, ["p3", "queued"])
    no_results = no_dir / (src_pairs.name + ".no")
    if not opts.force:
        fs.raise_if_exists(no_results)
    with no_results.open("w") as f:
        filter_results(str(src_results), False, f)

    phase = "p1"

    # 1.2.b. Add pairs to the "1st-pass classification done" set.
    merge_with_done_pairs(phase, src_pairs, opts)
        
    # 1.2.c. Cleanup files
    move_to_done(phase, src_pairs, src_results, opts)

    log.success(f"Completed pairs {src_pairs.name}")
    return 0


def run(command, opts, argv):
    if not argv:
        details = _usage_text()
        return usage.missing_argument(details)

    src_dir = config.path(opts.dir, ["p1", "eval"])
    pairs_path = src_dir / argv[0]
    fs.raise_if_not_file(pairs_path)
    results_path = _resolve_results_path(src_dir, pairs_path)

    return _complete(pairs_path, results_path, opts)
