# complete_pairs.py

import shutil

from pathlib import Path
from plumbum.cmd import cat, sort
from workflow import log, fs, config, usage
from src.filter import filter_results


def help_summary(name):
    return "pairs   — complete a pairs file evaluation"


def _format_help(command, opts, argv):
    return usage.format_help(command, help_summary(command), positional="PAIRS-FILE")


def show_help(command, opts, argv):
    text = _format_help(command, opts, argv)
    if argv:
        return usage.invalid_argument(argv[0], text)
    print(text, end="")
    return 0


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


def _filter_results_to(src_results: Path, yes: bool, dst_results: Path, opts) -> None:
    if not opts.force:
        fs.raise_if_exists(dst_results)
    with dst_results.open("w") as f:
        # NOTE: qwen35 27B default
        filter_results(str(src_results), yes, f, pmin=0.9)


# Workflow 1.2
def _complete(src_pairs: Path, src_results: Path, opts) -> int:
    log.info(f"found: {src_pairs.name}, {src_results.name}")
    phase = "p1"

    # 1.2.a.i. YES pairs go to p2's "need manual review" queue
    yes_pairs = config.path(opts.dir, ["p2", "queued"]) / (src_pairs.name + ".p1.yes")
    _filter_results_to(src_results, True, yes_pairs, opts)

    # 1.2.a.ii. NO pairs go to p3's "need another automated pass" queue
    # TODO: not sure this is technically correct.  also we're only filtering the
    #       top 10% of YES for Qwen35 27B. so there are a lot more "maybe NO" pairs
    #       to be passing along to p3 here.
    no_pairs = config.path(opts.dir, ["p3", "queued"]) / (src_pairs.name + ".p1.no")
    _filter_results_to(src_results, False, no_pairs, opts)

    # 1.2.b. Merge input pairs with "1st-pass classification done" pairs
    merge_with_done_pairs(phase, src_pairs, opts)
        
    # 1.2.c. Move src_pairs → p1/done/in, src_results → p1/done/out
    move_to_done(phase, src_pairs, src_results, opts)

    log.success(f"Completed pairs {src_pairs.name}")
    return 0


def run(command, opts, argv):
    if not argv:
        return usage.missing_argument(_format_help(command, opts, argv))

    src_dir = config.path(opts.dir, ["p1", "eval"])
    pairs_path = src_dir / argv[0]
    fs.raise_if_not_file(pairs_path)
    results_path = _resolve_results_path(src_dir, pairs_path)

    return _complete(pairs_path, results_path, opts)
