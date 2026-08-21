# filter_pairs.py

import argparse
from pathlib import Path
from workflow import log, fs, batch, config, names, setops, usage
from src.filter import filter_results


def help_summary(name):
    return "filter  — filter a p1 results file into p2/queued"


def _make_local_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--pm", "--prob-min", dest="prob_min", type=float, default=0.9,
                   metavar="PMIN", help="minimum YES probability (default: 0.9)")
    p.add_argument("--pr", "--prob-range", dest="prob_range", type=float, default=0.1,
                   metavar="PRANGE", help="probability range width (default: 0.1)")
    return p


def _format_help(command, opts, argv):
    return usage.format_help(command, help_summary(command),
                             local_parser=_make_local_parser(), positional="JSONL-FILE")


def show_help(command, opts, argv):
    text = _format_help(command, opts, argv)
    if argv:
        return usage.invalid_argument(argv[0], text)
    print(text, end="")
    return 0


def run(command, opts, argv):
    local = _make_local_parser()
    local_opts, rest = local.parse_known_args(argv)

    if not rest:
        return usage.missing_argument(_format_help(command, opts, argv))

    pmin = local_opts.prob_min
    prange = local_opts.prob_range

    jsonl_arg = Path(rest[0])
    if jsonl_arg.is_absolute():
        jsonl_path = jsonl_arg
    else:
        jsonl_path = config.path(opts.dir, ["p1", "done", "out"]) / rest[0]
    fs.raise_if_not_file(jsonl_path)

    slug = names.slug(jsonl_path.stem, pmin, prange)
    out_name = names.artifact(slug, "p1", "yes")

    queued_dir = config.path(opts.dir, ["p2", "queued"])
    # "Has this batch already been through p2?" -- three stats against known
    # paths, rather than scanning three slots for a matching name. The batch
    # directory exists iff the work is in flight; done/in answers the rest.
    for candidate in (queued_dir / out_name,
                      batch.path(opts, "p2", slug),
                      config.path(opts.dir, ["p2", "done", "in"]) / out_name):
        if candidate.exists() and not opts.force:
            raise ValueError(
                f"already submitted: {candidate.relative_to(opts.dir / config.CONFIG_ROOT)}")

    out_path = queued_dir / out_name
    if not opts.force:
        fs.raise_if_exists(out_path)

    # TODO: move complete_pairs.filter_results_to here. call here from complete_pairs/yes.
    # Emit through merge: filter_results writes in corpus order, and p2/queued is
    # later fed to `comm -23`, which silently misbehaves on unsorted input.
    unsorted = out_path.with_name(out_path.name + ".unsorted")
    try:
        with unsorted.open("w") as f:
            filter_results([jsonl_path], True, f, pmin=pmin, prng=prange)
        setops.merge([unsorted], out_path)
    finally:
        unsorted.unlink(missing_ok=True)

    n = sum(1 for _ in out_path.open())
    log.success(f"Filtered {n} pairs [{pmin:.2f}, {pmin + prange:.2f}) → {out_path.name}")
    return 0
