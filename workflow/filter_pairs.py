# filter_pairs.py

import argparse

from pathlib import Path

from src.filter import filter_results
from workflow import command, config, context, fs, log, names, setops, usage


class FilterPairs(command.Action):
    def __init__(self):
        super().__init__(
            summary="filter  — filter a p1 results file into p2/queued",
            positional="JSONL-FILE",
        )

    def parser(self):
        p = argparse.ArgumentParser(add_help=False)
        p.add_argument("--pm", "--prob-min", dest="prob_min", type=float, default=0.9,
                       metavar="PMIN", help="minimum YES probability (default: 0.9)")
        p.add_argument("--pr", "--prob-range", dest="prob_range", type=float, default=0.1,
                       metavar="PRANGE", help="probability range width (default: 0.1)")
        return p

    def run(self, command, opts, argv) -> int:
        rest = self.parse(opts, argv)
        if not rest:
            return usage.missing_argument(self.format_help(command))

        pmin = opts.prob_min
        prange = opts.prob_range

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
                          context.Context(root=opts.dir, phase="p2", slug=slug).batch_dir,
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


COMMAND = FilterPairs()
