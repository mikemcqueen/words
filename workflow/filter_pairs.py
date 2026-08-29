# filter_pairs.py
#
# UNREGISTERED. `wf filter` is not reachable from the CLI -- see COMMANDS in
# wf.py. The code is kept because the errand is still wanted; the
# implementation is not.
#
# TODO: bring `filter` up to the steps architecture.
#
# This is the last command written before `workflow/steps/` existed: it
# predates that refactor (7f03f48, vs ff0e359 which introduced steps) and was
# never converted. It shares no code with `complete p1` beyond the primitives
# both are entitled to -- `filter_results`, `setops.merge`, the `names`
# renderers -- and duplicates the sequencing around them, which is also spelled
# a third time in steps/filter.py.
#
# What retired it: it opens no bundle, so it has no Context and no bundle name.
# It renders the p2 queue name from the *result stem* instead, which
# `p1_extract` stopped doing -- evalpair spells that stem with `_`, which
# `check_name` refuses, so the artifact this command publishes is one
# `wf eval p2` cannot open. It also hand-rolls the "already submitted?" guard
# against three known paths, a question `bundle.in_flight` and the queue
# contract answer everywhere else.
#
# Converting it means giving it a bundle: `filter` is the "re-slice a result I
# already have at a different band" errand, so the bundle name has to come from
# the caller or from the archived pairs file the result was produced from, not
# from the jsonl. Once it has one, the naming divergence and the guard both
# disappear into machinery that already exists.

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

        bundle_name = names.bundle_name(jsonl_path.stem, pmin, prange)
        out_name = names.artifact(bundle_name, "p1", "yes")

        queued_dir = config.path(opts.dir, ["p2", "queued"])
        # "Has this bundle already been through p2?" -- three stats against
        # known paths, rather than scanning three slots for a matching name. The
        # bundle directory exists iff the work is in flight; done/in answers the
        # rest.
        for candidate in (queued_dir / out_name,
                          context.Context(root=opts.dir, phase="p2",
                                          bundle_name=bundle_name).bundle_dir,
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
