import argparse
import numpy as np
import sys

from pathlib import Path

from src import compare_native
from src.common import prefetch


MAX_PAIR_SET = 5_000_000


def _build_prob_mask(block, yes: bool, pmin: float, pmax: float, use_max: bool):
    labels = np.asarray(block.labels())[0]   # shape (rows, dirs)
    yes_label = compare_native.LABEL_YES
    if yes:
        probs = np.asarray(block.probs())[0]
        if use_max:
            yes_probs = np.where(labels == yes_label, probs, 0.0)
            max_probs = yes_probs.max(axis=1)
            return (max_probs >= pmin) & (max_probs < pmax)
        return (
            (labels == yes_label)
            & (probs >= pmin)
            & (probs < pmax)
        ).any(axis=1)
    return (labels != yes_label).all(axis=1)


def _pmax(pmin: float, prng: float) -> float:
    # [pmin, pmin+rng) unless pmin+rng == 1.0, then [pmin, 1.0] inclusive
    pmax = pmin + prng
    if pmax == 1.0:
        pmax += 0.1
    return pmax


def _load_pair_set(path: str) -> set:
    """Load a pair-list file (one 'word1,word2' per line) into a set."""
    pairs = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.add(line)
            if len(pairs) > MAX_PAIR_SET:
                raise SystemExit(f"pair set exceeds {MAX_PAIR_SET:,} entries (from {path})")
    return pairs


def filter_results(paths, yes: bool, out_file, pairs_path: str | None = None,
                   pmin = 0.5, prng = 1.0, use_max = False):
    """Write pairs matching the label/probability band to out_file.

    `paths` is a *corpus*: each file gets its own reader. A path list handed
    straight to iter_projected_blocks means something else entirely -- aligned
    files holding the same pairs, one per host -- and raises on a pair mismatch.

    `pairs_path` optionally restricts output to members of that pair set;
    None skips the identity mask.
    """
    if isinstance(paths, (str, Path)):
        # A bare path would iterate per character; each "file" then fails to open
        # and the warn-and-continue below would turn it into empty output.
        raise TypeError(f"filter_results() takes a list of paths, not {type(paths).__name__}")
    paths = list(paths)
    if not paths:
        raise SystemExit("no result files to filter")

    pmax = _pmax(pmin, prng)
    pair_set = None
    if pairs_path is not None:
        pair_set = _load_pair_set(pairs_path)
        print(f"loaded {len(pair_set):,} pairs from {pairs_path}", file=sys.stderr)

    # Skipping an unreadable file lets one corrupt member not kill a sweep over
    # a whole archived corpus. It must not turn "nothing could be read" into an
    # empty result reported as success -- which is what it did for the
    # single-file callers this function absorbed.
    readable = 0

    for results_file in paths:
        try:
            blocks = compare_native.iter_projected_blocks([str(results_file)], chunk_size=8192)
        except Exception as e:
            print(f"WARNING: skipping {results_file}: {e}", file=sys.stderr)
            continue
        readable += 1

        for block in prefetch(blocks):
            mask = _build_prob_mask(block, yes, pmin, pmax, use_max)
            if pair_set is not None:
                mask &= np.fromiter(
                    (p in pair_set for p in block.pairs()),
                    dtype=bool, count=block.size,
                )
            for idx in np.flatnonzero(mask):
                out_file.write(block.pair_at(idx) + "\n")

    if not readable:
        raise SystemExit(f"no readable result files among {len(paths)}")


def _filter_args(args):
    use_max = not args.any
    if args.dir is not None:
        paths = sorted(Path(args.dir).glob("*.jsonl"))
        if not paths:
            raise SystemExit(f"no .jsonl files in {args.dir}")
        filter_results(paths, args.yes, sys.stdout, pairs_path=args.file,
                       pmin=args.prob_min, prng=args.prob_range, use_max=use_max)
    else:
        filter_results([args.file], args.yes, sys.stdout,
                       pmin=args.prob_min, prng=args.prob_range, use_max=use_max)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Filter eval results by prob/label, optionally restricted to a pair list.")
    parser.add_argument("file", help="results .jsonl, or pair list when --dir is given")
    parser.add_argument("-d", "--dir", default=None, metavar="RESULTS_DIR",
                        help="directory of .jsonl results; positional becomes the pair list")

    yesno = parser.add_mutually_exclusive_group(required=True)
    yesno.add_argument("-y", "--yes", action="store_true")
    yesno.add_argument("-n", "--no", action="store_true")

    parser.add_argument("--pm", "--prob-min", dest="prob_min", type=float, default=0.5)
    parser.add_argument("--pr", "--prob-range", dest="prob_range", type=float, default=1.0)
    parser.add_argument("--any", dest="any", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    _filter_args(args)


if __name__ == "__main__":
    main()
