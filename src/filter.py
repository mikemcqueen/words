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


def filter_results(path: str, yes: bool, out_file, pmin = 0.5, prng = 1.0, use_max = False):
    pmax = _pmax(pmin, prng)
    blocks = compare_native.iter_projected_blocks([path], chunk_size=8192)

    for block in blocks:
        mask = _build_prob_mask(block, yes, pmin, pmax, use_max)
        for idx in np.flatnonzero(mask):
            out_file.write(block.pair_at(idx) + "\n")


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


def filter_pairs(pairs_path: str, results_dir: str, yes: bool, out_file,
                       pmin = 0.5, prng = 1.0, use_max = False):
    pmax = _pmax(pmin, prng)
    pair_set = _load_pair_set(pairs_path)
    print(f"loaded {len(pair_set):,} pairs from {pairs_path}", file=sys.stderr)

    files = sorted(Path(results_dir).glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no .jsonl files in {results_dir}")

    for results_file in files:
        try:
            blocks = compare_native.iter_projected_blocks([str(results_file)], chunk_size=8192)
        except Exception as e:
            print(f"WARNING: skipping {results_file}: {e}", file=sys.stderr)
            continue

        for block in prefetch(blocks):
            mask = _build_prob_mask(block, yes, pmin, pmax, use_max)
            pair_mask = np.fromiter(
                (p in pair_set for p in block.pairs()),
                dtype=bool, count=block.size,
            )
            mask &= pair_mask
            for idx in np.flatnonzero(mask):
                out_file.write(block.pair_at(idx) + "\n")


def _filter_args(args):
    use_max = not args.any
    if args.dir is not None:
        filter_results_dir(args.file, args.dir, args.yes, sys.stdout,
                           args.prob_min, args.prob_range, use_max)
    else:
        filter_results(args.file, args.yes, sys.stdout,
                       args.prob_min, args.prob_range, use_max)


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
