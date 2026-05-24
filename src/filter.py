import argparse
import numpy as np
import sys

from pathlib import Path

from src import compare_native


def filter_results(path: str, yes: bool, out_file, pmin = 0.5, prng = 1.0, use_max = False):
    # [pmin, pmin+rng) unless pmin+rng == 1.0, then [pmin, 1.0] inclusive
    pmax = pmin + prng
    if pmax == 1.0:
        pmax += 0.1

    blocks = compare_native.iter_projected_blocks([path], chunk_size=8192)

    yes_label = compare_native.LABEL_YES
    for block in blocks:
        labels = np.asarray(block.labels())[0]   # shape (rows, dirs)
        if yes:
            probs = np.asarray(block.probs())[0]
            if use_max:
                # pair qualifies only via its max YES-labeled prob
                yes_probs = np.where(labels == yes_label, probs, 0.0)
                max_probs = yes_probs.max(axis=1)
                mask = (max_probs >= pmin) & (max_probs < pmax)
            else:
                mask = (
                    (labels == yes_label)
                    & (probs >= pmin)
                    & (probs < pmax)
                ).any(axis=1)
        else:
            mask = (labels != yes_label).all(axis=1)
        #pairs = block.pairs()
        for idx in np.flatnonzero(mask):
            out_file.write(block.pair_at(idx) + "\n")


def _filter_args(args):
    filter_results(args.file, args.yes, sys.stdout, args.prob_min, args.prob_range, not args.any)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")

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
