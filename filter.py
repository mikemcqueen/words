import argparse
import sys

import numpy as np

import compare_native


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    yesno = parser.add_mutually_exclusive_group(required=True)
    yesno.add_argument("-y", "--yes", action="store_true")
    yesno.add_argument("-n", "--no", action="store_true")
    parser.add_argument("--pm", "--prob-min", dest="prob_min", type=float, default=0.5)
    parser.add_argument("--pr", "--prob-range", dest="prob_range", type=float, default=1.0)
    args = parser.parse_args()

    args.prob_max = args.prob_min + args.prob_range
    if args.prob_max == 1.0:
        args.prob_max += 0.1

    compare_native.require_native()
    blocks = compare_native.iter_projected_blocks([args.file], chunk_size=8192)

    out = sys.stdout.write
    yes_label = compare_native.LABEL_YES
    for block in blocks:
        labels = np.asarray(block.labels())[0]   # shape (rows, dirs)
        if args.yes:
            probs = np.asarray(block.probs())[0]
            mask = (
                (labels == yes_label)
                & (probs >= args.prob_min)
                & (probs < args.prob_max)
            ).any(axis=1)
        else:
            mask = (labels != yes_label).all(axis=1)
        pairs = block.pairs()
        for idx in np.flatnonzero(mask):
            out(pairs[idx] + "\n")

if __name__ == "__main__":
    main()
