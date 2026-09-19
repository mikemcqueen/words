import argparse
import math
import random
import sys

from pathlib import Path

import numpy as np

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


def _yes_prob_mask_and_scores(block, pmin: float, pmax: float, use_max: bool):
    labels = np.asarray(block.labels())[0]   # shape (rows, dirs)
    probs = np.asarray(block.probs())[0]
    yes_label = compare_native.LABEL_YES
    if use_max:
        direction_scores = np.where(labels == yes_label, probs, 0.0)
    else:
        in_band = (
            (labels == yes_label)
            & (probs >= pmin)
            & (probs < pmax)
        )
        direction_scores = np.where(in_band, probs, -np.inf)
    winners = direction_scores.argmax(axis=1)
    scores = direction_scores[np.arange(len(winners)), winners]
    mask = ((scores >= pmin) & (scores < pmax) if use_max
            else in_band.any(axis=1))
    reverse_wins = np.asarray(block.directions())[winners] == "rvs"
    return mask, scores, reverse_wins


def _pmax(pmin: float, prng: float) -> float:
    # [pmin, pmin+rng) unless pmin+rng == 1.0, then [pmin, 1.0] inclusive
    pmax = pmin + prng
    if pmax == 1.0:
        pmax += 0.1
    return pmax


def _load_pair_set(path: str) -> set:
    """Load unordered pair keys from a pair-list file."""
    pairs = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.add(_pair_lookup_key(line))
            if len(pairs) > MAX_PAIR_SET:
                raise SystemExit(f"pair set exceeds {MAX_PAIR_SET:,} entries (from {path})")
    return pairs


def _canonical_pair(pair: str) -> tuple[str, bool]:
    """Return an unordered-pair key and whether pair is reversed from it."""
    fields = pair.split(",")
    if len(fields) != 2:
        raise ValueError(f"cannot deduplicate malformed pair: {pair!r}")
    if fields[0] <= fields[1]:
        return pair, False
    return f"{fields[1]},{fields[0]}", True


def _pair_lookup_key(pair: str) -> str:
    # Preserve exact matching for malformed IDs, which the old mask accepted.
    try:
        return _canonical_pair(pair)[0]
    except ValueError:
        return pair


def _oriented_pair(canonical: str, is_reversed: bool) -> str:
    if not is_reversed:
        return canonical
    left, right = canonical.split(",")
    return f"{right},{left}"


def filter_results(paths, yes: bool, out_file, pairs_path: str | None = None,
                   pmin = 0.5, prng = 1.0, use_max = False,
                   report_pair_load = True, dedupe = None,
                   sample: int | None = None):
    """Write pairs matching the label/probability band to out_file.

    `paths` is a *corpus*: each file gets its own reader. A path list handed
    straight to iter_projected_blocks means something else entirely -- aligned
    files holding the same pairs, one per host -- and raises on a pair mismatch.

    `pairs_path` optionally restricts output to unordered members of that pair
    set; None skips the identity mask.

    YES pairs are deduplicated by default. `dedupe=False` emits every matching
    row in its stored orientation. Deduplication keeps the orientation whose
    matching direction has the highest in-band score. The score sign stores
    that orientation.

    `sample` limits output to a uniform sample of the final matching rows.
    """
    if isinstance(paths, (str, Path)):
        # A bare path would iterate per character; each "file" then fails to open
        # and the warn-and-continue below would turn it into empty output.
        raise TypeError(f"filter_results() takes a list of paths, not {type(paths).__name__}")
    paths = list(paths)
    if not paths:
        raise SystemExit("no result files to filter")
    if dedupe is None:
        dedupe = yes
    if dedupe and not yes:
        raise ValueError("dedupe requires yes=True")
    if sample is not None and sample < 0:
        raise ValueError("sample must be nonnegative")

    pmax = _pmax(pmin, prng)
    pair_set = None
    if pairs_path is not None:
        pair_set = _load_pair_set(pairs_path)
        if report_pair_load:
            print(f"loaded {len(pair_set):,} pairs from {pairs_path}",
                  file=sys.stderr)

    # Skipping an unreadable file lets one corrupt member not kill a sweep over
    # a whole archived corpus. It must not turn "nothing could be read" into an
    # empty result reported as success -- which is what it did for the
    # single-file callers this function absorbed.
    readable = 0
    best = {} if dedupe else None
    sampled = [] if sample is not None else None
    seen = 0

    def emit(pair):
        nonlocal seen
        if sampled is None:
            out_file.write(pair + "\n")
            return
        seen += 1
        if len(sampled) < sample:
            sampled.append(pair)
        elif sample:
            slot = random.randrange(seen)
            if slot < sample:
                sampled[slot] = pair

    for results_file in paths:
        try:
            blocks = compare_native.iter_projected_blocks([str(results_file)], chunk_size=8192)
        except Exception as e:
            print(f"WARNING: skipping {results_file}: {e}", file=sys.stderr)
            continue
        readable += 1

        for block in prefetch(blocks):
            if dedupe:
                mask, scores, reverse_wins = _yes_prob_mask_and_scores(
                    block, pmin, pmax, use_max)
            else:
                mask = _build_prob_mask(block, yes, pmin, pmax, use_max)
            if pair_set is not None:
                mask &= np.fromiter(
                    (_pair_lookup_key(p) in pair_set for p in block.pairs()),
                    dtype=bool, count=block.size,
                )
            for idx in np.flatnonzero(mask):
                pair = block.pair_at(idx)
                if not dedupe:
                    emit(pair)
                    continue

                canonical, is_reversed = _canonical_pair(pair)
                is_reversed ^= bool(reverse_wins[idx])
                score = float(scores[idx])
                previous = best.get(canonical)
                if previous is None or score > abs(previous):
                    best[canonical] = math.copysign(
                        score, -1.0 if is_reversed else 1.0)

    if not readable:
        raise SystemExit(f"no readable result files among {len(paths)}")

    if dedupe:
        for canonical, signed_score in best.items():
            is_reversed = math.copysign(1.0, signed_score) < 0.0
            emit(_oriented_pair(canonical, is_reversed))
    if sampled is not None:
        out_file.writelines(pair + "\n" for pair in sampled)


def _filter_args(args):
    use_max = not args.any
    if args.dir is not None:
        paths = sorted(Path(args.dir).glob("*.jsonl"))
        if not paths:
            raise SystemExit(f"no .jsonl files in {args.dir}")
        if args.file is None:
            print("WARNING: no pairs file supplied - displaying all pairs",
                  file=sys.stderr)
        filter_results(paths, args.yes, sys.stdout, pairs_path=args.file,
                       pmin=args.prob_min, prng=args.prob_range, use_max=use_max,
                       dedupe=args.yes and not args.ignore_ordering,
                       sample=args.sample)
    else:
        filter_results([args.file], args.yes, sys.stdout,
                       pmin=args.prob_min, prng=args.prob_range, use_max=use_max,
                       dedupe=args.yes and not args.ignore_ordering,
                       sample=args.sample)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Filter eval results by prob/label, optionally restricted to a pair list.")
    parser.add_argument(
        "file", nargs="?",
        help="results .jsonl, or optional pair list when --dir is given")
    parser.add_argument("-d", "--dir", default=None, metavar="RESULTS_DIR",
                        help="directory of .jsonl results; positional becomes the pair list")

    yesno = parser.add_mutually_exclusive_group(required=True)
    yesno.add_argument("-y", "--yes", action="store_true")
    yesno.add_argument("-n", "--no", action="store_true")

    parser.add_argument("--pm", "--prob-min", dest="prob_min", type=float, default=0.5)
    parser.add_argument("--pr", "--prob-range", dest="prob_range", type=float, default=1.0)
    parser.add_argument("--any", dest="any", action="store_true")
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "--ignore-ordering", action="store_true",
        help="emit every matching YES row, including reversed pair orderings")
    output.add_argument(
        "--sample", type=int, metavar="N",
        help="display a random sample of up to N matching pairs")
    args = parser.parse_args()
    if args.sample is not None and args.sample < 0:
        parser.error("--sample must be nonnegative")
    if args.file is None and args.dir is None:
        parser.error("file is required unless --dir is given")
    return args


def main():
    args = _parse_args()
    _filter_args(args)


if __name__ == "__main__":
    main()
