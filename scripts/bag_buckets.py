#!/usr/bin/env python3
"""Split a letter bag into overlapping buckets that each drop some rare letters.

    python scripts/bag_buckets.py $S1
    python scripts/bag_buckets.py $S1 --size 38
    python scripts/bag_buckets.py $S1SMAL --rare bfgkmpuvy
    python scripts/bag_buckets.py $S1 --size 40 --pairs pairs/p23.yes.sorted

Letters that occur at most --rare-max times in BAG are rare, unless --rare
names them. Use --rare on a count-reduced bag, where common letters may be
down to 1 or 2 copies. The rare letters are split into --groups groups, and
bucket i is BAG minus every copy of the letters in group i. A letter's copies
are always dropped together. Any pair whose rare letters touch fewer than all
groups fits in at least one bucket, so with 3 groups every pair using at most
2 rare letters is in some bucket (before --size cuts common letters).

With --size N, each bucket keeps all copies of its remaining rare letters and
cuts the other letters to N letters in total. Vowels and consonants get
separate shares so the bucket's CV is close to --cv-target, and each share is
split in proportion to the letter's count in BAG, keeping at least 1 copy,
or the number of times the letter appears in --keep.

Groupings are scored with words from --dict that can be made from BAG, and
with --samples random pairs of those words that BAG can make together, or
with the pairs in --pairs FILE (lines of word,word; pairs BAG cannot make are
skipped):
  pair coverage  share of sampled pairs that fit in some bucket
  biggest bucket most words formable from any one bucket, as a share of BAG's
  CV             consonants/vowels per bucket (y counts as a consonant),
                 compared with --cv-target
Lower score is better: (1 - pair coverage) * --w-cov
  + biggest bucket * --w-size + worst |ln(CV / target)| * --w-cv.

The anchor table lists each letter's count in BAG and how many formable words
contain it. Letters with few anchor words are the best ones to branch on.
"""

import argparse
import itertools
import math
import os
import sys
from collections import Counter

import numpy as np

VOWELS = set("aeiou")
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def default_dict():
    root = os.environ.get("WFROOT")
    if root:
        return os.path.join(root, ".wf", "dict", "dict.filtered")
    return None


def vector(counts):
    return np.array([counts.get(ch, 0) for ch in LETTERS], dtype=np.int16)


def load_words(path, bag, min_len):
    bagvec = vector(bag)
    words = []
    with open(path) as f:
        for line in f:
            w = line.strip().lower()
            if len(w) >= min_len and w.isalpha() and w.isascii():
                if (vector(Counter(w)) <= bagvec).all():
                    words.append(w)
    return words


def cv_ratio(bag):
    v = sum(n for ch, n in bag.items() if ch in VOWELS)
    c = sum(n for ch, n in bag.items() if ch not in VOWELS)
    return c / v if v else math.inf


def bag_string(bag):
    return "".join(ch * bag[ch] for ch in sorted(bag) if bag[ch] > 0)


def floors(counts, keep):
    return {ch: min(c, max(1, keep[ch])) for ch, c in counts.items()}


def allocate(n, counts, keep):
    """Split n copies over letters in proportion to counts, floor <= share <= count."""
    letters = sorted(counts)
    if not letters:
        return {}
    lo = floors(counts, keep)
    n = max(sum(lo.values()), min(n, sum(counts.values())))
    total = sum(counts.values())
    quota = {ch: n * counts[ch] / total for ch in letters}
    share = {ch: min(counts[ch], max(lo[ch], math.floor(quota[ch]))) for ch in letters}
    while sum(share.values()) < n:
        ch = max((c for c in letters if share[c] < counts[c]), key=lambda c: quota[c] - share[c])
        share[ch] += 1
    while sum(share.values()) > n:
        ch = min((c for c in letters if share[c] > lo[c]), key=lambda c: quota[c] - share[c])
        share[ch] -= 1
    return share


def make_bucket(bag, rare, dropped, size, cv_target, keep):
    b = Counter({ch: c for ch, c in bag.items() if ch not in dropped})
    if size is None:
        return b
    kept_rare = {ch: c for ch, c in b.items() if ch in rare}
    common = {ch: c for ch, c in b.items() if ch not in rare}
    budget = size - sum(kept_rare.values())
    cv = {ch: c for ch, c in common.items() if ch in VOWELS}
    cc = {ch: c for ch, c in common.items() if ch not in VOWELS}
    if not cv or not cc:
        share = allocate(budget, common, keep)
    else:
        vowels = round(size / (1 + cv_target)) - sum(c for ch, c in kept_rare.items() if ch in VOWELS)
        vowels = min(max(vowels, sum(floors(cv, keep).values())), budget - sum(floors(cc, keep).values()))
        share = {**allocate(vowels, cv, keep), **allocate(budget - vowels, cc, keep)}
    return Counter({**kept_rare, **share})


def set_partitions(k, n):
    """Yield group labels for k items in exactly n non-empty unlabeled groups."""
    labels = [0] * k

    def rec(i, used):
        if k - i < n - used:
            return
        if i == k:
            if used == n:
                yield list(labels)
            return
        for g in range(min(used + 1, n)):
            labels[i] = g
            yield from rec(i + 1, max(used, g + 1))

    yield from rec(0, 0)


def stirling2(k, n):
    return sum((-1) ** j * math.comb(n, j) * (n - j) ** k for j in range(n + 1)) // math.factorial(n)


def load_pairs(path, bagvec):
    seen = set()
    read = 0
    with open(path) as f:
        for line in f:
            fields = line.strip().lower().split(",")
            if len(fields) < 2:
                continue
            read += 1
            letters = "".join(ch for ch in fields[0] + fields[1] if "a" <= ch <= "z")
            key = "".join(sorted(letters))
            if letters and key not in seen and (vector(Counter(letters)) <= bagvec).all():
                seen.add(key)
    return read, np.array([vector(Counter(k)) for k in sorted(seen)]).reshape(-1, len(LETTERS))


def sample_pairs(wordvecs, bagvec, samples, rng):
    found = []
    have = 0
    for _ in range(1000):
        i = rng.integers(0, len(wordvecs), size=samples * 2)
        j = rng.integers(0, len(wordvecs), size=samples * 2)
        p = wordvecs[i] + wordvecs[j]
        p = p[(p <= bagvec).all(axis=1)]
        found.append(p)
        have += len(p)
        if have >= samples:
            break
    return np.concatenate(found)[:samples]


class Scorer:
    def __init__(self, bag, rare, words, n, args, rng):
        self.bag, self.rare, self.n, self.args = bag, rare, n, args
        self.wordvecs = np.array([vector(Counter(w)) for w in words])
        if args.pairs:
            self.pairs_read, self.pairvecs = load_pairs(args.pairs, vector(bag))
            if not len(self.pairvecs):
                sys.exit(f"no pair in {args.pairs} can be made from the bag")
        else:
            self.pairvecs = sample_pairs(self.wordvecs, vector(bag), args.samples, rng)
        self.cache = {}

    def bucket(self, dropped):
        key = frozenset(dropped)
        if key not in self.cache:
            a = self.args
            b = make_bucket(self.bag, self.rare, key, a.size, a.cv_target, a.keep)
            v = vector(b)
            self.cache[key] = (b, (self.wordvecs <= v).all(axis=1), (self.pairvecs <= v).all(axis=1))
        return self.cache[key]

    def buckets(self, labels):
        return [self.bucket({ch for ch, lab in zip(self.rare, labels) if lab == g}) for g in range(self.n)]

    def score(self, labels):
        bs = self.buckets(labels)
        words = np.logical_or.reduce([w for _, w, _ in bs])
        pairs = np.logical_or.reduce([p for _, _, p in bs])
        coverage = pairs.mean()
        word_coverage = words.mean()
        bucket_words = [int(w.sum()) for _, w, _ in bs]
        bucket_pairs = [p.mean() for _, _, p in bs]
        cvs = [cv_ratio(b) for b, _, _ in bs]
        cv_dev = max(abs(math.log(c / self.args.cv_target)) for c in cvs)
        biggest = max(bucket_words) / len(self.wordvecs)
        a = self.args
        s = (1 - coverage) * a.w_cov + biggest * a.w_size + cv_dev * a.w_cv
        return s, dict(coverage=coverage, word_coverage=word_coverage, bucket_words=bucket_words,
                       bucket_pairs=bucket_pairs, cvs=cvs)


def local_search(scorer, k, n, restarts, rng):
    best = []
    for _ in range(restarts):
        labels = list(rng.integers(0, n, size=k))
        cur = scorer.score(labels)[0] if len(set(labels)) == n else math.inf
        improved = True
        while improved:
            improved = False
            for i, g in itertools.product(range(k), range(n)):
                if labels[i] == g:
                    continue
                trial = list(labels)
                trial[i] = g
                if len(set(trial)) < n:
                    continue
                s = scorer.score(trial)[0]
                if s < cur:
                    labels, cur, improved = trial, s, True
        best.append((cur, canonical(labels)))
    return best


def canonical(labels):
    remap = {}
    return tuple(remap.setdefault(lab, len(remap)) for lab in labels)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bag", help="letters of the bag; non-letters are ignored")
    p.add_argument("--dict", default=default_dict(), help="word list, one per line (default $WFROOT/.wf/dict/dict.filtered)")
    p.add_argument("--groups", type=int, default=3)
    p.add_argument("--rare-max", type=int, default=2, help="letters with at most this many copies can be dropped")
    p.add_argument("--rare", help="letters that can be dropped, instead of using --rare-max")
    p.add_argument("--size", type=int, help="cut each bucket to this many letters")
    p.add_argument("--keep", default="",
                   help="minimum copies --size keeps, as repeated letters: ww keeps at least 2 w's")
    p.add_argument("--min-len", type=int, default=2, help="ignore shorter dictionary words")
    p.add_argument("--pairs", help="word,word lines to score pair coverage with, instead of sampled pairs")
    p.add_argument("--samples", type=int, default=20000, help="word pairs sampled for pair coverage")
    p.add_argument("--cv-target", type=float, default=1.6)
    p.add_argument("--w-cov", type=float, default=1.0)
    p.add_argument("--w-size", type=float, default=1.0)
    p.add_argument("--w-cv", type=float, default=0.5)
    p.add_argument("--top", type=int, default=3, help="groupings to print")
    p.add_argument("--max-exhaustive", type=int, default=200000,
                   help="above this many groupings, use local search instead")
    p.add_argument("--restarts", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not args.dict:
        sys.exit("no --dict given and $WFROOT is not set")
    bag = Counter(ch for ch in args.bag.lower() if "a" <= ch <= "z")
    if args.rare:
        rare = sorted(set(ch for ch in args.rare.lower() if bag[ch] > 0))
    else:
        rare = sorted(ch for ch, c in bag.items() if c <= args.rare_max)
    args.keep = Counter(ch for ch in args.keep.lower() if "a" <= ch <= "z")
    n = args.groups
    if len(rare) < n:
        sys.exit(f"only {len(rare)} rare letters ({''.join(rare)}); need at least --groups {n}")
    if args.size is not None:
        common = {ch: c for ch, c in bag.items() if ch not in rare}
        need = sum(bag[ch] for ch in rare) + sum(floors(common, args.keep).values())
        if args.size < need:
            sys.exit(f"--size {args.size} is too small: rare letters plus --keep (or 1) of each other letter is {need}")

    rng = np.random.default_rng(args.seed)
    words = load_words(args.dict, bag, args.min_len)
    scorer = Scorer(bag, rare, words, n, args, rng)

    print(f"bag: {bag_string(bag)}")
    print(f"  {sum(bag.values())} letters, {len(bag)} distinct, CV {cv_ratio(bag):.2f}, "
          f"{len(words)} formable words from {args.dict}")
    if args.pairs:
        with_rare = int((scorer.pairvecs[:, [LETTERS.index(ch) for ch in rare]] > 0).any(axis=1).sum())
        print(f"  {len(scorer.pairvecs)} distinct pairs the bag can make, of {scorer.pairs_read} in {args.pairs}; "
              f"{with_rare} use a rare letter")
    else:
        print(f"  {len(scorer.pairvecs)} sampled pairs")
    print(f"rare letters ({'--rare' if args.rare else f'<= {args.rare_max} copies'}): "
          f"{' '.join(ch * bag[ch] for ch in rare)}")

    total = stirling2(len(rare), n)
    if total <= args.max_exhaustive:
        results = [(scorer.score(lab)[0], tuple(lab)) for lab in set_partitions(len(rare), n)]
        how = f"all {total} groupings"
    else:
        results = local_search(scorer, len(rare), n, args.restarts, rng)
        how = f"local search, {args.restarts} restarts, of {total} groupings"
    results = sorted(set(results))[: args.top]
    print(f"searched {how}\n")

    for rank, (s, labels) in enumerate(results, 1):
        _, m = scorer.score(list(labels))
        groups = ["".join(ch * bag[ch] for ch, lab in zip(rare, labels) if lab == g) for g in range(n)]
        print(f"#{rank}  score {s:.3f}  pair coverage {m['coverage']:.1%}  "
              f"word coverage {m['word_coverage']:.1%}")
        for g, (b, _, _) in enumerate(scorer.buckets(labels)):
            print(f"  bucket {g + 1} (drops {groups[g]}): {bag_string(b)}")
            print(f"    {sum(b.values())} letters, {len(b)} distinct, CV {m['cvs'][g]:.2f}, "
                  f"{m['bucket_words'][g]} words ({m['bucket_words'][g] / len(words):.0%}), "
                  f"{m['bucket_pairs'][g]:.1%} of pairs")
        print()

    anchors = Counter(ch for w in words for ch in set(w))
    print("anchors (letter, copies in bag, formable words containing it), fewest first:")
    for ch in sorted(bag, key=lambda c: (anchors[c], c)):
        print(f"  {ch} {bag[ch]:2d} {anchors[ch]:6d}")


if __name__ == "__main__":
    main()
