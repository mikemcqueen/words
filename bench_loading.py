#!/usr/bin/env python3
"""Benchmark file loading vs processing in no-pairs 2-way diff mode."""

import argparse
import json
import time
import sys

from compare import eval_results_block_generator, _key_from_path
from score import score_eval_results
from diff import ENSEMBLE_RULES_2

FILES = [
    "final/s1/results/an.test4.4_third_p3_juniper.qwen35.jsonl",
    "final/s1/results/an.test4.4_third_p3_mini.qwen35_moe.jsonl",
]

def bench_loading_only(block_size):
    """Time just the JSON parsing / file reading (what eval_results_block_generator does)."""
    keys = [_key_from_path(f) for f in FILES]
    handles = [open(f) for f in FILES]
    total_pairs = 0
    t0 = time.perf_counter()
    while True:
        primary = {}
        for _ in range(block_size):
            line = handles[0].readline()
            if not line:
                break
            line = line.strip()
            if line:
                r = json.loads(line)
                primary[r["pair"]] = {"logprobs": r["logprobs"]}
        if not primary:
            break

        secondary = {}
        for _ in range(len(primary)):
            line = handles[1].readline()
            if not line:
                break
            line = line.strip()
            if line:
                r = json.loads(line)
                secondary[r["pair"]] = {"logprobs": r["logprobs"]}

        total_pairs += len(primary)

    t1 = time.perf_counter()
    for h in handles:
        h.close()
    return t1 - t0, total_pairs


def bench_full_pipeline():
    """Time the complete no-pairs 2-way pipeline (load + score + diff logic)."""
    rules = ENSEMBLE_RULES_2
    t0 = time.perf_counter()

    total_pairs = 0
    yes_counts = {}
    combined_yes = {r: 0 for r in rules}

    for block in eval_results_block_generator(FILES):
        file_keys = list(block.keys())
        for fk in file_keys:
            if fk not in yes_counts:
                yes_counts[fk] = 0

        block_size = len(next(iter(block.values())))
        total_pairs += block_size

        for fk in file_keys:
            score_eval_results(block[fk], 'top-token')

        pairs = next(iter(block.values())).keys()
        for pair in pairs:
            per_file_yes = {fk: block[fk][pair]["yes"] for fk in file_keys}
            for fk, is_yes in per_file_yes.items():
                if is_yes:
                    yes_counts[fk] += 1
            for r in rules:
                if r == 'OR':
                    pair_yes = any(per_file_yes.values())
                elif r == 'AND':
                    pair_yes = all(per_file_yes.values())
                else:
                    pair_yes = any(per_file_yes.values())
                if pair_yes:
                    combined_yes[r] += 1

    t1 = time.perf_counter()
    return t1 - t0, total_pairs


def bench_raw_readline():
    """Time just reading lines without JSON parsing."""
    t0 = time.perf_counter()
    total_lines = 0
    for f in FILES:
        with open(f) as h:
            for line in h:
                total_lines += 1
    t1 = time.perf_counter()
    return t1 - t0, total_lines


def bench_json_lines():
    """Time readline + JSON parse, no dict building."""
    t0 = time.perf_counter()
    total_lines = 0
    for f in FILES:
        with open(f) as h:
            for line in h:
                json.loads(line)
                total_lines += 1
    t1 = time.perf_counter()
    return t1 - t0, total_lines


def bench_json_batch(batch_size):
    """Time batched JSON parse: join lines into a JSON array, parse once."""
    t0 = time.perf_counter()
    total_lines = 0
    for f in FILES:
        with open(f) as h:
            while True:
                first = h.readline()
                if not first:
                    break
                first = first.strip()
                if not first:
                    continue
                batch = "[" + first
                count = 1
                for _ in range(batch_size - 1):
                    line = h.readline()
                    if not line:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    batch += "," + line
                    count += 1
                batch += "]"
                json.loads(batch)
                total_lines += count
    t1 = time.perf_counter()
    return t1 - t0, total_lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-r', '--runs', type=int, default=1)
    parser.add_argument('-b', '--block-size', type=int, default=100)
    args = parser.parse_args()

    print(f"Files: {FILES[0]}")
    print(f"       {FILES[1]}")
    print()

    avg_times = {}
    for label, fn, breakdown in [
        ("1. Raw readline (no parse)", bench_raw_readline, "raw"),
        ("2. readline + json.loads (per-line)", bench_json_lines, "json"),
        (f"3. readline + json.loads (batched {args.block_size})", lambda: bench_json_batch(args.block_size), "batch"),
        ("4. Block loading (json + dict build)", lambda: bench_loading_only(args.block_size), "load"),
        ("5. Full pipeline (load + score + diff)", bench_full_pipeline, "full"),
    ]:
        times = []
        count = 0
        for i in range(args.runs):
            t, c = fn()
            times.append(t)
            count = c
        best = min(times)
        avg = sum(times) / len(times)
        avg_times[breakdown] = avg
        print(f"{label}")
        print(f"   best: {best:.3f}s  avg: {avg:.3f}s  ({count:,} items, {args.runs} runs)")

    print()
    print("--- Breakdown ---")
    t_raw = avg_times["raw"]
    t_json = avg_times["json"]
    t_batch = avg_times["batch"]
    t_load = avg_times["load"]
    t_full = avg_times["full"]

    print(f"Raw I/O:            {t_raw:.3f}s ({100*t_raw/t_full:.1f}% of full)")
    print(f"JSON parse (line):  {t_json - t_raw:.3f}s ({100*(t_json-t_raw)/t_full:.1f}% of full)")
    print(f"JSON parse (batch): {t_batch - t_raw:.3f}s ({100*(t_batch-t_raw)/t_full:.1f}% of full)")
    print(f"Dict build:         {t_load - t_json:.3f}s ({100*(t_load-t_json)/t_full:.1f}% of full)")
    print(f"Processing:         {t_full - t_load:.3f}s ({100*(t_full-t_load)/t_full:.1f}% of full)")
    print(f"Full pipeline:      {t_full:.3f}s")
