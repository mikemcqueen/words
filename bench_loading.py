#!/usr/bin/env python3
"""Benchmark file loading vs processing in no-pairs 2-way diff mode."""

import argparse
import json
import time
import sys

from compare import (
    _aligned_chunk_generator,
    _block_generator_from_chunks,
    _parsed_eval_results_chunk_generator,
    eval_results_block_generator,
)
from score import score_eval_results
from diff import ENSEMBLE_RULES_2

FILES = [
    "final/s1/results/an.test4.4_third_p3_juniper.qwen35.jsonl",
    "final/s1/results/an.test4.4_third_p3_mini.qwen35_moe.jsonl",
]

def bench_loading_only(block_size, loader):
    """Time read + batched JSON parse only."""
    total_pairs = 0
    t0 = time.perf_counter()
    for chunk in _aligned_chunk_generator(FILES, chunk_size=block_size, loader=loader):
        chunk_keys = chunk.keys() if isinstance(chunk, dict) else chunk.keys()
        key = next(iter(chunk_keys))
        if isinstance(chunk, dict):
            total_pairs += len(chunk[key])
        else:
            total_pairs += chunk.size
    t1 = time.perf_counter()
    return t1 - t0, total_pairs


def bench_block_build_only(block_size):
    """Time block construction from already-parsed chunks."""
    chunks = list(_parsed_eval_results_chunk_generator(FILES, chunk_size=block_size))
    total_pairs = 0
    t0 = time.perf_counter()
    for block in _block_generator_from_chunks(chunks, block_size=block_size):
        total_pairs += len(next(iter(block.values())))
    t1 = time.perf_counter()
    return t1 - t0, total_pairs


def bench_full_pipeline(loader):
    """Time the complete no-pairs 2-way pipeline (load + score + diff logic)."""
    rules = ENSEMBLE_RULES_2
    t0 = time.perf_counter()

    total_pairs = 0
    yes_counts = {}
    combined_yes = {r: 0 for r in rules}

    for block in eval_results_block_generator(FILES, loader=loader):
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
    parser.add_argument('--loader', choices=['auto', 'python', 'native'], default='auto')
    args = parser.parse_args()

    print(f"Files: {FILES[0]}")
    print(f"       {FILES[1]}")
    print()

    avg_times = {}
    for label, fn, breakdown in [
        ("1. Raw readline (no parse)", bench_raw_readline, "raw"),
        ("2. readline + json.loads (per-line)", bench_json_lines, "json"),
        (f"3. readline + json.loads (batched {args.block_size})", lambda: bench_json_batch(args.block_size), "batch"),
        ("4. Parsed chunk loading (read + batch parse)", lambda: bench_loading_only(args.block_size, args.loader), "parse"),
        ("5. Block build from parsed chunks", lambda: bench_block_build_only(args.block_size), "build"),
        ("6. Full pipeline (load + score + diff)", lambda: bench_full_pipeline(args.loader), "full"),
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
    t_parse = avg_times["parse"]
    t_build = avg_times["build"]
    t_full = avg_times["full"]

    print(f"Raw I/O:            {t_raw:.3f}s ({100*t_raw/t_full:.1f}% of full)")
    print(f"JSON parse (line):  {t_json - t_raw:.3f}s ({100*(t_json-t_raw)/t_full:.1f}% of full)")
    print(f"JSON parse (batch): {t_batch - t_raw:.3f}s ({100*(t_batch-t_raw)/t_full:.1f}% of full)")
    print(f"Parsed chunks:      {t_parse:.3f}s ({100*t_parse/t_full:.1f}% of full)")
    print(f"Block build:        {t_build:.3f}s ({100*t_build/t_full:.1f}% of full)")
    print(f"Processing:         {t_full - t_parse - t_build:.3f}s ({100*(t_full-t_parse-t_build)/t_full:.1f}% of full)")
    print(f"Full pipeline:      {t_full:.3f}s")
