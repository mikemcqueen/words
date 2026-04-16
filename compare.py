#!/usr/bin/env python3
"""Compare prompt result files and report statistics.
"""

import argparse
import json
import signal
import sys

signal.signal(signal.SIGPIPE, signal.SIG_DFL)
from pathlib import Path

from common import (load_expected_pairs, load_eval_results,
                    parse_result_filename, discover_files_all,
                    resolve_key, print_bad_pairs,
                    add_print_keys_arg, print_displayed_keys)
from score import label_eval_results, resolve_all_pair_labels

import diff
import ensemble


def parse_args(argv=None):
    """Parse and validate command-line arguments. Returns (args, parser) or exits on error."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='+', metavar='path')
    parser.add_argument("-p", '--pairs', required=False, default=None, metavar='PAIRS',
                        help='pairs JSON file with expected values')
    parser.add_argument('--method', default='top-token', metavar='METHOD',
                        help='scoring method (default: top-token)')
    parser.add_argument('-e', '--ensemble', type=str.upper,
                        choices=['ALL', 'AND', 'OR', 'MAJORITY'],
                        default=None, metavar='RULE',
                        help='ensemble rule: ALL (default), AND, OR, MAJORITY (case-insensitive)')

    nway_group = parser.add_mutually_exclusive_group()
    nway_group.add_argument('-2', '--two-way', action='store_true',
                        help='2-way ensemble across all pairwise key/file combinations')
    nway_group.add_argument('-3', '--three-way', action='store_true',
                        help='3-way ensemble across all triple key/file combinations')
    nway_group.add_argument('-5', '--five-way', action='store_true',
                        help='5-way ensemble across all quintuple key/file combinations')

    parser.add_argument('-k', '--keys', metavar='KEYS',
                        help='comma-separated discovery keys for explicit ensemble; '
                             'positional arg must be a directory')
    parser.add_argument('-s', '--sort', default='score', metavar='FIELD',
                        help='sort field: diff=score/fixfp/fixfn/newfp/newfn/fp/fn; ensemble=score/FP/FN')
    parser.add_argument('--top', type=int, default=50, metavar='K',
                        help='max rows to display in output tables (default: 50)')
    parser.add_argument('--heap-size', type=int, default=100, metavar='N',
                        help='max discovery results to retain before final sort (default: 100)')
    parser.add_argument('--bad', action='store_true',
                        help='show FP and FN pairs for the single table entry; requires --top 1')
    add_print_keys_arg(parser, help_text='print displayed row labels as a comma-separated list (discovery mode only)')
    args = parser.parse_args(argv)


    """
    Usage:

    compare.py <dir>                                         — discovery 2-way diff pairwise across all .jsonl files
    compare.py <file>                                        — discovery 2-way diff anchor-based across all .jsonl files in same direcotry
    compare.py <file_a> <file_b>                             — explicit 2-way diff 
    compare.py <file_a> -k key1                              — explicit 2-way diff; alternate syntax
    compare.py <dir> -k key1,key2                            — explicit 2-way diff; alternate syntax
    compare.py <file_a> <file_b> -e RULE                     — explicit 2-way ensemble (OR, AND); -2 implied
    compare.py <file_a> -k key1 -e RULE                      — explicit 2-way ensemble (OR, AND); -2 implied, alternate syntax
    compare.py <dir|file> -2|-3|-5 [-e RULE]                 — n-way ensemble across all .jsonl files in same directory
    compare.py <dir> -k key1,key2[,...] [-2|]-3|-5 [-e RULE] — n-way ensemble across all key combinations; -N >= len(keys); -2 implied if two keys
    """

    # Resolve n_way from the mutually exclusive -2/-3/-5 flags.
    if args.five_way:
        args.n_way = 5
    elif args.three_way:
        args.n_way = 3
    elif args.two_way:
        args.n_way = 2
    else:
        args.n_way = None

    args.keys = [k.strip() for k in args.keys.split(',')] if args.keys else []

    # validate --keys (early: depends on original keys/files before fixups)
    if args.keys:
        if len(args.files) != 1:
            parser.error('--keys requires exactly one path argument')
        if len(args.keys) > 1 and not Path(args.files[0]).is_dir():
            parser.error('--keys with multiple keys requires the path to be a directory')

    # fixup args.files: 1-key alternate syntax
    if len(args.keys) == 1:
        file2 = resolve_key(args.files[0], args.keys[0])
        args.files.append(str(file2))

    # fixup args.files: 2-key alternate syntax (explicit 2-way diff)
    if len(args.keys) == 2 and not args.n_way:
        directory = args.files[0]
        args.files = [str(resolve_key(directory, k, enforce_unique=True)) for k in args.keys]
        args.keys = []

    # fixup args.n_way for implied -2 cases
    if not args.n_way:
        if len(args.files) == 2 and args.ensemble:
            # compare.py <file_a> <file_b> -e RULE                        - 2-way ensemble (OR, AND)
            args.n_way = 2
        elif len(args.files) == 1 and len(args.keys) == 2:
            # compare.py <dir> -k key1,key2[,...] [-2|]-3|-5 [-e RULE] — n-way ensemble across all keys
            args.n_way = 2

    # fixup args.ensemble for implied ALL cases
    if args.n_way and not args.ensemble:
        args.ensemble = "ALL"
            
    # no-pairs mode: require exactly 2 files after fixups
    if args.pairs is None:
        if len(args.files) != 2:
            parser.error('without --pairs, exactly 2 files are required (after key/fixup resolution)')
        return args, parser

    # validate files
    if len(args.files) == 2:
        if Path(args.files[0]).is_dir() or Path(args.files[1]).is_dir():
            parser.error('both paths must be files')
        if args.n_way and args.n_way != 2:
            parser.error('only -2 (2-way) is allowed when supplying two files or one file and one key')
    elif len(args.files) > 2:
        parser.error('supply 1 or 2 paths')

    # validate --keys (late: depends on n_way fixup)
    if len(args.keys) > 1:
        if not args.n_way:
            parser.error('supplying two or more --keys requires -2, -3, or -5')
        if args.n_way > len(args.keys):
            parser.error(f'-{args.n_way} requires at least {args.n_way} keys, got {len(args.keys)}')

    # validate --ensemble
    if args.ensemble:
        # FIRST: ensure args.n_way
        if not args.n_way:
            parser.error('--ensemble requires -2, -3, or -5')
        if args.ensemble == 'MAJORITY' and args.n_way % 2 == 0:
            parser.error(f'--ensemble MAJORITY is not valid with -{args.n_way}')

    """
    if args.keys:
        if len(args.files) != 1:
            parser.error('--keys requires exactly one positional argument')
        if not Path(args.files[0]).is_dir():
            parser.error(f'--keys: {args.files[0]!r} is not a directory')
        if n_ways == 0:
            parser.error('-k requires -2, -3, or -5')
        args.ensemble = args.ensemble or 'ALL'
        return args, parser

    if (args.two_way or args.three_way or args.five_way) and not args.ensemble:
        parser.error('-2/-3/-5 requires --ensemble')

    if args.ensemble == 'MAJORITY' and args.n_way % 2 == 0:
        parser.error('--ensemble MAJORITY requires -3 or -5')
    """

    # validate --bad
    # TODO: allow <dir> -k key1 possibly?
    if args.bad:
        allowed = (
            args.top == 1                                                          # --top 1
            or (len(args.files) == 2 and not args.ensemble)                        # explicit 2-way diff
            or (len(args.files) == 2 and args.ensemble and args.ensemble != "ALL") # explicit 2-way ensemble != ALL
        )
        if not allowed:
            parser.error('--bad requires --top 1, explicit 2-way diff, or explicit 2-way ensemble other than ALL')

    # discover = len(args.files) == 1
    # diff = !args.ensemble

    # validate --print-keys
    if args.print_keys and len(args.files) > 1:
        parser.error('--print-keys requires discovery mode')

    # validate --sort
    sort_lc = args.sort.lower()
    explicit_2way = len(args.files) == 2
    valid_sort = {'score', 'fp', 'fn'} if (args.ensemble and not explicit_2way) else {
        'score', 'fixfp', 'fixfn', 'newfp', 'newfn', 'fp', 'fn'
    }
    if sort_lc not in valid_sort:
        choices = ', '.join(sorted(valid_sort))
        parser.error(f'--sort: invalid value {args.sort!r}; choices: {choices}')
    if args.heap_size <= 0:
        parser.error('--heap-size must be > 0')
    if (len(args.files) == 1
            and Path(args.files[0]).is_dir()
            and not args.ensemble
            and args.top
            and args.top > args.heap_size):
        parser.error('--top cannot exceed --heap-size for directory 2-way diff discovery')

    return args, parser

def _key_from_path(path):
    """Parse a discovery key from a result filename. Exits on failure."""
    parsed = parse_result_filename(Path(path).name)
    if parsed is None:
        print(f"Error: could not parse key from filename: {Path(path).name}", file=sys.stderr)
        sys.exit(1)
    _, prompt_file, prompt_id, _, tag = parsed
    return f"{prompt_file}.{prompt_id}.{tag}" if tag else f"{prompt_file}.{prompt_id}"


def eval_results_block_generator(files_list):
    """Yield aligned blocks of eval results from multiple JSONL files.

    Each yielded value is a dict: {file_key: {pair: {"logprobs": ...}, ...}}
    with up to BLOCK_SIZE pairs, identical keys across all files.
    """
    BLOCK_SIZE = 1000
    keys = [_key_from_path(f) for f in files_list]
    handles = [open(f) for f in files_list]
    try:
        while True:
            block = {}
            # Load a block from file[0]
            primary = {}
            for _ in range(BLOCK_SIZE):
                line = handles[0].readline()
                if not line:
                    break
                line = line.strip()
                if line:
                    r = json.loads(line)
                    primary[r["pair"]] = {"logprobs": r["logprobs"]}
            if not primary:
                break
            block[keys[0]] = primary

            # Load matching block from each additional file
            for i in range(1, len(files_list)):
                secondary = {}
                for _ in range(len(primary)):
                    line = handles[i].readline()
                    if not line:
                        break
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        secondary[r["pair"]] = {"logprobs": r["logprobs"]}
                sym_diff = primary.keys() ^ secondary.keys()
                if sym_diff:
                    raise RuntimeError(
                        f"pair mismatch between {files_list[0]} and {files_list[i]}: "
                        f"{len(sym_diff)} differing pairs (e.g. {next(iter(sym_diff))})")
                block[keys[i]] = secondary

            yield block
    finally:
        for h in handles:
            h.close()


def load_files_from_keys(args):
    """Resolve keys to files, and load results. Returns {key: results}."""
    directory = args.files[0]
    files = {}
    for key in args.keys:
        path = resolve_key(directory, key)
        assert key not in files, f"duplicate key: {key}"
        files[key] = load_eval_results(path)
    return files


def load_files_explicit(filenames: [str]):
    files = {}
    for i in range(2):
        #parsed = parse_result_filename(filenames[i])
        #key = parsed[2] if parsed else Path(args.files[i]).stem
        key = _key_from_path(filenames[i])
        assert key not in files, f"duplicate key: {key}"
        files[key] = load_eval_results(filenames[i])
    return files


def load_result_files(expected, args):
    files = None
    if args.keys and len(args.keys) > 1:
        files = load_files_from_keys(args)
    elif len(args.files) == 2:
        files = load_files_explicit(args.files)
    elif len(args.files) == 1:
        files = discover_files_all(args.files[0])

    if files:
        if len(files) == 0:
            print("No files found.", file=sys.stderr)
            sys.exit(1)
        for results in files.values():
            label_eval_results(results, expected, args.method)
            resolve_all_pair_labels(results)
    return files


# --- top-level runners ---

def run_explicit_2way(files, expected, args):
    rule = args.ensemble if args.ensemble else 'OR'
    return diff.print_explicit_2way_diff(files, args, ensemble_rule=rule)


def run_explicit_nway(files, expected, args):
    return ensemble.print_discovery_ensemble(args, files)


def run_discovery(files, expected, args):
    p = Path(args.files[0])
    print(f"\nDirectory: {p if p.is_dir() else p.parent}")
    print(f"Found: {len(files)} file(s)\n")

    if args.ensemble:
        return ensemble.print_discovery_ensemble(args, files)
    elif Path(args.files[0]).is_dir():
        return diff.print_2way_diff_all_pairs(files, args)
    else:
        anchor_key = _key_from_path(args.files[0])
        return diff.print_2way_diff_anchored(anchor_key, files, args)


def main():
    args, _ = parse_args()

    if args.pairs is None:
        block_iter = eval_results_block_generator(args.files)
        rule = args.ensemble if args.ensemble else 'OR'
        diff.run_nopairs_2way(block_iter, rule, args.method)
        return

    expected = load_expected_pairs(args.pairs)
    files = load_result_files(expected, args)

    if args.keys and len(args.keys) > 1:
        rows = run_explicit_nway(files, expected, args)
    elif len(args.files) == 1:
        rows = run_discovery(files, expected, args)
    elif len(args.files) == 2:
        rows = run_explicit_2way(files, expected, args)

    if args.bad and rows:
        assert len(rows) == 1
        top_label, row_data = rows[0]
        parts = top_label.rsplit(' ', 1)
        if len(parts) == 2:
            keys, rule = parts[0].split(','), parts[1]
            if rule == 'DIFF':
                combined = row_data.get('or_results')
                if combined is None:
                    combined = ensemble.apply_ensemble_labeled([files[key] for key in keys], 'OR')
            else:
                combined = ensemble.apply_ensemble_labeled([files[key] for key in keys], rule)
            print_bad_pairs(combined, sources=[(key, files[key]) for key in keys])

if __name__ == '__main__':
    main()
