#!/usr/bin/env python3
"""Compare prompt result files and report statistics.
"""

import argparse
import queue
import signal
import sys
import threading

signal.signal(signal.SIGPIPE, signal.SIG_DFL)
from pathlib import Path

from src.common import (load_expected_pairs, load_eval_results,
                    key_from_path, discover_files_all,
                    resolve_key, print_bad_pairs,
                    add_print_keys_arg, print_displayed_keys)
from src.score import label_eval_results, resolve_all_pair_labels
from src.diff import SORT_DIFF_KEYS
from src.ensemble import SORT_ENSEMBLE_KEYS

from src import diff
from src import ensemble
from src import compare_native


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
    Usage: (likely a bit out of date)

    compare.py DIR                                         — discovery 2-way pairwise diff across all .jsonl files in DIR
    compare.py FILE                                        — discovery 2-way FILE-anchored diff across all .jsonl files in DIR
    compare.py FILE_A FILE_B                               — explicit 2-way diff 
    compare.py FILE_A -k key1                              — explicit 2-way diff; alternate syntax
    compare.py DIR -k key1,key2                            — explicit 2-way diff; alternate syntax
    compare.py FILE_A FILE_B -e RULE                       — explicit 2-way ensemble (OR, AND); -2 implied
    compare.py FILE_A -k key1 -e RULE                      — explicit 2-way ensemble (OR, AND); -2 implied, alternate syntax
    compare.py DIR|FILE -2|-3|-5 [-e RULE]                 — n-way ensemble across all .jsonl files in DIR
    compare.py DIR -k key1,key2[,...] [-2|]-3|-5 [-e RULE] — n-way ensemble across all key combinations; -N >= len(keys); -2 implied if two keys
    TODO:
    compare.py DIR -k key1 -2|-3|-5 shouldn't *key* always be included?
    """

    # Resolve n_way (always 2, 3, or 5; default 2). Track whether the user
    # explicitly chose an n-way mode to distinguish `-2` (ensemble) from the
    # implicit 2-way default (diff).
    nway_explicit = bool(args.two_way or args.three_way or args.five_way)
    if args.five_way:    args.n_way = 5
    elif args.three_way: args.n_way = 3
    else:                args.n_way = 2   # covers -2 and the unset default

    args.keys = [k.strip() for k in args.keys.split(',')] if args.keys else []

    # --keys requires exactly one positional path (the dir or the seed file)
    if args.keys and len(args.files) != 1:
        parser.error('--keys requires exactly one path argument')

    # Determine discovery_dir candidate and seed anchor list
    first = Path(args.files[0])
    if len(args.files) == 1 and first.is_dir():
        discovery_dir = first
        anchor_files: list[Path] = []
    else:
        anchor_files = [Path(f) for f in args.files]
        discovery_dir = anchor_files[0].parent

    # Resolve --keys against discovery_dir
    for key in args.keys:
        anchor_files.append(Path(resolve_key(str(discovery_dir), key, enforce_unique=True)))
    args.keys = []

    # All anchor files must share the same parent directory
    for f in anchor_files:
        if f.parent != discovery_dir:
            parser.error(f'anchor {f} must live in {discovery_dir}')

    # Validate anchor count vs n_way
    if len(anchor_files) > args.n_way:
        parser.error(f'-{args.n_way} accepts at most {args.n_way} anchor files; got {len(anchor_files)}')

    # Finalize args.files and args.discovery_dir
    args.files = [str(f) for f in anchor_files]
    args.discovery_dir = discovery_dir if len(anchor_files) < args.n_way else None

    # Implied ensemble rule. Any explicit -N flag or n_way > 2 implies
    # ensemble='ALL' when -e is absent. Plain `DIR` (no -N) stays as diff.
    if (nway_explicit or args.n_way > 2) and not args.ensemble:
        args.ensemble = 'ALL'

    # no-pairs mode: require exactly 2 files after fixups
    if args.pairs is None:
        if len(args.files) != 2:
            parser.error('without --pairs, exactly 2 files are required (after key resolution)')
        return args, parser

    # validate --ensemble
    if args.ensemble:
        if args.ensemble == 'MAJORITY' and args.n_way % 2 == 0:
            parser.error(f'--ensemble MAJORITY is not valid with -{args.n_way}')

    # validate --bad
    if args.bad:
        is_explicit_2way = (args.discovery_dir is None and args.n_way == 2)
        allowed = (
            args.top == 1
            or (is_explicit_2way and not args.ensemble)
            or (is_explicit_2way and args.ensemble and args.ensemble != 'ALL')
        )
        if not allowed:
            parser.error('--bad requires --top 1, explicit 2-way diff, or explicit 2-way ensemble other than ALL')

    # --print-keys: discovery only
    if args.print_keys and args.discovery_dir is None:
        parser.error('--print-keys requires discovery mode')

    # --sort: diff sort keys only for the 2-way diff path; ensemble keys otherwise
    is_2way_diff = (args.n_way == 2 and args.ensemble is None)
    valid_sort = set(SORT_DIFF_KEYS) if is_2way_diff else set(SORT_ENSEMBLE_KEYS)
    sort_lc = args.sort.lower()
    if sort_lc not in valid_sort:
        choices = ', '.join(sorted(valid_sort))
        parser.error(f'--sort: invalid value {args.sort!r}; choices: {choices}')
    if args.heap_size <= 0:
        parser.error('--heap-size must be > 0')

    # --top vs --heap-size: only the no-anchor 2-way diff path uses the heap
    # TODO: warning message, clamp to heap_size
    if (args.discovery_dir is not None
            and not args.files                # no anchors => all-pairs path
            and args.ensemble is None         # 2-way diff
            and args.top and args.top > args.heap_size):
        parser.error('--top cannot exceed --heap-size for directory 2-way diff discovery')

    return args, parser


def _prefetch(iterable):
    """Wrap an iterable so the next item is produced in a background thread."""
    q = queue.Queue(maxsize=1)
    sentinel = object()

    def producer():
        try:
            for item in iterable:
                q.put(item)
        except Exception as e:
            q.put(e)
        finally:
            q.put(sentinel)

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is sentinel:
            break
        if isinstance(item, Exception):
            raise item
        yield item
    t.join()


def load_result_files(expected, args):
    if args.discovery_dir is not None:
        files = discover_files_all(str(args.discovery_dir))
        for f in args.files:
            k = key_from_path(f)
            if k not in files:
                print(f'Error: anchor key {k!r} (from {f}) not found in '
                      f'discovery dir {args.discovery_dir}', file=sys.stderr)
                sys.exit(1)
    else:
        files = {}
        for f in args.files:
            k = key_from_path(f)
            if k in files:
                print(f'Error: duplicate key {k!r}', file=sys.stderr)
                sys.exit(1)
            files[k] = load_eval_results(f)
    if not files:
        print('No files found.', file=sys.stderr)
        sys.exit(1)
    for results in files.values():
        label_eval_results(results, expected, args.method)
        resolve_all_pair_labels(results)
    return files


# --- top-level runners ---

def run_discovery(files, expected, args):
    p = args.discovery_dir
    print(f'\nDirectory: {p}')
    print(f'Found: {len(files)} file(s)\n')
    anchor_keys = frozenset(key_from_path(f) for f in args.files)
    if args.n_way == 2 and args.ensemble is None:
        if not anchor_keys:
            return diff.print_2way_diff_all_pairs(files, args)
        (anchor,) = anchor_keys
        return diff.print_2way_diff_anchored(anchor, files, args)
    return ensemble.print_discovery_ensemble(args, files, anchor_keys=anchor_keys)


def run_explicit(files, expected, args):
    if args.n_way == 2:
        # TODO: probably not needed
        rule = args.ensemble or 'ALL'
        return diff.print_explicit_2way_diff(files, args, ensemble_rule=rule)
    anchor_keys = frozenset(files.keys())
    return ensemble.print_discovery_ensemble(args, files, anchor_keys=anchor_keys)


def main():
    args, _ = parse_args()

    if args.pairs is None:
        compare_native.require_native()
        block_iter = _prefetch(compare_native.iter_projected_blocks(args.files, chunk_size=1000))
        # TODO: probably not needed
        rule = args.ensemble if args.ensemble else 'ALL'
        diff.run_2way_nopairs_projected(block_iter, args.files, rule, args.method)
        return

    expected = load_expected_pairs(args.pairs)
    files = load_result_files(expected, args)
    assert files is not None

    if len(args.files) < args.n_way:
        rows = run_discovery(files, expected, args)
    else:  
        assert len(args.files) == args.n_way # arg validation guarantees this at time of writing
        rows = run_explicit(files, expected, args)

    if args.bad and rows:
        assert len(rows) == 1
        top_label, row_data = rows[0]
        parts = top_label.rsplit(' ', 1)
        if len(parts) == 2:
            keys, rule = parts[0].split(','), parts[1]
            combined = row_data.get('ens_results')
            if combined is None:
                combined = ensemble.apply_ensemble_labeled([files[key] for key in keys], rule)
            print_bad_pairs(combined, sources=[(key, files[key]) for key in keys])

if __name__ == '__main__':
    main()
