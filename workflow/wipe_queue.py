import argparse
import os
import shutil
import sys

from pathlib import Path
from workflow import log

#from workflow import queue

DESC = "wipe queue"

def _parser(prog):
    p = argparse.ArgumentParser(prog=prog, description=DESC, add_help=False)
    p.add_argument("num", type=int, metavar="NUM", help="queue number to wipe")
    return p


def help_summary(name: str):
    usage = _parser(name).format_usage().strip().removeprefix("usage: ")
    return f"{usage:<24} — {DESC}"


def elevate(command: str):
    if os.geteuid() != 0:
        log.info(f"{command}: sudo/root required")
        return False
    return True


def validate_queue_root(d: Path, opts) -> Path|None:
    q_root = d / 'q'
    if not q_root.exists():
        log.info(f"No queue found in {d}")
        return None
    if not q_root.is_dir():
        log.info(f"{q_root} exists and is not a directory")
        return None

    return q_root


def wipe_one(q_root: Path, n: int, opts) -> bool:
    q_dir = q_root / str(n)
    if q_dir.exists():
        if not q_dir.is_dir():
            log.warn(f"{q_dir} is not a directory - no action taken for queue {n}")
            return False
        shutil.rmtree(q_dir)

    return True

"""
def wipe_many(args, opts) -> bool:
    # validate queue directory first
    q_root = validate_queue_root(opts.dir, opts)
    if not q_root:
        return False

    success = True
    for n in args.nums:
        success &= wipe_one(q_root, n, opts)

    return success
"""

def run(command: str, opts, argv: list[str]) -> int:
    print(f"not implemented")
    return 1

    if not elevate(command):
        return 1

    q_root = validate_queue_root(opts.dir, opts)
    if not q_root:
        return False

    args = _parser(f"wf {command}").parse_args(argv)

    if not wipe_one(q_root, args.num, opts):
        log.error(f"Failed to wipe queue {args.num}")
        return 1

    log.success(f"Successfully wiped queue {args.num}")
    return 0


def help(command: str, opts, argv: list[str]) -> int:
    _parser(f"wf {command}").print_help()
    return 0
