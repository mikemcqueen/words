# show.py

import argparse

from workflow import show_all, show_queue, config, dispatch, log, fs, usage


def help_summary(name):
    return "show    — display workflow state"


def _make_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("-a", "--all", action="store_true")
    return p


def _parse_args(argv, opts):
    local_opts, rest = _make_parser().parse_known_args(argv)
    vars(opts).update(vars(local_opts))
    args = config.layout_args(rest)
    return args, opts


def show_help(command, opts, argv) -> int:
    show_opts, rest = _make_parser().parse_known_args(argv)
    args = config.layout_args(rest)
    return usage.show_layout_help(command, args, help_summary(command))


def _show(parts: list[str], opts) -> int:
    # hack coz i don't want to tear down .pairs-specific code yet
    opts.all = True

    path = config.path(opts.dir, parts)
    empty = True
    for p in path.iterdir():
        is_pairs = p.suffix == ".pairs" and p.is_file()
        if opts.all or is_pairs:
            empty = False
            print(p.name)
    if empty:
        log.info("directory is empty")
    return 0


def run(command, opts, argv) -> int:
    args, opts = _parse_args(argv, opts)
    if not args.ok:
        return usage.show_layout_help(command, args, help_summary(command))
    return _show(list(args.parts), opts)
