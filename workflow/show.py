# show.py

import argparse

from workflow import config, log, usage

COLLECT = ("queued", "eval")


def help_summary(name):
    return "show    — display workflow state"


def _make_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("-a", "--all", action="store_true")
    return p


def _parse_args(argv, opts):
    local_opts, rest = _make_parser().parse_known_args(argv)
    vars(opts).update(vars(local_opts))
    return rest, opts


def _build_aggregates(collect=COLLECT):
    result = {}

    def walk(node, path):
        for name, child in node.get("parts", {}).items():
            child_path = path + [name]
            if name in collect:
                result.setdefault(name, []).append(child_path)
            walk(child, child_path)

    walk(config.CONFIG_LAYOUT, [])
    return result


def show_target(parts: list[str], opts) -> bool:
    # hack coz i don't want to tear down .pairs-specific code yet
    opts.all = True

    path = config.path(opts.dir, parts)
    found = False
    for p in path.iterdir():
        if opts.all or (p.suffix == ".pairs" and p.is_file()):
            if not found:
                print(f"{'/'.join(parts)}:")
                found = True
            print(f"  {p.name}")
    return found


def _show_all(opts, subtarget) -> int:
    aggregates = _build_aggregates()
    if subtarget not in aggregates:
        return usage.invalid_argument(subtarget)
    found = False
    for parts in aggregates[subtarget]:
        found |= show_target(parts, opts)
    if not found:
        log.info("directory is empty")
    return 0


def _all_help_summary():
    return "all     — aggregate view across phases"


def _show_all_help(command) -> int:
    aggregates = _build_aggregates()
    subtargets = "|".join(aggregates)
    print(_all_help_summary())
    print(f"usage: wf {command} all {subtargets}")
    width = max(len(k) for k in aggregates)
    print("\nTargets:")
    for name, sources in aggregates.items():
        print(f"  {name.ljust(width)}  {', '.join('/'.join(s) for s in sources)}")
    return 0


def _show_help_with_all(command, args, summary) -> int:
    if not args.parts and not args.has_invalid:
        node = dict(args.node, parts=dict(args.node["parts"],
            all={"description": _all_help_summary()}))
        args = config.LayoutArgs(parts=args.parts, node=node, _invalid=args._invalid)
    return usage.show_layout_help(command, args, summary)


def show_help(command, opts, argv) -> int:
    local_opts, rest = _make_parser().parse_known_args(argv)
    if rest and rest[0] == "all":
        return _show_all_help(command)
    args = config.layout_args(rest)
    return _show_help_with_all(command, args, help_summary(command))


def run(command, opts, argv) -> int:
    rest, opts = _parse_args(argv, opts)
    if rest and rest[0] == "all":
        subtarget = rest[1] if len(rest) > 1 else None
        if not subtarget or subtarget in ("help", "-h", "--help"):
            return _show_all_help(command)
        return _show_all(opts, subtarget)
    args = config.layout_args(rest)
    if not args.ok:
        return _show_help_with_all(command, args, help_summary(command))
    found = show_target(list(args.parts), opts)
    if not found:
        log.info("directory is empty")
    return 0
