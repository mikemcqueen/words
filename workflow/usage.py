# usage.py

import argparse
import sys
from pathlib import Path
from workflow import config, log


def make_global_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("-d", "--dir", type=Path, metavar="DIR",
                   help="workflow root directory (default: $WFROOT or cwd)")
    p.add_argument("-f", "--force", action="store_true",
                   help="force overwrite existing files")
    p.add_argument("-h", "--help", action="store_true",
                   help="show this help message")
    return p


def _global_opts_section() -> str:
    text = make_global_parser().format_help()
    _, _, rest = text.partition("\n\n")
    return rest.rstrip()


def format_help(command: str | None = None,
                description: str | None = None,
                local_parser: argparse.ArgumentParser | None = None,
                positional: str | None = None,
                positional_help: tuple[tuple, ...] = ()) -> str:
    prog = f"wf {command}" if command else "wf"
    parents = [make_global_parser()]
    if local_parser is not None:
        parents.append(local_parser)
    p = argparse.ArgumentParser(prog=prog, description=description,
                                parents=parents, add_help=False)
    for positional_spec in positional_help:
        name, help_text, *nargs = positional_spec
        kwargs = {"nargs": nargs[0]} if nargs else {}
        p.add_argument(name.lower(), metavar=name, help=help_text, **kwargs)
    usage = p.format_usage().rstrip()
    if positional is not None and not positional_help:
        usage += f" {positional}"
    _, _, after_usage = p.format_help().partition("\n\n")
    return usage + "\n\n" + after_usage

def invalid_argument(tok: str, details: str | None = None) -> int:
    log.error(f"invalid argument: {tok!r}")
    if details:
        print(details, file=sys.stderr)
    return 2


def missing_argument(details: str | None = None) -> int:
    log.error("missing required argument")
    if details:
        print(details, file=sys.stderr)
    return 2


def default_help_text(summary: str, usage: str | None = None) -> str:
    lines = [summary]
    if usage:
        lines.append(usage)
    return "\n".join(lines)


def default_help(summary: str, argv: list[str], usage: str | None = None) -> int:
    details = default_help_text(summary, usage)
    if argv:
        return invalid_argument(argv[0], details)
    print(details)
    return 0


def _usage_text(command: str, args: config.LayoutArgs) -> str: #parts: list[str], node: dict) -> str:
    usage = f"wf {command}" if command else "wf"
    if args.parts:
        usage = f"{usage} {' '.join(args.parts)}"

    children = args.node.get("parts", {})
    if children:
        usage = f"{usage} "
        options = "|".join(children.keys())
        usage += f"[{options}]" if args.has_content else options

    return f"usage: {usage}"


def _format_targets(node: dict) -> list[str]:
    parts = node.get("parts", {})
    if not parts:
        return []

    width = max(len(name) for name in parts)
    lines = ["Targets:"]
    for name, child in parts.items():
        desc = child.get("description", "")
        lines.append(f"  {name.ljust(width)}  {desc}".rstrip())
    return lines


def _layout_help_text(command: str, args: config.LayoutArgs, summary: str) -> str:
    #parts = args.parts
    node = args.node

    lines: list[str] = []
    if summary:
        lines.append(summary)
    lines.append(_usage_text(command, args)) #parts, node))

    description = node.get("description", "")
    if description:
        lines.extend(["", description])

    targets = _format_targets(node)
    if targets:
        lines.extend(["", *targets])
    lines.extend(["", _global_opts_section()])
    return "\n".join(lines)


def show_layout_help(command: str, args: config.LayoutArgs, summary: str) -> int:
    details = _layout_help_text(command, args, summary)
    if args.has_invalid:
        assert args._invalid
        return invalid_argument(args._invalid, details)
    if args.has_missing:
        return missing_argument(details)
    print(details)
    return 0
