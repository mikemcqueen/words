# usage.py

import sys


def invalid_argument(tok: str, details: str | None = None) -> int:
    print(f"invalid argument: {tok!r}", file=sys.stderr)
    if details:
        print(details, file=sys.stderr)
    return 2


def missing_argument(details: str | None = None) -> int:
    print("missing required argument", file=sys.stderr)
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


def _usage_text(command: str, parts: list[str], node: dict) -> str:
    usage = f"wf {command}" if command else "wf"
    if parts:
        usage = f"{usage} {' '.join(parts)}"

    children = node.get("parts", {})
    if children:
        usage = f"{usage} " + "|".join(children.keys())

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


def _layout_help_text(command: str, args, summary: str) -> str:
    parts = list(args._parts)
    node = args._node

    lines: list[str] = []
    if summary:
        lines.append(summary)
    lines.append(_usage_text(command, parts, node))

    description = node.get("description")
    if description:
        lines.extend(["", description])

    targets = _format_targets(node)
    if targets:
        lines.extend(["", *targets])
    return "\n".join(lines)


def show_layout_help(command: str, args, summary: str) -> int:
    details = _layout_help_text(command, args, summary)
    if args.has_invalid:
        return invalid_argument(args._invalid, details)
    if args.has_missing:
        return missing_argument(details)
    print(details, file=file)
    return 0
