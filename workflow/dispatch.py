# dispatch.py

from workflow import config, usage


def _help_text(registry):
    lines = ["targets:"]
    lines += [f"  {registry[k].help_summary(k)}" for k in registry]
    return "\n".join(lines)


def registry_key(registry, key: str) -> str | None:
    if key in registry:
        return key
    # TODO: look up partial matches
    return None


def show_help(command, registry, opts, argv):
    if not argv:
        positional = "COMMAND" if command is None else "TARGET"
        print(usage.format_help(command, positional=positional), end="")
        print()
        print(_help_text(registry))
        return 0
    head = argv[0].lower()
    key = registry_key(registry, head)
    if key:
        child = f"{command} {key}" if command else key
        return registry[key].show_help(child, opts, argv[1:])
    return usage.invalid_argument(head, _help_text(registry))


def run(command, registry, opts, argv):
    if not argv:
        if command:
            return usage.missing_argument(_help_text(registry))
        return show_help(command, registry, opts, [])
    head = argv[0].lower()
    if head == "help":
        return show_help(command, registry, opts, argv[1:])
    key = registry_key(registry, head)
    if key:
        child = f"{command} {key}" if command else key
        rest = argv[1:]
        if rest and rest[0].lower() == "help":
            return registry[key].show_help(child, opts, rest[1:])
        return registry[key].run(child, opts, rest)
    return usage.invalid_argument(head, _help_text(registry))
