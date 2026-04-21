import sys


def _listing(registry):
    return "\n".join(registry[k].help_summary(k) for k in registry)


def _unknown(command, tok, registry):
    label = f"unknown {command.upper()} sub-command" if command else "unknown command"
    print(f"{label}: {tok}", file=sys.stderr)
    print(_listing(registry), file=sys.stderr)
    return 2


def registry_key(registry, key: str) -> str:
    if key in registry:
        return key
    # TODO: look up partial matches
    return None

def dispatch_run(command, registry, opts, argv):
    if not argv:
        return dispatch_help(command, registry, opts, [])
    head = argv[0].lower()
    if head == "help":
        return dispatch_help(command, registry, opts, argv[1:])
    key = registry_key(registry, head)
    if key:
        child = f"{command} {key}" if command else key
        rest = argv[1:]
        if rest and rest[0].lower() == "help":
            return registry[key].help(child, opts, rest[1:])
        return registry[key].run(child, opts, rest)
    return _unknown(command, argv[0], registry)


def dispatch_help(command, registry, opts, argv):
    if not argv:
        print(_listing(registry))
        return 0
    head = argv[0].lower()
    key = registry_key(command, registry, head)
    if key:
        child = f"{command} {key}" if command else key
        return registry[key].help(child, opts, argv[1:])
    return _unknown(command, argv[0], registry)
