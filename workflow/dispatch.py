import sys


def _listing(registry):
    return "\n".join(registry[k].help_summary() for k in registry)


def _unknown(command, tok, registry):
    label = f"unknown {command.upper()} sub-command" if command else "unknown command"
    print(f"{label}: {tok}", file=sys.stderr)
    print(_listing(registry), file=sys.stderr)
    return 2


def dispatch_run(command, registry, opts, argv):
    if not argv:
        return dispatch_help(command, registry, opts, [])
    head = argv[0].lower()
    if head == "help":
        return dispatch_help(command, registry, opts, argv[1:])
    if head in registry:
        child = f"{command} {head}" if command else head
        rest = argv[1:]
        if rest and rest[0].lower() == "help":
            return registry[head].help(child, opts, rest[1:])
        return registry[head].run(child, opts, rest)
    return _unknown(command, argv[0], registry)


def dispatch_help(command, registry, opts, argv):
    if not argv:
        print(_listing(registry))
        return 0
    head = argv[0].lower()
    if head in registry:
        child = f"{command} {head}" if command else head
        return registry[head].help(child, opts, argv[1:])
    return _unknown(command, argv[0], registry)
