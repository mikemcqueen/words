import sys


def _listing(registry):
    return "\n".join(registry[k].help_summary() for k in registry)


def dispatch_run(registry, argv):
    if not argv:
        return dispatch_help(registry, [])
    head = argv[0].lower()
    if head == "help":
        return dispatch_help(registry, argv[1:])
    if head in registry:
        rest = argv[1:]
        if rest and rest[0].lower() == "help":
            return registry[head].help(rest[1:])
        return registry[head].run(rest)
    print(f"unknown command: {argv[0]}", file=sys.stderr)
    print(_listing(registry), file=sys.stderr)
    return 2


def dispatch_help(registry, argv):
    if not argv:
        print(_listing(registry))
        return 0
    head = argv[0].lower()
    if head in registry:
        return registry[head].help(argv[1:])
    print(f"unknown command: {argv[0]}", file=sys.stderr)
    print(_listing(registry), file=sys.stderr)
    return 2
