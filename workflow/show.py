# show.py

from workflow import show_all, show_queue, config, dispatch, log, fs, usage


def help_summary(name):
    return "show    — display workflow state"


def _show(parts: list[str], opts) -> int:
    path = config.path(opts.dir, parts)
    any_files = False
    for p in sorted(path.iterdir()):
        if p.is_file():
            print(p.name)
            any_files = True
    if not any_files:
        log.info("Directory is empty.")
    return 0


def run(command, opts, argv) -> int:
    args = config.layout_args(argv)
    if not args.ok:
        return usage.show_layout_help(command, args, help_summary(command))
    return _show(list(args.parts), opts)


def show_help(command, opts, argv) -> int:
    args = config.layout_args(argv)
    return usage.show_layout_help(command, args, help_summary(command))
