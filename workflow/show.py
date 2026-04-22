# show.py

from workflow import show_all, show_queue, config, log, fs
from workflow.dispatch import dispatch_run, dispatch_help


def help_summary(name):
    return "show    — display workflow state (all | queue)"


def _show(opts, args):
    path = opts.dir / config.ROOT / args.root
    for part in args.path:
        path = path / part
    fs.raise_if_not_dir(path)
    for entry in sorted(path.iterdir()):
        if entry.is_file():
            log.info(entry.name)
    return 0


def run(command, opts, argv):
    parser = config.arg_parser()
    args = parser.parse_args(argv)

    config.validate_parsed_args(command, parser, args)
    return _show(opts, args)


def help(command, opts, argv):
    return dispatch_help(command, SUBCOMMANDS, opts, argv)
