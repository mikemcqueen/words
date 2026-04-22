# show.py

from workflow import show_all, show_queue, config, log, fs
from workflow.dispatch import dispatch_run, dispatch_help


def help_summary(name):
    return "show    — display workflow state"


def _show(opts, args):
    path = config.path(opts.dir, [args.root, *args.path])
    # TODO: Eh..
    for p in sorted(path.iterdir()):
        if p.is_file():
            log.info(p.name)
    return 0


def run(command, opts, argv):
    parser = config.arg_parser()
    args = parser.parse_args(argv)
    config.validate_parsed_args(command, parser, args)
    return _show(opts, args)


def help(command, opts, argv):
    print(help_summary(command))
    return 0
