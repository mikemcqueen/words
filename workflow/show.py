from workflow import show_all, show_queue, config, log
from workflow.dispatch import dispatch_run, dispatch_help


def help_summary(name):
    return "show    — display workflow state (all | queue)"


def run(command, opts, argv):
    parser = config.arg_parser()
    args = parser.parse_args(argv)

    config.validate_parsed_args(command, parser, args)
    log.success(f"valid: show {args.root} {' '.join(args.path)}")


def help(command, opts, argv):
    return dispatch_help(command, SUBCOMMANDS, opts, argv)
