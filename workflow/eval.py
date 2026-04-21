from workflow import eval_pairs
from workflow.dispatch import dispatch_run, dispatch_help

SUBCOMMANDS = {"pairs": eval_pairs}


def help_summary(name):
    return "eval    — run evaluators (pairs)"


def run(command, opts, argv):
    return dispatch_run(command, SUBCOMMANDS, opts, argv)


def help(command, opts, argv):
    return dispatch_help(command, SUBCOMMANDS, opts, argv)
