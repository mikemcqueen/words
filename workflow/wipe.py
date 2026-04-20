from workflow import wipe_queue
from workflow.dispatch import dispatch_run, dispatch_help

SUBCOMMANDS = {
    "queue": wipe_queue
}


def help_summary():
    return "wipe    — wipe workflow state (queue)"


def run(command, opts, argv):
    return dispatch_run(command, SUBCOMMANDS, opts, argv)


def help(command, opts, argv):
    return dispatch_help(command, SUBCOMMANDS, opts, argv)
