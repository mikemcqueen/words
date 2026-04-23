# wipe.py

from workflow import wipe_queue, usage, dispatch


SUBCOMMANDS = {
    "queue": wipe_queue
}


def help_summary(name):
    return "wipe    — wipe workflow state (queue)"


def run(command, opts, argv):
    return dispatch.run(command, SUBCOMMANDS, opts, argv)


def show_help(command, opts, argv):
    return dispatch.show_help(command, SUBCOMMANDS, opts, argv)
