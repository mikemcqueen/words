# complete.py

from workflow import complete_pairs, complete_yes, usage, dispatch


_TARGETS = {
    "pairs": complete_pairs,
    "yes": complete_yes
}


def help_summary(name):
    return "complete — complete evaluation (pairs|yes)"


def run(command, opts, argv):
    return dispatch.run(command, _TARGETS, opts, argv)


def show_help(command, opts, argv):
    if not argv:
        print(help_summary(command))
    return dispatch.show_help(command, _TARGETS, opts, argv)
