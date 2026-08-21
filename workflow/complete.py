# complete.py

from workflow import complete_pairs, complete_yes, usage, dispatch


_TARGETS = {
    "p1": complete_pairs,
    "p2": complete_yes
}


def help_summary(name):
    return "complete — complete evaluation (p1|p2)"


def run(command, opts, argv):
    return dispatch.run(command, _TARGETS, opts, argv)


def show_help(command, opts, argv):
    if not argv:
        print(help_summary(command))
    return dispatch.show_help(command, _TARGETS, opts, argv)
