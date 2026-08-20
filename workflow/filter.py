# filter.py

from workflow import filter_pairs, dispatch


_TARGETS = {
    "pairs": filter_pairs,
}


def help_summary(name):
    return "filter  — filter a p1 results file (pairs)"


def run(command, opts, argv):
    return dispatch.run(command, _TARGETS, opts, argv)


def show_help(command, opts, argv):
    if not argv:
        print(help_summary(command))
    return dispatch.show_help(command, _TARGETS, opts, argv)
