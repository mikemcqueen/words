# eval.py

from workflow import eval_pairs, eval_yes, eval_no, usage, dispatch


_TARGETS = {
    "p1": eval_pairs,
    "p2": eval_yes,
    "p3": eval_no
}


def help_summary(name):
    return "eval     — run evaluators (p1|p2|p3)"


def run(command, opts, argv):
    return dispatch.run(command, _TARGETS, opts, argv)


def show_help(command, opts, argv):
    if not argv:
        print(help_summary(command))
    return dispatch.show_help(command, _TARGETS, opts, argv)
