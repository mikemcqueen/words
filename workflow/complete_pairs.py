# complete_pairs.py

from workflow import context, fs, log, steps, usage
from workflow.steps import merge as merge_step
from workflow.steps import p1_advance, p1_archive, p1_extract


# extract → merge → archive → advance.
#
# extract produces into the batch directory: retryable, phase-private, nothing
# observable outside the phase. merge folds the batch into p1_done.pairs before
# anything moves. archive renames the inputs into done/{in,out}. advance
# publishes last, because publication is the only effect another `wf`
# invocation can observe.
STEPS = [p1_extract, merge_step, p1_archive, p1_advance]


def help_summary(name):
    return "p1      — complete a pairs file evaluation"


def _format_help(command, opts, argv):
    return usage.format_help(command, help_summary(command), positional="SLUG")


def show_help(command, opts, argv):
    text = _format_help(command, opts, argv)
    if argv:
        return usage.invalid_argument(argv[0], text)
    print(text, end="")
    return 0


def run(command, opts, argv):
    if not argv:
        return usage.missing_argument(_format_help(command, opts, argv))

    slug = argv[0]
    ctx = context.Context(root=opts.dir, phase="p1", force=opts.force, slug=slug)
    fs.raise_if_not_dir(ctx.batch_dir)

    code = steps.run_steps(STEPS, ctx)
    if code == 0:
        log.success(f"Completed {slug}")
    return code
