# complete_yes.py

from workflow import context, fs, log, steps, usage
from workflow.steps import merge as merge_step
from workflow.steps import p2_advance, p2_archive, p2_classify, p2_extract, p2_retrieve


STEPS = [p2_retrieve, p2_extract.YES, p2_classify, p2_extract.NO,
         merge_step, p2_archive, p2_advance]


def help_summary(name):
    return "p2      — complete a yes file manual review"


def _format_help(command, opts, argv):
    return usage.format_help(command, help_summary(command), positional="SLUG")


def show_help(command, opts, argv):
    text = _format_help(command, opts, argv)
    if argv:
        return usage.invalid_argument(argv[0], text)
    print(text, end="")
    return 0


def run(command, opts, argv) -> int:
    if not argv:
        return usage.missing_argument(_format_help(command, opts, argv))

    # The positional is the slug: it names the batch directory, and every
    # artifact inside is found by prefix so nothing takes a name apart.
    slug = argv[0]
    ctx = context.Context(root=opts.dir, phase="p2", force=opts.force, slug=slug)
    fs.raise_if_not_dir(ctx.batch_dir)

    code = steps.run_steps(STEPS, ctx)
    if code == 0:
        log.success(f"Completed {slug}")
    return code
