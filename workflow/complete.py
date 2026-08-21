# complete.py
#
# Drain an evaluated batch into the phase's done-set and close it out.
#
# Both phases are the same command: build a Context from the slug, check the
# batch exists, run the recipe. Only the recipe differs, so a phase is a
# STEPS list and nothing else.

from workflow import command, context, fs, log, steps, usage
from workflow.steps import merge as merge_step
from workflow.steps import p1_advance, p1_archive, p1_extract
from workflow.steps import p2_advance, p2_archive, p2_classify, p2_extract, p2_retrieve


class Complete(command.Action):
    def __init__(self, phase: str, step_list: list, summary: str):
        super().__init__(summary=summary, positional="SLUG")
        self.phase = phase
        self.steps = step_list

    def run(self, command, opts, argv) -> int:
        if not argv:
            return usage.missing_argument(self.format_help(command))

        # The positional is the slug: it names the batch directory, and every
        # artifact inside is found by prefix so nothing takes a name apart.
        slug = argv[0]
        ctx = context.Context(root=opts.dir, phase=self.phase,
                              force=opts.force, slug=slug)
        fs.raise_if_not_dir(ctx.batch_dir)

        code = steps.run_steps(self.steps, ctx)
        if code == 0:
            log.success(f"Completed {slug}")
        return code


# extract → merge → archive → advance.
#
# extract produces into the batch directory: retryable, phase-private, nothing
# observable outside the phase. merge folds the batch into p1_done.pairs before
# anything moves. archive renames the inputs into done/{in,out}. advance
# publishes last, because publication is the only effect another `wf`
# invocation can observe.
P1 = Complete("p1",
              [p1_extract, merge_step, p1_archive, p1_advance],
              "p1      — complete a pairs file evaluation")

P2 = Complete("p2",
              [p2_retrieve, p2_extract.YES, p2_classify, p2_extract.NO,
               merge_step, p2_archive, p2_advance],
              "p2      — complete a yes file manual review")
