# complete.py
#
# Drain an evaluated bundle into the phase's done-set and close it out.
#
# Both phases are the same command: build a Context from the bundle name, check
# the bundle exists, run the recipe. Only the recipe differs, so a phase is a
# STEPS list and nothing else.

import shlex

from workflow import bundle, command, context, fs, log, names, steps, usage
from workflow.steps import merge as merge_step
from workflow.steps import p1_advance, p1_archive, p1_extract
from workflow.steps import p2_archive, p2_classify, p2_close, p2_extract
from workflow.steps import p2_retrieve
from workflow.steps import words_close, words_extract, words_publish


def _cleanup_warning(ctx) -> None:
    path = str(ctx.bundle_dir)
    log.warn(f"Publication completed, but cleanup failed for {path}. Do not "
             f"rerun completion. Remove the disposable bundle manually:\n"
             f"rm -rf -- {shlex.quote(path)}")


class Complete(command.Action):
    def __init__(self, phase: str, step_list: list, summary: str):
        super().__init__(summary=summary, positional="BUNDLE-NAME")
        self.phase = phase
        self.steps = step_list
        self.archive = next(step for step in step_list
                            if step.NAME == "archive")

    def run(self, command, opts, argv) -> int:
        if not argv:
            return usage.missing_argument(self.format_help(command))
        if len(argv) > 1:
            return usage.invalid_argument(argv[1], self.format_help(command))

        # The positional names the bundle directory, and every
        # artifact inside is found by prefix so nothing takes a name apart.
        # Only the name is wanted: the next line requires the bundle to be
        # open, so a source resolved in some other slot answers nothing here.
        bundle_name, _ = bundle.resolve_source(opts.dir, self.phase, argv[0])
        ctx = context.Context(root=opts.dir, phase=self.phase,
                              force=opts.force, bundle_name=bundle_name)
        fs.raise_if_not_dir(ctx.bundle_dir)

        # Archive is the first irreversible part of completion. Discover every
        # collision before extraction or folding changes anything, so -f is a
        # decision for this invocation rather than a recovery from half an
        # archive.
        if not ctx.force and not steps.is_done(self.archive, ctx):
            fs.raise_if_any_exist(self.archive.outputs(ctx))

        if self.phase == "p2":
            archive_index = self.steps.index(self.archive)
            code = steps.run_steps(self.steps[:archive_index + 1], ctx)
            try:
                steps.run_steps(self.steps[archive_index + 1:], ctx)
            except (OSError, ValueError):
                _cleanup_warning(ctx)
                code = 0
        else:
            code = steps.run_steps(self.steps, ctx)
        if code == 0:
            log.success(f"Completed {bundle_name}")
        return code


# extract → merge → archive → advance.
#
# extract produces into the bundle directory: retryable, phase-private, nothing
# observable outside the phase. merge folds the bundle into p1_done.pairs before
# anything moves. archive renames the inputs into done/{in,out}. advance
# publishes last, because publication is the only effect another `wf`
# invocation can observe.
P1 = Complete("p1",
              [p1_extract, merge_step, p1_archive, p1_advance],
              "p1      — complete a pairs file evaluation")

# p2 is the same shape, with retrieval ahead of it and one extract producing
# both verdicts. The kinds are one step because a row ticked Y and N lands in
# both sets and only the pair of them shows it -- see p2_extract. `classify`
# is the first durable write, so everything that could reject the review has
# happened by the time it runs -- and it records both verdicts, so p2 ends by
# closing rather than by advancing anything.
P2 = Complete("p2",
              [p2_retrieve, p2_extract, p2_classify, merge_step,
               p2_archive, p2_close],
              "p2      — complete a yes file manual review")


class CompleteWords(command.Action):
    def __init__(self):
        super().__init__(summary="words   — complete a dictionary word review",
                         positional="NAME")

    def run(self, command_text, opts, argv) -> int:
        if not argv:
            return usage.missing_argument(self.format_help(command_text))
        if len(argv) > 1:
            return usage.invalid_argument(argv[1],
                                          self.format_help(command_text))
        bundle_name = names.check_name(argv[0], "bundle name")
        ctx = context.Context(root=opts.dir, phase="dict", force=opts.force,
                              bundle_name=bundle_name)
        fs.raise_if_not_dir(ctx.bundle_dir)

        # The source file standing in the bundle is the whole of "there is work
        # to do": p2_retrieve takes it as its input, so a published round -- from
        # which words_close unlinks it first, for this reason -- fails here
        # rather than being retrieved, extracted, and published again.
        steps.run_steps([p2_retrieve, words_extract], ctx)
        words_publish.run_step(ctx)
        try:
            words_close.run_step(ctx)
        except (OSError, ValueError):
            _cleanup_warning(ctx)
        log.success(f"Completed {bundle_name}")
        return 0


WORDS = CompleteWords()
