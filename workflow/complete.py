# complete.py
#
# Drain an evaluated bundle into the phase's done-set and close it out.
#
# Both phases are the same command: build a Context from the bundle name, check
# the bundle exists, run the recipe. Only the recipe differs, so a phase is a
# STEPS list and nothing else.

from pathlib import Path

from workflow import command, config, context, fs, log, names, steps, usage
from workflow.steps import merge as merge_step
from workflow.steps import p1_advance, p1_archive, p1_extract
from workflow.steps import p2_advance, p2_archive, p2_classify, p2_extract, p2_retrieve


def _resolve_bundle(root: Path, phase: str, positional: str) -> str:
    """The bundle directory named by complete's positional.

    The directory is canonical and its name is not ambiguous, so this is only
    an escape hatch: `eval` takes either the bundle name or the queued
    filename, and the string that opened a bundle has to be able to close it.
    A positional that names a directory is used as typed; only a miss falls
    back to taking the queue suffix off, and only if that names one.
    """
    if Path(positional).name != positional:
        return positional
    evals = config.path(root, [phase, "eval"])
    if (evals / positional).is_dir():
        return positional
    try:
        stem = names.queue_stem(phase, positional)
    except ValueError:
        # Not a queue shape -- or a phase with no queue contract at all. Either
        # way there is nothing to strip, so the miss is reported as typed.
        return positional
    return stem if (evals / stem).is_dir() else positional


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

        # The positional names the bundle directory, and every
        # artifact inside is found by prefix so nothing takes a name apart.
        bundle_name = _resolve_bundle(opts.dir, self.phase, argv[0])
        ctx = context.Context(root=opts.dir, phase=self.phase,
                              force=opts.force, bundle_name=bundle_name)
        fs.raise_if_not_dir(ctx.bundle_dir)

        # Archive is the first irreversible part of completion. Discover every
        # collision before extraction or folding changes anything, so -f is a
        # decision for this invocation rather than a recovery from half an
        # archive.
        if not ctx.force and not steps.is_done(self.archive, ctx):
            fs.raise_if_any_exist(self.archive.outputs(ctx))

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

P2 = Complete("p2",
              [p2_retrieve, p2_extract.YES, p2_classify, p2_extract.NO,
               merge_step, p2_archive, p2_advance],
              "p2      — complete a yes file manual review")
