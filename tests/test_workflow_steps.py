# test_workflow_steps.py
#
# Item 5: the operation protocol, its runner, and Context.

import dataclasses
import io
import tempfile
import unittest

from contextlib import redirect_stderr

from pathlib import Path

from tests import wf_fixture as fx
from workflow import names, steps
from workflow.context import Context


class FakeStep:
    """A step is anything with NAME/outputs/run_step; modules are the real ones."""

    def __init__(self, name, produces, own_is_done=None):
        self.NAME = name
        self._produces = produces
        self._own_is_done = own_is_done
        self.runs = 0

    def outputs(self, ctx):
        return list(self._produces)

    def run_step(self, ctx):
        self.runs += 1
        for p in self._produces:
            p.write_text("produced\n")

    def __getattr__(self, attr):
        # Only expose is_done when this fake declares one, so the default path
        # is exercised otherwise.
        if attr == "is_done" and self.__dict__.get("_own_is_done") is not None:
            return self.__dict__["_own_is_done"]
        raise AttributeError(attr)


class RunStepsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.ctx = Context(root=self.dir, phase="p1")

    def _run(self, step_list, ctx=None):
        """run_steps, with its skip logging kept out of the suite output."""
        with redirect_stderr(io.StringIO()):
            return steps.run_steps(step_list, self.ctx if ctx is None else ctx)

    def test_runs_a_step_whose_outputs_are_missing(self):
        step = FakeStep("a", [self.dir / "a.out"])
        self._run([step])
        self.assertEqual(1, step.runs)

    def test_skips_a_step_whose_outputs_already_exist(self):
        out = self.dir / "a.out"
        out.write_text("already\n")
        step = FakeStep("a", [out])
        self._run([step])
        self.assertEqual(0, step.runs)
        self.assertEqual("already\n", out.read_text())

    def test_resumes_a_partly_finished_recipe(self):
        done, pending = self.dir / "1.out", self.dir / "2.out"
        done.write_text("already\n")
        first, second = FakeStep("first", [done]), FakeStep("second", [pending])
        self._run([first, second])
        self.assertEqual((0, 1), (first.runs, second.runs))

    def test_force_ignores_is_done(self):
        out = self.dir / "a.out"
        out.write_text("already\n")
        step = FakeStep("a", [out])
        self._run([step], dataclasses.replace(self.ctx, force=True))
        self.assertEqual(1, step.runs)
        self.assertEqual("produced\n", out.read_text())

    def test_a_step_may_override_is_done(self):
        out = self.dir / "a.out"
        out.write_text("already\n")
        step = FakeStep("a", [out], own_is_done=lambda ctx: False)
        self._run([step])
        self.assertEqual(1, step.runs)

    def test_a_step_declaring_no_outputs_is_never_skipped(self):
        step = FakeStep("a", [])
        self._run([step])
        self.assertEqual(1, step.runs)


class ContextTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.opts, _ = fx.make_wf(self.root)

    def test_is_immutable(self):
        ctx = Context(root=self.root, phase="p1")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ctx.phase = "p2"

    def test_batch_dir_and_artifact_render_under_the_slug(self):
        slug = names.slug("batch", 0.9, 0.1)
        ctx = Context(root=self.root, phase="p2", slug=slug)
        self.assertEqual(fx.slot(self.opts, ["p2", "eval"]) / slug, ctx.batch_dir)
        self.assertEqual(ctx.batch_dir / f"{slug}.p2.yes",
                         ctx.artifact("p2", "yes"))

    def test_batch_dir_without_a_slug_is_an_error(self):
        with self.assertRaises(ValueError):
            Context(root=self.root, phase="p1").batch_dir


@fx.requires_native
class ExtractRecipeTests(unittest.TestCase):
    """extract-p1-yes is now STEPS = [filter]; the recipe is the whole command."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.opts, _ = fx.make_wf(self.root)
        fx.write_results(
            fx.slot(self.opts, ["p1", "done", "out"]) / "a.jsonl", fx.BAND_ROWS)
        self.dest = self.root / "out.pairs"

    def _extract(self, *extra):
        return fx.run_wf("-d", str(self.root), *extra,
                         "extract", "p1", "yes", "-o", str(self.dest), "all")

    def test_extracts_the_band(self):
        code, _, stderr = self._extract()
        self.assertEqual(0, code, stderr)
        self.assertEqual(
            ["mixed,split", "yes,divergent", "yes,edge",
             "yes,high", "yes,one", "yes,rvsonly"],
            self.dest.read_text().splitlines())

    def test_rerunning_skips_instead_of_redoing_the_work(self):
        self._extract()
        self.dest.write_text("sentinel\n")
        code, _, stderr = self._extract()
        self.assertEqual(0, code, stderr)
        self.assertIn("skip filter", stderr)
        self.assertEqual("sentinel\n", self.dest.read_text())

    def test_force_overwrites(self):
        self._extract()
        self.dest.write_text("sentinel\n")
        code, _, stderr = self._extract("-f")
        self.assertEqual(0, code, stderr)
        self.assertNotEqual("sentinel\n", self.dest.read_text())


if __name__ == "__main__":
    unittest.main()
