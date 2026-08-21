# test_workflow_p2.py
#
# Item 6: `complete p2` as a recipe. The note store is faked throughout -- the
# real one is a production side effect and `note` is installed on this machine.

import subprocess
import tempfile
import unittest

from pathlib import Path
from unittest import mock

from tests import wf_fixture as fx
from workflow import batch, names
from workflow.context import Context
from workflow.steps import p2_extract, p2_retrieve


_REAL_RUN = subprocess.run


# note title suffix -> the pairs a part yields, by kind
PARTS = {
    "aa": {"yes": ["alpha,two", "mid,three"], "no": ["zeta,one"]},
    "ab": {"yes": ["beta,five"], "no": ["yankee,four", "zeta,one"]},
}


class FakeNotes:
    """Stands in for the `note` binary in both steps."""

    def __init__(self):
        self.fetched = []
        self.fail_after = None

    def get(self, argv, **kwargs):
        title = argv[argv.index("--get") + 1]
        suffix = title[-2:]
        if self.fail_after is not None and len(self.fetched) >= self.fail_after:
            raise RuntimeError("network died")
        if suffix not in PARTS:
            return subprocess.CompletedProcess(argv, 1, "", "note not found")
        self.fetched.append(title)
        return subprocess.CompletedProcess(argv, 0, f"<enex>{suffix}</enex>", "")

    def route(self, argv, **kwargs):
        """p2_retrieve and p2_extract share one `subprocess` module object, so
        patching either patches both. Dispatch on the call instead."""
        if "--get" in argv:
            return self.get(argv, **kwargs)
        if "--parse-file" in argv:
            return self.parse(argv, **kwargs)
        # setops shells out through the same module object; let it through.
        return _REAL_RUN(argv, **kwargs)

    def parse(self, argv, **kwargs):
        source = Path(argv[argv.index("--parse-file") + 1])
        kind = argv[argv.index("--type") + 1]
        suffix = source.name.removesuffix(".enex")[-2:]
        for pair in PARTS[suffix][kind]:
            kwargs["stdout"].write(pair + "\n")
        return subprocess.CompletedProcess(argv, 0)


class P2RecipeTests(unittest.TestCase):
    SLUG = "s6.txt.pairs_third.90.10"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.opts, _ = fx.make_wf(self.root)

        self.batch_dir = fx.make_batch(self.opts, "p2", self.SLUG)
        self.queued = self.batch_dir / names.artifact(self.SLUG, "p1", "yes")
        fx.write_pairs(self.queued, ["alpha,two", "mid,three", "zeta,one"])

        self.notes = FakeNotes()
        self.ctx = Context(root=self.root, phase="p2", slug=self.SLUG)

    def _complete(self, *extra):
        with mock.patch.object(p2_retrieve.subprocess, "run", self.notes.route):
            return fx.run_wf("-d", str(self.root), *extra,
                             "complete", "p2", self.SLUG)

    def _slot(self, *parts):
        return fx.slot(self.opts, list(parts))

    # ---------------------------------------------------------------- happy path

    def test_completes_and_closes_the_batch(self):
        code, _, stderr = self._complete()
        self.assertEqual(0, code, stderr)
        self.assertFalse(self.batch_dir.exists())

    def test_retrieve_probes_until_a_part_is_missing(self):
        self._complete()
        self.assertEqual([f"{self.queued.name}.aa", f"{self.queued.name}.ab"],
                         self.notes.fetched)

    def test_archives_input_output_and_enex(self):
        self._complete()
        self.assertTrue((self._slot("p2", "done", "in") / self.queued.name).is_file())
        self.assertTrue((self._slot("p2", "done", "out")
                         / names.artifact(self.SLUG, "p2", "yes")).is_file())
        enex = self._slot("p2", "done", "out", "enex") / self.SLUG
        self.assertEqual([f"{self.queued.name}.aa.enex", f"{self.queued.name}.ab.enex"],
                         sorted(p.name for p in enex.iterdir()))

    def test_publishes_the_no_set_into_p3(self):
        self._complete()
        published = self._slot("p3", "queued") / names.artifact(self.SLUG, "p2", "no")
        self.assertEqual(["yankee,four", "zeta,one"],
                         published.read_text().splitlines())

    def test_folds_yes_into_the_done_set_and_the_classified_set(self):
        self._complete()
        expected = ["alpha,two", "mid,three", "zeta,one"]
        self.assertEqual(expected,
                         (self._slot("p2", "done") / "p2_done.pairs").read_text().splitlines())
        self.assertEqual(["alpha,two", "beta,five", "mid,three"],
                         (self._slot("classified", "yes") / "yes.pairs").read_text().splitlines())

    # ---------------------------------------------------------------- resume

    def test_retrieve_is_atomic_so_a_partial_fetch_is_not_mistaken_for_done(self):
        self.notes.fail_after = 1
        with self.assertRaises(RuntimeError):
            self._complete()
        self.assertFalse(p2_retrieve.enex_dir(self.ctx).exists())
        self.assertTrue(p2_retrieve.partial_dir(self.ctx).is_dir())
        self.assertFalse(p2_retrieve.is_done(self.ctx))

    def test_a_failed_retrieve_resumes_into_the_parts_already_fetched(self):
        self.notes.fail_after = 1
        with self.assertRaises(RuntimeError):
            self._complete()

        self.notes.fail_after = None
        code, _, stderr = self._complete()
        self.assertEqual(0, code, stderr)
        # .aa survived in enex.part/ across the failure and was fetched once
        # in total, not re-fetched on the resume; .ab was picked up after it.
        self.assertEqual(1, self.notes.fetched.count(f"{self.queued.name}.aa"))
        self.assertIn(f"{self.queued.name}.ab", self.notes.fetched)
        self.assertFalse(self.batch_dir.exists())

    def test_rerunning_after_merge_skips_forward_and_completes(self):
        from workflow import complete, steps as step_runner
        with mock.patch.object(p2_retrieve.subprocess, "run", self.notes.route):
            step_runner.run_steps(complete.P2.steps[:5], self.ctx)

        self.assertTrue(self.batch_dir.exists())
        code, _, stderr = self._complete()
        self.assertEqual(0, code, stderr)
        self.assertFalse(self.batch_dir.exists())
        # retrieve was skipped on the second pass, not repeated
        self.assertEqual(2, len(self.notes.fetched))


if __name__ == "__main__":
    unittest.main()
