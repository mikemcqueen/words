# test_workflow_intake.py
#
# The p2 queue contract and the classified sets: the two places a file enters
# the workflow from outside a bundle.

import tempfile
import unittest

from pathlib import Path
from unittest import mock

from tests import wf_fixture as fx
from workflow import config, eval as evaluate


class P2QueueContractTests(unittest.TestCase):
    """`submit p2` and `eval p2` must agree on what a queued name looks like."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.opts, _ = fx.make_wf(self.root)
        self.outside = self.root / "outside"
        self.outside.mkdir()

    def _submit(self, name, pairs=("beta,five", "alpha,two")):
        src = fx.write_pairs(self.outside / name, list(pairs))
        code, _, stderr = fx.run_wf("-d", str(self.root), "submit", "p2", str(src))
        self.assertEqual(0, code, stderr)
        return src

    def _queued(self):
        return sorted(p.name for p in fx.slot(self.opts, ["p2", "queued"]).iterdir())

    def test_an_unclassified_candidate_list_is_queued_as_pairs(self):
        self._submit("top.s2.m4.g4.1000")
        self.assertEqual(["top.s2.m4.g4.1000.pairs"], self._queued())

    def test_a_terminal_pairs_suffix_is_not_doubled(self):
        self._submit("top.s2.m4.g4.1000.pairs")
        self.assertEqual(["top.s2.m4.g4.1000.pairs"], self._queued())

    def test_an_advanced_p1_yes_keeps_its_name(self):
        self._submit("s6.pairs.90.10.p1.yes")
        self.assertEqual(["s6.pairs.90.10.p1.yes"], self._queued())

    def test_the_queued_copy_is_sorted_and_deduped(self):
        self._submit("top.1000", pairs=["zeta,one", "alpha,two", "alpha,two"])
        queued = fx.slot(self.opts, ["p2", "queued"]) / "top.1000.pairs"
        self.assertEqual(["alpha,two", "zeta,one"], queued.read_text().splitlines())

    # `eval` is what the old `submit p2` naming actually broke: it queued a
    # `*.yes` that `eval p2` had no glob for, so the file could never be opened.

    def _eval(self, bundle_name):
        with mock.patch.object(evaluate.EvalYes, "prepare"):
            return fx.run_wf("-d", str(self.root), "eval", "p2", bundle_name)

    def _assert_opens(self, submitted, queued_name, positional, opened_as=None):
        self._submit(submitted)
        code, _, stderr = self._eval(positional)
        self.assertEqual(0, code, stderr)
        bundle_dir = fx.slot(self.opts, ["p2", "eval"]) / (opened_as or positional)
        self.assertEqual([queued_name], [p.name for p in bundle_dir.iterdir()])

    def test_eval_opens_a_submitted_pairs_bundle(self):
        self._assert_opens("top.s2.m4.g4.1000",
                           "top.s2.m4.g4.1000.pairs", "top.s2.m4.g4.1000")

    def test_eval_opens_an_advanced_p1_yes_bundle(self):
        self._assert_opens("s6.pairs.90.10.p1.yes",
                           "s6.pairs.90.10.p1.yes", "s6.pairs.90.10")

    # Naming the queued file is the second way in. The bundle it opens is the
    # same one either spelling names -- the suffix comes off, so the eval
    # directory never records which producer filled the slot.

    def test_eval_accepts_the_queued_filename(self):
        self._assert_opens("top.s2.m4.g4.1000", "top.s2.m4.g4.1000.pairs",
                           "top.s2.m4.g4.1000.pairs",
                           opened_as="top.s2.m4.g4.1000")
        self.assertFalse(
            (fx.slot(self.opts, ["p2", "eval"]) / "top.s2.m4.g4.1000.pairs").exists())

    def test_eval_accepts_the_queued_filename_of_an_advanced_artifact(self):
        self._assert_opens("s6.90.10.p1.yes", "s6.90.10.p1.yes",
                           "s6.90.10.p1.yes", opened_as="s6.90.10")

    # p2 has two producers, so one bundle name can name two queued files. The
    # prefix is ambiguous by construction there; the filename is not.

    def _submit_both_shapes(self):
        self._submit("s6.90.10")            # -> s6.90.10.pairs
        self._submit("s6.90.10.p1.yes")     # -> kept verbatim
        self.assertEqual(["s6.90.10.p1.yes", "s6.90.10.pairs"], self._queued())

    def test_a_bundle_name_shared_by_two_queue_shapes_is_refused(self):
        self._submit_both_shapes()
        with self.assertRaises(ValueError) as caught:
            self._eval("s6.90.10")
        message = str(caught.exception)
        self.assertIn("s6.90.10.pairs", message)
        self.assertIn("s6.90.10.p1.yes", message)
        # Nothing was opened: the refusal is total.
        self.assertEqual([], list(fx.slot(self.opts, ["p2", "eval"]).iterdir()))

    def test_naming_the_file_opens_one_of_two_shapes_sharing_a_bundle_name(self):
        self._submit_both_shapes()
        code, _, stderr = self._eval("s6.90.10.p1.yes")
        self.assertEqual(0, code, stderr)
        bundle_dir = fx.slot(self.opts, ["p2", "eval"]) / "s6.90.10"
        self.assertEqual(["s6.90.10.p1.yes"], [p.name for p in bundle_dir.iterdir()])
        # The shape left behind is untouched, and waits its turn: the two
        # collapse to one bundle name, so it opens once this bundle completes.
        self.assertEqual(["s6.90.10.pairs"], self._queued())

    def test_the_second_shape_cannot_be_stacked_onto_an_open_bundle(self):
        self._submit_both_shapes()
        self.assertEqual(0, self._eval("s6.90.10.p1.yes")[0])

        # Refusing the plain form is what invites `-f`, so `-f` must refuse too:
        # a bundle holding two sources cannot be resolved and cannot be un-opened.
        for argv in (("eval", "p2", "s6.90.10.pairs"),
                     ("-f", "eval", "p2", "s6.90.10.pairs")):
            with self.subTest(argv=argv):
                with self.assertRaises(ValueError) as caught:
                    with mock.patch.object(evaluate.EvalYes, "prepare"):
                        fx.run_wf("-d", str(self.root), *argv)
                self.assertIn("s6.90.10.p1.yes", str(caught.exception))

        bundle_dir = fx.slot(self.opts, ["p2", "eval"]) / "s6.90.10"
        self.assertEqual(["s6.90.10.p1.yes"], [p.name for p in bundle_dir.iterdir()])
        self.assertEqual(["s6.90.10.pairs"], self._queued())

    def test_a_queued_name_outside_the_contract_is_rejected(self):
        fx.place(self.opts, ["p2", "queued"], "stray.txt", "alpha,two\n")
        with self.assertRaises(ValueError) as caught:
            self._eval("stray.txt")
        self.assertIn("queue shape", str(caught.exception))


class ClassifyTests(unittest.TestCase):
    """`wf classify KIND` normalizes and unions into classified/KIND/KIND.pairs."""

    KINDS = ("yes", "no")

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.opts, _ = fx.make_wf(self.root)

    def _classify(self, kind, name, pairs):
        src = self.root / name
        src.write_text("".join(f"{p}\n" for p in pairs))
        return fx.run_wf("-d", str(self.root), "classify", kind, str(src))

    def _aggregate(self, kind):
        return config.classified(self.root, kind)

    def test_normalizes_an_unsorted_input(self):
        for kind in self.KINDS:
            with self.subTest(kind=kind):
                code, _, stderr = self._classify(
                    kind, f"{kind}.pairs", ["zeta,one", "alpha,two", "alpha,two"])
                self.assertEqual(0, code, stderr)
                self.assertEqual(["alpha,two", "zeta,one"],
                                 self._aggregate(kind).read_text().splitlines())

    def test_unions_a_second_file_into_the_aggregate(self):
        for kind in self.KINDS:
            with self.subTest(kind=kind):
                self._classify(kind, f"{kind}.1.pairs", ["zeta,one"])
                self._classify(kind, f"{kind}.2.pairs", ["alpha,two", "zeta,one"])
                self.assertEqual(["alpha,two", "zeta,one"],
                                 self._aggregate(kind).read_text().splitlines())

    def test_each_kind_has_its_own_aggregate(self):
        self._classify("yes", "y.pairs", ["alpha,two"])
        self._classify("no", "n.pairs", ["cheese,map"])
        self.assertEqual(["alpha,two"],
                         self._aggregate("yes").read_text().splitlines())
        self.assertEqual(["cheese,map"],
                         self._aggregate("no").read_text().splitlines())

    def test_a_manual_yes_folds_into_what_p2_review_already_confirmed(self):
        # `complete p2` writes this same file, so a hand-made call and a
        # reviewed batch are indistinguishable once folded -- which is the point.
        reviewed = self._aggregate("yes")
        fx.write_pairs(reviewed, ["mid,three"])
        self._classify("yes", "manual.pairs", ["alpha,two"])
        self.assertEqual(["alpha,two", "mid,three"],
                         reviewed.read_text().splitlines())

    def test_warns_when_a_pair_lands_in_both_aggregates(self):
        self._classify("no", "n.pairs", ["cheese,map", "diamond,throat"])
        _, _, stderr = self._classify("yes", "y.pairs",
                                      ["cheese,map", "alpha,two"])
        self.assertIn("1 pair(s) now classified both YES and NO", stderr)
        self.assertIn("cheese,map", stderr)
        # A warning, not a refusal: there is no un-classify, so the reversal
        # the user just asked for still lands.
        self.assertEqual(["alpha,two", "cheese,map"],
                         self._aggregate("yes").read_text().splitlines())

    def test_is_silent_when_the_aggregates_do_not_overlap(self):
        self._classify("no", "n.pairs", ["cheese,map"])
        _, _, stderr = self._classify("yes", "y.pairs", ["alpha,two"])
        self.assertNotIn("classified both", stderr)

    def test_reports_new_and_total_counts(self):
        self._classify("no", "first.pairs", ["zeta,one"])
        _, stdout, _ = self._classify("no", "second.pairs",
                                      ["alpha,two", "zeta,one"])
        self.assertIn("1 new, 2 total", stdout)

    def test_a_missing_input_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            fx.run_wf("-d", str(self.root), "classify", "no",
                      str(self.root / "nope.pairs"))

    def test_a_missing_argument_reports_usage(self):
        code, _, stderr = fx.run_wf("-d", str(self.root), "classify", "yes")
        self.assertEqual(2, code)
        self.assertIn("PAIRS-FILE", stderr)


if __name__ == "__main__":
    unittest.main()
