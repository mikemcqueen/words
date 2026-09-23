# test_workflow_review.py
#
# `review p1|p2`: `submit` then `eval`, with eval's flags passed through.

import io
import tempfile
import unittest

from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from tests import wf_fixture as fx
from workflow import config, dictionary, notes, wf, eval as evaluate


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.opts, _ = fx.make_wf(self.root)
        self.outside = self.root / "outside"
        self.outside.mkdir()

    def _pairs(self, name, pairs=("beta,five", "alpha,two")):
        return fx.write_pairs(self.outside / name, list(pairs))

    def _review(self, *argv):
        with mock.patch.object(evaluate.EvalYes, "prepare") as prepare:
            code, _, stderr = fx.run_wf("-d", str(self.root), "review", *argv)
        return code, stderr, prepare

    def _names(self, phase, slot):
        return sorted(p.name for p in fx.slot(self.opts, [phase, slot]).iterdir())

    def test_p1_queues_and_opens_the_bundle(self):
        src = self._pairs("s6.txt")
        code, stderr, _ = self._review("p1", str(src))
        self.assertEqual(0, code, stderr)
        self.assertEqual([], self._names("p1", "queued"))
        bundle_dir = fx.slot(self.opts, ["p1", "eval"]) / "s6.txt"
        self.assertEqual(["s6.txt.pairs"],
                         [p.name for p in bundle_dir.iterdir()])

    def test_p2_queues_and_opens_the_bundle(self):
        src = self._pairs("top.1000")
        code, stderr, prepare = self._review("p2", str(src))
        self.assertEqual(0, code, stderr)
        self.assertEqual([], self._names("p2", "queued"))
        source = fx.slot(self.opts, ["p2", "eval"]) / "top.1000" / "top.1000.pairs"
        self.assertEqual(source, prepare.call_args.args[0])

    def test_p2_flags_reach_eval_on_either_side_of_the_file(self):
        fx.write_pairs(config.classified(self.root, "yes"), ["yes,known"])
        for i, place_flag_first in enumerate((True, False)):
            src = self._pairs(f"r{i}", pairs=("keep,new", "known,yes"))
            argv = (["--pcomm", str(src)] if place_flag_first
                    else [str(src), "--pcomm"])
            code, stderr, prepare = self._review("p2", *argv)
            self.assertEqual(0, code, stderr)
            filtered = prepare.call_args.args[0]
            self.assertEqual(f"r{i}.pairs.filtered", filtered.name)
            self.assertEqual(["keep,new"], filtered.read_text().splitlines())

    def test_dry_run_is_refused_before_anything_is_queued(self):
        src = self._pairs("top.1000")
        with self.assertRaisesRegex(ValueError,
                                    "--dry-run is not valid for review"):
            self._review("--dry-run", "p2", str(src))
        self.assertEqual([], self._names("p2", "queued"))

    def test_a_bad_eval_flag_is_refused_before_anything_is_queued(self):
        src = self._pairs("top.1000")
        missing = self.outside / "missing.yes"
        with self.assertRaises((OSError, ValueError)):
            self._review("p2", "--yes-pairs", str(missing), str(src))
        self.assertEqual([], self._names("p2", "queued"))

    def test_the_file_given_twice_is_refused(self):
        src = self._pairs("top.1000")
        with self.assertRaisesRegex(ValueError, "given more than once"):
            self._review("p2", "--yes-pairs", str(src), str(src))
        self.assertEqual([], self._names("p2", "queued"))

    def test_missing_and_extra_files_report_usage(self):
        code, stderr, _ = self._review("p2")
        self.assertEqual(2, code)
        self.assertIn("missing required argument", stderr)
        a, b = self._pairs("a"), self._pairs("b")
        code, stderr, _ = self._review("p2", str(a), str(b))
        self.assertEqual(2, code)
        self.assertIn("invalid argument", stderr)
        self.assertEqual([], self._names("p2", "queued"))

    def _review_failing(self, *argv, prepare_error=None):
        """Run review expecting eval to raise; return what reached stderr."""
        stderr = io.StringIO()
        with redirect_stderr(stderr), \
                mock.patch.object(evaluate.EvalYes, "prepare",
                                  side_effect=prepare_error), \
                self.assertRaises((OSError, ValueError)):
            wf.main(["-d", str(self.root), "review", *argv])
        return stderr.getvalue()

    def test_an_eval_failure_before_opening_names_the_queued_file(self):
        src = self._pairs("top.1000")
        fx.make_bundle(self.opts, "p2", "top.1000")
        stderr = self._review_failing("p2", str(src))
        self.assertEqual(["top.1000.pairs"], self._names("p2", "queued"))
        self.assertIn("top.1000.pairs is still queued; continue with "
                      "`wf eval p2 top.1000.pairs`", stderr)

    def test_an_eval_failure_after_opening_does_not_claim_it_is_queued(self):
        src = self._pairs("top.1000")
        stderr = self._review_failing(
            "p2", str(src), prepare_error=ValueError("notes failed"))
        self.assertEqual([], self._names("p2", "queued"))
        self.assertNotIn("still queued", stderr)

    def test_an_existing_queued_file_stops_review_at_submit(self):
        src = self._pairs("top.1000")
        fx.write_pairs(fx.slot(self.opts, ["p2", "queued"]) / "top.1000.pairs",
                       ["old,pair"])
        with self.assertRaises((OSError, ValueError)):
            self._review("p2", str(src))
        self.assertEqual([], self._names("p2", "eval"))


class ReviewWordsTests(unittest.TestCase):
    NAME = "top.s7.m4.g5.words"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.opts, _ = fx.make_wf(self.root)
        config.base_dictionary(self.root).write_text(
            "apple\nbanana\ncherry\n")
        dictionary.gen_dict(self.root)
        self.src = self.root / "ranked.words"
        self.src.write_text(" 9 banana\n 2 apple\n")

    def _review(self, *argv):
        with mock.patch.object(notes, "make", return_value=[]) as make:
            code, _, stderr = fx.run_wf("-d", str(self.root), "review",
                                        "words", *argv)
        return code, stderr, make

    def _names(self, slot):
        return sorted(p.name for p in fx.slot(self.opts, ["dict", slot]).iterdir())

    def test_queues_and_opens_the_bundle_under_the_as_name(self):
        config.reviewed_words(self.root).write_text("apple\n")
        for argv in ((str(self.src), "--as", self.NAME),
                     ("--as", self.NAME + ".b", str(self.src))):
            code, stderr, make = self._review(*argv)
            self.assertEqual(0, code, stderr)
            name = argv[1] if argv[0] == "--as" else self.NAME
            filtered = (fx.slot(self.opts, ["dict", "eval"]) / name
                        / f"{name}.filtered")
            self.assertEqual(" 9 banana\n", filtered.read_text())
            self.assertEqual(filtered, make.call_args.args[0])
        self.assertEqual([], self._names("queued"))

    def test_as_goes_to_submit_and_checked_to_eval(self):
        code, stderr, make = self._review("--checked", "yes", str(self.src),
                                          "--as", self.NAME)
        self.assertEqual(0, code, stderr)
        self.assertEqual([self.NAME], self._names("eval"))
        self.assertEqual("YES", make.call_args.args[1].checked)

    def test_help_lists_as_and_eval_flags(self):
        code, stdout, stderr = fx.run_wf("help", "review", "words")
        self.assertEqual(0, code, stderr)
        for flag in ("--as NAME", "--checked TYPE"):
            self.assertIn(flag, stdout)

    def test_the_file_given_twice_is_refused(self):
        with self.assertRaisesRegex(ValueError, "given more than once"):
            self._review(str(self.src), "--as", str(self.src))
        self.assertEqual([], self._names("queued"))

    def test_checked_no_is_accepted(self):
        code, stderr, make = self._review(str(self.src), "--checked", "NO")
        self.assertEqual(0, code, stderr)
        self.assertEqual("NO", make.call_args.args[1].checked)

    def test_without_as_the_file_name_is_the_bundle(self):
        code, stderr, _ = self._review(str(self.src))
        self.assertEqual(0, code, stderr)
        self.assertEqual(["ranked.words"], self._names("eval"))

    def test_missing_reviewed_words_is_refused_before_anything_is_queued(self):
        config.reviewed_words(self.root).unlink()
        with self.assertRaisesRegex(ValueError, "run `wf gen dict`"):
            self._review(str(self.src))
        self.assertEqual([], self._names("queued"))

    def test_dry_run_is_refused_before_anything_is_queued(self):
        with self.assertRaisesRegex(ValueError,
                                    "--dry-run is not valid for review"):
            fx.run_wf("-d", str(self.root), "--dry-run", "review", "words",
                      str(self.src))
        self.assertEqual([], self._names("queued"))

    def test_missing_and_extra_files_report_usage(self):
        code, stderr, _ = self._review()
        self.assertEqual(2, code)
        self.assertIn("missing required argument", stderr)
        code, stderr, _ = self._review(str(self.src), str(self.src))
        self.assertEqual(2, code)
        self.assertIn("invalid argument", stderr)
        self.assertEqual([], self._names("queued"))

    def test_an_eval_failure_names_the_queued_file(self):
        self.src.write_text(" 4 Apple\n")
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(ValueError):
            wf.main(["-d", str(self.root), "review", "words", str(self.src),
                     "--as", self.NAME])
        self.assertEqual([self.NAME], self._names("queued"))
        self.assertIn(f"{self.NAME} is still queued; continue with "
                      f"`wf eval words {self.NAME}`", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
