# test_workflow_primitives.py
#
# Item 2: the select / merge / diff primitives.

import os
import subprocess
import tempfile
import unittest

from pathlib import Path
from unittest import mock

from tests import wf_fixture as fx
from workflow import batch, names, setops
from workflow.context import Context
from workflow.select import select


class MergeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def _file(self, name, text) -> Path:
        p = self.dir / name
        p.write_text(text)
        return p

    def test_single_source_sorts_and_dedupes(self):
        src = self._file("a", "zeta\nalpha\nalpha\nmid\n")
        dst = setops.merge([src], self.dir / "out")
        self.assertEqual("alpha\nmid\nzeta\n", dst.read_text())

    def test_multiple_sources_are_unioned(self):
        a = self._file("a", "alpha\nmid\n")
        b = self._file("b", "mid\nzeta\n")
        setops.merge([a, b], self.dir / "out")
        self.assertEqual("alpha\nmid\nzeta\n", (self.dir / "out").read_text())

    def test_destination_may_be_one_of_its_own_sources(self):
        dst = self._file("done.pairs", "alpha\nmid\n")
        new = self._file("new.pairs", "beta\nmid\n")
        setops.merge([dst, new], dst)
        self.assertEqual("alpha\nbeta\nmid\n", dst.read_text())

    def test_failure_leaves_the_previous_destination_intact(self):
        dst = self._file("done.pairs", "original\n")
        with mock.patch.object(setops.subprocess, "run",
                               side_effect=subprocess.CalledProcessError(1, "sort")):
            with self.assertRaises(subprocess.CalledProcessError):
                setops.merge([self._file("new", "x\n")], dst)
        self.assertEqual("original\n", dst.read_text())
        self.assertEqual([], list(self.dir.glob("*.tmp")))

    def test_no_sources_is_an_error(self):
        with self.assertRaises(ValueError):
            setops.merge([], self.dir / "out")

    def test_runs_under_c_collation_regardless_of_ambient_locale(self):
        src = self._file("a", "x\n")
        seen = {}
        real = subprocess.run

        def spy(argv, **kwargs):
            seen.update(kwargs.get("env") or {})
            return real(argv, **kwargs)

        with mock.patch.dict(os.environ, {"LC_ALL": "en_US.UTF-8"}):
            with mock.patch.object(setops.subprocess, "run", spy):
                setops.merge([src], self.dir / "out")
        self.assertEqual("C", seen.get("LC_ALL"))


class DiffTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_keeps_lines_of_a_absent_from_b(self):
        a = self.dir / "a"; a.write_text("alpha\nbeta\nzeta\n")
        b = self.dir / "b"; b.write_text("beta\n")
        out = setops.diff(a, b, self.dir / "out")
        self.assertEqual("alpha\nzeta\n", out.read_text())

    def test_identical_sets_yield_nothing(self):
        a = self.dir / "a"; a.write_text("alpha\nbeta\n")
        b = self.dir / "b"; b.write_text("alpha\nbeta\n")
        self.assertEqual("", setops.diff(a, b, self.dir / "out").read_text())


class SelectTests(unittest.TestCase):
    SLOT = ["p1", "done", "out"]

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.opts, _ = fx.make_wf(self.root)
        self.out = fx.slot(self.opts, self.SLOT)
        for name in ("alpha.jsonl", "beta.jsonl", "notes.txt"):
            (self.out / name).write_text("")

    def _select(self, selector, **kw):
        return select(self.root, self.SLOT, selector, **kw)

    def test_all_honours_the_glob_and_sorts(self):
        self.assertEqual(["alpha.jsonl", "beta.jsonl"],
                         [p.name for p in self._select("all", glob="*.jsonl")])

    def test_all_without_a_glob_takes_everything(self):
        self.assertEqual(["alpha.jsonl", "beta.jsonl", "notes.txt"],
                         [p.name for p in self._select("all")])

    def test_all_on_an_empty_slot_is_an_error(self):
        with self.assertRaises(ValueError):
            select(self.root, ["p1", "queued"], "all")

    def test_named_file(self):
        self.assertEqual([self.out / "alpha.jsonl"], self._select("name:alpha.jsonl"))

    def test_bare_value_is_read_as_a_name(self):
        self.assertEqual(self._select("name:alpha.jsonl"), self._select("alpha.jsonl"))

    def test_missing_name_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            self._select("name:nope.jsonl")

    def test_stem_matches_by_prefix(self):
        self.assertEqual([self.out / "alpha.jsonl"],
                         self._select("stem:alpha", glob="*.jsonl"))

    def test_stem_matching_nothing_is_an_error(self):
        with self.assertRaises(ValueError):
            self._select("stem:nope", glob="*.jsonl")

    def test_ambiguous_stem_is_an_error(self):
        (self.out / "alpha.extra.jsonl").write_text("")
        with self.assertRaises(ValueError):
            self._select("stem:alpha", glob="*.jsonl")

    def test_absolute_path_bypasses_the_slot(self):
        outside = self.root / "elsewhere.jsonl"
        outside.write_text("")
        self.assertEqual([outside], self._select(str(outside)))

    def test_unknown_selector_kind_is_an_error(self):
        with self.assertRaises(ValueError):
            self._select("nonsense:alpha")


if __name__ == "__main__":
    unittest.main()


class NameRenderingTests(unittest.TestCase):
    """Item 3: names are rendered from dimensions, never edited or parsed."""

    def test_slice_renders_delimiter_free_values(self):
        self.assertEqual("90.10", names.slice_segment(0.9, 0.1))
        self.assertEqual("50.30", names.slice_segment(0.5, 0.3))

    def test_artifact_puts_invariant_dimensions_first(self):
        slug = names.slug("batch", 0.9, 0.1)
        self.assertEqual("batch.90.10.p1.yes", names.artifact(slug, "p1", "yes"))

    def test_slug_prefixes_every_artifact_of_its_batch(self):
        slug = names.slug("batch", 0.9, 0.1)
        for classifier in names.CLASSIFIERS:
            for kind in names.KINDS:
                with self.subTest(classifier=classifier, kind=kind):
                    self.assertTrue(
                        names.artifact(slug, classifier, kind).startswith(slug))

    def test_a_phase_transition_derives_a_sibling_not_a_rename(self):
        slug = names.slug("batch", 0.9, 0.1)
        p1 = names.artifact(slug, "p1", "yes")
        p2 = names.artifact(slug, "p2", "yes")
        self.assertNotEqual(p1, p2)
        self.assertTrue(p1.startswith(slug) and p2.startswith(slug))

    def test_different_bands_are_different_batches(self):
        self.assertNotEqual(names.slug("batch", 0.9, 0.1),
                            names.slug("batch", 0.5, 0.3))

    def test_unknown_classifier_or_kind_is_rejected(self):
        slug = names.slug("batch", 0.9, 0.1)
        with self.assertRaises(ValueError):
            names.artifact(slug, "p9", "yes")
        with self.assertRaises(ValueError):
            names.artifact(slug, "p1", "maybe")

    def test_empty_batch_or_slug_is_rejected(self):
        with self.assertRaises(ValueError):
            names.slug("", 0.9, 0.1)
        with self.assertRaises(ValueError):
            names.artifact("", "p1", "yes")

    def test_ensure_kind_is_idempotent(self):
        once = names.ensure_kind("ext1.txt", "pairs")
        self.assertEqual("ext1.txt.pairs", once)
        self.assertEqual(once, names.ensure_kind(once, "pairs"))


class BatchDirectoryTests(unittest.TestCase):
    """Item 4: the batch directory is the in-flight record."""

    SLUG = "s6.txt.pairs"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.opts, _ = fx.make_wf(self.root)
        self.queued = fx.slot(self.opts, ["p1", "queued"]) / self.SLUG
        fx.write_pairs(self.queued, ["alpha,two", "mid,three"])

    def _ctx(self, slug=None) -> Context:
        return Context(root=self.opts.dir, phase="p1", slug=slug or self.SLUG)

    def test_begin_moves_the_queued_artifact_into_a_new_directory(self):
        moved = batch.begin(self._ctx())
        self.assertEqual(self._ctx().batch_dir / self.SLUG, moved)
        self.assertTrue(moved.is_file())
        self.assertFalse(self.queued.exists())

    def test_directory_exists_exactly_while_work_is_in_flight(self):
        self.assertFalse(batch.in_flight(self._ctx()))
        batch.begin(self._ctx())
        self.assertTrue(batch.in_flight(self._ctx()))

    def test_begin_refuses_a_batch_already_in_flight(self):
        batch.begin(self._ctx())
        fx.write_pairs(self.queued, ["alpha,two"])
        with self.assertRaises(ValueError):
            batch.begin(self._ctx())

    def test_begin_refuses_an_unknown_slug(self):
        with self.assertRaises(ValueError):
            batch.begin(self._ctx("nope"))

    def test_one_finds_the_single_match(self):
        directory = batch.begin(self._ctx()).parent
        (directory / f"{self.SLUG}.jsonl").write_text("")
        self.assertEqual(directory / f"{self.SLUG}.jsonl",
                         batch.one(directory, "*.jsonl"))

    def test_one_rejects_zero_and_multiple_matches(self):
        directory = batch.begin(self._ctx()).parent
        with self.assertRaises(ValueError):
            batch.one(directory, "*.jsonl")
        (directory / f"{self.SLUG}.a.jsonl").write_text("")
        (directory / f"{self.SLUG}.b.jsonl").write_text("")
        with self.assertRaises(ValueError):
            batch.one(directory, "*.jsonl")

    def test_finish_removes_a_drained_directory(self):
        moved = batch.begin(self._ctx())
        moved.unlink()
        batch.finish(self._ctx())
        self.assertFalse(batch.in_flight(self._ctx()))

    def test_finish_refuses_to_drop_a_directory_still_holding_artifacts(self):
        batch.begin(self._ctx())
        with self.assertRaises(ValueError):
            batch.finish(self._ctx())
        self.assertTrue(batch.in_flight(self._ctx()))
