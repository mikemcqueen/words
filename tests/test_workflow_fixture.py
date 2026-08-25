# test_workflow_fixture.py
#
# Exercises the fixture harness itself, and pins the current src.filter masking
# behaviour so Item 1's unification can be diffed against it.

import io
import os
import tempfile
import unittest

from contextlib import redirect_stderr
from unittest import mock

from pathlib import Path

from src.filter import filter_results
from tests import wf_fixture as fx
from workflow import bundle, config, names
from workflow import eval as evaluate
from workflow.context import Context


def _walk_layout(node, path):
    """Every directory config.CONFIG_LAYOUT says should exist."""
    for name, child in node.get("parts", {}).items():
        child_path = path + [name]
        yield child_path
        yield from _walk_layout(child, child_path)


class FixtureHarnessTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.opts, self.wf_dir = fx.make_wf(self.root)

    def test_make_wf_builds_every_slot_in_the_layout(self):
        self.assertTrue(self.wf_dir.is_dir())
        for parts in _walk_layout(config.CONFIG_LAYOUT, []):
            with self.subTest(slot="/".join(parts)):
                self.assertTrue((self.wf_dir.joinpath(*parts)).is_dir())

    def test_place_writes_into_the_named_slot(self):
        path = fx.place(self.opts, ["p1", "queued"], "a.pairs", "one,two\n")
        self.assertEqual(self.wf_dir / "p1" / "queued" / "a.pairs", path)
        self.assertEqual("one,two\n", path.read_text())

    def test_run_wf_drives_the_cli_against_the_fixture_tree(self):
        fx.place(self.opts, ["p1", "queued"], "a.pairs", "one,two\n")
        code, stdout, _ = fx.run_wf("-d", str(self.root), "show", "p1", "queued")
        self.assertEqual(0, code)
        self.assertIn("a.pairs", stdout)

    def test_make_wf_is_isolated_per_fixture(self):
        with tempfile.TemporaryDirectory() as other:
            _, other_wf = fx.make_wf(Path(other))
            self.assertNotEqual(self.wf_dir, other_wf)
            self.assertEqual([], list((other_wf / "p1" / "queued").iterdir()))


@fx.requires_native
class FilterMaskTests(unittest.TestCase):
    """Pins src.filter behaviour on the canonical rows, pre-Item-1."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.results = fx.write_results(
            Path(self._tmp.name) / "sample.jsonl", fx.BAND_ROWS)

    def _filter(self, yes, **kwargs) -> list[str]:
        out = io.StringIO()
        filter_results([self.results], yes, out, **kwargs)
        return out.getvalue().splitlines()

    def test_yes_band_includes_the_edge_and_excludes_just_below(self):
        got = self._filter(True, pmin=0.9, prng=0.1)
        self.assertEqual(
            ["yes,high", "yes,edge", "yes,one", "yes,rvsonly",
             "mixed,split", "yes,divergent"],
            got)

    def test_no_filter_keeps_only_rows_with_no_yes_direction(self):
        got = self._filter(False, pmin=0.9, prng=0.1)
        self.assertEqual(["no,both", "unknown,token"], got)

    def test_use_max_is_inert_when_pmin_plus_prange_is_one(self):
        band = dict(pmin=0.9, prng=0.1)
        self.assertEqual(self._filter(True, **band),
                         self._filter(True, use_max=True, **band))

    def test_use_max_diverges_when_pmin_plus_prange_is_below_one(self):
        band = dict(pmin=0.5, prng=0.3)
        any_hits = self._filter(True, **band)
        max_hits = self._filter(True, use_max=True, **band)
        self.assertIn("yes,divergent", any_hits)
        self.assertNotIn("yes,divergent", max_hits)


if __name__ == "__main__":
    unittest.main()


@fx.requires_native
class ExtractP1YesSnapshotTests(unittest.TestCase):
    """The §4.1.1 acceptance snapshot, captured before Item 1 unifies src.filter.

    LC_ALL is pinned because _extract shells out to `sort -u` with the ambient
    environment, so its byte-level output ordering is locale-dependent.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.opts, _ = fx.make_wf(self.root)

        out = fx.slot(self.opts, ["p1", "done", "out"])
        fx.write_results(out / "alpha.jsonl", fx.BAND_ROWS[:5])
        fx.write_results(out / "beta.jsonl", fx.BAND_ROWS[5:])

        patcher = mock.patch.dict(os.environ, {"LC_ALL": "C"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _extract(self, source: str) -> list[str]:
        dst = self.root / f"extracted.{source}.pairs"
        code, _, stderr = fx.run_wf(
            "-d", str(self.root), "extract", "p1", "yes", "-o", str(dst), source)
        self.assertEqual(0, code, stderr)
        return dst.read_text().splitlines()

    def test_extracts_the_yes_band_across_the_whole_corpus(self):
        self.assertEqual(
            ["mixed,split", "yes,divergent", "yes,edge",
             "yes,high", "yes,one", "yes,rvsonly"],
            self._extract("all"))

    def test_extracts_from_a_single_named_result_file(self):
        self.assertEqual(["yes,edge", "yes,high", "yes,one"],
                         self._extract("alpha.jsonl"))

    def test_output_is_sorted_and_unique(self):
        got = self._extract("all")
        self.assertEqual(sorted(set(got)), got)

    def _extract_pairs(self, pairs: Path, *extra: str):
        dst = self.root / "extracted.filtered.pairs"
        code, _, stderr = fx.run_wf(
            "-d", str(self.root), "extract", "p1", "yes", "-o", str(dst),
            "--pairs", str(pairs), *extra)
        self.assertEqual(0, code, stderr)
        return dst.read_text().splitlines()

    def test_pairs_scans_the_archive_and_restricts_output(self):
        pairs = fx.write_pairs(
            self.root / "wanted.pairs", ["yes,high", "mixed,split", "no,both"])
        self.assertEqual(["mixed,split", "yes,high"], self._extract_pairs(pairs))

    def test_pairs_honors_the_requested_probability_band(self):
        pairs = fx.write_pairs(
            self.root / "wanted.pairs", ["yes,divergent", "yes,high"])
        self.assertEqual(
            ["yes,divergent"],
            self._extract_pairs(pairs, "--pm", "0.5", "--pr", "0.3"))

    def test_results_dir_overrides_the_archive_for_pairs_mode(self):
        results_dir = self.root / "other-results"
        results_dir.mkdir()
        fx.write_results(results_dir / "only.jsonl", fx.BAND_ROWS[5:])
        pairs = fx.write_pairs(
            self.root / "wanted.pairs", ["yes,high", "mixed,split"])

        self.assertEqual(
            ["mixed,split"],
            self._extract_pairs(pairs, "--results-dir", str(results_dir)))

    def test_results_dir_requires_pairs(self):
        dst = self.root / "unused.pairs"
        code, _, stderr = fx.run_wf(
            "-d", str(self.root), "extract", "p1", "yes", "-o", str(dst),
            "--results-dir", str(self.root), "all")
        self.assertEqual(2, code)
        self.assertIn("--results-dir requires --pairs", stderr)
        self.assertFalse(dst.exists())

    def test_pairs_mode_rejects_a_positional_source(self):
        pairs = fx.write_pairs(self.root / "wanted.pairs", ["yes,high"])
        dst = self.root / "unused.pairs"
        code, _, stderr = fx.run_wf(
            "-d", str(self.root), "extract", "p1", "yes", "-o", str(dst),
            "--pairs", str(pairs), "all")
        self.assertEqual(2, code)
        self.assertIn("invalid argument: 'all'", stderr)
        self.assertFalse(dst.exists())


@fx.requires_native
class UnifiedFilterTests(unittest.TestCase):
    """Item 1: one filter_results over a path list, with an optional pair mask."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.alpha = fx.write_results(self.dir / "alpha.jsonl", fx.BAND_ROWS[:5])
        self.beta = fx.write_results(self.dir / "beta.jsonl", fx.BAND_ROWS[5:])

    def _filter(self, paths, yes=True, **kwargs) -> list[str]:
        out = io.StringIO()
        filter_results(paths, yes, out, pmin=0.9, prng=0.1, **kwargs)
        return out.getvalue().splitlines()

    def test_reads_a_corpus_of_files_holding_different_pairs(self):
        # The reader treats a path list as *aligned* files and raises on a pair
        # mismatch, so filter_results must open one reader per file.
        self.assertEqual(
            ["yes,high", "yes,edge", "yes,one", "yes,rvsonly",
             "mixed,split", "yes,divergent"],
            self._filter([self.alpha, self.beta]))

    def test_pairs_path_restricts_output_to_set_members(self):
        pairs = fx.write_pairs(self.dir / "keep.pairs", ["yes,high", "mixed,split"])
        self.assertEqual(["yes,high", "mixed,split"],
                         self._filter([self.alpha, self.beta], pairs_path=str(pairs)))

    def test_pairs_path_none_skips_the_identity_mask(self):
        self.assertEqual(self._filter([self.alpha]),
                         self._filter([self.alpha], pairs_path=None))

    def test_unopenable_file_warns_and_continues(self):
        missing = self.dir / "gone.jsonl"
        err = io.StringIO()
        with redirect_stderr(err):
            got = self._filter([missing, self.alpha])
        self.assertEqual(["yes,high", "yes,edge", "yes,one"], got)
        self.assertIn("WARNING: skipping", err.getvalue())

    def test_malformed_row_still_raises_rather_than_being_skipped(self):
        # The guard covers opening the file, not parsing it -- carried over from
        # the old filter_pairs deliberately, so corrupt rows are not silent.
        bad = self.dir / "bad.jsonl"
        bad.write_text('{"pair": "a,b", "logprobs": {"fwd": [{"YES": 0.95}]}}\nnot json\n')
        with self.assertRaises(RuntimeError):
            self._filter([bad])

    def test_bare_path_is_rejected_instead_of_iterating_per_character(self):
        for bad in (str(self.alpha), self.alpha):
            with self.subTest(kind=type(bad).__name__):
                with self.assertRaises(TypeError):
                    self._filter(bad)

    def test_empty_path_list_is_an_error(self):
        with self.assertRaises(SystemExit):
            self._filter([])


@fx.requires_native
class FilterQueuePublishTests(unittest.TestCase):
    """`wf filter` publishes into p2/queued, which is later fed to `comm -23`."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.opts, _ = fx.make_wf(self.root)
        fx.write_results(
            fx.slot(self.opts, ["p1", "done", "out"]) / "sample.jsonl",
            fx.BAND_ROWS)

    def test_queued_output_is_a_sorted_unique_set(self):
        code, _, stderr = fx.run_wf(
            "-d", str(self.root), "filter", "sample.jsonl")
        self.assertEqual(0, code, stderr)

        queued = list(fx.slot(self.opts, ["p2", "queued"]).iterdir())
        self.assertEqual(1, len(queued), queued)
        lines = queued[0].read_text().splitlines()
        # filter_results emits in corpus order; unsorted here would make the
        # later `comm -23` in eval silently wrong rather than an error.
        self.assertEqual(sorted(set(lines)), lines)

    def test_no_scratch_file_is_left_behind(self):
        fx.run_wf("-d", str(self.root), "filter", "sample.jsonl")
        strays = [p.name for p in fx.slot(self.opts, ["p2", "queued"]).iterdir()
                  if p.suffix in (".tmp", ".unsorted")]
        self.assertEqual([], strays)


@fx.requires_native
class ProducedNameTests(unittest.TestCase):
    """Every artifact a bundle produces is prefixed by its bundle name."""

    # As in the real corpus, evalpair appends its own suffixes to the result
    # name, so the result stem is not the pairs name.
    RESULT = "s6.txt.pairs_third_p3_juniper.qwen35.jsonl"
    PAIRS = "s6.txt.pairs"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.opts, _ = fx.make_wf(self.root)
        ev = fx.make_bundle(self.opts, "p1", self.PAIRS)
        fx.write_pairs(ev / self.PAIRS, fx.pairs_of(fx.BAND_ROWS))
        fx.write_results(ev / self.RESULT, fx.BAND_ROWS)

    def _produced(self) -> list[str]:
        return sorted(
            [p.name for p in fx.slot(self.opts, ["p2", "queued"]).iterdir()]
            + [p.name for p in fx.slot(self.opts, ["p3", "queued"]).iterdir()])

    def test_complete_renders_every_artifact_under_one_bundle_name(self):
        code, _, stderr = fx.run_wf(
            "-d", str(self.root), "complete", "p1", self.PAIRS)
        self.assertEqual(0, code, stderr)

        bundle_name = names.bundle_name(Path(self.RESULT).stem, 0.9, 0.1)
        produced = self._produced()
        self.assertEqual(
            [names.artifact(bundle_name, "p1", "no"),
             names.artifact(bundle_name, "p1", "yes")],
            produced)
        for name in produced:
            with self.subTest(name=name):
                self.assertTrue(name.startswith(bundle_name))

    def test_complete_reports_pair_and_filter_counts_once(self):
        code, _, stderr = fx.run_wf(
            "-d", str(self.root), "complete", "p1", self.PAIRS)
        self.assertEqual(0, code, stderr)
        self.assertEqual(1, stderr.count("found 9 source pairs"))
        self.assertIn("filtered 6 90-100% YES pairs", stderr)
        self.assertIn("filtered 2 NO pairs", stderr)
        self.assertNotIn("loaded 9 pairs", stderr)

    def test_filter_and_complete_agree_on_the_bundle_name(self):
        # Both derive the bundle name from the p1 result stem, so re-slicing the same
        # band names the same artifact rather than a near-miss duplicate.
        fx.write_results(
            fx.slot(self.opts, ["p1", "done", "out"]) / self.RESULT, fx.BAND_ROWS)
        code, _, stderr = fx.run_wf("-d", str(self.root), "filter", self.RESULT)
        self.assertEqual(0, code, stderr)

        bundle_name = names.bundle_name(Path(self.RESULT).stem, 0.9, 0.1)
        self.assertEqual(
            [names.artifact(bundle_name, "p1", "yes")], self._produced())

    def test_a_different_band_renders_a_different_bundle_name(self):
        fx.write_results(
            fx.slot(self.opts, ["p1", "done", "out"]) / self.RESULT, fx.BAND_ROWS)
        fx.run_wf("-d", str(self.root), "filter", self.RESULT)
        fx.run_wf("-d", str(self.root), "filter", self.RESULT,
                  "--pm", "0.5", "--pr", "0.3")
        self.assertEqual(
            [names.artifact(
                names.bundle_name(Path(self.RESULT).stem, 0.5, 0.3),
                "p1", "yes"),
             names.artifact(
                 names.bundle_name(Path(self.RESULT).stem, 0.9, 0.1),
                 "p1", "yes")],
            self._produced())


class BundleInputSelectionTests(unittest.TestCase):
    """`eval` may filter its input; everything downstream follows that choice."""

    BUNDLE_NAME = "s6.txt.pairs_third.90.10"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.opts, _ = fx.make_wf(self.root)
        self.eval_dir = fx.make_bundle(self.opts, "p2", self.BUNDLE_NAME)
        self.queued = self.eval_dir / names.artifact(
            self.BUNDLE_NAME, "p1", "yes")
        fx.write_pairs(self.queued, ["alpha,two", "mid,three"])
        self.ctx = Context(
            root=self.root, phase="p2", bundle_name=self.BUNDLE_NAME)

    def test_source_is_always_the_queued_artifact(self):
        self.assertEqual(self.queued, bundle.source(self.ctx))

    def test_evaluated_is_the_queued_file_when_eval_did_not_filter(self):
        self.assertEqual(self.queued, bundle.evaluated(self.ctx))

    def test_evaluated_is_the_filtered_file_when_eval_produced_one(self):
        filtered = self.queued.with_name(self.queued.name + ".filtered")
        fx.write_pairs(filtered, ["alpha,two"])
        # Note titles and the done-set merge follow this; the *original* is
        # still what gets archived.
        self.assertEqual(filtered, bundle.evaluated(self.ctx))
        self.assertEqual(self.queued, bundle.source(self.ctx))

    def test_unknown_bundle_name_is_an_error(self):
        # wf.main() lets these propagate; the __main__ wrapper renders
        # (OSError, ValueError) as a message and a non-zero exit.
        with self.assertRaises((OSError, ValueError)):
            fx.run_wf("-d", str(self.root), "complete", "p2", "nope.90.10")


@fx.requires_native
class BundleLifecycleTests(unittest.TestCase):
    """`eval` opens a bundle directory; `complete` drains and removes it."""

    BUNDLE = "s6.txt"
    PAIRS = f"{BUNDLE}.pairs"
    RESULT = "s6.txt.pairs_third_p3_juniper.qwen35.jsonl"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.opts, _ = fx.make_wf(self.root)
        fx.write_pairs(fx.slot(self.opts, ["p1", "queued"]) / self.PAIRS,
                       fx.pairs_of(fx.BAND_ROWS))

    def _eval(self, positional=None):
        code, _, stderr = fx.run_wf("-d", str(self.root), "eval", "p1",
                                    positional or self.BUNDLE)
        self.assertEqual(0, code, stderr)
        return Context(
            root=self.opts.dir, phase="p1", bundle_name=self.BUNDLE).bundle_dir

    def test_eval_opens_a_bundle_directory_and_empties_the_queue(self):
        bundle_dir = self._eval()
        self.assertEqual([self.PAIRS], [p.name for p in bundle_dir.iterdir()])
        self.assertEqual([], list(fx.slot(self.opts, ["p1", "queued"]).iterdir()))

    def test_naming_the_queued_file_opens_the_same_bundle(self):
        # The suffix comes off: the directory is named for the bundle either way.
        self.assertEqual(self._eval(self.PAIRS),
                         fx.slot(self.opts, ["p1", "eval"]) / self.BUNDLE)

    def test_ls_eval_is_the_in_flight_list(self):
        self._eval()
        self.assertEqual([self.BUNDLE],
                         [p.name for p in fx.slot(self.opts, ["p1", "eval"]).iterdir()])

    def test_complete_drains_and_removes_the_bundle_directory(self):
        bundle_dir = self._eval()
        fx.write_results(bundle_dir / self.RESULT, fx.BAND_ROWS)

        code, _, stderr = fx.run_wf("-d", str(self.root), "complete", "p1", self.BUNDLE)
        self.assertEqual(0, code, stderr)
        self.assertFalse(bundle_dir.exists())
        self.assertEqual([], list(fx.slot(self.opts, ["p1", "eval"]).iterdir()))
        self.assertTrue((fx.slot(self.opts, ["p1", "done", "in"]) / self.PAIRS).is_file())
        self.assertTrue((fx.slot(self.opts, ["p1", "done", "out"]) / self.RESULT).is_file())

    def test_complete_accepts_the_queued_filename(self):
        # The escape hatch: whatever string opened the bundle closes it. The
        # directory name stays canonical -- only a miss falls back to stripping.
        bundle_dir = self._eval(self.PAIRS)
        fx.write_results(bundle_dir / self.RESULT, fx.BAND_ROWS)

        code, _, stderr = fx.run_wf("-d", str(self.root), "complete", "p1", self.PAIRS)
        self.assertEqual(0, code, stderr)
        self.assertFalse(bundle_dir.exists())
        self.assertTrue((fx.slot(self.opts, ["p1", "done", "in"]) / self.PAIRS).is_file())

    def test_complete_prefers_a_directory_of_exactly_that_name(self):
        # A bundle literally named `<stem>.pairs` is still reachable: the
        # directory wins over the strip, so the old spelling never shadows it.
        fx.make_bundle(self.opts, "p1", self.PAIRS)
        self._eval()
        self.assertTrue((fx.slot(self.opts, ["p1", "eval"]) / self.BUNDLE).is_dir())

        code, _, stderr = fx.run_wf("-d", str(self.root), "complete", "p1", self.PAIRS)
        self.assertEqual(0, code, stderr)
        # The empty directory of that exact name is what closed; the real
        # in-flight bundle was never touched.
        evals = fx.slot(self.opts, ["p1", "eval"])
        self.assertEqual([self.BUNDLE], [p.name for p in evals.iterdir()])

    def test_filter_refuses_a_bundle_already_in_flight_in_p2(self):
        fx.write_results(fx.slot(self.opts, ["p1", "done", "out"]) / self.RESULT,
                         fx.BAND_ROWS)
        bundle_name = names.bundle_name(Path(self.RESULT).stem, 0.9, 0.1)
        fx.make_bundle(self.opts, "p2", bundle_name)

        with self.assertRaises(ValueError):
            fx.run_wf("-d", str(self.root), "filter", self.RESULT)

    def test_filter_refuses_a_bundle_already_finished_in_p2(self):
        fx.write_results(fx.slot(self.opts, ["p1", "done", "out"]) / self.RESULT,
                         fx.BAND_ROWS)
        bundle_name = names.bundle_name(Path(self.RESULT).stem, 0.9, 0.1)
        done_in = fx.slot(self.opts, ["p2", "done", "in"])
        (done_in / names.artifact(bundle_name, "p1", "yes")).write_text("")

        with self.assertRaises(ValueError):
            fx.run_wf("-d", str(self.root), "filter", self.RESULT)

    def test_eval_p2_opens_a_bundle_directory_under_the_bundle_name(self):
        bundle_name = "s6.txt.pairs_third.90.10"
        fx.write_pairs(
            fx.slot(self.opts, ["p2", "queued"])
            / names.artifact(bundle_name, "p1", "yes"),
            ["alpha,two", "mid,three"])

        # `note --create --production` and split.sh are external side effects.
        with mock.patch.object(evaluate, "_split_pairs", return_value=[]), \
             mock.patch.object(evaluate, "_make_notes") as make_notes:
            code, _, stderr = fx.run_wf(
                "-d", str(self.root), "eval", "p2", bundle_name)

        self.assertEqual(0, code, stderr)
        make_notes.assert_called_once()
        bundle_dir = Context(
            root=self.opts.dir, phase="p2",
            bundle_name=bundle_name).bundle_dir
        self.assertEqual([names.artifact(bundle_name, "p1", "yes")],
                         [p.name for p in bundle_dir.iterdir()])
