# test_workflow_p2.py
#
# Item 6: `complete p2` as a recipe. The note store is faked throughout -- the
# real one is a production side effect and `note` is installed on this machine.

import subprocess
import tempfile
import unittest

from dataclasses import replace
from pathlib import Path
from unittest import mock

from tests import wf_fixture as fx
from workflow import bundle, names, notes
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
    BUNDLE_NAME = "s6.txt.pairs_third.90.10"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.opts, _ = fx.make_wf(self.root)

        self.bundle_dir = fx.make_bundle(
            self.opts, "p2", self.BUNDLE_NAME)
        self.queued = self.bundle_dir / names.artifact(
            self.BUNDLE_NAME, "p1", "yes")
        fx.write_pairs(self.queued, ["alpha,two", "mid,three", "zeta,one"])

        self.notes = FakeNotes()
        self.ctx = Context(
            root=self.root, phase="p2", bundle_name=self.BUNDLE_NAME)

    def _complete(self, *extra):
        with mock.patch.object(p2_retrieve.subprocess, "run", self.notes.route):
            return fx.run_wf("-d", str(self.root), *extra,
                             "complete", "p2", self.BUNDLE_NAME)

    def _slot(self, *parts):
        return fx.slot(self.opts, list(parts))

    # ---------------------------------------------------------------- happy path

    def test_completes_and_closes_the_bundle(self):
        code, _, stderr = self._complete()
        self.assertEqual(0, code, stderr)
        self.assertFalse(self.bundle_dir.exists())

    def test_retrieve_probes_until_a_part_is_missing(self):
        self._complete()
        self.assertEqual([f"{self.queued.name}.aa", f"{self.queued.name}.ab"],
                         self.notes.fetched)

    def test_archives_input_output_and_enex(self):
        self._complete()
        self.assertTrue((self._slot("p2", "done", "in") / self.queued.name).is_file())
        self.assertTrue((self._slot("p2", "done", "out")
                         / names.artifact(
                             self.BUNDLE_NAME, "p2", "yes")).is_file())
        enex = (self._slot("p2", "done", "out", "enex")
                / self.BUNDLE_NAME)
        self.assertEqual([f"{self.queued.name}.aa.enex", f"{self.queued.name}.ab.enex"],
                         sorted(p.name for p in enex.iterdir()))

    def test_publishes_the_no_set_into_p3(self):
        self._complete()
        published = self._slot("p3", "queued") / names.artifact(
            self.BUNDLE_NAME, "p2", "no")
        self.assertEqual(["yankee,four", "zeta,one"],
                         published.read_text().splitlines())

    def test_folds_yes_into_the_done_set_and_the_classified_set(self):
        self._complete()
        expected = ["alpha,two", "mid,three", "zeta,one"]
        self.assertEqual(expected,
                         (self._slot("p2", "done") / "p2_done.pairs").read_text().splitlines())
        self.assertEqual(["alpha,two", "beta,five", "mid,three"],
                         (self._slot("classified", "yes") / "yes.pairs").read_text().splitlines())

    def test_complete_preflights_archive_collisions_before_retrieval(self):
        archived = (self._slot("p2", "done", "out", "enex")
                    / self.BUNDLE_NAME)
        archived.mkdir()

        with self.assertRaisesRegex(ValueError, str(archived)):
            self._complete()

        self.assertEqual([], self.notes.fetched)
        self.assertTrue(self.queued.is_file())

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
        self.assertFalse(self.bundle_dir.exists())

    def test_rerunning_after_merge_skips_forward_and_completes(self):
        from workflow import complete, steps as step_runner
        names_ = [step.NAME for step in complete.P2.steps]
        through_merge = complete.P2.steps[:names_.index("merge") + 1]
        with mock.patch.object(p2_retrieve.subprocess, "run", self.notes.route):
            step_runner.run_steps(through_merge, self.ctx)

        self.assertTrue(self.bundle_dir.exists())
        code, _, stderr = self._complete()
        self.assertEqual(0, code, stderr)
        self.assertFalse(self.bundle_dir.exists())
        # retrieve was skipped on the second pass, not repeated
        self.assertEqual(2, len(self.notes.fetched))

    # ---------------------------------------------------------- forced refetch

    def _retrieve(self, force=False):
        with mock.patch.object(p2_retrieve.subprocess, "run", self.notes.route):
            p2_retrieve.run_step(replace(self.ctx, force=force))

    def _enex_parts(self):
        return sorted(p.name for p in p2_retrieve.enex_dir(self.ctx).iterdir())

    def test_force_replaces_a_completed_enex(self):
        self._retrieve()
        held = self._enex_parts()
        self.assertEqual(2, len(self.notes.fetched))

        self._retrieve(force=True)
        # -f on the one step that caches external state means the held copy is
        # stale, so every part is fetched again and replaces it.
        self.assertEqual(4, len(self.notes.fetched))
        self.assertEqual(held, self._enex_parts())
        self.assertFalse(p2_retrieve.partial_dir(self.ctx).exists())

    def test_a_forced_refetch_leaves_a_bundle_that_still_completes(self):
        # The regression: the forced rename used to raise ENOTEMPTY and strand
        # enex.part/, after which `bundle.finish` refused to close the bundle
        # ever again.
        self._retrieve()
        self._retrieve(force=True)

        code, _, stderr = self._complete()
        self.assertEqual(0, code, stderr)
        self.assertFalse(self.bundle_dir.exists())

    def test_a_failed_forced_refetch_keeps_the_copy_it_holds(self):
        self._retrieve()
        held = self._enex_parts()

        self.notes.fail_after = 2
        with self.assertRaises(RuntimeError):
            self._retrieve(force=True)

        # Nothing was cleared, because nothing had been staged to replace it.
        self.assertEqual(held, self._enex_parts())

        self.notes.fail_after = None
        code, _, stderr = self._complete()
        self.assertEqual(0, code, stderr)
        self.assertFalse(self.bundle_dir.exists())

    # ----------------------------------------------- contradictory checkboxes

    def _marked_both_ways(self, pair="mid,three"):
        """Both checkboxes ticked on one row: the parser answers each --type
        independently, so the pair comes back under both."""
        parts = {suffix: {kind: list(pairs) for kind, pairs in kinds.items()}
                 for suffix, kinds in PARTS.items()}
        parts["ab"]["no"].append(pair)
        return mock.patch.dict(PARTS, parts, clear=True)

    def _assert_nothing_was_written(self):
        # Neither set was placed: the check runs on the staged copies, so the
        # bundle is exactly as `retrieve` left it.
        self.assertFalse(self.ctx.artifact("p2", "yes").exists())
        self.assertFalse(self.ctx.artifact("p2", "no").exists())
        # `init` lays the classified set down empty; nothing folded into it.
        self.assertEqual("", (self._slot("classified", "yes")
                              / "yes.pairs").read_text())
        self.assertFalse((self._slot("p2", "done") / "p2_done.pairs").exists())
        self.assertEqual([], list(self._slot("p3", "queued").iterdir()))
        self.assertEqual([], list(self._slot("p2", "done", "in").iterdir()))
        # The bundle is still open on its source, so the review can be fixed
        # and the completion re-run.
        self.assertTrue(bundle.has_source(self.ctx))

    def test_a_pair_marked_both_ways_stops_the_bundle(self):
        with self._marked_both_ways():
            with self.assertRaisesRegex(ValueError, "mid,three"):
                self._complete()
        self._assert_nothing_was_written()

    def test_force_does_not_wave_a_contradiction_through(self):
        with self._marked_both_ways():
            with self.assertRaises(ValueError):
                self._complete("-f")
        self._assert_nothing_was_written()

    def test_correcting_the_note_and_re_running_completes(self):
        # The point of staging: the rejected extract placed nothing, so a
        # plain re-run redoes it whole and reads the corrected rows. No -f, no
        # cleaning up after the failure.
        with self._marked_both_ways():
            with self.assertRaises(ValueError):
                self._complete()

        code, _, stderr = self._complete()
        self.assertEqual(0, code, stderr)
        self.assertFalse(self.bundle_dir.exists())
        published = self._slot("p3", "queued") / names.artifact(
            self.BUNDLE_NAME, "p2", "no")
        self.assertEqual(["yankee,four", "zeta,one"],
                         published.read_text().splitlines())


class NoteNamingTests(unittest.TestCase):
    """The one rendering both ends of the note contract read."""

    def test_creation_and_retrieval_render_the_same_titles(self):
        source = Path("/tmp/s6.txt.pairs_third.90.10.p1.yes")
        paths = notes.part_paths(Path("/tmp"), source, 3)
        self.assertEqual([f"{source.name}.aa", f"{source.name}.ab",
                          f"{source.name}.ac"],
                         [path.name for path in paths])
        # What retrieve probes for is what creation named.
        self.assertEqual([path.name for path in paths],
                         [notes.title(source, i) for i in range(3)])

    def test_a_split_too_wide_to_name_is_refused(self):
        source = Path("/tmp/wide.pairs")
        notes.part_paths(Path("/tmp"), source, notes.MAX_PARTS)
        with self.assertRaisesRegex(ValueError, "at most 26"):
            notes.part_paths(Path("/tmp"), source, notes.MAX_PARTS + 1)


class NotesCommandTests(unittest.TestCase):
    """`wf notes p2` -- the note derivation, reached without moving anything.

    `note --create` and split.sh are external side effects, so the two calls
    that reach them are the assertion.
    """

    BUNDLE_NAME = "s6.txt.pairs_third.90.10"
    SOURCE = names.artifact(BUNDLE_NAME, "p1", "yes")
    PAIRS = ["alpha,two", "mid,three"]

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.opts, _ = fx.make_wf(self.root)

    def _place(self, *parts) -> Path:
        return fx.write_pairs(
            fx.slot(self.opts, list(parts)) / self.SOURCE, self.PAIRS)

    def _in_flight(self) -> Path:
        bundle_dir = fx.make_bundle(self.opts, "p2", self.BUNDLE_NAME)
        return fx.write_pairs(bundle_dir / self.SOURCE, self.PAIRS)

    # Both spellings the positional advertises. Every slot case runs under
    # each, because the source-file form is the one that has to reach past the
    # eval directory to answer at all.
    SPELLINGS = {"bundle name": BUNDLE_NAME, "source file": SOURCE}

    def _argv(self, force: bool, argv: tuple,
              positional: str | None = None) -> list[str]:
        return ["-d", str(self.root), *(["-f"] if force else []),
                "notes", "p2", positional or self.BUNDLE_NAME, *argv]

    def _notes(self, *argv, force=False, positional=None):
        with mock.patch.object(notes, "split", return_value=[]) as split, \
             mock.patch.object(notes, "create") as create:
            code, _, stderr = fx.run_wf(*self._argv(force, argv, positional))
        self.assertEqual(0, code, stderr)
        return split, create

    def _refuse(self, *argv, force=False, positional=None):
        with mock.patch.object(notes, "split") as split, \
             mock.patch.object(notes, "create") as create:
            with self.assertRaises((OSError, ValueError)) as caught:
                fx.run_wf(*self._argv(force, argv, positional))
        split.assert_not_called()
        create.assert_not_called()
        return str(caught.exception)

    def test_an_open_bundle_is_re_noted_from_whatever_eval_evaluated(self):
        source = self._in_flight()
        split, create = self._notes()
        split.assert_called_once_with(source)
        create.assert_called_once_with([], None)

        # `eval` follows the .filtered derivative when it wrote one, and so
        # does this: the notes cover the pairs actually under review.
        filtered = bundle.filtered(source)
        fx.write_pairs(filtered, ["alpha,two"])
        split, _ = self._notes()
        split.assert_called_once_with(filtered)

    def test_a_queued_review_names_the_eval_that_would_make_its_notes(self):
        for spelling, positional in self.SPELLINGS.items():
            with self.subTest(spelling=spelling):
                queued = self._place("p2", "queued")
                message = self._refuse(positional=positional)
                self.assertIn(f"wf eval p2 {queued.name}", message)
                # Nothing was moved: running the eval it names is the way in.
                self.assertTrue(queued.is_file())
                self.assertEqual(
                    [], list(fx.slot(self.opts, ["p2", "eval"]).iterdir()))
                queued.unlink()

    def test_an_archived_round_is_re_noted_only_under_force(self):
        for spelling, positional in self.SPELLINGS.items():
            with self.subTest(spelling=spelling):
                archived = self._place("p2", "done", "in")
                message = self._refuse(positional=positional)
                self.assertIn("-f", message)
                # The two ways the archived copy differs from what the round saw.
                self.assertIn("unfiltered", message)
                self.assertIn("--yes-pairs", message)

                split, _ = self._notes(force=True, positional=positional)
                split.assert_called_once_with(archived)
                self.assertTrue(archived.is_file())
                archived.unlink()

    def test_an_open_bundle_answers_to_the_source_file_it_holds(self):
        source = self._in_flight()
        split, _ = self._notes(positional=self.SOURCE)
        split.assert_called_once_with(source)

    def test_a_bundle_in_no_slot_at_all_is_reported_as_missing(self):
        for spelling, positional in self.SPELLINGS.items():
            with self.subTest(spelling=spelling):
                message = self._refuse(force=True, positional=positional)
                self.assertIn("bundle not found", message)
                # The miss is reported as typed rather than as a name the user
                # never used.
                self.assertIn(positional, message)

    def test_naming_one_of_two_queue_shapes_exactly_settles_the_ambiguity(self):
        """The reason the resolved path travels, and not just the name.

        p2's queue admits two shapes, so one bundle name matches both files.
        The bundle name cannot say which; the filename can, and that answer has
        to survive as far as the slot lookup.
        """
        candidates = self._place("p2", "queued")
        advanced = fx.write_pairs(
            fx.slot(self.opts, ["p2", "queued"]) / f"{self.BUNDLE_NAME}.pairs",
            self.PAIRS)

        # By bundle name, both shapes answer and the ambiguity is an error.
        self.assertIn("multiple", self._refuse())

        for named in (candidates, advanced):
            with self.subTest(named=named.name):
                message = self._refuse(positional=named.name)
                self.assertIn(f"wf eval p2 {named.name}", message)

    def test_yes_pairs_reaches_note_creation(self):
        self._in_flight()
        yes_pairs = fx.write_pairs(self.root / "best.pairs", ["alpha,two"])
        _, create = self._notes("--yes-pairs", str(yes_pairs))
        create.assert_called_once_with([], yes_pairs)

    def test_a_bad_yes_pairs_path_fails_before_a_single_note_is_made(self):
        self._in_flight()
        # Unlike eval, nothing here can be left half-done -- but a part
        # already raised cannot be taken back either, and the flag is not read
        # until `note --create`.
        for bad in [self.root / "no-such.pairs", self.root]:
            with self.subTest(bad=bad):
                self._refuse("--yes-pairs", str(bad))


if __name__ == "__main__":
    unittest.main()
