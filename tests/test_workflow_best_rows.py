"""Status rows, one at a time.

`derive_state` is a tuple of rows evaluated in order, so the suite addresses
them individually rather than arranging a whole tree to reach the ninth. Each
row is asserted to fire on exactly its own condition, to raise when reached out
of order, and to lose to every row above it.
"""

import os
import tempfile
import unittest

from pathlib import Path

from tests import wf_fixture as fx
from workflow import config, generation
from workflow.best import state


class RowTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.opts, _ = fx.make_wf(self.root)
        self.best = fx.slot(self.opts, ["best"])
        self.sentence_dir = self.best / "s2"
        self.dir = self.sentence_dir / "u-cdef" / "m4" / "g4"
        self.dir.mkdir(parents=True)
        self.results = self.root / "results"
        self.results.mkdir()
        self.target = state.one_target(self.opts.dir, "s2", "u-cdef", 4, 4)
        self._write(self.best / "idx" / state.INDEX_NAME, "index\n", 10)

    # ------------------------------------------------------------- placement

    def _write(self, path: Path, text: str = "", mtime=None) -> Path:
        path.write_text(text)
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        return path

    def _letters(self, mtime=10) -> Path:
        return self._write(self.sentence_dir / "letters", "abcdef\n", mtime)

    def _seed(self, mtime=10) -> Path:
        return self._write(
            self.sentence_dir / "seed.m4.idx2.85.15.pairs", "a,b\n", mtime)

    def _dfs(self, source: str, mtime=20) -> Path:
        # A symlink into results/, which is what gen publishes.
        rendered = self._write(
            self.results / f"dfs.{source}.out", "1 a b\n", mtime)
        link = self.dir / f"dfs.{source}"
        link.unlink(missing_ok=True)
        link.symlink_to(rendered)
        return link

    def _top(self, text="a,b\n", mtime=30, source: str | None = "seed",
             marker=None) -> Path:
        path = self._write(self.dir / "top.segments", text, mtime)
        if source is not None:
            self._write(generation.stamp(path), f"{source}\n",
                        mtime if marker is None else marker)
        return path

    def _best_pairs(self, text="a,b\n", mtime=50) -> Path:
        """A hand-edit: nothing generates it and it carries no marker."""
        return self._write(self.dir / "best.pairs", text, mtime)

    def _search_pairs(self, text="a,b\n", mtime=60) -> Path:
        """A legacy pre-migration dfs.best pair receipt."""
        return self._write(self.dir / "dfs.best.pairs", text, mtime)

    def _target_no(self, text="a,b\n", mtime=50) -> Path:
        """A hand-managed exclusion: nothing generates it, and it may be
        unsorted."""
        return self._write(self.dir / "no.pairs", text, mtime)

    def _dictionary(self, mtime=10) -> Path:
        """A rebuilt dictionary tree, shared by every target under the root.

        The base, the derived file, the reviewed done-set and the generation
        marker, all dated together: what `wf gen dict` leaves behind, dated
        back with the other hand-placed inputs so the fixture's small integer
        mtimes stay meaningful against a tree `wf init` created at real time.
        """
        root = self.opts.dir
        self._write(config.base_dictionary(root), "words\n", mtime)
        derived = self._write(config.dictionary(root), "words\n", mtime)
        self._write(config.reviewed_words(root), "", mtime)
        for directory in (config.removals(root),
                          config.reviewed_inputs(root)):
            os.utime(directory, (mtime, mtime))
        self._write(generation.stamp(derived), "", mtime)
        return derived

    def _classified(self, kind: str, mtime=10, text=None) -> Path:
        path = config.classified(self.root, kind)
        if text is not None:
            path.write_text(text)
        os.utime(path, (mtime, mtime))
        return path

    def _archived(self, ordinal=1, mtime=40) -> Path:
        return self._write(
            fx.slot(self.opts, ["p2", "done", "in"])
            / f"top.s2.m4.g4.u-cdef.1000.r{ordinal}.pairs", "a,b\n", mtime)

    def _steady(self) -> None:
        """The fixed point every row below is perturbed away from.

        A target-local BEST promotion makes the refine stage distinct from the
        sentence search. Its successful completion is recorded by the marker;
        the legacy dfs.best.pairs file is deliberately irrelevant.
        """
        self._letters()
        self._seed()
        self._dictionary()
        self._classified("yes", text="a,b\n")
        self._classified("no")
        self._dfs("seed", 20)
        self._top(mtime=30, marker=70)
        self._archived(mtime=40)
        self._best_pairs(mtime=10)
        self._dfs("best", 60)
        self._write(generation.stamp(self.dir / "dfs.best"), "", 60)
        self._search_pairs("a,b\n")

    def _inputs(self) -> state.Inputs:
        return state.Inputs(self.target)

    def _state(self) -> state.State:
        return state.derive_state(self.target)

    def _require_state(self, result: state.State | None) -> state.State:
        """Assert that the directly tested status row fired."""
        self.assertIsNotNone(result)
        assert result is not None
        return result

    def _commands(self, result: state.State | None) -> dict:
        result = self._require_state(result)
        return {choice.label: choice.command for choice in result.choices}

    # ------------------------------------------------------------------ rows

    def test_the_steady_state_is_converged(self):
        self._steady()
        self.assertEqual("converged", self._state().message)
        for row in state.ROWS:
            with self.subTest(row=row.name):
                self.assertIsNone(row.check(self._inputs()))

    def test_letters_and_seed_report_absence_and_raise_on_a_bad_file(self):
        self.assertEqual(
            state.State("letters missing", place=self.target.letters),
            state._letters_missing(self._inputs()))

        (self.sentence_dir / "letters").mkdir()
        with self.assertRaises(ValueError):
            state._letters_missing(self._inputs())
        (self.sentence_dir / "letters").rmdir()

        self._letters()
        self.assertIsNone(state._letters_missing(self._inputs()))
        self.assertEqual(
            state.State("seed missing", place=self.target.seed_glob),
            state._seed_missing(self._inputs()))
        self._seed()
        self.assertIsNone(state._seed_missing(self._inputs()))

    def test_no_frontier_reports_each_way_the_search_results_can_stand(self):
        self._letters()
        self._seed()

        # Nothing has been searched: the one bootstrap gate.
        result = self._require_state(state._no_frontier(self._inputs()))
        self.assertEqual("no search results yet", result.message)
        self.assertEqual(
            {"next": "wf best prepare s2 -u cdef -g 4 --source seed"},
            self._commands(result))

        # A dangling link is still nothing searched, and says which one.
        missing = self.root / "gone" / "dfs.out"
        (self.dir / "dfs.seed").symlink_to(missing)
        self.assertEqual(
            f"no search results yet (dangling symlink: {missing})",
            self._require_state(state._no_frontier(self._inputs())).message)
        (self.dir / "dfs.seed").unlink()

        for source in ("seed", "best"):
            with self.subTest(source=source):
                if source == "best":
                    self._best_pairs(mtime=10)
                self._dfs(source)
                result = self._require_state(state._no_frontier(self._inputs()))
                self.assertEqual("top.segments missing", result.message)
                self.assertEqual(
                    {"next": f"wf best gen s2 -u cdef -g 4 top.segments "
                             f"--source {source}"},
                    self._commands(result))
                (self.dir / f"dfs.{source}").unlink()
                if source == "best":
                    (self.dir / "best.pairs").unlink()

        self._dfs("seed")
        self._dfs("best")
        self._best_pairs(mtime=10)
        result = self._require_state(state._no_frontier(self._inputs()))
        self.assertEqual("top.segments missing", result.message)
        self.assertEqual(
            {"seed": "wf best gen s2 -u cdef -g 4 top.segments --source seed",
             "best": "wf best gen s2 -u cdef -g 4 top.segments --source best"},
            self._commands(result))

        # Deleting both target-local sources makes a leftover dfs.best
        # inapplicable as a frontier source.
        (self.dir / "dfs.seed").unlink()
        (self.dir / "best.pairs").unlink()
        result = self._require_state(state._no_frontier(self._inputs()))
        self.assertEqual("no search results yet", result.message)
        self.assertEqual(["next"], [choice.label for choice in result.choices])

        self._top()
        self.assertIsNone(state._no_frontier(self._inputs()))

    def test_review_needed_names_the_frontier_source(self):
        self._steady()
        # A frontier written after the last archived round is unreviewed.
        os.utime(self.dir / "top.segments", (45, 45))
        result = self._require_state(state._review_needed(self._inputs()))
        self.assertEqual("review needed (frontier from seed)", result.message)
        self.assertEqual({"next": "wf best review s2 -u cdef -g 4"},
                         self._commands(result))

        self._top(mtime=45, source="best", marker=70)
        self.assertEqual("review needed (frontier from best)",
                         self._require_state(
                             state._review_needed(self._inputs())).message)

    def test_a_dictionary_newer_than_the_frontier_skips_review(self):
        self._steady()
        # The frontier is otherwise unreviewed: it is newer than the archived
        # top round. A still newer dictionary makes that frontier obsolete.
        os.utime(self.dir / "top.segments", (45, 45))
        self._dictionary(90)
        self.assertIsNone(state._review_needed(self._inputs()))
        self.assertEqual(
            "top.segments behind its inputs (dictionary changed)",
            self._state().message)

    def test_an_open_oneoff_declines_top_rows_and_redirects_review_needed(self):
        self._steady()
        os.utime(self.dir / "top.segments", (45, 45))
        bundle = (fx.slot(self.opts, ["p2", "eval"])
                  / "oneoff.s2.m4.g4.u-cdef.1.r1")
        bundle.mkdir()
        self._write(bundle / f"{bundle.name}.pairs", "one,off\n")
        self._write(bundle / f"{bundle.name}.pairs.filtered", "one,off\n")

        inputs = self._inputs()
        self.assertIsNone(state._review_queued(inputs))
        self.assertIsNone(state._review_evaluating(inputs))
        result = self._require_state(state._review_needed(inputs))
        self.assertEqual("review needed (frontier from seed)", result.message)
        self.assertEqual(
            {"next": "wf best complete s2 -u cdef -g 4"},
            self._commands(result))

    def test_an_open_oneoff_does_not_hide_a_search_row(self):
        self._steady()
        bundle = (fx.slot(self.opts, ["p2", "eval"])
                  / "oneoff.s2.m4.g4.u-cdef.1.r1")
        bundle.mkdir()
        self._write(bundle / f"{bundle.name}.pairs", "one,off\n")
        self._write(bundle / f"{bundle.name}.pairs.filtered", "one,off\n")
        (self.dir / "dfs.best").unlink()

        result = self._state()
        self.assertEqual("dfs.best missing", result.message)
        self.assertEqual(
            {"refine": "wf best prepare s2 -u cdef -g 4 --source best"},
            self._commands(result))

    def test_legacy_dfs_best_pairs_is_ignored(self):
        self._steady()
        self._search_pairs("completely,different\n", mtime=90)
        self.assertEqual([], self._inputs().best_search_needed)
        self.assertEqual("converged", self._state().message)

    def test_top_segments_behind_ignores_a_dfs_file_that_is_itself_stale(self):
        self._steady()
        # A finished seed search the frontier was never generated from.
        os.utime(self.results / "dfs.seed.out", (80, 80))
        result = self._require_state(
            state._top_segments_behind_dfs(self._inputs()))
        self.assertEqual("dfs.seed generated after top.segments",
                         result.message)
        self.assertEqual(
            {"next": "wf best gen s2 -u cdef -g 4 top.segments --source seed"},
            self._commands(result))

        # Same file, now out of date itself: it wants re-running, not reading.
        self._seed(mtime=85)
        self.assertIsNone(state._top_segments_behind_dfs(self._inputs()))

        self._seed(mtime=10)
        os.utime(self.results / "dfs.best.out", (80, 80))
        result = self._require_state(
            state._top_segments_behind_dfs(self._inputs()))
        self.assertEqual("dfs.seed and dfs.best generated after top.segments",
                         result.message)
        self.assertEqual(
            {"seed": "wf best gen s2 -u cdef -g 4 top.segments --source seed",
             "best": "wf best gen s2 -u cdef -g 4 top.segments --source best"},
            self._commands(result))

    def test_next_search_offers_both_sides_when_a_round_moved_the_no_set(self):
        self._steady()
        self.assertIsNone(state._next_search(self._inputs()))

        # Completing a round folds new verdicts into the hard-NO set, which
        # puts both searches behind at once -- the ordinary steady state.
        self._classified("no", 90)
        result = self._require_state(state._next_search(self._inputs()))
        self.assertEqual(
            "dfs.seed out of date (hard-NO set changed); "
            "dfs.best out of date (hard-NO set changed)", result.message)
        self.assertEqual(
            {"reseed": "wf best prepare s2 -u cdef -g 4 --source seed",
             "refine": "wf best prepare s2 -u cdef -g 4 --source best"},
            self._commands(result))

        # An absent DFS file reads as missing rather than as out of date.
        self._classified("no", 10)
        (self.dir / "dfs.best").unlink()
        result = self._require_state(state._next_search(self._inputs()))
        self.assertEqual("dfs.best missing", result.message)
        self.assertEqual(["refine"],
                         [choice.label for choice in result.choices])

        # Pair contents are Nutrimatic's concern; the missing target-local
        # search remains actionable regardless of what this bag can spell.
        self._classified("yes", text="x,y\n")
        result = self._require_state(state._next_search(self._inputs()))
        self.assertEqual("dfs.best missing", result.message)

    def test_the_frontier_falls_behind_a_classify_until_a_regen(self):
        self._steady()
        self._classified("yes", 90, text="a,b\n")
        result = self._require_state(state._frontier_outdated(self._inputs()))
        self.assertEqual(
            "top.segments behind its inputs "
            "(confirmed-YES set changed)", result.message)
        # The regeneration is offered from the source the frontier already
        # records, not from a choice of both.
        self.assertEqual(
            {"next": "wf best gen s2 -u cdef -g 4 top.segments --source seed"},
            self._commands(result))

        self._classified("no", 95)
        self.assertEqual(
            "top.segments behind its inputs (confirmed-YES set "
            "changed, hard-NO set changed)",
            self._require_state(
                state._frontier_outdated(self._inputs())).message)

        # The marker is what clears it, and a byte-identical regeneration
        # still advances the marker -- which is what terminates the loop the
        # content clock could not.
        generation.mark_generated(self.dir / "top.segments", "seed\n")
        self.assertIsNone(state._frontier_outdated(self._inputs()))

    def test_a_dictionary_edit_makes_the_frontier_outdated(self):
        """Regenerating applies the changed dictionary to the recorded DFS.

        top-segments rejects a whole result row holding a segment whose word
        the dictionary no longer has, so the regen this row offers genuinely
        drops what the dictionary lost -- which is why the operator is offered
        the seconds here instead of being billed the hours of a re-search.
        """
        self._steady()
        self._dictionary(90)
        result = self._require_state(state._frontier_outdated(self._inputs()))
        self.assertEqual(
            "top.segments behind its inputs (dictionary changed)",
            result.message)
        self.assertEqual(
            {"next": "wf best gen s2 -u cdef -g 4 top.segments --source seed"},
            self._commands(result))

        generation.mark_generated(self.dir / "top.segments", "seed\n")
        self.assertIsNone(state._frontier_outdated(self._inputs()))

        # A dictionary placed before the frontier was made is the ordinary
        # case, and says nothing.
        self._dictionary(10)
        self.assertIsNone(state._frontier_outdated(self._inputs()))

    def test_the_g0_guards_require_both_dictionary_files(self):
        """Neither is optional now: four tools read the derived file.

        The base is hand-placed and the derived file is what `wf gen dict`
        makes of it. A status that dated a frontier against a file that is not
        there would be reporting on a search nothing could run, so both get the
        ordinary file-not-found diagnostic rather than a state of their own.
        """
        self._steady()
        self.assertIsNone(state._require_base_dictionary(self._inputs()))
        self.assertIsNone(state._require_dictionary(self._inputs()))

        config.dictionary(self.root).unlink()
        with self.assertRaises(FileNotFoundError):
            state._require_dictionary(self._inputs())
        # Present under that name and not a regular file is a broken tree,
        # the way a best.pairs that is not a file is.
        config.dictionary(self.root).mkdir()
        with self.assertRaises(ValueError):
            state._require_dictionary(self._inputs())

        config.base_dictionary(self.root).unlink()
        with self.assertRaises(FileNotFoundError):
            state._require_base_dictionary(self._inputs())

    # ------------------------------------------------------ dictionary stale

    def _stale(self) -> state.State | None:
        return state._dictionary_stale(self._inputs())

    def test_a_generation_newer_than_the_marker_offers_a_rebuild(self):
        self._steady()
        self.assertIsNone(self._stale())

        # A round recorded after the last rebuild: the file and the directory
        # both moved, which is the ordinary shape of an added record.
        record = self._write(
            config.removals(self.root) / "junk.txt.removed.1", "junk\n", 90)
        os.utime(record.parent, (90, 90))
        result = self._require_state(self._stale())
        self.assertEqual("dictionary behind its records (removals changed)",
                         result.message)
        self.assertEqual({"next": "wf gen dict"}, self._commands(result))

    def test_an_edit_in_place_is_caught_by_the_file_clock(self):
        """The retraction path, and the reason both clocks are read.

        Trimming a word out of a generation leaves the directory's own mtime
        where it was: only the file moved.
        """
        self._steady()
        record = self._write(
            config.removals(self.root) / "junk.txt.removed.1", "junk\n", 10)
        os.utime(record.parent, (10, 10))
        self.assertIsNone(self._stale())

        self._write(record, "", 90)
        os.utime(record.parent, (10, 10))
        self.assertEqual("dictionary behind its records (removals changed)",
                         self._require_state(self._stale()).message)

    def test_a_replaced_base_is_the_likeliest_real_trigger(self):
        """The base may be a symlink whose target the operator replaces."""
        self._steady()
        self._write(config.base_dictionary(self.root), "other\n", 90)
        self.assertEqual(
            "dictionary behind its records (base dictionary changed)",
            self._require_state(self._stale()).message)

    def test_a_reviewed_input_dates_the_derived_done_set(self):
        self._steady()
        record = self._write(
            config.reviewed_inputs(self.root) / "junk.txt.reviewed.1",
            "junk\n", 90)
        os.utime(record.parent, (90, 90))
        self.assertEqual(
            "dictionary behind its records (reviewed inputs changed)",
            self._require_state(self._stale()).message)

    def test_a_missing_reviewed_done_set_is_a_repairable_state(self):
        """Not a required input: its absence is what this row reports."""
        self._steady()
        config.reviewed_words(self.root).unlink()
        self.assertEqual(
            "dictionary behind its records (reviewed words missing)",
            self._require_state(self._stale()).message)

        # Present and not a regular file is a broken tree either way.
        config.reviewed_words(self.root).mkdir()
        with self.assertRaises(ValueError):
            self._stale()

    def test_one_rebuild_clears_the_row_even_when_it_changes_nothing(self):
        """stable_mtime pins the file; the marker is what moves regardless."""
        self._steady()
        self._write(config.base_dictionary(self.root), "words\n", 90)
        # Above the searches, so the seconds are offered before the hours.
        self.assertEqual(
            "dictionary behind its records (base dictionary changed)",
            self._state().message)

        code, _, stderr = fx.run_wf("-d", str(self.root), "gen", "dict")
        self.assertEqual(0, code, stderr)
        self.assertIsNone(self._stale())
        self.assertEqual("converged", self._state().message)

    def test_an_effective_removal_dates_both_searches(self):
        """The derived file's own mtime, not the marker: hours ride on it."""
        self._steady()
        inputs = self._inputs()
        self.assertEqual([], inputs.seed_search_needed)
        self.assertEqual([], inputs.best_search_needed)

        # A rebuild that changed nothing moves only the marker.
        self._write(generation.stamp(config.dictionary(self.root)), "", 90)
        inputs = self._inputs()
        self.assertEqual([], inputs.seed_search_needed)
        self.assertEqual([], inputs.best_search_needed)

        # A rebuild that removed a word moves the file itself.
        self._write(config.dictionary(self.root), "word\n", 90)
        inputs = self._inputs()
        self.assertEqual(["dictionary changed"], inputs.seed_search_needed)
        self.assertEqual(["dictionary changed"], inputs.best_search_needed)

    def test_declared_global_sources_date_both_searches(self):
        self._steady()
        sources = (
            (self._seed(), "seed changed"),
            (self._letters(), "letters changed"),
            (self.best / "idx" / state.INDEX_NAME, "index changed"),
            (self._dictionary(), "dictionary changed"),
            (self._classified("yes", text="a,b\n"),
             "confirmed-YES set changed"),
            (self._classified("no"), "hard-NO set changed"),
        )
        for path, reason in sources:
            with self.subTest(reason=reason):
                os.utime(path, (90, 90))
                inputs = self._inputs()
                self.assertIn(reason, inputs.seed_search_needed)
                self.assertIn(reason, inputs.best_search_needed)
                os.utime(path, (10, 10))

    def test_a_legacy_best_without_a_marker_is_stale_once(self):
        self._steady()
        marker = generation.stamp(self.dir / "dfs.best")
        marker.unlink()
        self.assertEqual(["generation marker missing"],
                         self._inputs().best_search_needed)
        generation.mark_generated(self.dir / "dfs.best")
        self.assertEqual([], self._inputs().best_search_needed)

    def test_a_review_is_still_offered_after_a_removal(self):
        """The frontier is unreviewed even where its bytes did not move.

        Any removal that actually removes a word makes the dictionary newer
        than top.segments, so a content-clock comparison would suppress this
        target's review indefinitely. The marker is what the row dates against.
        """
        self._steady()
        # A frontier written after the round that last read it, then a removal
        # that landed after the frontier's bytes but before its rebuild.
        self._top(mtime=50, marker=60)
        self._archived(mtime=40)
        self._write(config.dictionary(self.root), "word\n", 55)
        self.assertEqual("review needed (frontier from seed)",
                         self._require_state(
                             state._review_needed(self._inputs())).message)

        # A removal after the rebuild does make the frontier obsolete, and the
        # row that offers the regeneration is the one that wins.
        self._write(config.dictionary(self.root), "word\n", 70)
        self.assertIsNone(state._review_needed(self._inputs()))

    def test_the_stale_dictionary_yields_to_every_row_above_it(self):
        self._steady()
        self._write(config.base_dictionary(self.root), "words\n", 90)
        self.assertTrue(self._stale())

        # An unreviewed frontier, a missing one, and an open review each win.
        self._top(mtime=95, marker=95)
        self.assertEqual("review needed (frontier from seed)",
                         self._state().message)

        (self.dir / "top.segments").unlink()
        self.assertEqual("top.segments missing", self._state().message)

        self._top(mtime=30, marker=70)
        queued = self._write(
            fx.slot(self.opts, ["p2", "queued"])
            / "top.s2.m4.g4.u-cdef.1000.r2.pairs", "a,b\n")
        self.assertEqual(f"review submitted ({queued.name})",
                         self._state().message)

    # ---------------------------------------------------------------- guards

    def test_a_row_reached_out_of_order_raises_rather_than_dating_nothing(self):
        self._letters()
        self._seed()
        self._dictionary()
        self._dfs("seed")
        self._classified("yes", text="a,b\n")
        self._classified("no")
        for row in (state._review_needed, state._frontier_outdated):
            with self.subTest(row=row.__name__):
                with self.assertRaises(FileNotFoundError):
                    row(self._inputs())

        self._top()
        self.assertIsNone(state._frontier_outdated(self._inputs()))

        # best.pairs is optional, but a non-file under that name is an error.
        self.dir.joinpath("best.pairs").mkdir()
        with self.assertRaises(ValueError):
            _ = self._inputs().best_search_needed

    def test_search_conditions_refuse_to_answer_without_a_seed(self):
        self._letters()
        self._dfs("seed")
        with self.assertRaisesRegex(FileNotFoundError, "seed missing"):
            _ = self._inputs().seed_search_needed

    # ------------------------------------------------------------ precedence

    def test_rows_are_evaluated_in_the_documented_precedence(self):
        self.assertEqual(
            ["_letters_missing", "_seed_missing",
             "_require_base_dictionary", "_require_dictionary",
             "_review_queued", "_review_evaluating",
             "_no_frontier",
             "_review_needed",
             "_top_segments_behind_dfs",
             "_frontier_outdated",
             "_dictionary_stale",
             "_next_search"],
            [row.name for row in state.ROWS])

    def test_an_open_review_outranks_every_derived_row(self):
        self._steady()
        # Both searches behind and a finished search unread, so rows 10 and 11
        # would both fire -- and the review gate refuses the commands they
        # would print, so neither may.
        self._classified("no", 90)
        queued = self._write(
            fx.slot(self.opts, ["p2", "queued"])
            / "top.s2.m4.g4.u-cdef.1000.r2.pairs", "a,b\n")
        self.assertEqual(f"review submitted ({queued.name})",
                         self._state().message)

        evaluating = fx.slot(self.opts, ["p2", "eval"]) / queued.stem
        evaluating.mkdir()
        queued.rename(evaluating / queued.name)
        self.assertEqual(f"review awaiting completion ({evaluating.name})",
                         self._state().message)

    def test_an_unreviewed_frontier_outranks_a_finished_search(self):
        self._steady()
        os.utime(self.results / "dfs.seed.out", (80, 80))
        os.utime(self.dir / "top.segments", (45, 45))
        self.assertEqual("review needed (frontier from seed)",
                         self._state().message)

    def test_a_classify_stales_a_search_before_its_frontier_is_reused(self):
        self._steady()
        # Classified YES is now a declared search source, so the finished
        # search is itself stale and is not offered as a frontier source.
        self._classified("yes", 90, text="a,b\n")
        os.utime(self.results / "dfs.seed.out", (80, 80))
        self.assertTrue(self._inputs().frontier_outdated)
        self.assertEqual(
            "top.segments behind its inputs (confirmed-YES set changed)",
            self._state().message)

    def test_unspellable_pair_contents_do_not_block_search_choices(self):
        self._steady()
        self._classified("yes", 90, text="x,y\n")
        result = self._require_state(state._next_search(self._inputs()))
        self.assertEqual(["reseed", "refine"],
                         [choice.label for choice in result.choices])

    def test_target_source_presence_controls_whether_refine_exists(self):
        self._steady()
        (self.dir / "best.pairs").unlink()
        os.utime(self.results / "dfs.best.out", (90, 90))
        self.assertFalse((self.dir / "best.pairs").exists())
        self.assertEqual("converged", self._state().message)
        self.assertFalse(self._inputs().top_segments_behind("best"))

        self._best_pairs("b,a\n", mtime=90)
        self.assertEqual("dfs.best out of date (target-BEST set changed)",
                         self._state().message)

    def test_removing_the_last_target_source_invalidates_a_best_frontier(self):
        for source in ("best.pairs", "no.pairs"):
            with self.subTest(source=source):
                self._steady()
                (self.dir / "best.pairs").unlink()
                if source == "best.pairs":
                    self._best_pairs(mtime=10)
                else:
                    self._target_no(mtime=10)
                self._top(mtime=30, source="best", marker=70)

                (self.dir / source).unlink()
                inputs = self._inputs()
                self.assertIsNone(state._review_needed(inputs))
                result = self._require_state(state._frontier_outdated(inputs))
                self.assertEqual(
                    "top.segments behind its inputs "
                    "(target-local sources removed)", result.message)
                self.assertEqual(
                    {"next": "wf best gen s2 -u cdef -g 4 top.segments "
                             "--source seed"},
                    self._commands(result))
                self.assertEqual(result, self._state())

                self.dir.joinpath("best.pairs").unlink(missing_ok=True)
                self.dir.joinpath("no.pairs").unlink(missing_ok=True)

    # ------------------------------------------------- target-local no.pairs

    def test_a_local_exclusion_does_not_change_global_verdicts(self):
        self._steady()
        self._target_no("a,b\n", mtime=50)
        self.assertEqual(
            "a,b\n", config.classified(self.root, "yes").read_text())
        self.assertEqual([], self._inputs().seed_search_needed)

    def test_local_exclusion_contents_are_not_normalized_by_status(self):
        self._steady()
        local = self._target_no("b,a\na,b\nb,a\n", mtime=50)
        self.assertEqual([], self._inputs().best_search_needed)
        self.assertEqual("b,a\na,b\nb,a\n", local.read_text())

    def test_a_newer_local_exclusion_dates_only_the_complete_search(self):
        self._steady()
        # Excluding a pair the bag cannot spell leaves a usable bonus set, so
        # both searches are still worth offering -- and both are now behind.
        self._classified("yes", text="a,b\ntiger,lily\n")
        self._target_no("tiger,lily\n", mtime=90)
        inputs = self._inputs()
        self.assertEqual([], inputs.seed_search_needed)
        self.assertEqual(["target-NO set changed"], inputs.best_search_needed)

        # And an older one says nothing at all.
        self._target_no("tiger,lily\n", mtime=10)
        inputs = self._inputs()
        self.assertEqual([], inputs.seed_search_needed)
        self.assertEqual([], inputs.best_search_needed)
        self.assertEqual("converged", self._state().message)

    def test_a_newer_local_exclusion_dates_the_frontier_and_defers_review(self):
        self._steady()
        # An unreviewed frontier: newer than the archived round that read it.
        self._top(mtime=45, marker=45)
        self.assertEqual("review needed (frontier from seed)",
                         self._state().message)

        self._classified("yes", text="a,b\ntiger,lily\n")
        self._target_no("tiger,lily\n", mtime=90)
        # The frontier is obsolete rather than unreviewed, so the review row
        # stands down and the regeneration is what is offered.
        self.assertIsNone(state._review_needed(self._inputs()))
        result = self._require_state(state._frontier_outdated(self._inputs()))
        self.assertEqual(
            "top.segments behind its inputs (target-NO set changed)",
            result.message)
        self.assertEqual(
            {"next": "wf best gen s2 -u cdef -g 4 top.segments --source seed"},
            self._commands(result))

    def test_a_no_op_regeneration_reopens_a_genuinely_unreviewed_frontier(self):
        """The hole the content clock leaves, and why the marker closes it.

        Excluding a pair that is not on the frontier makes the regeneration a
        byte-for-byte no-op: stable_mtime leaves top.segments' own mtime
        where it was, so a content-clock comparison would suppress the review
        forever. The marker advances regardless, which is what lets the row
        fire once the regeneration has actually happened.
        """
        self._steady()
        self._top(mtime=45, marker=45)
        self._classified("yes", text="a,b\ntiger,lily\n")
        self._target_no("tiger,lily\n", mtime=90)
        self.assertIsNone(state._review_needed(self._inputs()))

        # A no-op regen: the marker moves, the content mtime does not.
        generation.mark_generated(self.dir / "top.segments", "seed\n")
        self.assertEqual(45, int((self.dir / "top.segments").stat().st_mtime))
        self.assertEqual("review needed (frontier from seed)",
                         self._state().message)

    def test_a_no_op_regeneration_does_not_reopen_a_reviewed_frontier(self):
        self._steady()
        self._classified("yes", text="a,b\ntiger,lily\n")
        self._search_pairs("a,b\n")
        self._target_no("tiger,lily\n", mtime=90)
        generation.mark_generated(self.dir / "top.segments", "seed\n")
        # The archived round is still newer than the frontier it read, so the
        # frontier has been reviewed and stays reviewed. Only the complete
        # search loads this target-local source.
        self.assertIsNone(state._review_needed(self._inputs()))
        self.assertEqual(
            "dfs.best out of date (target-NO set changed)",
            self._state().message)

    def test_local_exclusion_contents_do_not_create_a_dead_end_state(self):
        self._steady()
        self._top("a,b\nc,d\n", mtime=30, source="best", marker=70)
        self._target_no("a,b\n", mtime=90)

        result = self._state()
        self.assertEqual(
            "top.segments behind its inputs (target-NO set changed)",
            result.message)
        generation.mark_generated(self.dir / "top.segments", "best\n")
        result = self._state()
        self.assertEqual("dfs.best out of date (target-NO set changed)",
                         result.message)
        self.assertEqual(["refine"],
                         [choice.label for choice in result.choices])

    def test_an_open_review_outranks_a_newer_local_exclusion(self):
        self._steady()
        self._target_no("tiger,lily\n", mtime=90)
        queued = self._write(
            fx.slot(self.opts, ["p2", "queued"])
            / "top.s2.m4.g4.u-cdef.1000.r2.pairs", "a,b\n")
        self.assertEqual(f"review submitted ({queued.name})",
                         self._state().message)

    def test_a_local_exclusion_that_is_not_a_file_is_rejected(self):
        self._steady()
        (self.dir / "no.pairs").mkdir()
        with self.assertRaisesRegex(ValueError, "not a regular file"):
            self._inputs().target_no
        (self.dir / "no.pairs").rmdir()

        missing = self.root / "gone" / "no.pairs"
        (self.dir / "no.pairs").symlink_to(missing)
        with self.assertRaises(FileNotFoundError):
            self._inputs().target_no

    # ----------------------------------------------------- shared renderers

    def test_classified_yes_change_offers_both_searches(self):
        self._steady()
        self._classified("yes", 90, text="x,y\n")
        following = self._commands(state._next_search(self._inputs()))
        self.assertEqual(["reseed", "refine"], list(following))

    def test_both_rows_that_offer_a_generation_render_the_same_command(self):
        self._letters()
        self._seed()
        self._dictionary()
        self._dfs("seed")
        absent = self._commands(state._no_frontier(self._inputs()))

        self._classified("yes", text="a,b\n")
        self._classified("no")
        self._top(mtime=30, marker=30)
        self._archived(mtime=40)
        os.utime(self.results / "dfs.seed.out", (80, 80))
        behind = self._commands(state._top_segments_behind_dfs(self._inputs()))
        self.assertEqual(absent["next"], behind["next"])

    # -------------------------------------------------------------- markers

    def test_the_source_marker_is_read_leniently_and_never_guessed(self):
        self._top(source=None)
        marker = generation.stamp(self.dir / "top.segments")
        self.assertFalse(marker.exists())
        # A tree built before the frontier had a choice of source.
        self.assertEqual("seed", state.top_segments_source(self.target))

        marker.write_text("")
        self.assertEqual("seed", state.top_segments_source(self.target))
        marker.write_text("best\n")
        self.assertEqual("best", state.top_segments_source(self.target))
        marker.write_text("dfs.best\n")
        with self.assertRaisesRegex(ValueError, "unrecognised"):
            state.top_segments_source(self.target)

    # -------------------------------------------------------------- render

    def test_choices_render_singly_by_label_and_severally_under_a_heading(self):
        self.assertEqual([], state.render_choices(()))
        self.assertEqual(
            ["  next: wf best review s2 -u cdef -g 4"],
            state.render_choices((state.Choice(
                "next", "wf best review s2 -u cdef -g 4"),)))
        self.assertEqual(
            ["  widen: a"],
            state.render_choices((state.Choice("widen", "a"),)))
        self.assertEqual(
            ["  choose next:",
             "    widen:  a",
             "    reseed: b"],
            state.render_choices((state.Choice("widen", "a"),
                                  state.Choice("reseed", "b"))))

    # ---------------------------------------------------------------- walk

    def _labels(self) -> dict:
        verdicts = state.walk_rows(self.target)
        return {verdict.row.name: verdict for verdict in verdicts}

    def test_the_walk_asks_every_row_in_the_steady_state(self):
        self._steady()
        verdicts = state.walk_rows(self.target)
        self.assertEqual([row.name for row in state.ROWS],
                         [verdict.row.name for verdict in verdicts])
        for verdict in verdicts:
            with self.subTest(row=verdict.row.name):
                self.assertEqual((), verdict.unmet)
                self.assertFalse(verdict.fired)

    def test_the_walk_skips_the_rows_a_missing_file_would_raise_in(self):
        """The whole point of the requires table: no row reads what is absent.

        Straight iteration over ROWS here raises in _review_needed, which is
        what the walk exists to avoid. The dictionary is placed because its own
        guards are not part of that table: they are required inputs whose
        absence is an error at every row, not a file one row establishes for
        another.
        """
        self._dictionary()
        verdicts = self._labels()
        self.assertTrue(verdicts["_letters_missing"].fired)
        self.assertTrue(verdicts["_seed_missing"].fired)
        self.assertTrue(verdicts["_no_frontier"].fired)
        self.assertEqual(("top.segments",), verdicts["_review_needed"].unmet)
        self.assertEqual(("top.segments", "seed"),
                         verdicts["_top_segments_behind_dfs"].unmet)
        self.assertEqual(("top.segments",),
                         verdicts["_frontier_outdated"].unmet)
        self.assertEqual(("seed",), verdicts["_next_search"].unmet)
        # A row whose own condition needs nothing is still asked.
        self.assertEqual((), verdicts["_review_queued"].unmet)
        self.assertFalse(verdicts["_review_queued"].fired)

    def test_a_row_is_asked_once_the_row_providing_its_file_declines(self):
        self._letters()
        self._seed()
        self._dictionary()
        self._dfs("seed")
        self._top()
        verdicts = self._labels()
        self.assertEqual((), verdicts["_review_needed"].unmet)
        self.assertEqual((), verdicts["_top_segments_behind_dfs"].unmet)
        self.assertEqual((), verdicts["_frontier_outdated"].unmet)

    def test_the_walk_agrees_with_derive_state_down_to_the_winner(self):
        self._steady()
        # A frontier newer than the archived bundle that reviewed it, and a
        # finished dfs.best it was never generated from: rows 6 and 8 both
        # hold.
        self._top(mtime=55, marker=55)
        won = [verdict for verdict in state.walk_rows(self.target)
               if verdict.fired]
        self.assertEqual("_review_needed", won[0].row.name)
        self.assertEqual(self._state(), won[0].state)
        # And it keeps going: the finished search is unread too, which is the
        # diagnostic derive_state cannot show.
        self.assertEqual(["_review_needed", "_top_segments_behind_dfs"],
                         [verdict.row.name for verdict in won])

    def test_the_table_names_the_winner_apart_from_what_also_fired(self):
        self._steady()
        self._top(mtime=55, marker=55)
        lines = state.render_rows(state.walk_rows(self.target))
        self.assertEqual("  rows:", lines[0])
        rendered = {line.split()[1]: line.split(maxsplit=1)[0].rstrip(":")
                    for line in lines[1:] if line.split()[0].endswith(":")
                    and line.split()[1].startswith("_")}
        self.assertEqual("won", rendered["_review_needed"])
        self.assertEqual("also", rendered["_top_segments_behind_dfs"])
        self.assertEqual("no", rendered["_letters_missing"])

    def test_a_fired_row_carries_the_commands_it_offers(self):
        """Both fired rows print theirs, in the report's own rendering.

        The winner's line repeats what report printed above the table; the
        `also` row's is the one the table alone can show, indented under the
        row rather than beside the winner's.
        """
        self._steady()
        self._top(mtime=55, marker=55)
        lines = state.render_rows(state.walk_rows(self.target))
        winner = lines.index(next(
            line for line in lines if "won:" in line))
        self.assertEqual("      next: wf best review s2 -u cdef -g 4",
                         lines[winner + 1])
        also = lines.index(next(line for line in lines if "also:" in line))
        self.assertEqual(
            "      next: wf best gen s2 -u cdef -g 4 top.segments "
            "--source best",
            lines[also + 1])

        # A declined row prints nothing, and a row never asked prints nothing
        # -- neither has a state to offer commands from.
        self.assertEqual(
            len(state.ROWS) + 1 + 2,
            len(state.render_rows(state.walk_rows(self.target))))

        # Several alternatives keep the rendering report gives them.
        self._classified("no", 90)
        lines = state.render_rows(state.walk_rows(self.target))
        following = lines.index(next(
            line for line in lines if "_next_search" in line))
        self.assertEqual(
            ["      choose next:",
             "        reseed: wf best prepare s2 -u cdef -g 4 --source seed",
             "        refine: wf best prepare s2 -u cdef -g 4 --source best"],
            lines[following + 1:following + 4])

    def test_the_table_says_which_file_a_skipped_row_wanted(self):
        self._dictionary()
        lines = state.render_rows(state.walk_rows(self.target))
        line = next(line for line in lines if "_review_needed" in line)
        self.assertIn("n/a:", line)
        self.assertIn("not asked (needs top.segments)", line)


if __name__ == "__main__":
    unittest.main()
