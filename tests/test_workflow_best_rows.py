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
from workflow import config
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

    def _top(self, text="a,b\n", mtime=30, source="seed", marker=None) -> Path:
        path = self._write(self.dir / "top.segments", text, mtime)
        if source is not None:
            self._write(state._stamp(path), f"{source}\n",
                        mtime if marker is None else marker)
        return path

    def _best_pairs(self, text="a,b\n", mtime=50, marker=None) -> Path:
        path = self._write(self.dir / "best.pairs", text, mtime)
        self._write(state._stamp(path), "", mtime if marker is None else marker)
        return path

    def _classified(self, kind: str, mtime=10) -> Path:
        path = config.classified(self.root, kind)
        os.utime(path, (mtime, mtime))
        return path

    def _archived(self, ordinal=1, mtime=40) -> Path:
        return self._write(
            fx.slot(self.opts, ["p2", "done", "in"])
            / f"top.s2.m4.g4.u-cdef.1000.r{ordinal}.pairs", "a,b\n", mtime)

    def _steady(self) -> None:
        """The fixed point every row below is perturbed away from."""
        self._letters()
        self._seed()
        self._classified("yes")
        self._classified("no")
        self._dfs("seed", 20)
        self._top(mtime=30, marker=70)
        self._archived(mtime=40)
        self._best_pairs(mtime=50)
        self._dfs("best", 60)

    def _inputs(self) -> state.Inputs:
        return state.Inputs(self.target)

    def _state(self) -> state.State:
        return state.derive_state(self.target)

    def _commands(self, result: state.State) -> dict:
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
        result = state._no_frontier(self._inputs())
        self.assertEqual("no search results yet", result.message)
        self.assertEqual(
            {"next": "wf best prepare s2 -u cdef -g 4 --source seed"},
            self._commands(result))

        # A dangling link is still nothing searched, and says which one.
        missing = self.root / "gone" / "dfs.out"
        (self.dir / "dfs.seed").symlink_to(missing)
        self.assertEqual(
            f"no search results yet (dangling symlink: {missing})",
            state._no_frontier(self._inputs()).message)
        (self.dir / "dfs.seed").unlink()

        for source in ("seed", "best"):
            with self.subTest(source=source):
                self._dfs(source)
                result = state._no_frontier(self._inputs())
                self.assertEqual("top.segments missing", result.message)
                self.assertEqual(
                    {"next": f"wf best gen s2 -u cdef -g 4 top.segments "
                             f"--source {source}"},
                    self._commands(result))
                (self.dir / f"dfs.{source}").unlink()

        self._dfs("seed")
        self._dfs("best")
        result = state._no_frontier(self._inputs())
        self.assertEqual("top.segments missing", result.message)
        self.assertEqual(
            {"seed": "wf best gen s2 -u cdef -g 4 top.segments --source seed",
             "best": "wf best gen s2 -u cdef -g 4 top.segments --source best"},
            self._commands(result))

        self._top()
        self.assertIsNone(state._no_frontier(self._inputs()))

    def test_review_needed_names_the_frontier_source(self):
        self._steady()
        # A frontier written after the last archived round is unreviewed.
        os.utime(self.dir / "top.segments", (45, 45))
        result = state._review_needed(self._inputs())
        self.assertEqual("review needed (frontier from seed)", result.message)
        self.assertEqual({"next": "wf best review s2 -u cdef -g 4"},
                         self._commands(result))

        self._top(mtime=45, source="best", marker=70)
        self.assertEqual("review needed (frontier from best)",
                         state._review_needed(self._inputs()).message)

    def test_best_pairs_empty_offers_a_widen_before_either_search(self):
        self._steady()
        self._top("a,b\nc,d\n", mtime=30, source="best", marker=70)
        self._best_pairs("", mtime=50)

        result = state._best_pairs_empty(self._inputs())
        self.assertEqual("review confirmed no pairs", result.message)
        self.assertEqual("(2 frontier pairs, 0 in classified/yes)",
                         result.detail)
        # The widen carries the recorded source and the frontier it already
        # has, and it is the only row that renders a count.
        self.assertEqual(
            {"widen": "wf best gen s2 -u cdef -g 4 top.segments "
                      "--source best -n 1002"},
            self._commands(result))
        self.assertEqual(
            (f"or retract NO verdicts in {config.classified(self.root, 'no')}",
             "   and run: wf best review s2 -u cdef -g 4"),
            result.note)

        # A reseed joins it only when the seed search is actually behind, and
        # a refine never does: dfs.best refuses an empty best.pairs.
        self._classified("no", 90)
        result = state._best_pairs_empty(self._inputs())
        self.assertEqual(
            ["widen", "reseed"], [choice.label for choice in result.choices])

    def test_top_segments_behind_ignores_a_dfs_file_that_is_itself_stale(self):
        self._steady()
        # A finished seed search the frontier was never generated from.
        os.utime(self.results / "dfs.seed.out", (80, 80))
        result = state._top_segments_behind_dfs(self._inputs())
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
        result = state._top_segments_behind_dfs(self._inputs())
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
        result = state._next_search(self._inputs())
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
        result = state._next_search(self._inputs())
        self.assertEqual("dfs.best missing", result.message)
        self.assertEqual(["refine"],
                         [choice.label for choice in result.choices])

        # And an empty best.pairs takes the refine off the table entirely.
        self._best_pairs("", mtime=50)
        self.assertIsNone(state._next_search(self._inputs()))

    # ---------------------------------------------------------------- guards

    def test_a_row_reached_out_of_order_raises_rather_than_dating_nothing(self):
        self._letters()
        self._seed()
        self._dfs("seed")
        with self.assertRaises(FileNotFoundError):
            state._review_needed(self._inputs())

        self._top()
        with self.assertRaises(FileNotFoundError):
            state._best_pairs_out_of_date(self._inputs())
        with self.assertRaises(FileNotFoundError):
            state._best_pairs_empty(self._inputs())

        self.dir.joinpath("best.pairs").mkdir()
        with self.assertRaises(ValueError):
            state._best_pairs_out_of_date(self._inputs())

    def test_search_conditions_refuse_to_answer_without_a_seed(self):
        self._letters()
        self._dfs("seed")
        with self.assertRaisesRegex(FileNotFoundError, "seed missing"):
            _ = self._inputs().seed_search_needed

    # ------------------------------------------------------------ precedence

    def test_rows_are_evaluated_in_the_documented_precedence(self):
        self.assertEqual(
            ["_letters_missing", "_seed_missing",
             "_review_queued", "_review_evaluating",
             "_no_frontier",
             "_review_needed",
             "_best_pairs_missing", "_best_pairs_out_of_date",
             "_best_pairs_empty",
             "_top_segments_behind_dfs",
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

    def test_a_stale_best_pairs_outranks_both_searches_and_the_frontier(self):
        self._steady()
        # Seconds to rebuild; running either search off a set that is behind
        # the classified sets costs hours.
        self._classified("no", 90)
        os.utime(self.results / "dfs.seed.out", (80, 80))
        self.assertEqual("best.pairs out of date (hard-NO set changed)",
                         self._state().message)

    def test_an_empty_best_pairs_outranks_an_available_reseed(self):
        self._steady()
        self._best_pairs("", mtime=50)
        self._classified("no", 45)
        self.assertEqual("review confirmed no pairs", self._state().message)

    def test_a_missing_best_pairs_outranks_both(self):
        self._steady()
        (self.dir / "best.pairs").unlink()
        state._stamp(self.dir / "best.pairs").unlink()
        self._classified("no", 90)
        result = self._state()
        self.assertEqual("best.pairs missing", result.message)
        self.assertEqual({"next": "wf best gen s2 -u cdef -g 4 best.pairs"},
                         self._commands(result))

    # ----------------------------------------------------- shared renderers

    def test_both_rows_that_offer_a_reseed_render_the_same_command(self):
        self._steady()
        self._best_pairs("", mtime=50)
        self._classified("no", 90)
        empty = self._commands(state._best_pairs_empty(self._inputs()))

        self._best_pairs("a,b\n", mtime=50)
        following = self._commands(state._next_search(self._inputs()))
        self.assertEqual(empty["reseed"], following["reseed"])

    def test_both_rows_that_offer_a_generation_render_the_same_command(self):
        self._letters()
        self._seed()
        self._dfs("seed")
        absent = self._commands(state._no_frontier(self._inputs()))

        self._classified("yes")
        self._classified("no")
        self._top(mtime=30, marker=30)
        self._archived(mtime=40)
        self._best_pairs(mtime=50)
        os.utime(self.results / "dfs.seed.out", (80, 80))
        behind = self._commands(state._top_segments_behind_dfs(self._inputs()))
        self.assertEqual(absent["next"], behind["next"])

    # -------------------------------------------------------------- markers

    def test_the_source_marker_is_read_leniently_and_never_guessed(self):
        self._top(source=None)
        marker = state._stamp(self.dir / "top.segments")
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

    def test_a_marker_dates_a_generation_the_artifact_cannot(self):
        top = self._top(mtime=30, marker=30)
        # A missing marker leaves the artifact dating itself.
        state._stamp(top).unlink()
        self.assertEqual(top, state._generated(top))

        state.mark_generated(top, "best\n")
        self.assertEqual(state._stamp(top), state._generated(top))
        self.assertEqual("best\n", state._stamp(top).read_text())

        # A byte-identical remark still advances the clock: content is what
        # stable_mtime pins, and the generation clock has to move regardless.
        os.utime(state._stamp(top), (30, 30))
        state.mark_generated(top, "best\n")
        self.assertGreater(state._stamp(top).stat().st_mtime_ns, 30 * 10 ** 9)
        self.assertEqual(30, int(top.stat().st_mtime))

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
        what the walk exists to avoid.
        """
        verdicts = self._labels()
        self.assertTrue(verdicts["_letters_missing"].fired)
        self.assertTrue(verdicts["_seed_missing"].fired)
        self.assertTrue(verdicts["_no_frontier"].fired)
        self.assertEqual(("top.segments",), verdicts["_review_needed"].unmet)
        self.assertEqual(("best.pairs", "top.segments"),
                         verdicts["_best_pairs_out_of_date"].unmet)
        self.assertEqual(("best.pairs", "top.segments", "seed"),
                         verdicts["_best_pairs_empty"].unmet)
        self.assertEqual(("top.segments", "seed"),
                         verdicts["_top_segments_behind_dfs"].unmet)
        self.assertEqual(("seed",), verdicts["_next_search"].unmet)
        # A row whose own condition needs nothing is still asked.
        self.assertTrue(verdicts["_best_pairs_missing"].fired)

    def test_a_row_is_asked_once_the_row_providing_its_file_declines(self):
        self._letters()
        self._seed()
        self._dfs("seed")
        self._top()
        verdicts = self._labels()
        self.assertEqual((), verdicts["_review_needed"].unmet)
        self.assertEqual(("best.pairs",),
                         verdicts["_best_pairs_out_of_date"].unmet)
        self.assertEqual((), verdicts["_top_segments_behind_dfs"].unmet)

    def test_the_walk_agrees_with_derive_state_down_to_the_winner(self):
        self._steady()
        # A frontier newer than the archived bundle that reviewed it and
        # newer than the best.pairs derived from it: rows 6 and 8 both hold.
        os.utime(self.dir / "top.segments", (55, 55))
        won = [verdict for verdict in state.walk_rows(self.target)
               if verdict.fired]
        self.assertEqual("_review_needed", won[0].row.name)
        self.assertEqual(self._state(), won[0].state)
        # And it keeps going: best.pairs is stale against the frontier too,
        # which is the diagnostic derive_state cannot show.
        self.assertEqual(["_review_needed", "_best_pairs_out_of_date"],
                         [verdict.row.name for verdict in won])

    def test_the_table_names_the_winner_apart_from_what_also_fired(self):
        self._steady()
        os.utime(self.dir / "top.segments", (55, 55))
        lines = state.render_rows(state.walk_rows(self.target))
        self.assertEqual("  rows:", lines[0])
        rendered = {line.split()[1]: line.split(maxsplit=1)[0].rstrip(":")
                    for line in lines[1:] if line.split()[0].endswith(":")
                    and line.split()[1].startswith("_")}
        self.assertEqual("won", rendered["_review_needed"])
        self.assertEqual("also", rendered["_best_pairs_out_of_date"])
        self.assertEqual("no", rendered["_letters_missing"])

    def test_a_fired_row_carries_the_commands_it_offers(self):
        """Both fired rows print theirs, in the report's own rendering.

        The winner's line repeats what report printed above the table; the
        `also` row's is the one the table alone can show, indented under the
        row rather than beside the winner's.
        """
        self._steady()
        os.utime(self.dir / "top.segments", (55, 55))
        lines = state.render_rows(state.walk_rows(self.target))
        winner = lines.index(next(
            line for line in lines if "won:" in line))
        self.assertEqual("      next: wf best review s2 -u cdef -g 4",
                         lines[winner + 1])
        also = lines.index(next(line for line in lines if "also:" in line))
        self.assertEqual("      next: wf best gen s2 -u cdef -g 4 best.pairs",
                         lines[also + 1])

        # A declined row prints nothing, and a row never asked prints nothing
        # -- neither has a state to offer commands from.
        self.assertEqual(
            len(state.ROWS) + 1 + 2,
            len(state.render_rows(state.walk_rows(self.target))))

        # Several alternatives keep the rendering report gives them.
        self._classified("no", 90)
        self._best_pairs("a,b\n", mtime=95, marker=95)
        lines = state.render_rows(state.walk_rows(self.target))
        following = lines.index(next(
            line for line in lines if "_next_search" in line))
        self.assertEqual(
            ["      choose next:",
             "        reseed: wf best prepare s2 -u cdef -g 4 --source seed",
             "        refine: wf best prepare s2 -u cdef -g 4 --source best"],
            lines[following + 1:following + 4])

    def test_the_table_says_which_file_a_skipped_row_wanted(self):
        lines = state.render_rows(state.walk_rows(self.target))
        line = next(line for line in lines if "_review_needed" in line)
        self.assertIn("n/a:", line)
        self.assertIn("not asked (needs top.segments)", line)


if __name__ == "__main__":
    unittest.main()
