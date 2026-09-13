import os
import tempfile
import unittest

from pathlib import Path
from unittest import mock

from tests import wf_fixture as fx
from workflow import config, generation
from workflow.best import commands, generate, state


PRODUCERS = ("dfs-anagrams", "top-segments")


class BestEndToEndTests(unittest.TestCase):
    """Drive the CLI and durable state, stubbing only external producers."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.results = self.root / "results"
        self.results.mkdir()

    def _dictionary(self, mtime=5) -> Path:
        """The dictionary tree a rebuild leaves behind, dated back.

        Dated with the other hand-placed inputs because the frontier's marker
        is forced back below, and a dictionary at real time would read as an
        edit made after it.
        """
        config.base_dictionary(self.root).write_text("words\n")
        derived = config.dictionary(self.root)
        derived.write_text("words\n")
        config.reviewed_words(self.root).write_text("")
        generation.mark_generated(derived)
        for path in (config.base_dictionary(self.root), derived,
                     config.reviewed_words(self.root),
                     generation.stamp(derived),
                     config.removals(self.root),
                     config.reviewed_inputs(self.root)):
            os.utime(path, (mtime, mtime))
        return derived

    def _wf(self, *argv: str) -> tuple[str, str]:
        code, stdout, stderr = fx.run_wf("-d", str(self.root), *argv)
        self.assertEqual(0, code, stderr)
        return stdout, stderr

    def _require_state(self, result: state.State | None) -> state.State:
        """Assert that the directly tested status row fired."""
        self.assertIsNotNone(result)
        assert result is not None
        return result

    def _run_producers(self, outputs: list[str],
                       *argv: str) -> tuple[list[list[str]], str]:
        """Drive a wf best command, standing in for its external producers.

        Each output is what the next producer writes to stdout, in the order
        the command runs them -- one for a gen, two for a prepare. Anything
        that is not a producer is passed straight through.
        """
        calls = []
        real_run = generate.subprocess.run

        def run(command, **kwargs):
            if command[0] not in PRODUCERS:
                return real_run(command, **kwargs)
            calls.append(command)
            kwargs["stdout"].write(outputs[len(calls) - 1])
            return mock.Mock(returncode=0)

        with mock.patch.object(generate.shutil, "which", return_value="/bin/fake"), \
                mock.patch.object(generate.subprocess, "run", side_effect=run):
            stdout, _ = self._wf("best", *argv)

        self.assertEqual(len(outputs), len(calls))
        return calls, stdout

    def _gen(self, output: str, *argv: str) -> tuple[list[str], str]:
        calls, stdout = self._run_producers([output], "gen", *argv)
        return calls[0], stdout

    def test_complete_workflow_with_injected_p2_verdicts(self):
        stdout, _ = self._wf("init")
        self.assertIn("Initialized", stdout)

        wf_dir = self.root / ".wf"
        best_dir = wf_dir / "best"
        (best_dir / "idx" / generate.INDEX_NAME).write_text("index\n")
        dictionary = self._dictionary()

        sentence_dir = best_dir / "s2"
        sentence_dir.mkdir()
        (sentence_dir / "letters").write_text(
            "goodonetwofinalanswersecondlookcdef\n")
        seed = sentence_dir / "seed.m4.idx2.85.15.pairs"
        seed.write_text("seed,pair\n")
        universe = sentence_dir / "u-cdef" / "m4"

        dfs_seed_command, stdout = self._gen(
            "100 good,one\n90 good,two\n80 rejected,pair\n",
            "-f", "s2", "-u", "cdef", "-g", "4", "-r", str(self.results),
            "-n", "3", "dfs.seed")
        self.assertEqual("dfs-anagrams", dfs_seed_command[0])
        self.assertEqual(str(self.root),
                         dfs_seed_command[
                             dfs_seed_command.index("--wfroot") + 1])
        self.assertEqual("s2",
                         dfs_seed_command[dfs_seed_command.index("-t") + 1])
        self.assertIn("s2/u-cdef/m4/g4: top.segments missing", stdout)

        top_command, stdout = self._gen(
            "good,one\ngood,two\nrejected,pair\n",
            "s2", "-u", "cdef", "-g", "4", "-n", "3", "--source", "seed",
            "top.segments")
        self.assertEqual("top-segments", top_command[0])
        # The frontier is filtered against both classified sets, so it holds
        # candidates with no standing verdict rather than rows already
        # answered.
        self.assertEqual(
            [str(self.root), "-y", str(universe / "g4" / "dfs.seed")],
            top_command[top_command.index("--wfroot") + 1:])
        self.assertIn("s2/u-cdef/m4/g4: review needed (frontier from seed)",
                      stdout)

        # The review gate asks whether a round completed after top.segments
        # was written, and mtime granularity here is coarse enough (~4ms) for
        # a whole round to land inside one tick. Date the two stages apart so
        # the gate is answering the question the test is asking.
        target = universe / "g4"
        for path in (best_dir / "idx" / generate.INDEX_NAME, dictionary,
                     sentence_dir / "letters", seed,
                     config.classified(self.root, "yes"),
                     config.classified(self.root, "no")):
            os.utime(path, (5, 5))
        os.utime(target / "dfs.seed", (10, 10))
        for path in (target / "top.segments",
                     generation.stamp(target / "top.segments")):
            os.utime(path, (20, 20))

        with mock.patch.object(commands.evaluate.P2, "prepare") as prepare:
            stdout, _ = self._wf(
                "best", "review", "s2", "-u", "cdef", "-g", "4")
        prepare.assert_called_once()
        self.assertIn("review awaiting completion", stdout)

        bundle_name = "top.s2.m4.g4.u-cdef.3.r1"
        bundle_dir = wf_dir / "p2" / "eval" / bundle_name
        source = bundle_dir / f"{bundle_name}.pairs"
        self.assertEqual(
            "good,one\ngood,two\nrejected,pair\n", source.read_text())

        # Stand in for the note round by placing the verdict artifacts that
        # its retrieve/extract steps would have produced.
        (bundle_dir / "enex").mkdir()
        (bundle_dir / f"{bundle_name}.p2.yes").write_text(
            "good,one\ngood,two\n")
        (bundle_dir / f"{bundle_name}.p2.no").write_text(
            "rejected,pair\n")

        stdout, _ = self._wf("best", "complete", "s2", "-u", "cdef", "-g", "4")
        target = universe / "g4"
        # Nothing derives a per-target set: verdicts land in the classified
        # sets and Nutrimatic reads them through --wfroot.
        self.assertFalse((target / "best.pairs").exists())
        self.assertEqual(
            "good,one\ngood,two\n",
            config.classified(self.root, "yes").read_text())
        self.assertFalse(bundle_dir.exists())
        self.assertTrue(
            (wf_dir / "p2" / "done" / "in" / source.name).is_file())
        self.assertEqual(
            "rejected,pair\n",
            config.classified(self.root, "no").read_text())
        # Completing the review moves both classified sets past the marker
        # that dates the frontier, so the same DFS file refills 1000 fresh
        # candidates -- seconds, offered before any search.
        self.assertIn(
            "s2/u-cdef/m4/g4: top.segments behind its inputs "
            "(confirmed-YES set changed, hard-NO set changed)",
            stdout)

        # Take the operator's decision to accept the frontier as it stands, so
        # the rest of the tail is reachable in one test: date the fold back
        # behind the artifacts that were derived before it.
        for kind in ("yes", "no"):
            os.utime(config.classified(self.root, kind), (25, 25))
        os.utime(target / "dfs.seed", (30, 30))
        for path in (target / "top.segments",
                     generation.stamp(target / "top.segments")):
            os.utime(path, (40, 40))
        (target / "best.pairs").write_text("good,one\ngood,two\n")
        os.utime(target / "best.pairs", (25, 25))
        stdout, _ = self._wf("best", "status", "s2/u-cdef/m4/g4")
        self.assertIn("s2/u-cdef/m4/g4: dfs.best missing", stdout)

        dfs_best_command, stdout = self._gen(
            "200 final,answer\n",
            "s2", "-u", "cdef", "-g", "4", "-r", str(self.results),
            "-n", "1", "dfs.best")
        self.assertEqual("dfs-anagrams", dfs_best_command[0])
        self.assertEqual(
            "s2/u-cdef/m4/g4",
            dfs_best_command[dfs_best_command.index("-t") + 1])
        self.assertNotIn("--pairs", dfs_best_command)
        self.assertFalse((target / "dfs.best.pairs").exists())
        self.assertTrue(generation.stamp(target / "dfs.best").is_file())
        # A finished search the frontier was never generated from.
        self.assertIn(
            "s2/u-cdef/m4/g4: dfs.best generated after top.segments", stdout)

        # ------------------------------------------- one inner-loop round
        #
        # prepare re-runs that search and generates the frontier from it in
        # one command, which is what a round of the inner loop is.
        commands_run, stdout = self._run_producers(
            ["200 final,answer\n150 good,one\n100 second,look\n",
             "final,answer\ngood,one\nsecond,look\n"],
            "prepare", "s2", "-u", "cdef", "-g", "4", "-r", str(self.results),
            "--source", "best", "--dfs-count", "3", "--top-count", "3")
        self.assertEqual(["dfs-anagrams", "top-segments"],
                         [call[0] for call in commands_run])
        self.assertNotIn("--pairs", commands_run[0])
        self.assertEqual(
            [str(self.root), "-y", str(target / "dfs.best")],
            commands_run[1][commands_run[1].index("--wfroot") + 1:])
        self.assertEqual("best\n",
                         generation.stamp(target / "top.segments").read_text())
        self.assertIn("s2/u-cdef/m4/g4: review needed (frontier from best)",
                      stdout)

        # Same coarse-mtime problem as round 1, and the same answer: date the
        # search and the frontier apart so the clocks answer the question the
        # rest of the round is asking.
        os.utime(target / "dfs.best", (60, 60))
        for path in (target / "top.segments",
                     generation.stamp(target / "top.segments")):
            os.utime(path, (70, 70))

        # The confirmed-YES pairs are subtracted, so good,one is not asked
        # about again, and the round is the next of one shared sequence.
        with mock.patch.object(commands.evaluate.P2, "prepare") as prepare:
            stdout, _ = self._wf(
                "best", "review", "s2", "-u", "cdef", "-g", "4")
        prepare.assert_called_once()
        round_two = "top.s2.m4.g4.u-cdef.3.r2"
        bundle_dir = wf_dir / "p2" / "eval" / round_two
        source = bundle_dir / f"{round_two}.pairs"
        self.assertEqual("final,answer\nsecond,look\n", source.read_text())
        self.assertIn("review awaiting completion", stdout)

        (bundle_dir / "enex").mkdir()
        (bundle_dir / f"{round_two}.p2.yes").write_text(
            "final,answer\nsecond,look\n")
        (bundle_dir / f"{round_two}.p2.no").write_text("")

        stdout, _ = self._wf("best", "complete", "s2", "-u", "cdef", "-g", "4")
        # good,two is not in this round's frontier and was never re-reviewed,
        # and it is still here: the confirmed-YES set accumulates across
        # rounds, and every target reads the same one.
        self.assertEqual("final,answer\ngood,one\ngood,two\nsecond,look\n",
                         config.classified(self.root, "yes").read_text())
        # The frontier is behind the round that just landed, which is the
        # cheap thing to fix and outranks the hours below it.
        self.assertIn(
            "s2/u-cdef/m4/g4: top.segments behind its inputs", stdout)
        # Classified YES is a declared source for both searches, so the
        # completed round conservatively stales this target too.
        target_state = state.one_target(self.root, "s2", "u-cdef", 4, 4)
        self.assertEqual(
            ["confirmed-YES set changed"],
            state.Inputs(target_state).best_search_needed)

        # The DFS output names carry no generation or review ordinal: two
        # rounds of one target write the one path their cutoff names.
        self.assertEqual(
            ["dfs.s2.idx2.85.15.m4.x2.g4.3.u-cdef",
             "dfs.s2.idx2.85.15.m4.x2.g4.best.1.u-cdef",
             "dfs.s2.idx2.85.15.m4.x2.g4.best.3.u-cdef"],
            sorted(path.name for path in (self.results / "s2").iterdir()))

    def test_a_removal_reaches_the_next_search_through_the_derived_file(self):
        """`remove words` then `gen dfs.seed`: one round trip, end to end.

        The searches read what the rebuild produced, not the hand-placed base,
        so a word removed here is a word the next search cannot spell.
        """
        self._wf("init")
        best_dir = self.root / ".wf" / "best"
        (best_dir / "idx" / generate.INDEX_NAME).write_text("index\n")
        config.base_dictionary(self.root).write_text(
            "apple\nbanana\ncherry\n")
        sentence_dir = best_dir / "s2"
        sentence_dir.mkdir()
        (sentence_dir / "letters").write_text("abcdef\n")
        (sentence_dir / "seed.m4.idx2.85.15.pairs").write_text("seed,pair\n")

        junk = self.root / "junk.txt"
        junk.write_text("  1234 banana\n")
        stdout, _ = self._wf("remove", "words", str(junk))
        self.assertIn("1 words submitted, 1 new to the removal union, "
                      "1 newly removed from the dictionary", stdout)
        derived = config.dictionary(self.root)
        self.assertEqual("apple\ncherry\n", derived.read_text())

        command, _ = self._gen(
            "100 good,one\n",
            "-f", "s2", "-u", "cdef", "-g", "4", "-r", str(self.results),
            "-n", "1", "dfs.seed")
        self.assertEqual("dfs-anagrams", command[0])
        self.assertEqual(str(self.root),
                         command[command.index("--wfroot") + 1])
        self.assertNotIn("--dict", command)

        # And the search it just ran is now behind the next removal.
        os.utime(derived, (10 ** 10, 10 ** 10))
        target = state.one_target(self.root, "s2", "u-cdef", 4, 4)
        self.assertEqual(["dictionary changed"],
                         state.Inputs(target).seed_search_needed)

    def test_oneoff_lifecycle_archives_source_and_records_yes_globally(self):
        self._wf("init")
        wf_dir = self.root / ".wf"
        target = wf_dir / "best" / "s2" / "u-cdef" / "m4" / "g4"
        target.mkdir(parents=True)
        (target / "top.segments").write_text("frontier,unknown\n")
        self._dictionary()
        supplied = self.root / "arbitrary-input"
        supplied.write_text(
            "known,yes\nnew,yes\nknown,no\nnew,no\nknown,yes\n")
        config.classified(self.root, "yes").write_text("known,yes\n")
        config.classified(self.root, "no").write_text("known,no\n")

        with mock.patch.object(commands.evaluate.P2, "prepare"):
            stdout, _ = self._wf(
                "best", "review", "s2", "-u", "cdef", "-g", "4",
                str(supplied))
        self.assertIn("one-off review in flight", stdout)
        supplied.unlink()

        bundle_name = "oneoff.s2.m4.g4.u-cdef.4.r1"
        bundle = wf_dir / "p2" / "eval" / bundle_name
        source = bundle / f"{bundle_name}.pairs"
        filtered = source.with_name(source.name + ".filtered")
        self.assertEqual(
            "known,no\nknown,yes\nnew,no\nnew,yes\n", source.read_text())
        self.assertEqual("new,no\nnew,yes\n", filtered.read_text())

        (bundle / "enex").mkdir()
        (bundle / f"{bundle_name}.p2.yes").write_text("new,yes\n")
        (bundle / f"{bundle_name}.p2.no").write_text("new,no\n")
        self._wf("best", "complete", "s2", "-u", "cdef", "-g", "4")

        archived = wf_dir / "p2" / "done" / "in" / source.name
        self.assertEqual(
            "known,no\nknown,yes\nnew,no\nnew,yes\n", archived.read_text())
        # The verdicts land in the classified sets and nowhere else: a one-off
        # no longer accumulates into a per-target file.
        self.assertEqual("known,yes\nnew,yes\n",
                         config.classified(self.root, "yes").read_text())
        self.assertEqual("known,no\nnew,no\n",
                         config.classified(self.root, "no").read_text())
        self.assertEqual("new,no\nnew,yes\n",
                         (wf_dir / "p2" / "done" / "p2_done.pairs").read_text())
        self.assertFalse(filtered.exists())
        self.assertFalse((target / "best.pairs").exists())
        target_state = state.one_target(self.root, "s2", "u-cdef", 4, 4)
        self.assertEqual(
            "review needed (frontier from seed)",
            self._require_state(
                state._review_needed(state.Inputs(target_state))).message)


if __name__ == "__main__":
    unittest.main()
