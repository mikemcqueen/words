import os
import tempfile
import unittest

from pathlib import Path
from unittest import mock

from tests import wf_fixture as fx
from workflow import config
from workflow.best import commands, generate, state


class BestEndToEndTests(unittest.TestCase):
    """Drive the CLI and durable state, stubbing only external producers."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.results = self.root / "results"
        self.results.mkdir()

    def _wf(self, *argv: str) -> tuple[str, str]:
        code, stdout, stderr = fx.run_wf("-d", str(self.root), *argv)
        self.assertEqual(0, code, stderr)
        return stdout, stderr

    def _run_producers(self, outputs: list[str],
                       *argv: str) -> tuple[list[list[str]], str]:
        """Drive a wf best command, standing in for its external producers.

        Each output is what the next subprocess writes to stdout, in the order
        the command runs them -- one for a gen, two for a prepare.
        """
        calls = []

        def run(command, **kwargs):
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
        (best_dir / "dict" / generate.DICTIONARY_NAME).write_text("words\n")

        sentence_dir = best_dir / "s2"
        sentence_dir.mkdir()
        (sentence_dir / "letters").write_text("abcdef\n")
        seed = sentence_dir / "seed.m4.idx2.85.15.pairs"
        seed.write_text("seed,pair\n")
        universe = sentence_dir / "u-cdef" / "m4"

        dfs_seed_command, stdout = self._gen(
            "100 good,one\n90 good,two\n80 rejected,pair\n",
            "-f", "s2", "-u", "cdef", "-g", "4", "-r", str(self.results),
            "-n", "3", "dfs.seed")
        self.assertEqual("dfs-anagrams", dfs_seed_command[0])
        self.assertEqual(
            str(seed),
            dfs_seed_command[dfs_seed_command.index("--pairs") + 1])
        self.assertIn("s2/u-cdef/m4/g4: top.segments missing", stdout)

        top_command, stdout = self._gen(
            "good,one\ngood,two\nrejected,pair\n",
            "s2", "-u", "cdef", "-g", "4", "-n", "3", "--source", "seed",
            "top.segments")
        self.assertEqual("top-segments", top_command[0])
        self.assertIn("s2/u-cdef/m4/g4: review needed (frontier from seed)",
                      stdout)

        # The review gate asks whether a round completed after top.segments
        # was written, and mtime granularity here is coarse enough (~4ms) for
        # a whole round to land inside one tick. Date the two stages apart so
        # the gate is answering the question the test is asking.
        target = universe / "g4"
        for path in (seed, config.classified(self.root, "no")):
            os.utime(path, (5, 5))
        os.utime(target / "dfs.seed", (10, 10))
        for path in (target / "top.segments",
                     state._stamp(target / "top.segments")):
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
        self.assertEqual("good,one\ngood,two\n",
                         (target / "best.pairs").read_text())
        self.assertEqual(
            "good,one\ngood,two\n",
            config.classified(self.root, "yes").read_text())
        self.assertFalse(bundle_dir.exists())
        self.assertTrue(
            (wf_dir / "p2" / "done" / "in" / source.name).is_file())
        self.assertEqual(
            "rejected,pair\n",
            config.classified(self.root, "no").read_text())
        # Completing the review is what grows the hard-NO set, so the seed the
        # search ran from is now genuinely stale and the report says so.
        self.assertIn(
            "s2/u-cdef/m4/g4: dfs.seed out of date (hard-NO set changed)",
            stdout)

        # Take the operator's decision to accept the seed as it stands, so the
        # rest of the tail is reachable in one test: date the fold back behind
        # the artifacts that were derived before it.
        for kind in ("yes", "no"):
            os.utime(config.classified(self.root, kind), (25, 25))
        os.utime(target / "dfs.seed", (30, 30))
        for path in (target / "top.segments",
                     state._stamp(target / "top.segments")):
            os.utime(path, (40, 40))
        stdout, _ = self._wf("best", "status", "s2/u-cdef/m4/g4")
        self.assertIn("s2/u-cdef/m4/g4: dfs.best missing", stdout)

        dfs_best_command, stdout = self._gen(
            "200 final,answer\n",
            "s2", "-u", "cdef", "-g", "4", "-r", str(self.results),
            "-n", "1", "dfs.best")
        self.assertEqual("dfs-anagrams", dfs_best_command[0])
        self.assertEqual(
            str(target / "best.pairs"),
            dfs_best_command[dfs_best_command.index("--pairs") + 1])
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
        self.assertEqual(
            str(target / "best.pairs"),
            commands_run[0][commands_run[0].index("--pairs") + 1])
        self.assertEqual(
            str(target / "dfs.best"), commands_run[1][-1])
        self.assertEqual("best\n",
                         state._stamp(target / "top.segments").read_text())
        self.assertIn("s2/u-cdef/m4/g4: review needed (frontier from best)",
                      stdout)

        # Same coarse-mtime problem as round 1, and the same answer: date the
        # search and the frontier apart so the clocks answer the question the
        # rest of the round is asking.
        os.utime(target / "dfs.best", (60, 60))
        for path in (target / "top.segments",
                     state._stamp(target / "top.segments")):
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
        # and it is still here: best.pairs accumulates across rounds.
        self.assertEqual("final,answer\ngood,one\ngood,two\nsecond,look\n",
                         (target / "best.pairs").read_text())
        self.assertIn("s2/u-cdef/m4/g4: dfs.best out of date "
                      "(best.pairs changed)", stdout)
        self.assertIn(
            "refine: wf best prepare s2 -u cdef -g 4 --source best", stdout)

        # The DFS output names carry no generation or review ordinal: two
        # rounds of one target write the one path their cutoff names.
        self.assertEqual(
            ["dfs.s2.idx2.85.15.m4.x2.g4.3.u-cdef",
             "dfs.s2.idx2.85.15.m4.x2.g4.best.1.u-cdef",
             "dfs.s2.idx2.85.15.m4.x2.g4.best.3.u-cdef"],
            sorted(path.name for path in (self.results / "s2").iterdir()))


if __name__ == "__main__":
    unittest.main()
