import os
import tempfile
import unittest

from pathlib import Path
from unittest import mock

from tests import wf_fixture as fx
from workflow import best, config


class BestTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.opts, _ = fx.make_wf(self.root)
        self.best = fx.slot(self.opts, ["best"])

    def _target(self, sentence="s2", m=4, g=4) -> Path:
        target = self.best / sentence / f"m{m}" / f"g{g}"
        target.mkdir(parents=True)
        return target

    def _write(self, path: Path, text="", mtime=None) -> Path:
        path.write_text(text)
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        return path

    def _shared_inputs(self, universe: Path) -> tuple[Path, Path, Path]:
        index = self.best / "idx" / best.INDEX_NAME
        dictionary = self.best / "dict" / best.DICTIONARY_NAME
        seed = universe / "seed.idx2.85.15.pairs"
        self._write(index, "index\n")
        self._write(dictionary, "words\n")
        self._write(universe.parent / "letters", "abcdef\n", 10)
        self._write(seed, "a,b\n", 10)
        os.utime(config.classified(self.root, "no"), (10, 10))
        return index, dictionary, seed

    def _complete_files(self, target: Path) -> None:
        sentence_dir = target.parents[1]
        universe_dir = target.parent
        for kind in ("yes", "no"):
            os.utime(config.classified(self.root, kind), (10, 10))
        self._write(sentence_dir / "letters", "abcdef\n", 10)
        self._write(universe_dir / "seed.idx2.85.15.pairs", "a,b\n", 10)
        results = self.root / "results"
        results.mkdir()
        dfs_seed = self._write(results / "dfs.seed.out", "1 a b\n", 20)
        (target / "dfs.seed").symlink_to(dfs_seed)
        self._write(target / "top.segments", "a,b\n", 30)
        prefix = (f"top.{sentence_dir.name}.{universe_dir.name}."
                  f"{target.name}.1000.r1.pairs")
        self._write(fx.slot(self.opts, ["p2", "done", "in"]) / prefix,
                    "a,b\n", 40)
        self._write(target / "best.pairs", "a,b\n", 50)
        dfs_best = self._write(results / "dfs.best.out", "1 a b\n", 60)
        (target / "dfs.best").symlink_to(dfs_best)

    def test_init_builds_static_crown_and_show_points_to_status(self):
        self.assertTrue((self.best / "idx").is_dir())
        self.assertTrue((self.best / "dict").is_dir())

        code, _, stderr = fx.run_wf(
            "-d", str(self.root), "show", "best")
        self.assertEqual(2, code)
        self.assertIn("wf best status", stderr)

        code, _, stderr = fx.run_wf(
            "-d", str(self.root), "show", "balls")
        self.assertEqual(2, code)
        self.assertNotIn("BEST PAIRS workflow state", stderr)

    def test_status_walks_targets_under_a_valid_prefix(self):
        self._target(g=5)
        self._target(g=4)
        (self.best / "not-a-sentence").mkdir()

        code, stdout, stderr = fx.run_wf(
            "-d", str(self.root), "best", "status", "s2/m4")
        self.assertEqual(0, code, stderr)
        self.assertEqual(
            ["s2/m4/g4: letters missing", "s2/m4/g5: letters missing"],
            [line for line in stdout.splitlines() if not line.startswith("  ")])

        with self.assertRaisesRegex(ValueError, "expected m<N>"):
            fx.run_wf("-d", str(self.root), "best", "status", "s2/g4")

    def test_status_reports_dangling_and_stale_dfs_seed(self):
        target = self._target()
        self._write(target.parents[1] / "letters", "abcdef\n", 10)
        self._write(target.parent / "seed.pairs", "a,b\n", 10)
        missing = self.root / "gone" / "dfs.out"
        (target / "dfs.seed").symlink_to(missing)

        code, stdout, stderr = fx.run_wf(
            "-d", str(self.root), "best", "status", "s2/m4/g4")
        self.assertEqual(0, code, stderr)
        self.assertIn(f"dfs.seed missing (dangling symlink: {missing})", stdout)

        (target / "dfs.seed").unlink()
        self._write(target / "dfs.seed", "1 a b\n", 20)
        hard_no = config.classified(self.root, "no")
        self._write(hard_no, "a,b\n", 30)
        code, stdout, stderr = fx.run_wf(
            "-d", str(self.root), "best", "status", "s2/m4/g4")
        self.assertEqual(0, code, stderr)
        self.assertIn("dfs.seed out of date (hard-NO set changed)", stdout)
        self.assertIn("next: wf best gen s2 -g 4 dfs.seed", stdout)

    def test_status_derives_review_gate_and_fully_fresh_state(self):
        target = self._target()
        self._complete_files(target)

        code, stdout, stderr = fx.run_wf(
            "-d", str(self.root), "best", "status", "s2/m4/g4")
        self.assertEqual(0, code, stderr)
        self.assertIn("s2/m4/g4: up to date", stdout)

        os.utime(target / "dfs.seed", (35, 35))
        _, stdout, _ = fx.run_wf(
            "-d", str(self.root), "best", "status", "s2/m4/g4")
        self.assertIn("top.segments out of date (dfs.seed changed)", stdout)
        os.utime(target / "dfs.seed", (20, 20))

        confirmed_yes = config.classified(self.root, "yes")
        os.utime(confirmed_yes, (55, 55))
        _, stdout, _ = fx.run_wf(
            "-d", str(self.root), "best", "status", "s2/m4/g4")
        self.assertIn(
            "best.pairs out of date (confirmed-YES set changed)", stdout)
        os.utime(confirmed_yes, (10, 10))

        os.utime(target / "best.pairs", (65, 65))
        _, stdout, _ = fx.run_wf(
            "-d", str(self.root), "best", "status", "s2/m4/g4")
        self.assertIn("dfs.best out of date (best.pairs changed)", stdout)
        os.utime(target / "best.pairs", (50, 50))

        done = fx.slot(self.opts, ["p2", "done", "in"])
        next(done.iterdir()).unlink()
        code, stdout, stderr = fx.run_wf(
            "-d", str(self.root), "best", "status", "s2/m4/g4")
        self.assertEqual(0, code, stderr)
        self.assertIn("s2/m4/g4: review needed", stdout)
        self.assertIn("next: wf best review s2 -g 4", stdout)

        queued = fx.slot(self.opts, ["p2", "queued"])
        self._write(queued / "top.s2.m4.g4.1000.r1.pairs", "a,b\n")
        code, stdout, stderr = fx.run_wf(
            "-d", str(self.root), "best", "status", "s2/m4/g4")
        self.assertEqual(0, code, stderr)
        self.assertIn("review submitted (top.s2.m4.g4.1000.r1.pairs)", stdout)
        self.assertIn("next: wf eval p2 top.s2.m4.g4.1000.r1.pairs", stdout)

        queued_file = queued / "top.s2.m4.g4.1000.r1.pairs"
        evaluating = fx.slot(self.opts, ["p2", "eval"]) / queued_file.stem
        evaluating.mkdir()
        queued_file.rename(evaluating / queued_file.name)
        code, stdout, stderr = fx.run_wf(
            "-d", str(self.root), "best", "status", "s2/m4/g4")
        self.assertEqual(0, code, stderr)
        self.assertIn(
            "review awaiting completion (top.s2.m4.g4.1000.r1)", stdout)
        self.assertIn("next: wf best complete s2 -g 4", stdout)

    def test_gen_dfs_seed_validates_then_creates_and_publishes_atomically(self):
        universe = self.best / "s2" / "m4"
        universe.mkdir(parents=True)
        _, _, seed = self._shared_inputs(universe)
        (self.best / "idx" / best.INDEX_NAME).unlink()
        results = self.root / "results"
        results.mkdir()
        target = universe / "g4"
        calls = []

        def run(argv, **kwargs):
            calls.append(argv)
            kwargs["stdout"].write("9 alpha,beta\n8 gamma,delta\n")
            return mock.Mock(returncode=0)

        with mock.patch.object(best.shutil, "which", return_value="/bin/fake"):
            with self.assertRaises(FileNotFoundError):
                fx.run_wf("-d", str(self.root), "best", "gen", "s2",
                          "-g", "4", "-r", str(results), "dfs.seed")
            self.assertFalse(target.exists())
            with self.assertRaises(FileNotFoundError):
                fx.run_wf("-d", str(self.root), "-f", "best", "gen", "s2",
                          "-g", "4", "-r", str(results), "dfs.seed")
        self.assertFalse(target.exists())

        self._write(self.best / "idx" / best.INDEX_NAME, "index\n")
        with mock.patch.object(best.shutil, "which", return_value="/bin/fake"), \
                mock.patch.object(best.subprocess, "run", side_effect=run), \
                mock.patch.object(best.time, "monotonic", side_effect=[0, 2]):
            code, stdout, stderr = fx.run_wf(
                "-d", str(self.root), "-f", "best", "gen", "s2",
                "-g", "4", "-r", str(results), "dfs.seed")

        self.assertEqual(0, code, stderr)
        link = target / "dfs.seed"
        self.assertTrue(link.is_symlink())
        rendered = results / "s2" / "dfs.s2.idx2.85.15.m4.x2.g4.1000000"
        self.assertEqual(rendered, link.resolve())
        self.assertEqual("9 alpha,beta\n8 gamma,delta\n", rendered.read_text())
        self.assertFalse(rendered.with_name(rendered.name + ".tmp").exists())
        self.assertEqual("dfs-anagrams", calls[0][0])
        self.assertEqual(str(seed), calls[0][calls[0].index("--pairs") + 1])
        self.assertEqual("1000000", calls[0][calls[0].index("-n") + 1])
        self.assertIn("$(cat ", stderr)
        self.assertIn("Generated 2 results in 2s", stdout)
        self.assertIn("s2/m4/g4: top.segments missing", stdout)

        def fail(argv, **kwargs):
            kwargs["stdout"].write("truncated\n")
            raise best.subprocess.CalledProcessError(1, argv)

        with mock.patch.object(best.shutil, "which", return_value="/bin/fake"), \
                mock.patch.object(best.subprocess, "run", side_effect=fail):
            with self.assertRaises(best.subprocess.CalledProcessError):
                fx.run_wf("-d", str(self.root), "best", "gen", "s2",
                          "-g", "4", "-r", str(results), "dfs.seed")
        self.assertEqual(rendered, link.resolve())
        self.assertEqual("9 alpha,beta\n8 gamma,delta\n", rendered.read_text())
        self.assertEqual(
            "truncated\n",
            rendered.with_name(rendered.name + ".tmp").read_text())

    def test_gen_top_segments_passes_count_and_preserves_unchanged_mtime(self):
        target = self._target()
        universe = target.parent
        self._shared_inputs(universe)
        dfs_seed = self._write(target / "dfs.seed", "9 alpha,beta\n", 20)
        calls = []

        def run(argv, **kwargs):
            calls.append(argv)
            kwargs["stdout"].write("alpha,beta\ngamma,delta\n")
            return mock.Mock(returncode=0)

        with mock.patch.object(best.shutil, "which", return_value="/bin/fake"), \
                mock.patch.object(best.subprocess, "run", side_effect=run):
            code, stdout, stderr = fx.run_wf(
                "-d", str(self.root), "best", "gen", "s2",
                "-g", "4", "top.segments")
            top = target / "top.segments"
            os.utime(top, (30, 30))
            code2, stdout2, stderr2 = fx.run_wf(
                "-d", str(self.root), "best", "gen", "s2",
                "-g", "4", "-n", "2", "top.segments")

        self.assertEqual((0, 0), (code, code2), stderr + stderr2)
        self.assertEqual(
            ["top-segments", "--pairs", str(dfs_seed)], calls[0])
        self.assertEqual(
            ["top-segments", "--pairs", "-n", "2", str(dfs_seed)], calls[1])
        self.assertEqual(30, int(top.stat().st_mtime))
        self.assertIn("Generated 2 top segments", stdout + stdout2)
        self.assertIn("s2/m4/g4: review needed", stdout2)
        self.assertIn(
            f"top-segments --pairs -n 2 {dfs_seed}", stderr2)

    def test_gen_top_segments_rejects_force_before_running(self):
        target = self._target()
        self._write(target / "dfs.seed", "9 alpha,beta\n")
        with mock.patch.object(best.shutil, "which") as which:
            with self.assertRaisesRegex(ValueError, "only valid for gen dfs.seed"):
                fx.run_wf("-d", str(self.root), "-f", "best", "gen", "s2",
                          "-g", "4", "top.segments")
        which.assert_not_called()

    def test_exclude_classifies_hard_no_and_reports_target_status(self):
        target = self._target()
        self._complete_files(target)
        excluded = self._write(self.root / "excluded.pairs", "hard,no\n")

        code, stdout, stderr = fx.run_wf(
            "-d", str(self.root), "best", "exclude", "s2",
            "-g", "4", str(excluded))

        self.assertEqual(0, code, stderr)
        self.assertEqual(
            "hard,no\n", config.classified(self.root, "no").read_text())
        self.assertIn("Classified NO: 1 new, 1 total", stdout)
        self.assertIn("dfs.seed out of date (hard-NO set changed)", stdout)

    def test_review_subtracts_hard_no_and_opens_unfiltered_round(self):
        target = self._target()
        universe = target.parent
        self._shared_inputs(universe)
        self._write(target / "dfs.seed", "9 alpha,beta\n", 20)
        top = self._write(target / "top.segments", "", 30)
        hard_no = config.classified(self.root, "no")
        self._write(hard_no, "hard,no\n", 10)

        with self.assertRaisesRegex(ValueError, "top.segments is empty"):
            fx.run_wf("-d", str(self.root), "best", "review", "s2",
                      "-g", "4")
        self.assertEqual([], list(fx.slot(self.opts, ["p2", "queued"]).iterdir()))

        self._write(top, "keep,known\nhard,no\nnew,pair\n", 30)
        self._write(fx.slot(self.opts, ["p2", "done"]) / "p2_done.pairs",
                    "keep,known\n")
        with mock.patch.object(best.evaluate.P2, "prepare") as prepare:
            code, stdout, stderr = fx.run_wf(
                "-d", str(self.root), "best", "review", "s2", "-g", "4")

        self.assertEqual(0, code, stderr)
        bundle_name = "top.s2.m4.g4.3.r1"
        bundle_dir = fx.slot(self.opts, ["p2", "eval"]) / bundle_name
        source = bundle_dir / f"{bundle_name}.pairs"
        self.assertEqual("keep,known\nnew,pair\n", source.read_text())
        self.assertFalse(source.with_name(source.name + ".filtered").exists())
        prepare.assert_called_once()
        self.assertIn(f"review awaiting completion ({bundle_name})", stdout)

        with self.assertRaisesRegex(ValueError, "already in flight"):
            fx.run_wf("-d", str(self.root), "best", "review", "s2",
                      "-g", "4")

        archived = fx.slot(self.opts, ["p2", "done", "in"]) / source.name
        source.rename(archived)
        bundle_dir.rmdir()
        with mock.patch.object(best.evaluate.P2, "prepare"):
            code, _, stderr = fx.run_wf(
                "-d", str(self.root), "best", "review", "s2", "-g", "4")
        self.assertEqual(0, code, stderr)
        self.assertTrue(
            (fx.slot(self.opts, ["p2", "eval"])
             / "top.s2.m4.g4.3.r2").is_dir())

    def test_gen_best_pairs_accumulates_and_preserves_unchanged_mtime(self):
        target = self._target()
        self._complete_files(target)
        top = self._write(target / "top.segments", "a,b\nbad,pair\n", 30)
        old_best = self._write(
            target / "best.pairs", "bad,pair\nx,y\n", 50)
        self._write(config.classified(self.root, "yes"),
                    "a,b\nbad,pair\nx,y\n", 45)
        self._write(config.classified(self.root, "no"), "bad,pair\n", 10)

        with self.assertRaisesRegex(ValueError, "-n is not valid"):
            fx.run_wf("-d", str(self.root), "best", "gen", "s2",
                      "-g", "4", "-n", "2", "best.pairs")

        code, stdout, stderr = fx.run_wf(
            "-d", str(self.root), "best", "gen", "s2",
            "-g", "4", "best.pairs")
        self.assertEqual(0, code, stderr)
        self.assertEqual("a,b\nx,y\n", old_best.read_text())
        self.assertEqual("a,b\nbad,pair\n", top.read_text())
        self.assertIn("Generated 2 best pairs (1 added, 1 dropped)", stdout)
        self.assertIn("dfs.best out of date (best.pairs changed)", stdout)

        mtime = old_best.stat().st_mtime_ns
        code, stdout, stderr = fx.run_wf(
            "-d", str(self.root), "best", "gen", "s2",
            "-g", "4", "best.pairs")
        self.assertEqual(0, code, stderr)
        self.assertEqual(mtime, old_best.stat().st_mtime_ns)
        self.assertIn("Generated 2 best pairs (0 added, 0 dropped)", stdout)

    def test_complete_selects_target_bundle_and_generates_best_pairs(self):
        target = self._target()
        universe = target.parent
        self._shared_inputs(universe)
        self._write(target / "dfs.seed", "9 alpha,beta\n", 20)
        self._write(target / "top.segments", "a,b\nnew,pair\n", 30)
        self._write(config.classified(self.root, "yes"), "a,b\nnew,pair\n", 10)
        bundle_name = "top.s2.m4.g4.2.r1"
        bundle_dir = fx.slot(self.opts, ["p2", "eval"]) / bundle_name
        bundle_dir.mkdir()

        def complete(command_text, opts, argv):
            self.assertEqual(("complete p2", [bundle_name]),
                             (command_text, argv))
            bundle_dir.rmdir()
            self._write(
                fx.slot(self.opts, ["p2", "done", "in"])
                / f"{bundle_name}.pairs", "a,b\nnew,pair\n", 40)
            return 0

        with mock.patch.object(best.complete_phase.P2, "run",
                               side_effect=complete) as run:
            code, stdout, stderr = fx.run_wf(
                "-d", str(self.root), "best", "complete", "s2", "-g", "4")

        self.assertEqual(0, code, stderr)
        run.assert_called_once()
        self.assertEqual("a,b\nnew,pair\n",
                         (target / "best.pairs").read_text())
        self.assertIn("Generated 2 best pairs (2 added, 0 dropped)", stdout)
        self.assertIn("s2/m4/g4: dfs.best missing", stdout)


if __name__ == "__main__":
    unittest.main()
