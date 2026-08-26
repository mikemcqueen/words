import os
import tempfile
import unittest

from pathlib import Path

from tests import wf_fixture as fx
from workflow import config


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


if __name__ == "__main__":
    unittest.main()
