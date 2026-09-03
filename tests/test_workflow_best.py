import io
import os
import re
import tempfile
import unittest

from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from tests import wf_fixture as fx
from workflow import config
from workflow.best import commands, generate, state


PRODUCERS = ("dfs-anagrams", "top-segments")


def only_producers(stub):
    """Wrap a subprocess.run stub so only the external producers are stubbed.

    setops shares this module's subprocess, and the final DFS leg runs sort
    and comm inside the command to build its --pairs -- as does every status
    report printed after one. Anything that is not a producer runs for real.
    """
    real = generate.subprocess.run

    def run(command, **kwargs):
        if command[0] not in PRODUCERS:
            return real(command, **kwargs)
        return stub(command, **kwargs)

    return run


class BestTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.opts, _ = fx.make_wf(self.root)
        self.best = fx.slot(self.opts, ["best"])

    def _target(self, sentence="s2", letters="u-cdef", m=4, g=4) -> Path:
        target = self.best / sentence / letters / f"m{m}" / f"g{g}"
        target.mkdir(parents=True)
        return target

    def _write(self, path: Path, text="", mtime=None) -> Path:
        path.write_text(text)
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        return path

    def _shared_inputs(self, universe: Path) -> tuple[Path, Path, Path]:
        index = self.best / "idx" / generate.INDEX_NAME
        dictionary = self.best / "dict" / generate.DICTIONARY_NAME
        sentence_dir = universe.parents[1]
        sentence_dir.mkdir(parents=True, exist_ok=True)
        seed = sentence_dir / f"seed.{universe.name}.idx2.85.15.pairs"
        self._write(index, "index\n")
        self._write(dictionary, "words\n")
        self._write(sentence_dir / "letters", "abcdef\n", 10)
        self._write(seed, "a,b\n", 10)
        os.utime(config.classified(self.root, "no"), (10, 10))
        return index, dictionary, seed

    def _complete_files(self, target: Path) -> None:
        universe_dir = target.parent
        letter_set_dir = universe_dir.parent
        sentence_dir = letter_set_dir.parent
        # The bag under u-cdef is `ab`, and `a,b` is a confirmed pair it can
        # spell: the union dfs.best searches with is non-empty without any
        # best.pairs, which nothing generates and need not be there.
        self._write(config.classified(self.root, "yes"), "a,b\n", 10)
        os.utime(config.classified(self.root, "no"), (10, 10))
        self._write(sentence_dir / "letters", "abcdef\n", 10)
        self._write(
            sentence_dir / f"seed.{universe_dir.name}.idx2.85.15.pairs",
            "a,b\n", 10)
        results = self.root / "results"
        results.mkdir()
        dfs_seed = self._write(results / "dfs.seed.out", "1 a b\n", 20)
        (target / "dfs.seed").symlink_to(dfs_seed)
        self._write(target / "top.segments", "a,b\n", 30)
        prefix = (f"top.{sentence_dir.name}.{universe_dir.name}."
                  f"{target.name}.{letter_set_dir.name}.1000.r1.pairs")
        self._write(fx.slot(self.opts, ["p2", "done", "in"]) / prefix,
                    "a,b\n", 40)
        dfs_best = self._write(results / "dfs.best.out", "1 a b\n", 60)
        (target / "dfs.best").symlink_to(dfs_best)
        # What that search was weighted by, published beside its results.
        self._write(target / "dfs.best.pairs", "a,b\n", 60)
        # The frontier was last generated from dfs.best, after it landed: the
        # content clock stays at 30 because the regeneration was a no-op, and
        # the generation clock is what says the finished search was read.
        self._write(state._stamp(target / "top.segments"), "best\n", 65)

    def test_gen_help_describes_options_and_positionals(self):
        code, stdout, stderr = fx.run_wf(
            "-d", str(self.root), "best", "gen", "help")

        self.assertEqual(0, code, stderr)
        self.assertRegex(
            stdout, r"SENTENCE\s+sentence identifier under \.wf/best")
        self.assertRegex(stdout, r"STAGE\s+artifact to generate: dfs\.seed")
        self.assertIn("-g COUNT", stdout)
        self.assertIn("number of segments", stdout)
        self.assertIn("-m LENGTH", stdout)
        self.assertIn("min word length", stdout)
        self.assertIn("-n COUNT", stdout)
        self.assertIn("maximum results to output", stdout)
        rendered = "".join(line.strip() for line in stdout.splitlines())
        self.assertIn("dfs-anagrams and top-segments", rendered)

    def test_review_help_describes_the_optional_pairs_file(self):
        code, stdout, stderr = fx.run_wf(
            "-d", str(self.root), "best", "review", "help")

        self.assertEqual(0, code, stderr)
        self.assertIn("SENTENCE [PAIRS-FILE]", stdout)
        self.assertRegex(stdout, r"PAIRS-FILE\s+optional one-off pairs file")

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
            "-d", str(self.root), "best", "status", "s2/u-cdef/m4")
        self.assertEqual(0, code, stderr)
        self.assertEqual(
            ["s2/u-cdef/m4/g4: letters missing",
             "s2/u-cdef/m4/g5: letters missing"],
            [line for line in stdout.splitlines() if not line.startswith("  ")])

        with self.assertRaisesRegex(ValueError, r"expected \[ou\]-<letters>"):
            fx.run_wf("-d", str(self.root), "best", "status", "s2/m4")
        with self.assertRaisesRegex(ValueError, "expected m<N>"):
            fx.run_wf("-d", str(self.root), "best", "status", "s2/u-cdef/g4")

    def test_status_all_reports_every_row_and_leaves_the_verdict_alone(self):
        self._target()
        plain_code, plain, stderr = fx.run_wf(
            "-d", str(self.root), "best", "status", "s2/u-cdef/m4/g4")
        code, stdout, stderr = fx.run_wf(
            "-d", str(self.root), "best", "status", "--all",
            "s2/u-cdef/m4/g4")
        self.assertEqual(0, code, stderr)
        self.assertEqual(plain_code, code)
        # The report is unchanged; the table is added under it.
        self.assertTrue(stdout.startswith(plain), stdout)
        self.assertNotIn("rows:", plain)

        table = stdout[len(plain):].splitlines()
        self.assertEqual("  rows:", table[0])
        rows = [line for line in table[1:]
                if any(line.startswith(f"    {label}:")
                       for label in ("won", "also", "no", "n/a"))]
        self.assertEqual(len(state.ROWS), len(rows))
        # Nothing is placed, so the first row wins and the rows reading files
        # it reported missing are named rather than run.
        self.assertIn("won:", table[1])
        self.assertIn("_letters_missing", table[1])
        self.assertIn("not asked (needs top.segments)",
                      next(line for line in table if "_review_needed" in line))
        # A fired row carries the command it offers, under it.
        frontier = next(i for i, line in enumerate(table)
                        if "_no_frontier" in line)
        self.assertIn("also:", table[frontier])
        self.assertEqual("      next: wf best prepare s2 -u cdef -g 4 "
                         "--source seed", table[frontier + 1])

        # -a is the same switch.
        self.assertEqual(
            stdout,
            fx.run_wf("-d", str(self.root), "best", "status", "-a",
                      "s2/u-cdef/m4/g4")[1])

    def test_status_reports_no_search_results_and_a_dangling_link(self):
        target = self._target()
        self._write(target.parents[2] / "letters", "abcdef\n", 10)
        self._write(target.parents[2] / "seed.m4.idx2.85.15.pairs",
                    "a,b\n", 10)
        missing = self.root / "gone" / "dfs.out"
        (target / "dfs.seed").symlink_to(missing)

        code, stdout, stderr = fx.run_wf(
            "-d", str(self.root), "best", "status", "s2/u-cdef/m4/g4")
        self.assertEqual(0, code, stderr)
        self.assertIn(
            f"no search results yet (dangling symlink: {missing})", stdout)
        self.assertIn(
            "next: wf best prepare s2 -u cdef -g 4 --source seed", stdout)

        # One search result and no frontier is a generation, not a search:
        # the bootstrap gate is "no results at all", not "no dfs.seed".
        (target / "dfs.seed").unlink()
        self._write(target / "dfs.seed", "1 a b\n", 20)
        code, stdout, stderr = fx.run_wf(
            "-d", str(self.root), "best", "status", "s2/u-cdef/m4/g4")
        self.assertEqual(0, code, stderr)
        self.assertIn("s2/u-cdef/m4/g4: top.segments missing", stdout)
        self.assertIn(
            "next: wf best gen s2 -u cdef -g 4 top.segments --source seed",
            stdout)

        # Two, and the operator picks which frontier to review.
        self._write(target / "dfs.best", "1 a b\n", 20)
        code, stdout, stderr = fx.run_wf(
            "-d", str(self.root), "best", "status", "s2/u-cdef/m4/g4")
        self.assertEqual(0, code, stderr)
        self.assertIn("choose next:", stdout)
        self.assertIn(
            "seed: wf best gen s2 -u cdef -g 4 top.segments --source seed",
            stdout)
        self.assertIn(
            "best: wf best gen s2 -u cdef -g 4 top.segments --source best",
            stdout)

    def test_status_derives_review_gate_and_fully_fresh_state(self):
        target = self._target()
        self._complete_files(target)

        code, stdout, stderr = fx.run_wf(
            "-d", str(self.root), "best", "status", "s2/u-cdef/m4/g4")
        self.assertEqual(0, code, stderr)
        self.assertIn("s2/u-cdef/m4/g4: converged", stdout)

        os.utime(target / "dfs.seed", (70, 70))
        _, stdout, _ = fx.run_wf(
            "-d", str(self.root), "best", "status", "s2/u-cdef/m4/g4")
        self.assertIn("dfs.seed generated after top.segments", stdout)
        os.utime(target / "dfs.seed", (20, 20))

        confirmed_yes = config.classified(self.root, "yes")
        os.utime(confirmed_yes, (70, 70))
        _, stdout, _ = fx.run_wf(
            "-d", str(self.root), "best", "status", "s2/u-cdef/m4/g4")
        self.assertIn(
            "top.segments behind the classified sets "
            "(confirmed-YES set changed)", stdout)
        os.utime(confirmed_yes, (10, 10))

        # A hand-added pair the bag can spell is one dfs.best never saw, and
        # the published list is what says so.
        best_pairs = self._write(target / "best.pairs", "b,a\n", 65)
        _, stdout, _ = fx.run_wf(
            "-d", str(self.root), "best", "status", "s2/u-cdef/m4/g4")
        self.assertIn("dfs.best out of date (usable pair set changed)", stdout)
        best_pairs.unlink()

        done = fx.slot(self.opts, ["p2", "done", "in"])
        next(done.iterdir()).unlink()
        code, stdout, stderr = fx.run_wf(
            "-d", str(self.root), "best", "status", "s2/u-cdef/m4/g4")
        self.assertEqual(0, code, stderr)
        self.assertIn(
            "s2/u-cdef/m4/g4: review needed (frontier from best)", stdout)
        self.assertIn("next: wf best review s2 -u cdef -g 4", stdout)

        queued = fx.slot(self.opts, ["p2", "queued"])
        self._write(queued / "top.s2.m4.g4.u-cdef.1000.r1.pairs", "a,b\n")
        code, stdout, stderr = fx.run_wf(
            "-d", str(self.root), "best", "status", "s2/u-cdef/m4/g4")
        self.assertEqual(0, code, stderr)
        self.assertIn(
            "review submitted (top.s2.m4.g4.u-cdef.1000.r1.pairs)", stdout)
        self.assertIn(
            "next: wf eval p2 top.s2.m4.g4.u-cdef.1000.r1.pairs", stdout)

        queued_file = queued / "top.s2.m4.g4.u-cdef.1000.r1.pairs"
        evaluating = fx.slot(self.opts, ["p2", "eval"]) / queued_file.stem
        evaluating.mkdir()
        queued_file.rename(evaluating / queued_file.name)
        code, stdout, stderr = fx.run_wf(
            "-d", str(self.root), "best", "status", "s2/u-cdef/m4/g4")
        self.assertEqual(0, code, stderr)
        self.assertIn(
            "review awaiting completion (top.s2.m4.g4.u-cdef.1000.r1)", stdout)
        self.assertIn("next: wf best complete s2 -u cdef -g 4", stdout)

    def test_gen_dfs_seed_validates_then_creates_and_publishes_atomically(self):
        universe = self.best / "s2" / "u-cdef" / "m4"
        universe.mkdir(parents=True)
        _, _, seed = self._shared_inputs(universe)
        (self.best / "idx" / generate.INDEX_NAME).unlink()
        results = self.root / "results"
        results.mkdir()
        target = universe / "g4"
        calls = []

        def run(argv, **kwargs):
            calls.append(argv)
            kwargs["stdout"].write("9 alpha,beta\n8 gamma,delta\n")
            return mock.Mock(returncode=0)

        with mock.patch.object(generate.shutil, "which", return_value="/bin/fake"):
            with self.assertRaises(FileNotFoundError):
                fx.run_wf("-d", str(self.root), "best", "gen", "s2",
                          "-u", "cdef", "-g", "4", "-r", str(results),
                          "dfs.seed")
            self.assertFalse(target.exists())
            with self.assertRaises(FileNotFoundError):
                fx.run_wf("-d", str(self.root), "-f", "best", "gen", "s2",
                          "-u", "cdef", "-g", "4", "-r", str(results),
                          "dfs.seed")
        self.assertFalse(target.exists())

        self._write(self.best / "idx" / generate.INDEX_NAME, "index\n")
        with mock.patch.object(generate.shutil, "which", return_value="/bin/fake"), \
                mock.patch.object(generate.subprocess, "run",
                                  side_effect=only_producers(run)), \
                mock.patch.object(generate.time, "monotonic", side_effect=[0, 2]):
            code, stdout, stderr = fx.run_wf(
                "-d", str(self.root), "-f", "best", "gen", "s2", "-u", "cdef",
                "-g", "4", "-r", str(results), "dfs.seed")

        self.assertEqual(0, code, stderr)
        link = target / "dfs.seed"
        self.assertTrue(link.is_symlink())
        rendered = results / "s2" / "dfs.s2.idx2.85.15.m4.x2.g4.1000000.u-cdef"
        self.assertEqual(rendered, link.resolve())
        self.assertEqual("9 alpha,beta\n8 gamma,delta\n", rendered.read_text())
        self.assertFalse(rendered.with_name(rendered.name + ".tmp").exists())
        self.assertEqual("dfs-anagrams", calls[0][0])
        self.assertEqual(str(seed), calls[0][calls[0].index("--pairs") + 1])
        self.assertEqual("1000000", calls[0][calls[0].index("-n") + 1])
        self.assertIn("$(cat ", stderr)
        self.assertIn("Generated 2 results in 2s", stdout)
        self.assertIn("s2/u-cdef/m4/g4: top.segments missing", stdout)

        def fail(argv, **kwargs):
            kwargs["stdout"].write("truncated\n")
            raise generate.subprocess.CalledProcessError(1, argv)

        with mock.patch.object(generate.shutil, "which", return_value="/bin/fake"), \
                mock.patch.object(generate.subprocess, "run",
                                  side_effect=only_producers(fail)):
            with self.assertRaises(generate.subprocess.CalledProcessError):
                fx.run_wf("-d", str(self.root), "best", "gen", "s2",
                          "-u", "cdef", "-g", "4", "-r", str(results),
                          "dfs.seed")
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

        with mock.patch.object(generate.shutil, "which", return_value="/bin/fake"), \
                mock.patch.object(generate.subprocess, "run",
                                  side_effect=only_producers(run)):
            code, stdout, stderr = fx.run_wf(
                "-d", str(self.root), "best", "gen", "s2", "-u", "cdef",
                "-g", "4", "--source", "seed", "top.segments")
            top = target / "top.segments"
            os.utime(top, (30, 30))
            code2, stdout2, stderr2 = fx.run_wf(
                "-d", str(self.root), "best", "gen", "s2", "-u", "cdef",
                "-g", "4", "-n", "2", "--source", "seed", "top.segments")

        self.assertEqual((0, 0), (code, code2), stderr + stderr2)
        # --wfroot rejects the hard-NO lines whole and -y stops the standing
        # YES pairs from being counted, so the frontier holds candidates with
        # no verdict rather than rows already answered.
        self.assertEqual(
            ["top-segments", "--pairs", "--wfroot", str(self.root), "-y",
             str(dfs_seed)], calls[0])
        self.assertEqual(
            ["top-segments", "--pairs", "-n", "2", "--wfroot", str(self.root),
             "-y", str(dfs_seed)], calls[1])
        self.assertEqual(30, int(top.stat().st_mtime))
        self.assertIn("Generated 2 top segments", stdout + stdout2)
        self.assertIn("s2/u-cdef/m4/g4: review needed (frontier from seed)",
                      stdout2)
        self.assertIn(
            f"top-segments --pairs -n 2 --wfroot {self.root} -y {dfs_seed}",
            stderr2)
        # The marker records the source and advances even where the stable
        # placement left top.segments untouched.
        self.assertEqual("seed\n", state._stamp(top).read_text())
        self.assertGreater(state._stamp(top).stat().st_mtime_ns,
                           top.stat().st_mtime_ns)

    def test_no_op_top_segments_gen_clears_a_reran_dfs_seed(self):
        target = self._target()
        self._complete_files(target)
        top = target / "top.segments"

        def run(argv, **kwargs):
            kwargs["stdout"].write("a,b\n")
            return mock.Mock(returncode=0)

        # A dfs.seed rerun that exhausts the search renames its target fresh,
        # so a finished search sits ahead of the frontier on identical input.
        os.utime(target / "dfs.seed", (70, 70))
        _, stdout, _ = fx.run_wf(
            "-d", str(self.root), "best", "status", "s2/u-cdef/m4/g4")
        self.assertIn("dfs.seed generated after top.segments", stdout)
        self.assertIn(
            "next: wf best gen s2 -u cdef -g 4 top.segments --source seed",
            stdout)

        with mock.patch.object(generate.shutil, "which", return_value="/bin/fake"), \
                mock.patch.object(generate.subprocess, "run",
                                  side_effect=only_producers(run)):
            code, stdout, stderr = fx.run_wf(
                "-d", str(self.root), "best", "gen", "s2", "-u", "cdef",
                "-g", "4", "--source", "seed", "top.segments")

        self.assertEqual(0, code, stderr)
        self.assertEqual(30, int(top.stat().st_mtime))
        self.assertEqual("seed\n", state._stamp(top).read_text())
        self.assertNotIn("generated after top.segments", stdout)
        self.assertIn("s2/u-cdef/m4/g4: converged", stdout)

    def test_best_pairs_is_no_longer_a_stage_and_needs_no_generation(self):
        target = self._target()
        self._complete_files(target)

        code, _, stderr = fx.run_wf(
            "-d", str(self.root), "best", "gen", "s2", "-u", "cdef",
            "-g", "4", "best.pairs")
        self.assertEqual(2, code)
        self.assertIn("invalid argument: 'best.pairs'", stderr)

        # Absent is the ordinary shape, and a hand-edit is read straight into
        # the union without anything generating or dating it.
        self.assertFalse((target / "best.pairs").exists())
        _, stdout, _ = fx.run_wf(
            "-d", str(self.root), "best", "status", "s2/u-cdef/m4/g4")
        self.assertIn("s2/u-cdef/m4/g4: converged", stdout)
        self.assertNotIn("best.pairs", stdout)

    def test_status_reports_a_malformed_target_and_keeps_listing(self):
        universe = self._target().parent
        self._shared_inputs(universe)
        self._write(universe.parents[1] / "seed.m4.idx2.90.10.pairs",
                    "a,b\n", 10)
        self._target(sentence="s3")

        code, stdout, stderr = fx.run_wf("-d", str(self.root), "best", "status")

        self.assertEqual(1, code)
        self.assertIn("s2/u-cdef/m4/g4: multiple seeds in", stderr)
        self.assertIn("s3/u-cdef/m4/g4: letters missing", stdout)

    def test_gen_dfs_best_searches_with_the_bag_filtered_union(self):
        target = self._target()
        self._complete_files(target)
        self._shared_inputs(target.parent)
        # A hand-edited best.pairs joins classified/yes in the union, and a
        # standing NO comes back out of it. `alpha,beta` the bag cannot spell,
        # so it is filtered out and never reaches the search -- where it could
        # not have changed a score either, enumeration being bag-bounded.
        self._write(target / "best.pairs", "b,a\nalpha,beta\nbad,pair\n", 50)
        self._write(config.classified(self.root, "no"), "bad,pair\n", 10)
        results = self.root / "results"
        calls = []
        pairs_seen = []

        def run(argv, **kwargs):
            calls.append(argv)
            pairs_seen.append(
                Path(argv[argv.index("--pairs") + 1]).read_text())
            kwargs["stdout"].write("9 alpha,beta\n8 gamma,delta\n")
            return mock.Mock(returncode=0)

        with mock.patch.object(generate.shutil, "which", return_value="/bin/fake"), \
                mock.patch.object(generate.subprocess, "run",
                                  side_effect=only_producers(run)), \
                mock.patch.object(generate.time, "monotonic", side_effect=[0, 3]):
            code, stdout, stderr = fx.run_wf(
                "-d", str(self.root), "best", "gen", "s2", "-u", "cdef",
                "-g", "4", "-r", str(results), "-n", "25", "dfs.best")

        self.assertEqual(0, code, stderr)
        link = target / "dfs.best"
        rendered = (results / "s2"
                    / "dfs.s2.idx2.85.15.m4.x2.g4.best.25.u-cdef")
        self.assertTrue(link.is_symlink())
        self.assertEqual(rendered, link.resolve())
        self.assertEqual("9 alpha,beta\n8 gamma,delta\n", rendered.read_text())
        self.assertEqual(["a,b\nb,a\n"], pairs_seen)
        # The same list, published beside the results so a later status can
        # tell a classify that changed something here from one that did not.
        self.assertEqual("a,b\nb,a\n", (target / "dfs.best.pairs").read_text())
        self.assertEqual("25", calls[0][calls[0].index("-n") + 1])
        self.assertIn("Searched with 2 of 3 confirmed pairs", stdout)
        self.assertIn("Generated 2 results in 3s", stdout)
        # A finished search whose frontier was never generated: seconds of
        # work, and status offers it before the hours of another search.
        self.assertIn(
            "s2/u-cdef/m4/g4: dfs.best generated after top.segments", stdout)
        self.assertIn(
            "next: wf best gen s2 -u cdef -g 4 top.segments --source best",
            stdout)

        with mock.patch.object(generate.shutil, "which") as which:
            with self.assertRaisesRegex(ValueError,
                                        "only valid for gen dfs.seed"):
                fx.run_wf("-d", str(self.root), "-f", "best", "gen", "s2",
                          "-u", "cdef", "-g", "4", "-r", str(results),
                          "dfs.best")
        which.assert_not_called()

    def test_gen_top_segments_rejects_force_before_running(self):
        target = self._target()
        self._write(target / "dfs.seed", "9 alpha,beta\n")
        with mock.patch.object(generate.shutil, "which") as which:
            with self.assertRaisesRegex(ValueError, "only valid for gen dfs.seed"):
                fx.run_wf("-d", str(self.root), "-f", "best", "gen", "s2",
                          "-u", "cdef", "-g", "4", "--source", "seed",
                          "top.segments")
        which.assert_not_called()

    def test_exclude_classifies_hard_no_and_reports_target_status(self):
        target = self._target()
        self._complete_files(target)
        excluded = self._write(self.root / "excluded.pairs", "hard,no\n")

        code, stdout, stderr = fx.run_wf(
            "-d", str(self.root), "best", "exclude", "s2", "-u", "cdef",
            "-g", "4", str(excluded))

        self.assertEqual(0, code, stderr)
        self.assertEqual(
            "hard,no\n", config.classified(self.root, "no").read_text())
        self.assertIn("Classified NO: 1 new, 1 total", stdout)
        # The frontier comes first: refilling it costs seconds, and running
        # either search off one that is behind the verdicts costs hours.
        self.assertIn(
            "top.segments behind the classified sets (hard-NO set changed)",
            stdout)

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
                      "-u", "cdef", "-g", "4")
        self.assertEqual([], list(fx.slot(self.opts, ["p2", "queued"]).iterdir()))

        self._write(top, "keep,known\nhard,no\nnew,pair\n", 30)
        self._write(fx.slot(self.opts, ["p2", "done"]) / "p2_done.pairs",
                    "keep,known\n")
        with mock.patch.object(commands.evaluate.P2, "prepare") as prepare:
            code, stdout, stderr = fx.run_wf(
                "-d", str(self.root), "best", "review", "s2", "-u", "cdef",
                "-g", "4")

        self.assertEqual(0, code, stderr)
        bundle_name = "top.s2.m4.g4.u-cdef.3.r1"
        bundle_dir = fx.slot(self.opts, ["p2", "eval"]) / bundle_name
        source = bundle_dir / f"{bundle_name}.pairs"
        self.assertEqual("keep,known\nnew,pair\n", source.read_text())
        self.assertFalse(source.with_name(source.name + ".filtered").exists())
        prepare.assert_called_once()
        self.assertIn(f"review awaiting completion ({bundle_name})", stdout)

        with self.assertRaisesRegex(ValueError, "already in flight"):
            fx.run_wf("-d", str(self.root), "best", "review", "s2",
                      "-u", "cdef", "-g", "4")

        archived = fx.slot(self.opts, ["p2", "done", "in"]) / source.name
        source.rename(archived)
        bundle_dir.rmdir()
        with mock.patch.object(commands.evaluate.P2, "prepare"):
            code, _, stderr = fx.run_wf(
                "-d", str(self.root), "best", "review", "s2", "-u", "cdef",
                "-g", "4")
        self.assertEqual(0, code, stderr)
        self.assertTrue(
            (fx.slot(self.opts, ["p2", "eval"])
             / "top.s2.m4.g4.u-cdef.3.r2").is_dir())

    def test_oneoff_review_keeps_the_full_source_and_evaluates_only_unknowns(self):
        self._target()
        supplied = self._write(
            self.root / "outside-name.txt",
            "yes,known\nunknown,pair\nno,known\nyes,known\n")
        self._write(config.classified(self.root, "yes"), "yes,known\n")
        self._write(config.classified(self.root, "no"), "no,known\n")

        with mock.patch.object(commands.evaluate.P2, "prepare") as prepare:
            code, stdout, stderr = fx.run_wf(
                "-d", str(self.root), "best", "review", "s2", "-u", "cdef",
                "-g", "4", str(supplied))

        self.assertEqual(0, code, stderr)
        bundle_name = "oneoff.s2.m4.g4.u-cdef.3.r1"
        bundle = fx.slot(self.opts, ["p2", "eval"]) / bundle_name
        source = bundle / f"{bundle_name}.pairs"
        filtered = source.with_name(source.name + ".filtered")
        self.assertEqual(
            "no,known\nunknown,pair\nyes,known\n", source.read_text())

        with self.assertRaisesRegex(ValueError, "already in flight"):
            fx.run_wf("-d", str(self.root), "best", "review", "s2",
                      "-u", "cdef", "-g", "4")
        self.assertEqual("unknown,pair\n", filtered.read_text())
        self.assertEqual(filtered, prepare.call_args.args[0])
        self.assertIn("one-off review in flight", stdout)

        supplied.unlink()
        self.assertEqual(
            "no,known\nunknown,pair\nyes,known\n", source.read_text())

    def test_review_kinds_have_independent_round_sequences(self):
        target_dir = self._target()
        self._write(target_dir / "top.segments", "top,new\n")
        done = fx.slot(self.opts, ["p2", "done", "in"])
        self._write(done / "top.s2.m4.g4.u-cdef.1.r7.pairs", "old,top\n")
        self._write(
            done / "oneoff.s2.m4.g4.u-cdef.1.r1.pairs", "old,oneoff\n")
        supplied = self._write(self.root / "supplied.pairs", "oneoff,new\n")

        with mock.patch.object(commands.evaluate.P2, "prepare"):
            code, _, stderr = fx.run_wf(
                "-d", str(self.root), "best", "review", "s2", "-u", "cdef",
                "-g", "4", str(supplied))
        self.assertEqual(0, code, stderr)
        oneoff = (fx.slot(self.opts, ["p2", "eval"])
                  / "oneoff.s2.m4.g4.u-cdef.1.r2")
        self.assertTrue(oneoff.is_dir())

        for child in oneoff.iterdir():
            child.unlink()
        oneoff.rmdir()
        with mock.patch.object(commands.evaluate.P2, "prepare"):
            code, _, stderr = fx.run_wf(
                "-d", str(self.root), "best", "review", "s2", "-u", "cdef",
                "-g", "4")
        self.assertEqual(0, code, stderr)
        self.assertTrue((fx.slot(self.opts, ["p2", "eval"])
                         / "top.s2.m4.g4.u-cdef.1.r8").is_dir())

    def test_round_discovery_is_typed_and_rejects_duplicates_within_a_kind(self):
        self._target()
        done = fx.slot(self.opts, ["p2", "done", "in"])
        top = self._write(done / "top.s2.m4.g4.u-cdef.2.r1.pairs", "a,b\n")
        oneoff = self._write(
            done / "oneoff.s2.m4.g4.u-cdef.2.r1.pairs", "c,d\n")
        target = state.one_target(self.root, "s2", "u-cdef", 4, 4)

        _, _, rounds = state.review_locations(target)
        self.assertEqual(
            [(oneoff, "oneoff", 1), (top, "top", 1)],
            [(round_.path, round_.kind, round_.ordinal) for round_ in rounds])

        self._write(done / "top.s2.m4.g4.u-cdef.3.r1.pairs", "e,f\n")
        with self.assertRaisesRegex(ValueError, "two top review rounds"):
            state.review_locations(target)

    def test_oneoff_review_rejects_empty_subset_without_moving_state(self):
        self._target()
        supplied = self._write(
            self.root / "classified.pairs", "yes,known\nno,known\n")
        self._write(config.classified(self.root, "yes"), "yes,known\n")
        self._write(config.classified(self.root, "no"), "no,known\n")

        with self.assertRaisesRegex(ValueError, "all 2 pairs"):
            fx.run_wf("-d", str(self.root), "best", "review", "s2",
                      "-u", "cdef", "-g", "4", str(supplied))
        self.assertEqual([], list(fx.slot(self.opts, ["p2", "queued"]).iterdir()))
        self.assertEqual([], list(fx.slot(self.opts, ["p2", "eval"]).iterdir()))

    # ------------------------------------------------------------ best notes

    ROUND_1 = "top.s2.m4.g4.u-cdef.1000.r1"

    def _run_notes(self, *extra):
        """`best notes`, with the primitive stubbed: what it is handed is the
        composite's whole job."""
        with mock.patch.object(commands.notes.P2, "run",
                               return_value=0) as run:
            code, stdout, stderr = fx.run_wf(
                "-d", str(self.root), *extra, "best", "notes", "s2",
                "-u", "cdef", "-g", "4")
        self.assertEqual(0, code, stderr)
        return run.call_args.args[2], stdout

    def test_notes_re_notes_the_in_flight_round(self):
        target = self._target()
        self._complete_files(target)
        evaluating = (fx.slot(self.opts, ["p2", "eval"])
                      / "top.s2.m4.g4.u-cdef.1000.r2")
        evaluating.mkdir()
        self._write(evaluating / f"{evaluating.name}.pairs", "a,b\n")

        # The bundle no longer holds confirmed-YES pairs -- best review
        # subtracts them -- so there is nothing left for --yes-pairs to mark
        # and no wf best command types it.
        argv, stdout = self._run_notes()
        self.assertEqual([evaluating.name], argv)
        self.assertIn("review awaiting completion", stdout)

        self._write(target / "best.pairs", "b,a\n")
        argv, _ = self._run_notes()
        self.assertEqual([evaluating.name], argv)

    def test_oneoff_notes_use_filtered_input_and_archived_recreation_is_refused(self):
        self._target()
        bundle_name = "oneoff.s2.m4.g4.u-cdef.2.r1"
        evaluating = fx.slot(self.opts, ["p2", "eval"]) / bundle_name
        evaluating.mkdir()
        source = self._write(evaluating / f"{bundle_name}.pairs", "a,b\nc,d\n")
        filtered = self._write(
            source.with_name(source.name + ".filtered"), "c,d\n")

        with mock.patch.object(commands.notes, "make", return_value=[]) as make:
            code, _, stderr = fx.run_wf(
                "-d", str(self.root), "best", "notes", "s2", "-u", "cdef",
                "-g", "4")
        self.assertEqual(0, code, stderr)
        self.assertEqual(filtered, make.call_args.args[0])
        self.assertEqual(f"{filtered.name}.aa", commands.notes.title(filtered, 0))

        filtered.unlink()
        archived = fx.slot(self.opts, ["p2", "done", "in"]) / source.name
        source.rename(archived)
        evaluating.rmdir()
        with self.assertRaisesRegex(
                ValueError, re.escape(f"archived one-off source: {archived}")):
            fx.run_wf("-d", str(self.root), "-f", "best", "notes", "s2",
                      "-u", "cdef", "-g", "4")

    def test_notes_selects_the_highest_round_across_a_gapped_archive(self):
        target = self._target()
        self._complete_files(target)
        done_in = fx.slot(self.opts, ["p2", "done", "in"])
        # r1 is _complete_files'; r3 leaves r2 as a gap, which is what makes
        # counting and sorting both wrong.
        self._write(done_in / "top.s2.m4.g4.u-cdef.1000.r3.pairs", "a,b\n", 40)
        self._write(done_in / "top.s2.m4.g4.u-cdef.1000.r10.pairs", "a,b\n", 40)

        argv, _ = self._run_notes("-f")
        self.assertEqual("top.s2.m4.g4.u-cdef.1000.r10", argv[0])

        # And the next review is numbered past it, not by how many there are.
        self._write(target / "top.segments", "a,b\nnew,pair\n", 30)
        with mock.patch.object(commands.evaluate.P2, "prepare"):
            code, _, stderr = fx.run_wf(
                "-d", str(self.root), "best", "review", "s2", "-u", "cdef",
                "-g", "4")
        self.assertEqual(0, code, stderr)
        self.assertTrue((fx.slot(self.opts, ["p2", "eval"])
                         / "top.s2.m4.g4.u-cdef.2.r11").is_dir())

    def test_notes_refuses_a_target_with_no_review_at_all(self):
        target = self._target()
        self._complete_files(target)
        next(fx.slot(self.opts, ["p2", "done", "in"]).iterdir()).unlink()
        with self.assertRaisesRegex(ValueError, "no review to recreate"):
            fx.run_wf("-d", str(self.root), "best", "notes", "s2",
                      "-u", "cdef", "-g", "4")

    def test_a_queued_review_is_refused_by_notes_and_complete_alike(self):
        target = self._target()
        self._complete_files(target)
        queued = self._write(
            fx.slot(self.opts, ["p2", "queued"])
            / "top.s2.m4.g4.u-cdef.1000.r2.pairs", "a,b\n")

        # Both refusals render the eval the same way, so what the operator is
        # told to type is what best review would have run.
        expected = re.escape(f"wf eval p2 {queued.name}") + "$"
        for verb in ("notes", "complete"):
            with self.subTest(verb=verb):
                with self.assertRaisesRegex(ValueError, expected):
                    fx.run_wf("-d", str(self.root), "best", verb, "s2",
                              "-u", "cdef", "-g", "4")
        self.assertTrue(queued.is_file())

    def test_complete_selects_the_target_bundle_and_reports_what_moved(self):
        target = self._target()
        universe = target.parent
        self._shared_inputs(universe)
        self._write(target / "dfs.seed", "9 alpha,beta\n", 20)
        self._write(target / "top.segments", "a,b\nnew,pair\n", 30)
        self._write(config.classified(self.root, "yes"), "a,b\nnew,pair\n", 10)
        bundle_name = "top.s2.m4.g4.u-cdef.2.r1"
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

        with mock.patch.object(commands.complete_phase.P2, "run",
                               side_effect=complete) as run:
            code, stdout, stderr = fx.run_wf(
                "-d", str(self.root), "best", "complete", "s2", "-u", "cdef",
                "-g", "4")

        self.assertEqual(0, code, stderr)
        run.assert_called_once()
        # Nothing derives a per-target set: the round's verdicts are already
        # in the classified sets and reach --pairs from there.
        self.assertFalse((target / "best.pairs").exists())
        self.assertIn("s2/u-cdef/m4/g4: dfs.best missing", stdout)

    def test_gen_dfs_seed_passes_the_letter_set_both_ways(self):
        self._shared_inputs(self.best / "s2" / "u-cdef" / "m4")
        results = self.root / "results"
        results.mkdir()
        calls = []

        def run(argv, **kwargs):
            calls.append(argv)
            kwargs["stdout"].write("9 alpha,beta\n")
            return mock.Mock(returncode=0)

        with mock.patch.object(generate.shutil, "which", return_value="/bin/fake"), \
                mock.patch.object(generate.subprocess, "run",
                                  side_effect=only_producers(run)):
            code, _, used = fx.run_wf(
                "-d", str(self.root), "-f", "best", "gen", "s2", "-u", "cdef",
                "-g", "4", "-r", str(results), "dfs.seed")
            self.assertEqual(0, code, used)
            code, _, only = fx.run_wf(
                "-d", str(self.root), "-f", "best", "gen", "s2", "-o", "abc",
                "-g", "4", "-r", str(results), "dfs.seed")
            self.assertEqual(0, code, only)

        index = str(self.best / "idx" / generate.INDEX_NAME)
        self.assertEqual(["dfs-anagrams", index, "abcdef", "-u", "cdef"],
                         calls[0][:5])
        self.assertEqual(["dfs-anagrams", index, "abc", "-m"], calls[1][:4])
        self.assertNotIn("-u", calls[1])
        # Each form renders its own output path, so neither gen replaces the
        # other's results.
        rendered = "dfs.s2.idx2.85.15.m4.x2.g4.1000000"
        self.assertTrue((results / "s2" / f"{rendered}.u-cdef").is_file())
        self.assertTrue((results / "s2" / f"{rendered}.o-abc").is_file())
        # The frozen bag is abbreviated; an o- label is short and shown as is.
        self.assertIn("$(cat ", used)
        self.assertNotIn("$(cat ", only)
        self.assertIn(" abc -m 4", only)

    def test_letter_set_check_refuses_a_bad_or_duplicate_label(self):
        self._shared_inputs(self.best / "s2" / "u-cdef" / "m4")
        (self.best / "s2" / "u-cdef" / "m4" / "g4").mkdir(parents=True)

        def gen(*flags):
            return fx.run_wf("-d", str(self.root), "-f", "best", "gen", "s2",
                             *flags, "-g", "4", "dfs.seed")

        with self.assertRaisesRegex(ValueError, "not a subset"):
            gen("-u", "xyz")
        for form in ("-u", "-o"):
            with self.assertRaisesRegex(ValueError, "not a proper subset"):
                gen(form, "abcdef")
        # A transposition is an anagram, so it reduces to the sibling's bag.
        with self.assertRaisesRegex(ValueError, "same letters as u-cdef"):
            gen("-u", "cdfe")
        # o-ab describes that same working bag from the other end.
        with self.assertRaisesRegex(ValueError, "same letters as u-cdef"):
            gen("-o", "ab")
        # A typo that changes the bag passes every check that can be written;
        # the -f refusal is what catches it, naming the level it would create.
        with self.assertRaisesRegex(FileNotFoundError, "u-cdf"):
            fx.run_wf("-d", str(self.root), "best", "gen", "s2", "-u", "cdf",
                      "-g", "4", "dfs.seed")
        # An established letter set is never a duplicate of itself.
        code, stdout, stderr = fx.run_wf(
            "-d", str(self.root), "best", "status", "s2/u-cdef/m4/g4")
        self.assertEqual(0, code, stderr)
        self.assertIn("no search results yet", stdout)

    def test_status_synthesizes_a_fully_qualified_absent_target(self):
        self._shared_inputs(self.best / "s2" / "u-cdef" / "m4")

        code, stdout, stderr = fx.run_wf(
            "-d", str(self.root), "best", "status", "s2/u-cdef/m4/g4")
        self.assertEqual(0, code, stderr)
        self.assertIn("s2/u-cdef/m4/g4: no search results yet", stdout)
        self.assertIn(
            "next: wf best prepare s2 -u cdef -g 4 --source seed -f", stdout)

        # Which is where a duplicate label is found out, before any DFS.
        (self.best / "s2" / "u-cdef").mkdir()
        code, _, stderr = fx.run_wf(
            "-d", str(self.root), "best", "status", "s2/o-ab/m4/g4")
        self.assertEqual(1, code)
        self.assertIn("o-ab searches the same letters as u-cdef", stderr)

        # A partial prefix names a subtree, and an absent one stays an error.
        with self.assertRaises(FileNotFoundError):
            fx.run_wf("-d", str(self.root), "best", "status", "s2/u-typo")
        # So does an absent sentence, which holds both hand-placed files.
        with self.assertRaises(FileNotFoundError):
            fx.run_wf("-d", str(self.root), "best", "status",
                      "s9/u-cdef/m4/g4")

    def test_status_diagnoses_the_pre_letter_set_tree(self):
        stale = self.best / "s2" / "m4" / "g4"
        stale.mkdir(parents=True)

        with self.assertRaisesRegex(ValueError, "s2 predates the letter set"):
            fx.run_wf("-d", str(self.root), "best", "status")

        stale.rmdir()
        stale.parent.rmdir()
        code, stdout, stderr = fx.run_wf("-d", str(self.root), "best", "status")
        self.assertEqual(0, code, stderr)
        self.assertIn("no BEST PAIRS targets", stdout)

    # ---------------------------------------------------------- search source

    def _run_producers(self, outputs, *argv):
        """Drive a wf best command, standing in for its external producers."""
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            kwargs["stdout"].write(outputs[len(calls) - 1])
            return mock.Mock(returncode=0)

        with mock.patch.object(generate.shutil, "which",
                               return_value="/bin/fake"), \
                mock.patch.object(generate.subprocess, "run",
                                  side_effect=only_producers(run)):
            code, stdout, stderr = fx.run_wf("-d", str(self.root), *argv)
        return code, calls, stdout, stderr

    def test_source_is_required_by_top_segments_and_refused_elsewhere(self):
        target = self._target()
        self._complete_files(target)
        self._shared_inputs(target.parent)

        # One parser serves every stage, so --source parses everywhere and is
        # refused where it means nothing.
        for stage in ("dfs.seed", "dfs.best"):
            with self.subTest(stage=stage):
                with self.assertRaisesRegex(
                        ValueError, "only valid for gen top.segments"):
                    fx.run_wf("-d", str(self.root), "best", "gen", "s2",
                              "-u", "cdef", "-g", "4", "--source", "seed",
                              stage)

        code, _, stderr = fx.run_wf(
            "-d", str(self.root), "best", "gen", "s2", "-u", "cdef", "-g", "4",
            "top.segments")
        self.assertEqual(2, code)
        self.assertIn("missing required argument", stderr)

        with self.assertRaisesRegex(ValueError, "expected seed or best"):
            fx.run_wf("-d", str(self.root), "best", "gen", "s2", "-u", "cdef",
                      "-g", "4", "--source", "dfs.best", "top.segments")

    def test_gen_top_segments_reads_the_source_it_is_given(self):
        target = self._target()
        self._complete_files(target)

        for source in ("best", "seed"):
            with self.subTest(source=source):
                code, calls, stdout, _ = self._run_producers(
                    ["a,b\n"], "best", "gen", "s2", "-u", "cdef", "-g", "4",
                    "--source", source, "top.segments")
                self.assertEqual(0, code)
                self.assertEqual(
                    ["top-segments", "--pairs", "--wfroot", str(self.root),
                     "-y", str(target / f"dfs.{source}")],
                    calls[0])
                self.assertEqual(
                    f"{source}\n",
                    state._stamp(target / "top.segments").read_text())

    def test_gen_top_segments_warns_when_the_dfs_file_is_exhausted(self):
        target = self._target()
        self._complete_files(target)

        code, calls, _, stderr = self._run_producers(
            ["a,b\nc,d\n"], "best", "gen", "s2", "-u", "cdef", "-g", "4",
            "-n", "5", "--source", "seed", "top.segments")

        self.assertEqual(0, code)
        self.assertEqual(["-n", "5"], calls[0][2:4])
        self.assertIn("top.segments holds 2 of the 5 requested", stderr)
        self.assertIn("exhausted at this cutoff", stderr)

    # --------------------------------------------------------------- prepare

    def test_prepare_help_names_the_source_and_the_two_cutoffs(self):
        code, stdout, stderr = fx.run_wf(
            "-d", str(self.root), "best", "prepare", "help")
        self.assertEqual(0, code, stderr)
        self.assertIn("--source SOURCE", stdout)
        self.assertIn("--dfs-count COUNT", stdout)
        self.assertIn("--top-count COUNT", stdout)
        rendered = "".join(line.strip() for line in stdout.splitlines())
        self.assertIn("maximum dfs-anagrams results (default: 1000000)",
                      rendered)
        self.assertIn("maximum top.segments pairs (default: 1000)", rendered)

        code, stdout, stderr = fx.run_wf("-d", str(self.root), "best", "help")
        self.assertEqual(0, code, stderr)
        self.assertIn("prepare  — run a DFS search and generate the frontier",
                      stdout)

    def test_prepare_seed_runs_both_legs_with_the_default_cutoffs(self):
        universe = self.best / "s2" / "u-cdef" / "m4"
        _, _, seed = self._shared_inputs(universe)
        results = self.root / "results"
        results.mkdir()
        target = universe / "g4"

        code, calls, stdout, stderr = self._run_producers(
            ["9 alpha,beta\n8 gamma,delta\n", "alpha,beta\ngamma,delta\n"],
            "-f", "best", "prepare", "s2", "-u", "cdef", "-g", "4",
            "-r", str(results), "--source", "seed")

        self.assertEqual(0, code, stderr)
        self.assertEqual(["dfs-anagrams", "top-segments"],
                         [call[0] for call in calls])
        # Neither default changes what runs today: DFS_LIMIT is what
        # gen dfs.seed already used, and 1000 is top-segments' own default.
        self.assertEqual(str(seed), calls[0][calls[0].index("--pairs") + 1])
        self.assertEqual("1000000", calls[0][calls[0].index("-n") + 1])
        self.assertEqual(
            ["top-segments", "--pairs", "-n", "1000", "--wfroot",
             str(self.root), "-y", str(target / "dfs.seed")], calls[1])
        rendered = results / "s2" / "dfs.s2.idx2.85.15.m4.x2.g4.1000000.u-cdef"
        self.assertEqual(rendered, (target / "dfs.seed").resolve())
        self.assertEqual("alpha,beta\ngamma,delta\n",
                         (target / "top.segments").read_text())
        self.assertEqual("seed\n",
                         state._stamp(target / "top.segments").read_text())
        self.assertIn("s2/u-cdef/m4/g4: review needed (frontier from seed)",
                      stdout)

    def test_prepare_takes_the_two_cutoffs_independently(self):
        universe = self.best / "s2" / "u-cdef" / "m4"
        self._shared_inputs(universe)
        results = self.root / "results"
        results.mkdir()

        code, calls, _, stderr = self._run_producers(
            ["9 alpha,beta\n", "alpha,beta\n"],
            "-f", "best", "prepare", "s2", "-u", "cdef", "-g", "4",
            "-r", str(results), "--source", "seed",
            "--dfs-count", "7", "--top-count", "3")

        self.assertEqual(0, code, stderr)
        self.assertEqual("7", calls[0][calls[0].index("-n") + 1])
        self.assertEqual(["-n", "3"], calls[1][2:4])
        self.assertTrue(
            (results / "s2" / "dfs.s2.idx2.85.15.m4.x2.g4.7.u-cdef").is_file())

        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            fx.run_wf("-d", str(self.root), "-f", "best", "prepare", "s2",
                      "-u", "cdef", "-g", "4", "--source", "seed",
                      "--top-count", "-1")

    def test_prepare_best_weights_the_search_and_refuses_force(self):
        target = self._target()
        self._complete_files(target)
        self._shared_inputs(target.parent)
        results = self.root / "results"

        code, calls, stdout, stderr = self._run_producers(
            ["9 alpha,beta\n", "alpha,beta\n"],
            "best", "prepare", "s2", "-u", "cdef", "-g", "4",
            "-r", str(results), "--source", "best", "--dfs-count", "1",
            "--top-count", "1")

        self.assertEqual(0, code, stderr)
        self.assertEqual("a,b\n", (target / "dfs.best.pairs").read_text())
        self.assertEqual(str(target / "dfs.best"), calls[1][-1])
        self.assertEqual(
            results / "s2" / "dfs.s2.idx2.85.15.m4.x2.g4.best.1.u-cdef",
            (target / "dfs.best").resolve())
        self.assertEqual("best\n",
                         state._stamp(target / "top.segments").read_text())

        # -f is what creates the levels below the sentence, and only a seed
        # search may create them.
        with mock.patch.object(generate.shutil, "which") as which:
            with self.assertRaisesRegex(
                    ValueError, "only valid for prepare --source seed"):
                fx.run_wf("-d", str(self.root), "-f", "best", "prepare", "s2",
                          "-u", "cdef", "-g", "4", "-r", str(results),
                          "--source", "best")
        which.assert_not_called()

    def test_prepare_seed_creates_the_target_tree_only_under_force(self):
        universe = self.best / "s2" / "u-cdef" / "m4"
        self._shared_inputs(universe)
        results = self.root / "results"
        results.mkdir()

        with mock.patch.object(generate.shutil, "which") as which:
            with self.assertRaisesRegex(FileNotFoundError, "use -f to force"):
                fx.run_wf("-d", str(self.root), "best", "prepare", "s2",
                          "-u", "cdef", "-g", "4", "-r", str(results),
                          "--source", "seed")
        which.assert_not_called()
        self.assertFalse((universe / "g4").exists())

        code, _, _, stderr = self._run_producers(
            ["9 alpha,beta\n", "alpha,beta\n"],
            "-f", "best", "prepare", "s2", "-u", "cdef", "-g", "4",
            "-r", str(results), "--source", "seed")
        self.assertEqual(0, code, stderr)
        self.assertTrue((universe / "g4" / "top.segments").is_file())

    def test_a_union_this_bag_cannot_spell_refuses_both_final_searches(self):
        target = self._target()
        self._complete_files(target)
        self._shared_inputs(target.parent)
        # Confirmed pairs exist; none of them fits the `ab` bag, so dfs.best
        # would be a strictly worse dfs.seed and must not cost the hours.
        self._write(config.classified(self.root, "yes"), "x,y\nz,z\n", 10)
        results = self.root / "results"
        before = sorted((results).iterdir())
        started = []

        for argv in (["gen", "s2", "-u", "cdef", "-g", "4",
                      "-r", str(results), "dfs.best"],
                     ["prepare", "s2", "-u", "cdef", "-g", "4",
                      "-r", str(results), "--source", "best"]):
            with self.subTest(command=argv[0]):
                with mock.patch.object(generate.shutil, "which",
                                       return_value="/bin/fake"), \
                        mock.patch.object(
                            generate.subprocess, "run",
                            side_effect=only_producers(started.append)):
                    with self.assertRaisesRegex(
                            ValueError, "fits s2/u-cdef/m4/g4's letters"):
                        fx.run_wf("-d", str(self.root), "best", *argv)
        self.assertEqual([], started)
        self.assertEqual(before, sorted((results).iterdir()))
        # And the record of what the last finished search used is untouched:
        # only a run that finished writes it.
        self.assertEqual("a,b\n", (target / "dfs.best.pairs").read_text())

    # -------------------------------------------------------- review gating

    def test_a_review_in_flight_blocks_the_frontier_but_not_a_search(self):
        target = self._target()
        self._complete_files(target)
        self._shared_inputs(target.parent)
        results = self.root / "results"
        queued = self._write(
            fx.slot(self.opts, ["p2", "queued"])
            / "top.s2.m4.g4.u-cdef.1000.r2.pairs", "a,b\n")

        blocked = (
            ["gen", "s2", "-u", "cdef", "-g", "4", "--source", "seed",
             "top.segments"],
            ["prepare", "s2", "-u", "cdef", "-g", "4", "-r", str(results),
             "--source", "seed"],
            ["prepare", "s2", "-u", "cdef", "-g", "4", "-r", str(results),
             "--source", "best"],
        )
        for argv in blocked:
            with self.subTest(command=" ".join(argv[:1] + argv[-2:])):
                with mock.patch.object(generate.shutil, "which") as which:
                    with self.assertRaisesRegex(
                            ValueError, "review bundle in flight"):
                        fx.run_wf("-d", str(self.root), "best", *argv)
                which.assert_not_called()

        # A DFS run does not overwrite the frontier, so it stays allowed --
        # status simply never advertises it, because completing the round
        # would date it stale the moment it landed.
        code, calls, _, stderr = self._run_producers(
            ["9 alpha,beta\n"], "best", "gen", "s2", "-u", "cdef", "-g", "4",
            "-r", str(results), "dfs.seed")
        self.assertEqual(0, code, stderr)
        self.assertEqual("dfs-anagrams", calls[0][0])
        self.assertTrue(queued.is_file())

    def test_an_open_oneoff_allows_frontier_generation_and_is_a_status_footnote(self):
        target = self._target()
        self._complete_files(target)
        bundle = (fx.slot(self.opts, ["p2", "eval"])
                  / "oneoff.s2.m4.g4.u-cdef.1.r1")
        bundle.mkdir()
        self._write(bundle / f"{bundle.name}.pairs", "one,off\n")
        self._write(bundle / f"{bundle.name}.pairs.filtered", "one,off\n")

        code, calls, stdout, stderr = self._run_producers(
            ["new,frontier\n"], "best", "gen", "s2", "-u", "cdef",
            "-g", "4", "--source", "seed", "top.segments")

        self.assertEqual(0, code, stderr)
        self.assertEqual("top-segments", calls[0][0])
        self.assertIn("review needed (frontier from seed)", stdout)
        self.assertIn("next: wf best complete s2 -u cdef -g 4", stdout)
        self.assertIn(f"one-off review in flight ({bundle.name})", stdout)

    def test_a_failed_frontier_keeps_the_search_and_names_the_recovery(self):
        target_dir = self._target()
        self._complete_files(target_dir)
        self._shared_inputs(target_dir.parent)
        results = self.root / "results"
        target = state.one_target(self.opts.dir, "s2", "u-cdef", 4, 4)
        marker = state._stamp(target_dir / "top.segments")
        before = (target_dir / "top.segments").read_text()

        def run(command, **kwargs):
            if command[0] == "dfs-anagrams":
                kwargs["stdout"].write("9 alpha,beta\n")
                return mock.Mock(returncode=0)
            raise generate.subprocess.CalledProcessError(1, command)

        stderr = io.StringIO()
        with redirect_stderr(stderr), \
                mock.patch.object(generate.shutil, "which",
                                  return_value="/bin/fake"), \
                mock.patch.object(generate.subprocess, "run",
                                  side_effect=only_producers(run)):
            with self.assertRaises(generate.subprocess.CalledProcessError):
                generate.prepare(target, source="seed", force=False,
                                 results_dir=results, dfs_count=1,
                                 top_count=2)

        self.assertEqual(
            results / "s2" / "dfs.s2.idx2.85.15.m4.x2.g4.1.u-cdef",
            (target_dir / "dfs.seed").resolve())
        self.assertEqual(before, (target_dir / "top.segments").read_text())
        self.assertEqual("best\n", marker.read_text())
        self.assertEqual(65, int(marker.stat().st_mtime))
        self.assertIn(
            "rerun: wf best gen s2 -u cdef -g 4 top.segments --source seed "
            "-n 2", stderr.getvalue())

    # -------------------------------------------------- review bundle contents

    def test_review_excludes_both_standing_sets(self):
        target = self._target()
        self._complete_files(target)
        self._write(target / "top.segments",
                    "hard,no\nkeep,new\nyes,known\n", 30)
        self._write(config.classified(self.root, "no"), "hard,no\n", 10)
        self._write(config.classified(self.root, "yes"), "yes,known\n", 10)

        with mock.patch.object(commands.evaluate.P2, "prepare") as prepare:
            code, stdout, stderr = fx.run_wf(
                "-d", str(self.root), "best", "review", "s2", "-u", "cdef",
                "-g", "4")

        self.assertEqual(0, code, stderr)
        prepare.assert_called_once()
        bundle_name = "top.s2.m4.g4.u-cdef.3.r2"
        source = (fx.slot(self.opts, ["p2", "eval"]) / bundle_name
                  / f"{bundle_name}.pairs")
        # Re-confirming a standing YES buys nothing: the verdict is global
        # and reaches --pairs straight out of classified/yes.
        self.assertEqual("keep,new\n", source.read_text())

    def test_an_empty_review_bundle_converges_instead_of_raising(self):
        target = self._target()
        self._complete_files(target)
        self._write(target / "top.segments", "hard,no\nyes,known\n", 30)
        self._write(config.classified(self.root, "no"), "hard,no\n", 90)
        self._write(config.classified(self.root, "yes"), "a,b\nyes,known\n", 10)

        code, stdout, stderr = fx.run_wf(
            "-d", str(self.root), "best", "review", "s2", "-u", "cdef", "-g", "4")

        self.assertEqual(0, code, stderr)
        self.assertIn("s2/u-cdef/m4/g4: no review candidates remain "
                      "(2 frontier pairs, all already classified)", stdout)
        # The classify that produced those verdicts moved a classified set
        # past the frontier's marker, so refilling it is the cheap way out and
        # is named ahead of the hours.
        self.assertIn("choose next:", stdout)
        self.assertIn(
            "refresh: wf best gen s2 -u cdef -g 4 top.segments --source best",
            stdout)
        self.assertIn("reseed:  wf best prepare s2 -u cdef -g 4 --source seed",
                      stdout)
        self.assertIn("refine:  wf best prepare s2 -u cdef -g 4 --source best",
                      stdout)
        self.assertEqual([], list(fx.slot(self.opts, ["p2", "queued"]).iterdir()))
        self.assertEqual([], list(fx.slot(self.opts, ["p2", "eval"]).iterdir()))

    def test_no_best_command_types_yes_pairs_but_the_primitives_still_do(self):
        target = self._target()
        self._complete_files(target)
        self._write(target / "top.segments", "keep,new\n", 30)

        with mock.patch.object(commands.evaluate.P2, "run",
                               return_value=0) as run:
            code, stdout, stderr = fx.run_wf(
                "-d", str(self.root), "best", "review", "s2", "-u", "cdef",
                "-g", "4")
        self.assertEqual(0, code, stderr)
        self.assertNotIn("--yes-pairs", run.call_args.args[2])
        self.assertNotIn("--yes-pairs", stdout + stderr)

        # The plumbing stays wired for a later revival of the marked bundle.
        for command_text in (("eval", "p2"), ("notes", "p2")):
            with self.subTest(command=command_text):
                code, stdout, stderr = fx.run_wf(
                    "-d", str(self.root), *command_text, "help")
                self.assertEqual(0, code, stderr)
                self.assertIn("--yes-pairs", stdout)

    def test_letter_set_flag_is_required_and_singular(self):
        self._target()

        code, _, stderr = fx.run_wf(
            "-d", str(self.root), "best", "review", "s2", "-g", "4")
        self.assertEqual(2, code)
        self.assertIn("missing required argument", stderr)
        self.assertIn("wf best review", stderr)

        with self.assertRaisesRegex(ValueError, "give exactly one"):
            fx.run_wf("-d", str(self.root), "best", "review", "s2",
                      "-o", "abc", "-u", "cdef", "-g", "4")

        code, stdout, stderr = fx.run_wf(
            "-d", str(self.root), "best", "gen", "help")
        self.assertEqual(0, code, stderr)
        self.assertIn("-o LETTERS", stdout)
        self.assertIn("use only these letters", stdout)
        self.assertIn("-u LETTERS", stdout)
        self.assertIn("the sentence less these letters", stdout)


if __name__ == "__main__":
    unittest.main()
