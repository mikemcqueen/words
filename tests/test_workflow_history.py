import os
import tempfile
import unittest

from pathlib import Path

from tests import wf_fixture as fx
from workflow import config, generation


class HistoryTests(unittest.TestCase):
    def test_reports_artifact_backed_history_in_time_order_and_limits_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            opts, _ = fx.make_wf(root)

            def write(path: Path, text: str, mtime: int) -> Path:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
                os.utime(path, (mtime, mtime))
                return path

            # The dictionary has a surviving reviewed record but its editable
            # removal record was deleted after the last generation marker.
            write(config.base_dictionary(root),
                  "apple\nbanana\ncherry\n", 100)
            derived = write(config.dictionary(root), "apple\ncherry\n", 120)
            write(config.reviewed_words(root), "banana\n", 120)
            write(generation.stamp(derived), "", 250)
            write(config.reviewed_inputs(root) / "round.reviewed.3",
                  "banana\ncherry\n", 260)
            os.utime(config.removals(root), (280, 280))

            # One dictionary review is both the latest queued source and the
            # latest open bundle, and therefore renders as one logical event.
            words_bundle = fx.make_bundle(opts, "dict", "ranked-words")
            words_source = write(words_bundle / "ranked-words",
                                 " 2 banana\n 1 apple\n", 200)
            write(words_source.with_name(words_source.name + ".filtered"),
                  " 2 banana\n", 290)

            # A recognized BEST top review supplies the grouped P2 open event.
            target = (fx.slot(opts, ["best"]) /
                      "s2" / "u-cdef" / "m4" / "g4")
            target.mkdir(parents=True)
            review_name = "top.s2.m4.g4.u-cdef.1000.r1"
            review_bundle = fx.make_bundle(opts, "p2", review_name)
            review_source = write(review_bundle / f"{review_name}.pairs",
                                  "a,b\nc,d\ne,f\n", 220)
            write(review_source.with_name(review_source.name + ".filtered"),
                  "a,b\nc,d\n", 300)

            # This ordinary underscore-bearing queue name must be collected
            # without attempting names.queue_stem().
            write(fx.slot(opts, ["p2", "queued"]) / "manual_batch.pairs",
                  "x,y\n", 210)

            # A legacy completion can have YES but no explicit NO artifact.
            completed = "old.90.10"
            write(fx.slot(opts, ["p2", "done", "in"]) /
                  f"{completed}.p1.yes", "g,h\ni,j\n", 150)
            write(fx.slot(opts, ["p2", "done", "out"]) /
                  f"{completed}.p2.yes", "g,h\n", 270)
            write(fx.slot(opts, ["p2", "done", "out", "enex"]) /
                  completed / f"{completed}.p1.yes.aa.enex", "note", 265)

            write(config.classified(root, "yes"), "a,b\nc,d\n", 180)
            write(config.classified(root, "no"), "x,y\n", 170)

            # BEST clocks deliberately disagree with link/content mtimes.
            results = root / "results"
            results.mkdir()
            seed_results = write(results / "seed.results", "one\ntwo\n", 310)
            seed_link = target / "dfs.seed"
            seed_link.symlink_to(seed_results)
            os.utime(seed_link, (50, 50), follow_symlinks=False)

            best_results = write(results / "best.results", "best row\n", 390)
            (target / "dfs.best").symlink_to(best_results)
            write(target / "dfs.best.pairs", "a,b\n", 320)

            top = write(target / "top.segments", "a,b\nc,d\n", 130)
            write(generation.stamp(top), "best\n", 330)
            write(target / "best.pairs", "a,b\nc,d\n", 340)
            write(target / "no.pairs", "x,y\n", 350)

            code, output, stderr = fx.run_wf(
                "-d", str(root), "history", "50")
            self.assertEqual(0, code, stderr)
            self.assertLess(output.index("no.pairs changed"),
                            output.index("best.pairs changed"))
            self.assertLess(output.index("top.segments published"),
                            output.index("dfs.best published"))
            self.assertLess(output.index("dfs.best published"),
                            output.index("dfs.seed published"))
            self.assertLess(output.index("P2 queued and opened"),
                            output.index("Dictionary review queued and opened"))
            self.assertLess(output.index("Dictionary review queued and opened"),
                            output.index("P2 completed"))
            self.assertEqual(1, output.count("P2 queued and opened"))
            self.assertNotIn("P2 opened:", output)
            self.assertNotIn("queue time:", output)
            self.assertIn(
                "submitted: 3, filtered: 2, notes: 1", output)
            self.assertIn("BEST top review: s2/u-cdef/m4/g4, round 1", output)
            self.assertIn(
                "YES: 1 pairs; NO: not recorded; note parts: 1", output)
            self.assertIn("Dictionary round 3 recorded; publication is not confirmed",
                          output)
            self.assertIn("removal record was deleted", output)
            self.assertIn("removal records changed since the last rebuild", output)
            self.assertIn("rows: 2", output)
            self.assertIn("source: best", output)
            self.assertIn("content last changed:", output)
            self.assertIn("results: 2 rows", output)

            code, default, stderr = fx.run_wf("-d", str(root), "history")
            self.assertEqual(0, code, stderr)
            headings = [line for line in default.splitlines()
                        if not line.startswith("  ")]
            self.assertEqual(5, len(headings))
            self.assertNotIn("P2 queued and opened", default)

            code, smaller, stderr = fx.run_wf(
                "-d", str(root), "history", "2")
            self.assertEqual(0, code, stderr)
            headings = [line for line in smaller.splitlines()
                        if not line.startswith("  ")]
            self.assertEqual(2, len(headings))
            self.assertIn("no.pairs changed", smaller)
            self.assertIn("best.pairs changed", smaller)
            self.assertNotIn("top.segments published", smaller)


if __name__ == "__main__":
    unittest.main()
