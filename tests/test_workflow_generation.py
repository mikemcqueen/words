"""The generation clock: the marker that dates a derived artifact.

Shared by top.segments and by the derived dictionary, which is why it lives
outside BEST. The whole of what it exists for is that an artifact placed with
stable_mtime cannot date itself against its own inputs.
"""

import os
import tempfile
import unittest

from pathlib import Path

from workflow import generation


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def _artifact(self, name="top.segments", mtime=30) -> Path:
        path = self.dir / name
        path.write_text("a,b\n")
        os.utime(path, (mtime, mtime))
        return path

    def test_the_marker_is_a_hidden_sibling_of_what_it_dates(self):
        artifact = self._artifact()
        self.assertEqual(self.dir / ".top.segments.gen",
                         generation.stamp(artifact))
        # A dotted name, so it never matches a glob that collects the artifacts
        # beside it.
        self.assertTrue(generation.stamp(artifact).name.startswith("."))

    def test_a_missing_marker_leaves_the_artifact_dating_itself(self):
        """What a tree built before the marker, or a hand-placed file, does."""
        artifact = self._artifact()
        self.assertEqual(artifact, generation.generated(artifact))

        generation.mark_generated(artifact, "best\n")
        self.assertEqual(generation.stamp(artifact),
                         generation.generated(artifact))

    def test_a_marker_carries_what_the_generation_was_made_from(self):
        artifact = self._artifact()
        generation.mark_generated(artifact, "best\n")
        self.assertEqual("best\n", generation.stamp(artifact).read_text())
        # Contents are optional: the dictionary has one input and marks empty.
        generation.mark_generated(artifact)
        self.assertEqual("", generation.stamp(artifact).read_text())

    def test_a_byte_identical_remark_still_advances_the_clock(self):
        """Content is what stable_mtime pins; the clock has to move anyway.

        Without this the staleness rows that date against the marker are
        permanently stuck rows: the input moves, the regeneration is a no-op,
        and the one write that would clear the row is the one suppressed.
        """
        artifact = self._artifact(mtime=30)
        generation.mark_generated(artifact, "best\n")
        os.utime(generation.stamp(artifact), (30, 30))

        generation.mark_generated(artifact, "best\n")
        self.assertGreater(generation.stamp(artifact).stat().st_mtime_ns,
                           30 * 10 ** 9)
        self.assertEqual(30, int(artifact.stat().st_mtime))


if __name__ == "__main__":
    unittest.main()
