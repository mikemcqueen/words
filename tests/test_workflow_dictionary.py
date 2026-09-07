"""The workflow-managed dictionary: what a round records, and what it derives.

Two records answering two questions -- what was removed, and what was looked at
-- and two derived files rebuilt from them. The properties worth holding on to
are that recording is cheap, that a rebuild which changes nothing costs nothing
downstream, and that a retraction is an edit plus one idempotent command.
"""

import tempfile
import unittest

from pathlib import Path
from unittest import mock

from tests import wf_fixture as fx
from workflow import config, dictionary, generation, setops


BASE = "apple\nbanana\ncherry\ndate\nfig\n"


class DictionaryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.opts, _ = fx.make_wf(self.root)
        self.base = config.base_dictionary(self.root)
        self.derived = config.dictionary(self.root)
        self.removals = config.removals(self.root)
        self.reviewed_inputs = config.reviewed_inputs(self.root)
        self.reviewed_words = config.reviewed_words(self.root)

    # ------------------------------------------------------------- placement

    def _base(self, text=BASE) -> Path:
        self.base.write_text(text)
        return self.base

    def _submission(self, name: str, text: str) -> Path:
        path = self.root / name
        path.write_text(text)
        return path

    def _remove(self, name: str, text: str) -> tuple[int, str, str]:
        return fx.run_wf("-d", str(self.root), "remove", "words",
                         str(self._submission(name, text)))

    def _gen(self) -> tuple[int, str, str]:
        return fx.run_wf("-d", str(self.root), "gen", "dict")

    def _lines(self, path: Path) -> list[str]:
        return path.read_text().splitlines()

    # ------------------------------------------------------------- recording

    def test_a_removal_drops_the_word_and_leaves_the_base_alone(self):
        self._base()
        code, stdout, stderr = self._remove("junk.txt", "banana\nfig\n")

        self.assertEqual(0, code, stderr)
        self.assertEqual(["apple", "cherry", "date"], self._lines(self.derived))
        self.assertEqual(BASE, self.base.read_text())
        self.assertIn("2 words submitted, 2 new to the removal union, "
                      "2 newly removed from the dictionary", stdout)

    def test_a_removal_of_a_word_not_in_the_base_costs_nothing_downstream(self):
        """The stable_mtime property: no search may be billed for a no-op."""
        self._base()
        self._remove("first.txt", "banana\n")
        placed = self.derived.stat().st_mtime_ns
        marker = generation.stamp(self.derived).stat().st_mtime_ns

        code, stdout, stderr = self._remove("second.txt", "notinbase\n")
        self.assertEqual(0, code, stderr)
        self.assertEqual(placed, self.derived.stat().st_mtime_ns)
        self.assertGreater(generation.stamp(self.derived).stat().st_mtime_ns,
                           marker)
        # Recorded all the same: what was submitted is a fact worth keeping.
        self.assertTrue((self.removals / "second.txt.removed.2").is_file())
        self.assertIn("1 words submitted, 1 new to the removal union, "
                      "0 newly removed from the dictionary", stdout)
        self.assertIn("words.filtered 4 (unchanged)", stdout)

    def test_resubmitting_an_already_removed_word_counts_as_zero(self):
        """The delta is against the removal union, not against the base.

        A removed word remains in words.big, so counting every submitted base
        word would make a resubmission look effective.
        """
        self._base()
        self._remove("junk.txt", "banana\n")
        _, stdout, _ = self._remove("again.txt", "banana\n")

        self.assertIn("1 words submitted, 0 new to the removal union, "
                      "0 newly removed from the dictionary", stdout)
        self.assertEqual(["apple", "cherry", "date", "fig"],
                         self._lines(self.derived))

    def test_the_counter_is_global_and_the_stem_is_the_input_name(self):
        self._base()
        self._remove("junk.txt", "banana\n")
        self._remove("other.txt", "fig\n")
        # Two submissions of one filename get distinct generations.
        self._remove("junk.txt", "cherry\n")

        self.assertEqual(
            ["junk.txt.removed.1", "junk.txt.removed.3", "other.txt.removed.2"],
            sorted(path.name for path in self.removals.iterdir()))
        self.assertEqual(
            ["junk.txt.reviewed.1", "junk.txt.reviewed.3",
             "other.txt.reviewed.2"],
            sorted(path.name for path in self.reviewed_inputs.iterdir()))
        self.assertEqual(["banana"],
                         self._lines(self.removals / "junk.txt.removed.1"))
        self.assertEqual(["cherry"],
                         self._lines(self.removals / "junk.txt.removed.3"))

    def test_allocation_is_one_max_over_both_namespaces(self):
        self._base()
        (self.removals / "a.txt.removed.1").write_text("apple\n")
        (self.reviewed_inputs / "b.txt.reviewed.3").write_text("apple\n")
        # A name not ending in a positive-integer suffix is ignored rather
        # than parsed, and a greedy stem reads its own generation.
        (self.removals / "c.txt.removed.0").write_text("cherry\n")
        (self.removals / "d.txt.removed.99~").write_text("date\n")

        self._remove("next.txt", "fig\n")
        self.assertTrue((self.removals / "next.txt.removed.4").is_file())

        self.assertEqual(
            [(1, self.removals / "a.txt.removed.1"),
             (4, self.removals / "next.txt.removed.4")],
            dictionary.records(self.removals, "removed"))

    def test_a_greedy_stem_reads_its_own_generation(self):
        self._base()
        (self.removals / "foo.removed.3.removed.9").write_text("apple\n")
        self._remove("next.txt", "fig\n")
        self.assertTrue((self.removals / "next.txt.removed.10").is_file())

    def test_the_archive_is_named_after_the_basename_not_the_typed_path(self):
        self._base()
        nested = self.root / "lists"
        nested.mkdir()
        (nested / "junk.txt").write_text("banana\n")
        code, _, stderr = fx.run_wf("-d", str(self.root), "remove", "words",
                                    str(nested / "junk.txt"))

        self.assertEqual(0, code, stderr)
        self.assertTrue((self.removals / "junk.txt.removed.1").is_file())

    def test_a_basename_that_names_nothing_is_refused(self):
        self._base()
        for spelled in (".", ".."):
            with self.subTest(path=spelled):
                with self.assertRaisesRegex(ValueError, "cannot name an "
                                            "archive"):
                    dictionary.archive_stem(Path(spelled))

    def test_a_symlink_input_is_refused_before_anything_is_written(self):
        self._base()
        real = self._submission("junk.txt", "banana\n")
        link = self.root / "link.txt"
        link.symlink_to(real)

        with self.assertRaisesRegex(ValueError, "symlink input not allowed"):
            dictionary.remove_words(self.root, link)
        self.assertEqual([], list(self.removals.iterdir()))
        self.assertFalse(self.derived.exists())

    # ------------------------------------------------------- the submission

    def test_a_count_prefixed_slice_and_a_bare_word_list_agree(self):
        self._base()
        self._remove("counted.txt", "  1234 banana\n\n    9 fig\n")
        self._remove("bare.txt", "banana\nfig\n")

        self.assertEqual(
            self._lines(self.removals / "counted.txt.removed.1"),
            self._lines(self.removals / "bare.txt.removed.2"))

    def test_a_row_that_is_not_a_word_is_refused_and_nothing_is_written(self):
        self._base()
        with self.assertRaisesRegex(ValueError, "not a word: 'good,pair'"):
            dictionary.remove_words(
                self.root, self._submission("pairs.txt", "banana\ngood,pair\n"))
        self.assertEqual([], list(self.removals.iterdir()))
        self.assertFalse(self.derived.exists())

    def test_an_unsorted_submission_still_subtracts_every_entry(self):
        """comm under-subtracts in silence on an unsorted right-hand side."""
        self._base()
        self._remove("junk.txt", "fig\nbanana\ncherry\n")

        self.assertEqual(["banana", "cherry", "fig"],
                         self._lines(self.removals / "junk.txt.removed.1"))
        self.assertEqual(["apple", "date"], self._lines(self.derived))

    # -------------------------------------------------------- the derivation

    def test_a_stray_name_in_removed_does_not_resurrect_a_retracted_word(self):
        self._base()
        self._remove("junk.txt", "banana\n")
        record = self.removals / "junk.txt.removed.1"
        # What vim leaves behind beside the file the operator just trimmed.
        (self.removals / "junk.txt.removed.1~").write_text(record.read_text())
        record.write_text("")

        self._gen()
        self.assertEqual(["apple", "banana", "cherry", "date", "fig"],
                         self._lines(self.derived))

    def test_a_stray_name_in_done_in_does_not_become_a_reviewed_verdict(self):
        self._base()
        self._remove("junk.txt", "banana\n")
        record = self.reviewed_inputs / "junk.txt.reviewed.1"
        (self.reviewed_inputs / "junk.txt.reviewed.1~").write_text(
            record.read_text())
        record.write_text("")

        self._gen()
        self.assertEqual([], self._lines(self.reviewed_words))

    def test_a_word_is_restored_only_once_every_generation_drops_it(self):
        self._base()
        self._remove("first.txt", "banana\nfig\n")
        self._remove("second.txt", "banana\n")

        # Trimming one occurrence leaves the word removed.
        (self.removals / "first.txt.removed.1").write_text("fig\n")
        self._gen()
        self.assertNotIn("banana", self._lines(self.derived))

        (self.removals / "second.txt.removed.2").write_text("")
        self._gen()
        self.assertIn("banana", self._lines(self.derived))
        self.assertNotIn("fig", self._lines(self.derived))

        # Deleting a generation removes its whole contribution.
        (self.removals / "first.txt.removed.1").unlink()
        self._gen()
        self.assertEqual(["apple", "banana", "cherry", "date", "fig"],
                         self._lines(self.derived))

    def test_a_retraction_leaves_the_reviewed_done_set_alone(self):
        """The two records answer different questions, and diverge on purpose.

        A retraction changes whether a word is removed; it cannot change
        whether the word was reviewed, so the word does not return as a
        candidate.
        """
        self._base()
        self._remove("junk.txt", "banana\n")
        (self.removals / "junk.txt.removed.1").write_text("")
        self._gen()

        self.assertIn("banana", self._lines(self.derived))
        self.assertEqual(["banana"], self._lines(self.reviewed_words))

    def test_deleting_a_reviewed_input_reopens_only_unshared_words(self):
        self._base()
        self._remove("first.txt", "banana\nfig\n")
        self._remove("second.txt", "banana\n")

        (self.reviewed_inputs / "first.txt.reviewed.1").unlink()
        self._gen()
        self.assertEqual(["banana"], self._lines(self.reviewed_words))
        # The removals are untouched: they live in removed/ only.
        self.assertEqual(["apple", "cherry", "date"],
                         self._lines(self.derived))

    def test_the_kept_words_are_recoverable_without_a_kept_directory(self):
        """reviewed.words minus the removal union, over an overlapping run."""
        self._base()
        self._remove("first.txt", "banana\ncherry\n")
        self._remove("second.txt", "banana\nnotinbase\n")
        (self.removals / "first.txt.removed.1").write_text("banana\n")
        (self.removals / "second.txt.removed.2").write_text("banana\n")
        self._gen()

        with tempfile.TemporaryDirectory() as tmp:
            union = setops.merge(sorted(self.removals.iterdir()),
                                 Path(tmp) / "removed")
            kept = setops.diff(self.reviewed_words, union, Path(tmp) / "kept")
            self.assertEqual(["cherry", "notinbase"], self._lines(kept))

    def test_an_empty_tree_copies_the_base_and_writes_an_empty_done_set(self):
        """Neither empty aggregate asks setops.merge to merge no sources."""
        self._base()
        code, stdout, stderr = self._gen()

        self.assertEqual(0, code, stderr)
        self.assertEqual(BASE, self.derived.read_text())
        self.assertEqual("", self.reviewed_words.read_text())
        self.assertIn("words.filtered 5 (new)", stdout)

    def test_both_commands_create_the_derived_file_and_require_the_base(self):
        """words.filtered is the output; words.big is the input.

        wf init deliberately creates neither derived file, so the first build
        is a reachable state and not an error.
        """
        for command in (lambda: self._gen(),
                        lambda: self._remove("junk.txt", "banana\n")):
            with self.subTest(command=command):
                with self.assertRaises(FileNotFoundError):
                    command()
                self.assertFalse(self.derived.exists())

        self._base()
        self._gen()
        self.assertTrue(self.derived.is_file())
        self.derived.unlink()
        self._remove("junk.txt", "banana\n")
        self.assertTrue(self.derived.is_file())

    def test_the_reviewed_archive_shares_the_stem_and_the_generation(self):
        self._base()
        self._remove("junk.txt", "banana\nnotinbase\n")

        self.assertEqual(
            self._lines(self.removals / "junk.txt.removed.1"),
            self._lines(self.reviewed_inputs / "junk.txt.reviewed.1"))
        # Both files, never one hard link: the removal record is editable on
        # its own.
        self.assertNotEqual(
            (self.removals / "junk.txt.removed.1").stat().st_ino,
            (self.reviewed_inputs / "junk.txt.reviewed.1").stat().st_ino)
        # A word not in the base was still looked at.
        self.assertEqual(["banana", "notinbase"],
                         self._lines(self.reviewed_words))

    # -------------------------------------------------------- the placement

    def test_full_batch_preflight_fails_before_either_archive_lands(self):
        self._base()
        self.reviewed_words.mkdir()

        with self.assertRaises(ValueError):
            dictionary.remove_words(
                self.root, self._submission("junk.txt", "banana\n"))
        self.assertEqual([], list(self.removals.iterdir()))
        self.assertEqual([], list(self.reviewed_inputs.iterdir()))
        self.assertFalse(self.derived.exists())
        self.assertFalse(generation.stamp(self.derived).exists())

    def test_a_set_operation_failure_is_a_diagnostic_and_not_a_traceback(self):
        """comm exits 1 on unsorted input, which is the operator's to fix."""
        self._base()
        self._remove("junk.txt", "banana\n")
        placed = self.derived.read_text()
        marker = generation.stamp(self.derived).stat().st_mtime_ns

        # A base replaced by hand with one that is not C-sorted. Nothing here
        # revalidates it; comm refuses and the boundary names the operation.
        self._base("fig\napple\ncherry\n")
        with self.assertRaisesRegex(ValueError,
                                    "dictionary derivation failed over"):
            dictionary.gen_dict(self.root)
        self.assertEqual(placed, self.derived.read_text())
        self.assertEqual(marker,
                         generation.stamp(self.derived).stat().st_mtime_ns)

    def test_a_prospective_failure_lands_before_any_archive(self):
        self._base("fig\napple\n")

        with self.assertRaises(ValueError):
            dictionary.remove_words(
                self.root, self._submission("junk.txt", "apple\n"))
        self.assertEqual([], list(self.removals.iterdir()))
        self.assertEqual([], list(self.reviewed_inputs.iterdir()))
        self.assertFalse(self.derived.exists())
        self.assertFalse(generation.stamp(self.derived).exists())

    def test_the_marker_survives_a_crash_after_the_derived_file(self):
        """Which is what makes the staleness row offer the rebuild again."""
        self._base()
        self._gen()
        before = generation.stamp(self.derived).stat().st_mtime_ns

        self._base("apple\nbanana\n")
        # The marker is written after the batch commits, so failing that one
        # write is the whole crash window between the two clocks.
        with mock.patch.object(dictionary.generation, "mark_generated",
                               side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                dictionary.gen_dict(self.root)
        self.assertEqual("apple\nbanana\n", self.derived.read_text())
        self.assertEqual(before,
                         generation.stamp(self.derived).stat().st_mtime_ns)

    def test_a_reviewed_words_failure_stops_before_the_dictionary(self):
        self._base()
        self._gen()
        placed = self.derived.read_text()

        real = setops.place

        def place(staged):
            if staged.dst == self.reviewed_words:
                raise OSError("no")
            return real(staged)

        self._base("apple\nbanana\n")
        self.reviewed_words.unlink()
        with mock.patch.object(dictionary.setops, "place", side_effect=place):
            with self.assertRaises(OSError):
                dictionary.gen_dict(self.root)
        self.assertEqual(placed, self.derived.read_text())

    def test_the_staging_directory_does_not_outlive_the_command(self):
        self._base()
        self._remove("junk.txt", "banana\n")
        self.assertEqual(
            [], [path for path in config.path(self.root, ["dict"]).iterdir()
                 if path.name.startswith(dictionary.STAGING_PREFIX)])

    # ---------------------------------------------------------- the report

    def test_the_count_line_renders_a_move_over_an_equal_count(self):
        """A rebuild that applies one removal and picks up one retraction."""
        self._base()
        self._remove("first.txt", "banana\n")
        self._remove("second.txt", "cherry\n")
        self.assertEqual(["apple", "date", "fig"], self._lines(self.derived))

        (self.removals / "first.txt.removed.1").write_text("date\n")
        _, stdout, _ = self._gen()
        self.assertEqual(["apple", "banana", "fig"], self._lines(self.derived))
        # An equal count with different bytes is a move, and renders as one.
        self.assertIn("words.filtered 3 -> 3", stdout)
        self.assertNotIn("words.filtered 3 (unchanged)", stdout)

    def test_the_report_names_both_archives_under_one_generation(self):
        self._base()
        _, stdout, _ = self._remove("junk.txt", "banana\n")

        self.assertIn("recorded as generation 1 -> "
                      "dict/removed/junk.txt.removed.1", stdout)
        self.assertIn("-> dict/done/in/junk.txt.reviewed.1", stdout)

    def test_gen_dict_does_not_recommend_itself(self):
        self._base()
        self._gen()
        self.reviewed_words.unlink()
        code, stdout, stderr = self._gen()

        self.assertEqual(0, code, stderr)
        self.assertTrue(self.reviewed_words.is_file())
        self.assertNotIn("wf gen dict", stdout + stderr)

    # ---------------------------------------------------------------- layout

    def test_init_creates_the_dictionary_tree(self):
        self.assertTrue(config.path(self.root, ["dict"]).is_dir())
        self.assertTrue(self.removals.is_dir())
        self.assertTrue(self.reviewed_inputs.is_dir())
        # The derived files are not created: deriving them needs inputs init
        # has no business requiring.
        self.assertFalse(self.derived.exists())
        self.assertFalse(self.reviewed_words.exists())

    def test_show_dict_lists_files_and_subparts_together(self):
        self._base()
        self._remove("junk.txt", "banana\n")
        code, stdout, stderr = fx.run_wf("-d", str(self.root), "show", "dict")

        self.assertEqual(0, code, stderr)
        self.assertIn("words.big", stdout)
        self.assertIn("words.filtered", stdout)
        self.assertIn("removed/", stdout)


if __name__ == "__main__":
    unittest.main()
