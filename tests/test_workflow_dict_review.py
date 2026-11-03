"""Dictionary word-review lifecycle and its partition/publication boundary."""

import json
import subprocess
import tempfile
import unittest

from pathlib import Path
from unittest import mock

from tests import wf_fixture as fx
from workflow import bundle, config, dictionary, notes
from workflow.context import Context
from workflow.steps import p2_retrieve, words_extract


_REAL_RUN = subprocess.run


class FakeWordNotes:
    def __init__(self, parts):
        self.parts = parts
        self.fetched = []
        self.parsed = []

    def route(self, argv, **kwargs):
        if "--get" in argv:
            title = argv[argv.index("--get") + 1]
            suffix = title[-2:]
            if suffix not in self.parts:
                return subprocess.CompletedProcess(argv, 1, "",
                                                   "note not found")
            self.fetched.append(title)
            return subprocess.CompletedProcess(
                argv, 0, json.dumps(self.parts[suffix]), "")
        if "--parse-file" in argv:
            self.parsed.append(argv)
            source = Path(argv[argv.index("--parse-file") + 1])
            note_type = argv[argv.index("--type") + 1]
            snapshot = json.loads(source.read_text())
            for row in snapshot.get(note_type.lower(), []):
                kwargs["stdout"].write(row + "\n")
            return subprocess.CompletedProcess(argv, 0)
        return _REAL_RUN(argv, **kwargs)


class DictionaryReviewTests(unittest.TestCase):
    NAME = "top.s7.m4.g5.words"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.opts, _ = fx.make_wf(self.root)
        config.base_dictionary(self.root).write_text(
            "apple\nbanana\ncherry\ndate\n")
        config.small_base_dictionary(self.root).write_text("apple\n")
        dictionary.gen_dict(self.root)

    def run_cli(self, *argv):
        return fx.run_wf("-d", str(self.root), *argv)

    def submit(self, text=" 9 banana\n 2 apple\n", name=NAME):
        source = self.root / "ranked.words"
        source.write_bytes(text.encode())
        code, stdout, stderr = self.run_cli(
            "submit", "words", str(source), "--as", name)
        self.assertEqual(0, code, stderr)
        return source, stdout

    def evaluate(self, name=NAME):
        with mock.patch.object(notes, "make", return_value=[]) as make:
            code, stdout, stderr = self.run_cli("eval", "words", name)
        self.assertEqual(0, code, stderr)
        return make, stdout

    def complete(self, fake, *extra, name=NAME, flags=()):
        with mock.patch.object(p2_retrieve.subprocess, "run", fake.route):
            return self.run_cli(*extra, "complete", "words", name, *flags)

    def review(self, name, text, parts, *extra, flags=()):
        """Submit, evaluate and complete one round under `name`."""
        self.submit(text, name)
        self.evaluate(name)
        return self.complete(FakeWordNotes({"aa": parts}), *extra, name=name,
                             flags=flags)

    def test_init_adds_the_dictionary_review_layout(self):
        for parts in (["dict", "queued"], ["dict", "eval"],
                      ["dict", "done", "out", "enex"]):
            self.assertTrue(config.path(self.root, parts).is_dir())

    def test_submit_is_byte_exact_and_as_names_the_queue(self):
        source, stdout = self.submit(" 2 pear\n\n 1 apple")
        queued = config.path(self.root, ["dict", "queued"]) / self.NAME
        self.assertEqual(source.read_bytes(), queued.read_bytes())
        self.assertEqual(b" 2 pear\n\n 1 apple", queued.read_bytes())
        self.assertIn(self.NAME, stdout)

    def test_submit_rejects_a_symlink_without_publishing(self):
        source = self.root / "real.words"
        source.write_text("apple\n")
        link = self.root / "linked.words"
        link.symlink_to(source)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.run_cli("submit", "words", str(link))
        self.assertEqual([], list(config.path(
            self.root, ["dict", "queued"]).iterdir()))

    def test_eval_filters_by_identity_and_retains_ranked_rows(self):
        config.yes_words(self.root).write_text("apple\n")
        config.no_words(self.root).write_text("cherry\n")
        config.reviewed_words(self.root).write_text("banana\n")
        self.submit(" 8 apple\n 7 banana\n 6 cherry\n")
        make, _ = self.evaluate()
        filtered = (config.path(self.root, ["dict", "eval"]) / self.NAME
                    / f"{self.NAME}.filtered")
        self.assertEqual(" 7 banana\n", filtered.read_text())
        self.assertEqual(notes.TWO_CHECKBOXES, make.call_args.args[2])

    def _note_create_argv(self, *argv):
        """Run a words command with `note` faked; return its --create argv."""
        calls = []
        def fake(args, **kwargs):
            if args[0] != "note":
                return _REAL_RUN(args, **kwargs)
            calls.append(args)
            return subprocess.CompletedProcess(args, 0)
        with mock.patch.object(notes.subprocess, "run", fake):
            code, _, stderr = self.run_cli(*argv)
        self.assertEqual(0, code, stderr)
        return [c for c in calls if "--create" in c]

    def test_eval_and_notes_create_two_unticked_checkboxes(self):
        self.submit()
        for argv in (("eval", "words", self.NAME),
                     ("notes", "words", self.NAME)):
            with self.subTest(command=argv[0]):
                created = self._note_create_argv(*argv)
                self.assertTrue(created)
                for args in created:
                    self.assertIn("--two-checkboxes", args)
                    self.assertNotIn("--checked", args)

    def test_complete_refuses_dry_run_before_retrieving(self):
        self.submit()
        self.evaluate()
        fake = FakeWordNotes({"aa": {"yes": ["2 apple"], "no": []}})
        with self.assertRaisesRegex(ValueError,
                                    "--dry-run is not valid for complete words"):
            self.complete(fake, "--dry-run")
        self.assertEqual([], fake.fetched)
        self.assertTrue((config.path(self.root, ["dict", "eval"])
                         / self.NAME).exists())

    def test_eval_failure_leaves_the_source_queued(self):
        self.submit(" 4 Apple\n")
        queued = config.path(self.root, ["dict", "queued"]) / self.NAME
        with self.assertRaisesRegex(ValueError, r":1: not a word"):
            self.run_cli("eval", "words", self.NAME)
        self.assertTrue(queued.is_file())
        self.assertEqual([], list(config.path(
            self.root, ["dict", "eval"]).iterdir()))

    def test_eval_refuses_an_empty_unclassified_set_before_opening(self):
        config.no_words(self.root).write_text("apple\n")
        self.submit(" 4 apple\n\n")
        with self.assertRaisesRegex(ValueError, "no unclassified words"):
            self.run_cli("eval", "words", self.NAME)
        self.assertTrue((config.path(self.root, ["dict", "queued"])
                         / self.NAME).is_file())

    def test_complete_records_yes_no_and_undecided_words(self):
        self.submit(" 9 banana\n 2 apple\n 1 cherry\n")
        self.evaluate()
        fake = FakeWordNotes({
            "aa": {"yes": ["2 apple"], "no": ["9 banana"],
                   "none": ["1 cherry"]},
        })
        code, stdout, stderr = self.complete(fake)
        self.assertEqual(0, code, stderr)
        inputs = config.reviewed_inputs(self.root)
        removed = config.removals(self.root) / f"{self.NAME}.removed.1"
        archive = (config.path(self.root, ["dict", "done", "out", "enex"])
                   / f"{self.NAME}.reviewed.1")
        self.assertEqual("apple\n",
                         (inputs / f"{self.NAME}.yes.1").read_text())
        self.assertEqual("banana\n", removed.read_text())
        self.assertEqual("apple\nbanana\ncherry\n",
                         (inputs / f"{self.NAME}.reviewed.1").read_text())
        self.assertEqual("apple\n", config.yes_words(self.root).read_text())
        self.assertEqual("banana\n", config.no_words(self.root).read_text())
        self.assertEqual(["apple", "cherry", "date"],
                         config.dictionary(self.root).read_text().splitlines())
        self.assertTrue(archive.is_dir())
        self.assertFalse((config.path(self.root, ["dict", "eval"])
                          / self.NAME).exists())
        self.assertIn("3 words reviewed: 1 YES, 1 NO, 1 undecided", stdout)
        self.assertIn(f"recorded as generation 1 -> dict/done/in/"
                      f"{self.NAME}.yes.1", stdout)
        for argv in fake.parsed:
            self.assertIn("--two-checkboxes", argv)

        code, stdout, stderr = self.run_cli("history")
        self.assertEqual(0, code, stderr)
        self.assertIn("YES: 1 words", stdout)

    def test_a_flipped_verdict_fails_whatever_the_flags(self):
        # Both reviews are open before either completes, so eval's filter
        # cannot keep the second from classifying the first's words.
        self.submit("banana\napple\n", "second")
        self.evaluate("second")
        code, _, stderr = self.review("first", "banana\napple\n",
                                      {"yes": ["apple"], "no": ["banana"]})
        self.assertEqual(0, code, stderr)
        derived = config.dictionary(self.root).read_text()
        fake = FakeWordNotes({"aa": {"yes": ["banana"], "no": ["apple"]}})
        for extra, flags in (((), ()), (("-f",), ()),
                             (("-f",), ("--force-verdicts",))):
            with self.subTest(extra=extra, flags=flags):
                with self.assertRaisesRegex(
                        ValueError,
                        "cannot record 1 NO words already classified YES:\n"
                        "  apple  dict/done/in/first.yes.1\n"
                        "cannot record 1 YES words already classified NO:\n"
                        "  banana  dict/removed/first.removed.1\n"):
                    self.complete(fake, *extra, name="second", flags=flags)
        self.assertEqual(["first.removed.1"],
                         [p.name for p in config.removals(self.root).iterdir()])
        self.assertEqual(derived, config.dictionary(self.root).read_text())
        self.assertTrue((config.path(self.root, ["dict", "eval"])
                         / "second").is_dir())

    def test_a_late_verdict_needs_force_verdicts(self):
        code, _, stderr = self.review("first", "banana\napple\n",
                                      {"yes": ["apple"], "none": ["banana"]})
        self.assertEqual(0, code, stderr)
        self.submit("banana\napple\n", "second")
        self.evaluate("second")
        fake = FakeWordNotes({"aa": {"no": ["banana"]}})
        for extra in ((), ("-f",)):
            with self.subTest(extra=extra):
                with self.assertRaisesRegex(ValueError,
                                            "rerun with --force-verdicts"):
                    self.complete(fake, *extra, name="second")
        self.assertEqual(["first.removed.1"],
                         [p.name for p in config.removals(self.root).iterdir()])

        code, stdout, stderr = self.complete(fake, "-f", name="second",
                                             flags=("--force-verdicts",))
        self.assertEqual(0, code, stderr)
        self.assertIn("banana was reviewed 1 times without a verdict", stderr)
        self.assertEqual("banana\n", config.no_words(self.root).read_text())

    def test_invalid_partition_places_no_results_or_dictionary_record(self):
        self.submit()
        self.evaluate()
        fake = FakeWordNotes({"aa": {"yes": ["apple"], "no": []}})
        with self.assertRaisesRegex(ValueError,
                                    "(?s)missing 1.*missing from the notes:"
                                    "\n  banana.*wf -f complete words"):
            self.complete(fake)
        ctx = Context(self.root, "dict", bundle_name=self.NAME)
        for path in words_extract.outputs(ctx):
            self.assertFalse(path.exists())
        self.assertEqual([], list(config.removals(self.root).iterdir()))
        self.assertTrue(bundle.has_source(ctx))

    def test_remote_correction_needs_force_to_replace_the_enex_snapshot(self):
        self.submit()
        self.evaluate()
        parts = {"aa": {"yes": ["apple"], "no": []}}
        fake = FakeWordNotes(parts)
        with self.assertRaises(ValueError):
            self.complete(fake)
        parts["aa"]["no"] = ["banana"]
        with self.assertRaises(ValueError):
            self.complete(fake)
        code, _, stderr = self.complete(fake, "-f")
        self.assertEqual(0, code, stderr)

    def test_count_only_rows_are_malformed_for_direct_remove_too(self):
        bad = self.root / "bad.words"
        bad.write_text(" 123 \n")
        with self.assertRaisesRegex(ValueError, "not a word"):
            dictionary.remove_words(self.root, bad)


class WordValidationTests(unittest.TestCase):
    def test_content_equality_not_equal_counts_controls_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            reviewed = directory / "reviewed"
            yes = directory / "yes"
            no = directory / "no"
            reviewed.write_text("apple\nbanana\n")
            yes.write_text("apple\n")
            no.write_text("cherry\n")
            undecided = directory / "undecided"
            undecided.write_text("")
            with self.assertRaisesRegex(ValueError, "extra 1, missing 1"):
                words_extract.validate_results(reviewed, yes, no, undecided)

    def test_a_word_ticked_both_ways_fails_and_undecided_is_not_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            paths = {}
            for name, text in (("reviewed", "apple\nbanana\ncherry\n"),
                               ("yes", "apple\n"), ("no", "apple\n"),
                               ("undecided", "banana\ncherry\n")):
                paths[name] = directory / name
                paths[name].write_text(text)
            with self.assertRaises(ValueError) as caught:
                words_extract.validate_results(
                    paths["reviewed"], paths["yes"], paths["no"],
                    paths["undecided"])
            message = str(caught.exception)
            self.assertIn("conflicting 1, extra 0, missing 0", message)
            self.assertIn("ticked both YES and NO:\n  apple", message)
            self.assertNotIn("missing from the notes", message)


class SharedNoteCreationRecoveryTests(unittest.TestCase):
    def test_p2_and_words_list_manual_cleanup_and_full_recreation(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            cases = [
                (notes.TWO_CHECKBOXES, "wf notes p2 review", None),
                (notes.TWO_CHECKBOXES, "wf notes words review", "words"),
            ]
            for mode, recovery, scope in cases:
                with self.subTest(mode=mode):
                    source = directory / "review" / "review"
                    source.parent.mkdir(exist_ok=True)
                    source.write_text("apple\n")
                    paths = [directory / "review.aa", directory / "review.ab"]
                    opts = fx.make_opts(directory)
                    opts.yes_pairs = None
                    failure = subprocess.CalledProcessError(1, ["note"])
                    with mock.patch.object(notes, "split", return_value=paths), \
                         mock.patch.object(notes.subprocess, "run",
                                           side_effect=[mock.DEFAULT, failure]):
                        with self.assertRaises(ValueError) as caught:
                            if scope == "words":
                                notes.make(source, opts, mode, recovery)
                            else:
                                notes.make(source, opts)
                    message = str(caught.exception)
                    self.assertIn("review.aa", message)
                    self.assertIn("review.ab", message)
                    self.assertIn(recovery, message)

    def test_words_note_creation_failure_keeps_the_active_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            opts, _ = fx.make_wf(root)
            config.base_dictionary(root).write_text("apple\nbanana\n")
            config.small_base_dictionary(root).write_text("apple\n")
            dictionary.gen_dict(root)
            source = root / "source.words"
            source.write_text("apple\nbanana\n")
            fx.run_wf("-d", str(root), "submit", "words", str(source),
                      "--as", "review")
            parts = [root / "review.aa", root / "review.ab"]
            failure = subprocess.CalledProcessError(1, ["note"])
            with mock.patch.object(notes, "split", return_value=parts), \
                 mock.patch.object(notes.subprocess, "run",
                                   side_effect=[mock.DEFAULT, failure]):
                with self.assertRaisesRegex(ValueError, "wf notes words review"):
                    fx.run_wf("-d", str(root), "eval", "words", "review")
            ctx = Context(root, "dict", bundle_name="review")
            self.assertTrue(bundle.has_source(ctx))

    def test_p2_note_creation_failure_keeps_the_bundle_for_notes_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            opts, _ = fx.make_wf(root)
            queued = fx.place(opts, ["p2", "queued"], "review.pairs",
                              "alpha,beta\n")
            parts = [root / "review.pairs.aa", root / "review.pairs.ab"]
            failure = subprocess.CalledProcessError(1, ["note"])
            with mock.patch.object(notes, "split", return_value=parts), \
                 mock.patch.object(notes.subprocess, "run",
                                   side_effect=[mock.DEFAULT, failure]):
                with self.assertRaisesRegex(ValueError, "wf notes p2 review"):
                    fx.run_wf("-d", str(root), "eval", "p2", "review")
            ctx = Context(root, "p2", bundle_name="review")
            self.assertTrue(bundle.has_source(ctx))
            self.assertFalse(queued.exists())
            with mock.patch.object(notes, "split", return_value=[]), \
                 mock.patch.object(notes, "create") as create:
                code, _, stderr = fx.run_wf(
                    "-d", str(root), "notes", "p2", "review")
            self.assertEqual(0, code, stderr)
            create.assert_called_once()


if __name__ == "__main__":
    unittest.main()
