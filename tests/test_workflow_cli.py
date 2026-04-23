import io
import unittest

from contextlib import redirect_stderr, redirect_stdout

from workflow import wf


class WorkflowCliTests(unittest.TestCase):
    def _run(self, *argv):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = wf.main(list(argv))
        return code, stdout.getvalue(), stderr.getvalue()

    def test_invalid_root_command_shows_root_help(self):
        code, stdout, stderr = self._run("balls")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("invalid argument: 'balls'", stderr)
        self.assertIn("init    — initialize a workflow (stub)", stderr)
        self.assertIn("show    — display workflow state", stderr)
        self.assertIn("submit  — submit items (pairs)", stderr)

    def test_invalid_show_root_target_shows_show_help(self):
        code, stdout, stderr = self._run("show", "balls")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("invalid argument: 'balls'", stderr)
        self.assertIn("show    — display workflow state", stderr)
        self.assertIn("usage: wf show p1|p2|p3|classified", stderr)
        self.assertIn("Targets:", stderr)

    def test_invalid_show_nested_target_shows_parent_help(self):
        code, stdout, stderr = self._run("show", "p1", "balls")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("invalid argument: 'balls'", stderr)
        self.assertIn("usage: wf show p1 queued|running|done", stderr)
        self.assertIn("Targets:", stderr)
        self.assertIn("queued", stderr)

    def test_invalid_show_extra_argument_shows_leaf_help(self):
        code, stdout, stderr = self._run("show", "p1", "queued", "balls")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("invalid argument: 'balls'", stderr)
        self.assertIn("usage: wf show p1 queued", stderr)
        self.assertNotIn("Targets:", stderr)

    def test_invalid_init_argument_shows_init_help(self):
        code, stdout, stderr = self._run("init", "balls")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("invalid argument: 'balls'", stderr)
        self.assertIn("init    — initialize a workflow (stub)", stderr)

    def test_invalid_init_help_argument_shows_init_help(self):
        code, stdout, stderr = self._run("help", "init", "balls")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("invalid argument: 'balls'", stderr)
        self.assertIn("init    — initialize a workflow (stub)", stderr)

    def test_invalid_submit_pairs_help_argument_shows_default_help(self):
        code, stdout, stderr = self._run("help", "submit", "pairs", "balls")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("invalid argument: 'balls'", stderr)
        self.assertIn("pairs   — submit a pairs file into p1/queued (sorted, deduped)", stderr)
        self.assertIn("usage: wf submit pairs <pair-file>", stderr)

    def test_invalid_eval_pairs_help_argument_shows_default_help(self):
        code, stdout, stderr = self._run("help", "eval", "pairs", "balls")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("invalid argument: 'balls'", stderr)
        self.assertIn("pairs   — move a queued pairs file to the running stage", stderr)
        self.assertIn("usage: wf eval pairs [pair-file]", stderr)

    def test_invalid_complete_pairs_help_argument_shows_default_help(self):
        code, stdout, stderr = self._run("help", "complete", "pairs", "balls")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("invalid argument: 'balls'", stderr)
        self.assertIn("pairs   — complete a running pairs file", stderr)
        self.assertIn("usage: wf complete pairs [pair-file]", stderr)

    def test_incomplete_show_command_reports_missing_required_argument(self):
        code, stdout, stderr = self._run("show")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("missing required argument", stderr)
        self.assertIn("usage: wf show p1|p2|p3|classified", stderr)

    def test_incomplete_show_parent_path_reports_missing_required_argument(self):
        code, stdout, stderr = self._run("show", "p1")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("missing required argument", stderr)
        self.assertIn("usage: wf show p1 queued|running|done", stderr)

    def test_incomplete_submit_command_reports_missing_required_argument(self):
        code, stdout, stderr = self._run("submit")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("missing required argument", stderr)
        self.assertIn("pairs   — submit a pairs file into p1/queued (sorted, deduped)", stderr)

    def test_incomplete_eval_command_reports_missing_required_argument(self):
        code, stdout, stderr = self._run("eval")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("missing required argument", stderr)
        self.assertIn("pairs   — move a queued pairs file to the running stage", stderr)

    def test_incomplete_complete_command_reports_missing_required_argument(self):
        code, stdout, stderr = self._run("complete")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("missing required argument", stderr)
        self.assertIn("pairs   — complete a running pairs file", stderr)


if __name__ == "__main__":
    unittest.main()
