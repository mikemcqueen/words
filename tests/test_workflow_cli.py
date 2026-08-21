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
        self.assertIn("submit  — submit items (p1|p2)", stderr)

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
        self.assertIn("usage: wf show p1 queued|eval|done", stderr)
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
        self.assertIn("init — initialize a workflow (stub)", stderr)

    def test_invalid_submit_p1_help_argument_shows_default_help(self):
        code, stdout, stderr = self._run("help", "submit", "p1", "balls")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("invalid argument: 'balls'", stderr)
        self.assertIn("p1 — submit a pairs file into p1/queued (sorted, deduped)", stderr)
        self.assertIn("usage: wf submit p1 [-d DIR] [-f] [-h] PAIRS-FILE", stderr)

    def test_invalid_eval_p1_help_argument_shows_default_help(self):
        code, stdout, stderr = self._run("help", "eval", "p1", "balls")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("invalid argument: 'balls'", stderr)
        self.assertIn("p1 — evaluate pairs", stderr)
        self.assertIn(
            "usage: wf eval p1 [-d DIR] [-f] [-h] [--no-filter] SLUG",
            stderr,
        )

    def test_invalid_complete_p1_help_argument_shows_default_help(self):
        code, stdout, stderr = self._run("help", "complete", "p1", "balls")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("invalid argument: 'balls'", stderr)
        self.assertIn("p1 — complete a pairs file evaluation", stderr)
        self.assertIn("usage: wf complete p1 [-d DIR] [-f] [-h] SLUG", stderr)

    def test_incomplete_show_command_reports_missing_required_argument(self):
        code, stdout, stderr = self._run("show")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("missing required argument", stderr)
        self.assertIn("usage: wf show p1|p2|p3|classified|all", stderr)

    def test_incomplete_show_parent_path_reports_missing_required_argument(self):
        code, stdout, stderr = self._run("show", "p1")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("missing required argument", stderr)
        self.assertIn("usage: wf show p1 queued|eval|done", stderr)

    def test_incomplete_submit_command_reports_missing_required_argument(self):
        code, stdout, stderr = self._run("submit")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("missing required argument", stderr)
        self.assertIn("p1      — submit a pairs file into p1/queued (sorted, deduped)", stderr)

    def test_incomplete_eval_command_reports_missing_required_argument(self):
        code, stdout, stderr = self._run("eval")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("missing required argument", stderr)
        self.assertIn("p1      — evaluate pairs", stderr)

    def test_incomplete_complete_command_reports_missing_required_argument(self):
        code, stdout, stderr = self._run("complete")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("missing required argument", stderr)
        self.assertIn("p1      — complete a pairs file evaluation", stderr)


if __name__ == "__main__":
    unittest.main()
