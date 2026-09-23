import io
import os
import tempfile
import unittest

from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from workflow import wf


class WorkflowCliTests(unittest.TestCase):
    def _run(self, *argv):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = wf.main(list(argv))
        return code, stdout.getvalue(), stderr.getvalue()

    def test_invalid_root_command_is_rejected(self):
        code, stdout, stderr = self._run("balls")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("invalid argument: 'balls'", stderr)

    def test_wfroot_is_the_default_workflow_root(self):
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.dict(os.environ, {"WFROOT": root}):
                with mock.patch.object(
                    wf.Path, "cwd", side_effect=FileNotFoundError
                ):
                    code, _, stderr = self._run("init")
            self.assertEqual(0, code, stderr)
            self.assertTrue((Path(root) / ".wf").is_dir())

    def test_explicit_dir_overrides_wfroot(self):
        with tempfile.TemporaryDirectory() as default_root:
            with tempfile.TemporaryDirectory() as explicit_root:
                with mock.patch.dict(os.environ, {"WFROOT": default_root}):
                    code, _, stderr = self._run(
                        "-d", explicit_root, "init")
                self.assertEqual(0, code, stderr)
                self.assertTrue((Path(explicit_root) / ".wf").is_dir())
                self.assertFalse((Path(default_root) / ".wf").exists())

    def test_invalid_show_root_target_is_rejected(self):
        code, stdout, stderr = self._run("show", "balls")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("invalid argument: 'balls'", stderr)

    def test_invalid_show_nested_target_is_rejected(self):
        code, stdout, stderr = self._run("show", "p1", "balls")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("invalid argument: 'balls'", stderr)

    def test_invalid_show_extra_argument_is_rejected(self):
        code, stdout, stderr = self._run("show", "p1", "queued", "balls")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("invalid argument: 'balls'", stderr)

    def test_invalid_init_argument_is_rejected(self):
        code, stdout, stderr = self._run("init", "balls")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("invalid argument: 'balls'", stderr)

    def test_invalid_init_help_argument_is_rejected(self):
        code, stdout, stderr = self._run("help", "init", "balls")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("invalid argument: 'balls'", stderr)

    def test_invalid_submit_p1_help_argument_is_rejected(self):
        code, stdout, stderr = self._run("help", "submit", "p1", "balls")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("invalid argument: 'balls'", stderr)

    def test_invalid_eval_p1_help_argument_is_rejected(self):
        code, stdout, stderr = self._run("help", "eval", "p1", "balls")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("invalid argument: 'balls'", stderr)

    def test_invalid_complete_p1_help_argument_is_rejected(self):
        code, stdout, stderr = self._run("help", "complete", "p1", "balls")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("invalid argument: 'balls'", stderr)

    def test_incomplete_show_command_reports_missing_required_argument(self):
        code, stdout, stderr = self._run("show")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("missing required argument", stderr)

    def test_incomplete_show_parent_path_reports_missing_required_argument(self):
        code, stdout, stderr = self._run("show", "p1")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("missing required argument", stderr)

    def test_incomplete_submit_command_reports_missing_required_argument(self):
        code, stdout, stderr = self._run("submit")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("missing required argument", stderr)

    def test_incomplete_eval_command_reports_missing_required_argument(self):
        code, stdout, stderr = self._run("eval")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("missing required argument", stderr)

    def test_incomplete_complete_command_reports_missing_required_argument(self):
        code, stdout, stderr = self._run("complete")

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("missing required argument", stderr)


if __name__ == "__main__":
    unittest.main()
