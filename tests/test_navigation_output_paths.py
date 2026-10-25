"""Log setup must fail before GPU startup and preserve existing evidence."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('output_path_runner',
    Path(__file__).resolve().parents[1]/'source/rexmi_rl/nav/checkpoint_runner.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class NavigationOutputTests(unittest.TestCase):
    def test_new_parent_is_checked_without_consuming_joint_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)/'new run'
            runner.validate_output_paths(SimpleNamespace(log_file=str(run/'run.csv'),joint_log_dir=str(run/'joints')))
            self.assertTrue(run.is_dir());self.assertEqual(list(run.iterdir()),[])

    def test_existing_joint_data_is_rejected_and_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            marker=Path(folder)/'events.jsonl';marker.write_text('preserve')
            with self.assertRaisesRegex(ValueError,'already exists'):
                runner.validate_output_paths(SimpleNamespace(log_file=None,joint_log_dir=folder))
            self.assertEqual(marker.read_text(),'preserve')

    def test_unwritable_output_explains_empty_run_variable(self):
        with patch.object(runner.tempfile,'TemporaryFile',side_effect=PermissionError('denied')):
            with self.assertRaisesRegex(ValueError,'RUN may be unset'):
                runner.validate_output_paths(SimpleNamespace(log_file='/run.csv',joint_log_dir='/joints'))

    def test_failure_report_permission_error_does_not_mask_original_error(self):
        stream=io.StringIO()
        with patch.object(Path,'write_text',side_effect=PermissionError('denied')),contextlib.redirect_stdout(stream):
            runner.record_run_failure('/run.csv',RuntimeError('original failure'))
        self.assertIn('original failure',stream.getvalue())
        self.assertIn('Could not save failure report',stream.getvalue())


if __name__=='__main__':unittest.main()
