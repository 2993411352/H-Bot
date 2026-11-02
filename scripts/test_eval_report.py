"""CPU-only checks for report completeness and owned-process cleanup."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock


def module(name):
    path = Path(__file__).with_name(name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


report = module('report-pi05-eval')
runner = module('run-full60k-eval')


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.current = self.root / 'current'
        self.previous = self.root / 'previous'
        self.current.mkdir()
        self.previous.mkdir()
        self.payload = {
            'status': 'succeeded', 'eval_average_score': .5,
            'eval_average_success_rate': .2,
            'task_results': {
                name: {'episode_count': 50, 'failed_episode_count': 0,
                       'score': .5, 'success_rate': .2,
                       'stage_success_rate': {'first': .8, 'last': .2}}
                for name in report.ORDER
            },
        }
        (self.previous / 'summary.json').write_text(json.dumps(self.payload))

    def tearDown(self):
        self.temp.cleanup()

    def test_completed_results_get_exact_percentage_point_comparison(self):
        value = copy.deepcopy(self.payload)
        value['task_results']['stack_cubes']['success_rate'] = .3
        (self.current / 'summary.json').write_text(json.dumps(value))
        self.assertTrue(report.generate(self.current, self.previous))
        output = json.loads((self.current / 'comparison_60000_vs_30000.json').read_text())
        stack = next(row for row in output['tasks'] if row['task'] == 'stack_cubes')
        self.assertAlmostEqual(stack['success_delta_pp'], 10)
        self.assertTrue((self.current / 'report_60000_vs_30000.md').is_file())

    def test_execution_failure_is_not_classified_as_complete(self):
        self.payload['task_results']['stack_cubes']['failed_episode_count'] = 1
        (self.current / 'summary.json').write_text(json.dumps(self.payload))
        self.assertFalse(report.generate(self.current, self.previous))

    def test_missing_final_summary_generates_incomplete_report_not_zero_score(self):
        self.assertFalse(report.generate(self.current, self.previous))
        output = json.loads((self.current / 'comparison_60000_vs_30000.json').read_text())
        self.assertIsNone(output['score_60000'])
        self.assertEqual(output['tasks'], [])

    def test_finished_owned_process_is_not_signalled_again(self):
        process = Mock()
        process.poll.return_value = 0
        runner.terminate_owned(process)
        process.terminate.assert_not_called()
        process.kill.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
