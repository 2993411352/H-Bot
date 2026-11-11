"""CPU tests: real routing/TLS, synthetic observations, no model or GPU allocation."""
import copy
import json
import logging
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from official_server import validate_actions, validate_uri, verify_frozen
from legacy_server import checkpoint_metadata
from pi05_server_config import example_observation, make_adapter
from robo_orchard_server.policy.pi05.policy import Pi05Policy
from robo_orchard_server.server import PolicyWebsocketServer
from smoke_wss import run_smoke

PACKAGE = Path(__file__).resolve().parents[1]


class CounterPipeline:
    """A deterministic mock whose mutable counter must be session-local."""
    def __init__(self):
        self.counter = 0

    def infer(self, observation):
        self.counter += 1
        state = observation['observation.state'].copy()
        state[:7] += self.counter * .001
        return {'actions': np.repeat(state[None], 15, axis=0).astype(np.float32)}


class PortabilityTests(unittest.TestCase):
    def test_backend_selects_separate_python_before_imports(self):
        env = os.environ.copy()
        env.pop('HBOT_PYTHON', None)
        env['HBOT_RUNTIME'] = '/tmp/hbot-portability-test'
        env['HBOT_BACKEND'] = 'jax'
        result = subprocess.run(['bash', '-c', 'source "$1/common.sh"; printf "%s" "$HBOT_PYTHON"',
                                 'test', str(PACKAGE), '--backend', 'loongforge'],
                                env=env, capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout, '/tmp/hbot-portability-test/loongforge/.venv/bin/python')

    def test_uri_validation(self):
        self.assertEqual(validate_uri(' wss://example.com/session/ws '), 'wss://example.com/session/ws')
        for value in ['ws://example.com', 'wss://', 'https://example.com',
                      'wss://user:secret@example.com/ws', 'wss://example.com/#fragment']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_uri(value)

    def test_integrity(self):
        verify_frozen()

    def test_checkpoint_root_and_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'params').mkdir()
            stats = root / 'assets/hbot_panda_joint_delta_h15'
            stats.mkdir(parents=True)
            row = {'mean': [0]*8, 'std': [1]*8, 'q01': [-1]*8, 'q99': [1]*8}
            (stats / 'norm_stats.json').write_text(json.dumps({'norm_stats': {'state': row, 'actions': row}}))
            item = {'key_metadata': [{'key': k} for k in ['params', 'test', 'value']],
                    'value_metadata': {'write_shape': [8, 32]}}
            (root / 'params/_METADATA').write_text(json.dumps({'tree_metadata': {'test': item}}))
            shapes, _ = checkpoint_metadata(root, stats.name, 'hbot-full')
            self.assertEqual(shapes, {'test': (8, 32)})
            with self.assertRaises(ValueError):
                checkpoint_metadata(root / 'params', stats.name, 'hbot-full')
            with self.assertRaises(ValueError):
                checkpoint_metadata(root, stats.name, 'hbot-lora')

    def test_camera_joint_and_gripper_contract(self):
        settings = yaml.safe_load((PACKAGE / 'model.yaml').read_text())['hbot-full']
        adapter = make_adapter('hbot-full', settings)
        obs = example_observation()
        raw = adapter.build_model_input(obs)
        self.assertEqual(raw['observation.state'].shape, (8,))
        self.assertAlmostEqual(float(raw['observation.state'][7]), .5)
        for name in ('ext1_camera', 'ext2_camera', 'wrist_camera'):
            np.testing.assert_array_equal(raw['observation.images.' + name], obs['cameras'][name]['rgb'][0])
        actions = np.repeat(raw['observation.state'][None], 15, axis=0)
        canonical = adapter.build_action_sequence(actions, obs, 15)
        validate_actions(canonical)
        np.testing.assert_allclose(canonical[0]['values'][0, 7:], [.4, .4])
        broken = copy.deepcopy(canonical)
        broken[0]['values'][0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_actions(broken)

    def test_original_wss_suite_with_mock_weights(self):
        settings = yaml.safe_load((PACKAGE / 'model.yaml').read_text())['hbot-full']
        shared = Pi05Policy(CounterPipeline(), make_adapter('hbot-full', settings), steps=15)
        server = PolicyWebsocketServer(policy_factory=shared.new_session)
        previous = {k: os.environ.get(k) for k in ('SSL_CERT_FILE', 'no_proxy', 'NO_PROXY')}
        with tempfile.TemporaryDirectory() as directory:
            report = run_smoke(server, Path(directory), real_weights=False)
        self.assertEqual(report['status'], 'MOCK_POLICY_LOCAL_WSS_PASSED')
        self.assertFalse(report['weights_loaded'])
        self.assertFalse(report['real_observation_tested'])
        self.assertEqual(report['fixture_source'], 'synthetic_rgb')
        for key in ('cross_client_rng_isolation_passed', 'reset_discards_pending_actions_passed',
                    'close_recreate_clean_session_passed', 'tls_verification'):
            self.assertTrue(report[key], key)
        self.assertEqual(previous, {k: os.environ.get(k) for k in previous})

    def test_official_entry_self_test_uses_selected_backend_without_platform(self):
        import official_server
        settings = yaml.safe_load((PACKAGE / 'model.yaml').read_text())['hbot-full']
        shared = Pi05Policy(CounterPipeline(), make_adapter('hbot-full', settings), steps=15)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            root = Path(directory)
            tokenizer = root/'big_vision/paligemma_tokenizer.model'
            tokenizer.parent.mkdir()
            tokenizer.write_bytes(b'mocked-tokenizer-not-read-by-mock-backend')
            stats = root/'norm_stats.json'
            stats.write_text('{}')
            os.environ['OPENPI_DATA_HOME'] = str(root)
            prepared = dict(load=lambda: shared, check=lambda: None, stats=stats,
                            observation=example_observation)
            output = root/'run'
            with patch('backends.prepare_backend', return_value=prepared) as prepare, \
                    patch('legacy_server.check_resources') as resources, \
                    patch('smoke_wss.run_smoke', side_effect=lambda *a, **k: run_smoke(*a, **k, real_weights=False)), \
                    patch('sys.argv', ['official_server', '--backend', 'loongforge', '--self-test',
                                      '--checkpoint', str(root), '--output-dir', str(output)]):
                try:
                    official_server.main()
                finally:
                    for handler in logging.getLogger().handlers[:]:
                        handler.close()
                        logging.getLogger().removeHandler(handler)
            self.assertEqual(prepare.call_args.args[0], 'loongforge')
            resources.assert_called_once()
            ready = json.loads((output/'ready.json').read_text())
            self.assertEqual(ready['backend'], 'loongforge')
            self.assertEqual(ready['action_steps'], 15)
            self.assertEqual(ready['denoise_steps'], 10)
            result = json.loads((output/'wss_smoke.json').read_text())
            self.assertFalse(result['official_platform_connected'])
            self.assertTrue(json.loads((output/'connection_opened.json').read_text())['self_test'])
            self.assertFalse((output/'connection_error.json').exists())


if __name__ == '__main__':
    unittest.main()
