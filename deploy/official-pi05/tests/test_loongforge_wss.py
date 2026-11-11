"""Exercise the actual Torch session adapter through the original TLS/WSS test."""
import os
from pathlib import Path
import tempfile
import unittest

from backends import enable_loongforge
from robo_orchard_server.server import PolicyWebsocketServer
from smoke_wss import run_smoke


@unittest.skipUnless(os.environ.get('HBOT_BACKEND') == 'loongforge', 'Requires LoongForge venv')
class LoongForgeWssTests(unittest.TestCase):
    def test_torch_session_rng_reset_and_single_gripper(self):
        root = enable_loongforge()
        import torch
        from hbot_inference import Codec, SessionPolicy, example_observation
        class MockEngine:
            def predict(self, encoded, generator):
                return torch.randn(15, 32, generator=generator).numpy()
        tokenizer = Path(os.environ['OPENPI_DATA_HOME'])/'big_vision/paligemma_tokenizer.model'
        codec = Codec(root/'fixtures/norm_stats.json', root/'fixtures/hbot_contract.json', tokenizer)
        shared = SessionPolicy(MockEngine(), codec, action_steps=15)
        server = PolicyWebsocketServer(policy_factory=shared.new_session)
        with tempfile.TemporaryDirectory() as directory:
            report = run_smoke(server, Path(directory), real_weights=False,
                               observation_factory=example_observation)
        self.assertEqual(report['status'], 'MOCK_POLICY_LOCAL_WSS_PASSED')
        self.assertTrue(report['cross_client_rng_isolation_passed'])
        self.assertTrue(report['reset_discards_pending_actions_passed'])
        self.assertFalse(report['official_platform_connected'])


if __name__ == '__main__':
    unittest.main()
