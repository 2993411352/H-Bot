"""CPU-only tests: real frozen preprocessing and protocol, never model inference."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', OMP_NUM_THREADS='2')
import copy
from contextlib import ExitStack
import importlib.util
import json
import math
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE/'vendor'))
import numpy as np
import torch
torch.set_num_threads(2)
from hbot_inference import Codec, PandaAdapter, SessionPolicy, example_observation, read
from model_runtime import check_checkpoint, check_model_config, check_shapes, load_exact_core, verify_runtime


class FakeEngine:
    def predict(self, encoded, generator):
        # Deliberately stochastic: interleaving must not mix client RNGs.
        return torch.randn(15, 32, generator=generator).numpy()


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.codec = Codec(BASE/'fixtures/norm_stats.json', BASE/'fixtures/hbot_contract.json',
                          Path(os.environ['OPENPI_DATA_HOME'])/'big_vision/paligemma_tokenizer.model')
        spec = importlib.util.spec_from_file_location('training_reference',
                    BASE/'fixtures/training_transform_reference.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls.reference = mod.HBotTransform.__new__(mod.HBotTransform)
        cls.reference.stats = cls.codec.stats
        cls.reference._tokenizer = cls.codec.tokenizer

    def session(self, steps=15):
        return SessionPolicy(FakeEngine(), self.codec, action_steps=steps, seed=42)

    def test_preprocessing_matches_frozen_training_exactly(self):
        rng = np.random.default_rng(7)
        for dims in ((48, 64), (224, 224), (319, 641), (641, 319)):
            for scale in (0., 1., 3.):
                obs = example_observation()
                obs['instruction'] = '  Stack_the cubes:\nblue, lime, red, cyan.  '
                obs['manipulators']['single_arm']['joint_position'] *= scale
                for item in obs['cameras'].values():
                    item['rgb'] = rng.integers(0, 256, (1, *dims, 3), dtype=np.uint8)
                raw = PandaAdapter().build_model_input(obs)
                encoded = self.codec.encode(raw)
                train = dict(raw, task=raw['prompt'], action=rng.normal(size=(15, 8)).astype(np.float32))
                for name in ('ext1_camera', 'wrist_camera', 'ext2_camera'):
                    key = 'observation.images.'+name
                    train[key] = raw[key].transpose(2, 0, 1).astype(np.float32)/255
                expected = self.reference(train)
                for image, ref in zip(encoded['images'], expected['images_list'], strict=True):
                    np.testing.assert_array_equal(image, ref.numpy())
                np.testing.assert_array_equal(encoded['input_ids'], expected['input_ids'].numpy())
                np.testing.assert_array_equal(encoded['attention_mask'], expected['attention_mask'].numpy())
                np.testing.assert_array_equal(self.codec.decode(expected['action'].numpy(), encoded['state']),
                    self.reference.decode_actions(expected['action'].numpy(), encoded['state']).astype(np.float32))
                np.testing.assert_allclose(self.codec.decode(expected['action'].numpy(), encoded['state']),
                    train['action'], atol=2e-6, rtol=1e-6)

    def test_camera_order_and_gripper(self):
        obs = example_observation()
        raw = PandaAdapter().build_model_input(obs)
        self.assertAlmostEqual(float(raw['observation.state'][7]), .5)
        encoded = self.codec.encode(raw)
        self.assertTrue(encoded['images'][0].mean() < encoded['images'][1].mean() < encoded['images'][2].mean())
        actions = np.tile(np.arange(8, dtype=np.float32), (15, 1))
        physical = PandaAdapter().build_action_sequence(actions, obs, 7)
        self.assertEqual(len(physical), 7)
        np.testing.assert_array_equal(physical[0]['values'][0, :7], actions[0, :7])
        self.assertEqual(physical[0]['joint_names'][-1:], ['finger_joint'])
        self.assertEqual(physical[0]['values'].shape, (1, 8))
        np.testing.assert_allclose(physical[0]['values'][0, 7:], [7*math.pi/4])

    def test_panda_droid_gripper_real_binding_roundtrip(self):
        adapter = PandaAdapter()
        for measured in (0., math.pi/8, math.pi/4):
            obs = example_observation()
            obs['manipulators']['single_arm']['gripper_position'][:] = measured
            state = adapter.build_model_input(obs)['observation.state']
            self.assertAlmostEqual(float(state[-1]), measured*4/math.pi, places=6)
            physical = adapter.build_action_sequence(np.tile(state, (15, 1)), obs, 1)
            np.testing.assert_allclose(physical[0]['values'][0, -1], measured, atol=1e-7)

    def test_panda_droid_rejects_franka_two_finger_layout(self):
        obs = example_observation()
        spec = obs['action_layout']['manipulators']['single_arm']
        spec['gripper_joint_names'] = ['finger_a', 'finger_b']
        with self.assertRaises(ValueError):
            PandaAdapter().build_model_input(obs)

    def test_invalid_observations_rejected(self):
        raw = PandaAdapter().build_model_input(example_observation())
        for key, value in [('prompt', ' '), ('prompt', 'abc '*500),
                           ('observation.state', np.full(8, np.nan)),
                           ('observation.images.ext1_camera', np.zeros((10, 10, 3), np.float32))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.codec.encode(dict(raw, **{key: value}))
        with self.assertRaises(KeyError):
            self.codec.encode({k:v for k,v in raw.items() if k != 'observation.images.ext2_camera'})
        with self.assertRaises(ValueError):
            self.codec.decode(np.full((15, 32), np.nan), raw['observation.state'])
        obs = example_observation()
        obs['action_layout']['embodiment_type'] = 'franka_panda'
        with self.assertRaises(ValueError):
            PandaAdapter().build_model_input(obs)

    def test_session_interleaving_and_reset(self):
        sessions, controls = [self.session() for _ in range(3)], [self.session() for _ in range(3)]
        obs = example_observation()
        for i in (0, 1, 0, 2, 1, 0, 2, 2):
            actual, expected = sessions[i].act_sequence(obs), controls[i].act_sequence(obs)
            np.testing.assert_array_equal(actual[0]['values'], expected[0]['values'])
        cached, control = self.session(), self.session()
        first = cached.act(obs)
        expected_first = control.act_sequence(obs)[0]
        np.testing.assert_array_equal(first['values'], expected_first['values'])
        self.assertEqual(len(cached._cache), 14)
        cached.reset()
        self.assertFalse(cached._cache)
        np.testing.assert_array_equal(cached.act(obs)['values'], control.act_sequence(obs)[0]['values'])
        changed = copy.deepcopy(obs)
        changed['action_layout']['manipulators']['single_arm']['gripper_policy_scale'] = 2.
        with self.assertRaises(ValueError):
            cached.act(changed)

    def test_execution_lengths(self):
        for steps in (1, 5, 8, 12, 15):
            self.assertEqual(len(self.session(steps).act_sequence(example_observation())), steps)
        for steps in (0, 16):
            with self.assertRaises(ValueError):
                self.session(steps)

    def test_strict_loader_preserves_distinct_embedding(self):
        model = torch.nn.Module()
        model.lm_head = torch.nn.Linear(3, 2, bias=False)
        model.embed_tokens = torch.nn.Embedding(2, 3)
        values = {'model.lm_head.weight': torch.full((2, 3), 11.),
                  'model.embed_tokens.weight': torch.full((2, 3), 23.)}
        load_exact_core(model, values)
        self.assertTrue(torch.all(model.lm_head.weight == 11))
        self.assertTrue(torch.all(model.embed_tokens.weight == 23))
        with self.assertRaises(RuntimeError):
            load_exact_core(model, {'model.lm_head.weight': torch.zeros(2, 3)})
        with self.assertRaises(ValueError):
            load_exact_core(model, {'lm_head.weight': torch.zeros(2, 3)})

    def test_frozen_runtime_and_meta_parameter_shapes(self):
        verify_runtime()
        report = check_shapes(read(BASE/'fixtures/checkpoint_header.json'))
        self.assertEqual(report['parameter_tensors'], 813)
        print('META_REPORT', json.dumps(report), flush=True)
        with patch('model_runtime.core_model', return_value=torch.nn.Linear(2, 3)):
            with self.assertRaises(ValueError):
                check_shapes({'model.weight': {'shape': [3, 2], 'dtype': 'F32'}})
            with self.assertRaises(ValueError):
                check_shapes({'model.weight': {'shape': [3, 2], 'dtype': 'BF16'},
                              'model.bias': {'shape': [3], 'dtype': 'F32'}})
        with self.assertRaises(ValueError):
            check_model_config({'action_horizon': 50})

    def test_test_checkpoint_rejected(self):
        marker = dict(status='VERIFIED', kind='inference_weights', step=2, run_id='acceptance_test')
        with patch('model_runtime.read', return_value=marker), self.assertRaisesRegex(ValueError, 'Wrong run'):
            check_checkpoint(BASE/'fixtures')

    def test_tiny_core_cached_denoising_api_on_cpu(self):
        # API regression only: random miniature model, NOT the real trained model.
        from loongforge.embodied.model.pi05 import modeling_pi05 as m
        from transformers import PaliGemmaConfig
        def small_vision_config():
            return PaliGemmaConfig(vision_config=dict(hidden_size=32, num_hidden_layers=1,
                num_attention_heads=4, intermediate_size=64, patch_size=14))
        mapping = {'paligemma': small_vision_config, 'gemma': m.CONFIG_MAPPING['gemma']}
        small = m._GemmaConfig(32, 2, 64, 2, 1, 16)
        original = m.PaliGemmaForConditionalGenerationWithPiGemma
        def small_projector(config):
            config.vision_config.projection_dim = 32
            return original(config)
        with patch.object(m, 'CONFIG_MAPPING', mapping), patch.object(m, '_get_gemma_config', return_value=small), \
                patch.object(m, 'PaliGemmaForConditionalGenerationWithPiGemma', small_projector):
            model = m.PI05Pytorch(m._PI05InternalConfig(dtype='float32', chunk_size=15,
                n_action_steps=15, image_resolution=(28, 28), compile_model=False)).eval()
        tokens = torch.tensor([[2, 17, 28, 0]], dtype=torch.long)
        noise = torch.randn(1, 15, 32, generator=torch.Generator().manual_seed(9))
        with torch.inference_mode():
            out = model.sample_actions([torch.zeros(1, 3, 28, 28)]*3,
                [torch.tensor([True])]*3, tokens, tokens != 0, noise=noise, num_steps=2)
        self.assertEqual(tuple(out.shape), (1, 15, 32))
        self.assertTrue(torch.isfinite(out).all())
        self.assertFalse(torch.equal(out, noise))

    def test_three_real_websocket_clients_with_fake_model(self):
        from robo_orchard_server.server.policy_server import PolicyWebsocketServer, _encode_message, _decode_message
        from websockets.sync.server import serve
        from websockets.sync.client import connect
        shared = self.session(8)
        service = PolicyWebsocketServer(policy_factory=shared.new_session)
        with serve(service.handle_client, '127.0.0.1', 0, max_size=200*1024*1024, compression=None) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.socket.getsockname()[1]
            try:
                with ExitStack() as stack:
                    clients = [stack.enter_context(connect(f'ws://127.0.0.1:{port}', proxy=None)) for _ in range(3)]
                    controls = [self.session(8) for _ in clients]
                    def request(client, payload):
                        client.send(_encode_message(payload))
                        response = _decode_message(client.recv(timeout=10))
                        self.assertNotIn('error', response)
                        return response
                    obs = example_observation()
                    for i in (0, 1, 2, 0, 2, 1):
                        actions = request(clients[i], dict(type='act_sequence', obs_data=dict(obs, format='canonical')))['actions']
                        expected = controls[i].act_sequence(obs)
                        self.assertEqual(len(actions), 8)
                        command = actions[0]['__joint_command__']
                        self.assertEqual(command['joint_names'], expected[0]['joint_names'])
                        np.testing.assert_array_equal(command['values'], expected[0]['values'])
                    for client in clients:
                        self.assertTrue(request(client, dict(type='reset'))['ok'])
            finally:
                server.shutdown()
                thread.join(timeout=5)
                self.assertFalse(thread.is_alive())


if __name__ == '__main__':
    unittest.main(verbosity=2)
