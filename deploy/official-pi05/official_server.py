"""Official JAX/LoongForge reverse-connection entry; never reconnects itself."""
from __future__ import annotations

import argparse
from datetime import datetime
import getpass
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

PACKAGE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('HBOT_HOME', str(PACKAGE / 'runtime'))).expanduser().resolve()
CASES = (
    ('pick_color', 'Take white charging head.'),
    ('pick_spatial', 'Pick up the ballpoint pen behind the spice jar.'),
    ('place_a2b', 'Pick bottle opener and place in bowl.'),
    ('stack_cubes', 'Stack the cubes bottom-to-top: red, yellow, lime, cyan.'),
    ('open_drawer', 'open the top drawer of the cabinet'),
    ('joint_direction', 'open the door of the microwave oven'),
)


def validate_uri(uri):
    parsed = urlsplit(uri.strip())
    if (parsed.scheme != 'wss' or not parsed.hostname
            or parsed.username or parsed.password or parsed.fragment):
        raise ValueError('Use the full platform wss:// URL; no credentials in the authority or fragment')
    return uri.strip()


def verify_frozen():
    for name, expected in json.loads((PACKAGE / 'inference_manifest.json').read_text())['sha256'].items():
        if hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen inference configuration changed: {name}')
    manifest = json.loads((PACKAGE / 'vendor_manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        if hashlib.sha256((PACKAGE / 'vendor' / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen protocol implementation changed: {name}')


def validate_actions(actions, observation=None):
    import numpy as np
    if len(actions) != 15:
        raise ValueError('Expected exactly 15 actions')
    names = [f'panda_joint{i}' for i in range(1, 8)] + ['finger_a', 'finger_b']
    if observation is not None:
        layout = observation['action_layout']
        names = [joint for manipulator in layout['manipulator_order']
                 for joint in (layout['manipulators'][manipulator]['arm_joint_names']
                               + layout['manipulators'][manipulator]['gripper_joint_names'])]
    for action in actions:
        values = action['values']
        if (values.shape != (1, len(names)) or values.dtype != np.float32
                or not np.isfinite(values).all()
                or action['joint_names'] != names):
            raise ValueError('Invalid canonical float32 action for the requested joint layout')


def warmup(shared, observation_factory):
    import numpy as np
    rows = []
    # Dedicated sessions keep the template and evaluation RNG unchanged.
    for task, instruction in CASES:
        session = shared.new_session()
        obs = observation_factory()
        obs['instruction'] = instruction
        for image in obs['cameras'].values():
            image['rgb'] = np.zeros((1, 480, 640, 3), np.uint8)
        started = time.perf_counter()
        validate_actions(session.act_sequence(obs), obs)
        elapsed = time.perf_counter() - started
        session.close()
        rows.append({'task': task, 'seconds': elapsed, 'actions': 15})
        print(json.dumps({'warmup': rows[-1]}), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['jax', 'loongforge'], default=os.environ.get('HBOT_BACKEND', 'jax'))
    parser.add_argument('--gpu', default=os.environ.get('HBOT_GPU', '0'), help='Physical NVIDIA GPU index')
    parser.add_argument('--checkpoint', type=Path, default=os.environ.get('HBOT_PI05_CHECKPOINT'),
                        help='Checkpoint root for the selected backend; see README')
    parser.add_argument('--fixtures', type=Path, default=os.environ.get('HBOT_SMOKE_DATASET_ROOT'),
                        help='Optional six-task LeRobot root for original real-observation WSS test')
    parser.add_argument('--supervisor-uri')
    parser.add_argument('--supervisor-uri-file', type=Path)
    parser.add_argument('--team-id', default=os.environ.get('TEAM_ID'))
    parser.add_argument('--evaluation-id', default=os.environ.get('SESSION_ID'))
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.checkpoint is None:
        parser.error('Set HBOT_PI05_CHECKPOINT in local.env or pass --checkpoint')
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_dir():
        parser.error('Checkpoint directory does not exist')
    if not args.gpu.isdigit():
        parser.error('--gpu must be one physical NVIDIA GPU index')
    tokenizer = Path(os.environ.get('OPENPI_DATA_HOME', str(ROOT / 'cache'))) / 'big_vision/paligemma_tokenizer.model'
    if not tokenizer.is_file():
        parser.error('Missing offline tokenizer; run setup.sh first')
    if sum(bool(x) for x in (args.supervisor_uri, args.supervisor_uri_file)) > 1:
        parser.error('Choose one URL input')
    if args.check and args.self_test:
        parser.error('Choose check or self-test')
    if (args.check or args.self_test) and (args.supervisor_uri or args.supervisor_uri_file):
        parser.error('Checks never connect to a platform evaluation')
    uri = args.supervisor_uri
    if args.supervisor_uri_file:
        uri = args.supervisor_uri_file.read_text().strip()
    if uri:
        uri = validate_uri(uri)
    if not (args.check or args.self_test or uri or sys.stdin.isatty()):
        parser.error('Provide a URL file/argument or use an interactive terminal')
    verify_frozen()
    os.environ.update(HBOT_PI05_CHECKPOINT=str(checkpoint),
                      CUDA_VISIBLE_DEVICES='' if args.check else args.gpu,
                      JAX_PLATFORMS='cpu' if args.check else 'cuda',
                      XLA_PYTHON_CLIENT_PREALLOCATE='false')
    from legacy_server import check_resources
    from backends import prepare_backend
    prepared = prepare_backend(args.backend, checkpoint, tokenizer)
    if args.check:
        prepared['check']()
        return
    # Port zero is an ephemeral preflight probe, never a public listener.
    check_resources(args.gpu, '127.0.0.1', 0, False)
    output = args.output_dir or ROOT / 'logs' / (
        args.backend + '_official_' + time.strftime('%Y%m%d_%H%M%S') + '_' + str(os.getpid()))
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, force=True, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(output / 'server.log'), logging.StreamHandler()])
    from robo_orchard_server.server import PolicyWebsocketServer
    from robo_orchard_server.server.policy import BasePolicy
    print(f'Loading {checkpoint.name} (H-Bot full SFT) on GPU {args.gpu}; logs={output}', flush=True)
    started = time.perf_counter()
    shared = prepared['load']()
    warmed = warmup(shared, prepared['observation'])
    manifest = dict(checkpoint=str(checkpoint), profile='hbot-full', backend=args.backend, action_steps=15,
                    denoise_steps=10, cameras=3, gpu=args.gpu, event_rules=False,
                    warmup=warmed, load_and_warmup_seconds=time.perf_counter()-started,
                    normalization_sha256=hashlib.sha256(prepared['stats'].read_bytes()).hexdigest(),
                    ready_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                    official_connection_tested=False)
    (output / 'ready.json').write_text(json.dumps(manifest, indent=2))
    count = 0
    metrics = (output / 'requests.jsonl').open('a', buffering=1)

    class LoggedSession(BasePolicy):
        def __init__(self, identity):
            self.identity = identity
            self.policy = shared.new_session()
            self.reset_count = 0
            self.request_in_episode = 0
            self.instruction = None

        def call(self, kind, function, *values):
            begin = time.perf_counter()
            error_type = None
            returned_steps = None
            try:
                result = function(*values)
                if kind == 'act_sequence':
                    returned_steps = len(result)
                elif kind == 'act':
                    returned_steps = 1
                ok = True
                return result
            except BaseException as error:
                ok = False
                error_type = type(error).__name__
                raise
            finally:
                metrics.write(json.dumps(dict(timestamp=datetime.now().astimezone().isoformat(),
                    monotonic_seconds=time.perf_counter(), session=self.identity,
                    episode_from_reset=self.reset_count,
                    request_in_episode=self.request_in_episode,
                    instruction=self.instruction, operation=kind,
                    returned_steps=returned_steps, seconds=time.perf_counter()-begin,
                    ok=ok, error_type=error_type), ensure_ascii=False) + '\n')

        def reset(self):
            self.reset_count += 1
            self.request_in_episode = 0
            self.instruction = None
            return self.call('reset', self.policy.reset)

        def close(self):
            return self.call('close', self.policy.close)

        def act(self, obs):
            self.request_in_episode += 1
            self.instruction = obs.get('instruction')
            return self.call('act', self.policy.act, obs)

        def act_sequence(self, obs):
            self.request_in_episode += 1
            self.instruction = obs.get('instruction')
            return self.call('act_sequence', self.policy.act_sequence, obs)

    def factory():
        nonlocal count
        count += 1
        return LoggedSession(count)

    class ObservedServer(PolicyWebsocketServer):
        def handle_sessions(self, websocket):
            (output / 'connection_opened.json').write_text(json.dumps({
                'opened_at': datetime.now().astimezone().isoformat(),
                'protocol': 'sessions-v1', 'negotiation_completed': True,
                'self_test': args.self_test}))
            logging.info('WSS_NEGOTIATION_PASSED; serving logical Client sessions')
            try:
                return super().handle_sessions(websocket)
            finally:
                details = dict(closed_at=datetime.now().astimezone().isoformat(),
                               close_code=websocket.close_code,
                               close_reason=websocket.close_reason,
                               self_test=args.self_test,
                               note='Transport closure is not proof of evaluation completion.')
                (output / 'transport_closed.json').write_text(json.dumps(details, ensure_ascii=False, indent=2))
                logging.warning('WSS_CLOSED code=%s reason=%s', details['close_code'], details['close_reason'])

    server = ObservedServer(policy_factory=factory,
                                   logging_tag=args.backend+'-fixed15-official')
    try:
        if args.self_test:
            from smoke_wss import run_smoke
            result = run_smoke(server, output, dataset_root=args.fixtures, observation_factory=prepared['observation'])
            (output / 'wss_smoke.json').write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
            return
        print('MODEL_READY: weights loaded, six-task warmup passed; no platform connection yet.', flush=True)
        if not uri:
            uri = validate_uri(getpass.getpass('Paste the complete platform WSS URL (hidden): '))
        print('Connecting once to the platform; no automatic reconnect/retry.', flush=True)
        try:
            server.run_reverse(uri, team_id=args.team_id, evaluation_id=args.evaluation_id)
        except Exception as error:
            (output / 'connection_error.json').write_text(json.dumps({
                'at': datetime.now().astimezone().isoformat(),
                'error_type': type(error).__name__,
                'error': 'Connection failed; session URL omitted from diagnostics'}))
            raise RuntimeError(f'WSS connection failed ({type(error).__name__}); no automatic retry') from None
        (output / 'connection_closed.json').write_text(json.dumps({
            'closed_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'note': 'Connection ended; verify final platform status. This is not a score/completion receipt.'}))
    finally:
        metrics.close()


if __name__ == '__main__':
    main()
