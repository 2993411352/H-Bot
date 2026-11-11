"""Real-weight integration tests through verified local TLS and official routing.

Uses training observations only as protocol fixtures; never measures task scores.
The client tensor codec below is independent of the policy-server codec.
"""
import copy
from concurrent.futures import Future
import json
import os
from pathlib import Path
import ssl
import struct
import subprocess
import threading
import time

import numpy as np


def encode(payload):
    blocks = []

    def visit(value):
        if isinstance(value, np.ndarray):
            index = len(blocks)
            blocks.append(value.tobytes())
            return {'__tensor__': {'dtype': value.dtype.name, 'shape': list(value.shape)},
                    '__tensor_idx__': index}
        if isinstance(value, dict):
            return {k: visit(v) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(v) for v in value]
        return value

    header = json.dumps(visit(payload), separators=(',', ':')).encode()
    return b''.join(struct.pack('!Q', len(v)) + v for v in [header, *blocks])


def decode(message):
    blocks, cursor = [], 0
    while cursor < len(message):
        size = struct.unpack_from('!Q', message, cursor)[0]
        cursor += 8
        blocks.append(message[cursor:cursor+size])
        cursor += size

    def visit(value):
        if isinstance(value, list):
            return [visit(v) for v in value]
        if not isinstance(value, dict):
            return value
        if '__tensor__' in value:
            info = value['__tensor__']
            return np.frombuffer(blocks[1+value['__tensor_idx__']],
                                 dtype=info['dtype']).reshape(info['shape']).copy()
        return {k: visit(v) for k, v in value.items()}

    return visit(json.loads(blocks[0]))


def observations(dataset_root=None, observation_factory=None):
    if observation_factory is None:
        from pi05_server_config import example_observation
        observation_factory = example_observation
    if dataset_root is None:
        from official_server import CASES
        result = []
        for task, prompt in CASES:
            obs = observation_factory()
            obs['format'] = 'canonical'
            obs['instruction'] = prompt
            for camera in obs['cameras'].values():
                camera['rgb'] = np.zeros((1, 480, 640, 3), np.uint8)
            result.append((task, obs))
        return result
    import av
    import pyarrow.parquet as pq
    root = Path(dataset_root).expanduser().resolve()
    tasks = ['pick_color', 'spatial_pick', 'place_a2b', 'stack_cubes',
             'open_drawer', 'joint_direction']
    result = []
    for task in tasks:
        path = root / f'panda_droid-{task}-20261013'
        row = pq.read_table(path / 'data/chunk-000/episode_000000.parquet',
                            columns=['observation.state', 'task_index']).slice(0, 1).to_pylist()[0]
        prompts = [json.loads(line) for line in (path / 'meta/tasks.jsonl').read_text().splitlines()]
        prompt = next(item['task'] for item in prompts if item['task_index'] == row['task_index'])
        obs = observation_factory()
        obs['format'] = 'canonical'
        obs['instruction'] = prompt
        state = np.asarray(row['observation.state'], np.float32)
        obs['manipulators']['single_arm']['joint_position'] = state[None, :7].copy()
        obs['manipulators']['single_arm']['gripper_position'] = np.full((1, len(obs['action_layout']['manipulators']['single_arm']['gripper_joint_names'])),
            state[7]/obs['action_layout']['manipulators']['single_arm']['gripper_policy_scale'], np.float32)
        for name in obs['cameras']:
            with av.open(str(path / f'videos/chunk-000/observation.images.{name}/episode_000000.mp4')) as video:
                obs['cameras'][name]['rgb'] = next(video.decode(video=0)).to_ndarray(format='rgb24')[None]
        result.append((task, obs))
    return result


def action_array(response, expected_count=15, joint_names=None):
    joint_names = joint_names or [f'panda_joint{i}' for i in range(1, 8)] + ['finger_a', 'finger_b']
    if 'error' in response:
        raise AssertionError(response['error'])
    actions = response['actions']
    if expected_count == 1 and isinstance(actions, dict):
        actions = [actions]
    assert len(actions) == expected_count
    arrays = []
    for action in actions:
        command = action['__joint_command__']
        value = command['values']
        assert value.dtype == np.float32 and value.shape == (1, len(joint_names))
        assert np.isfinite(value).all()
        assert command['joint_names'] == joint_names
        arrays.append(value[0])
    return np.stack(arrays)


def run_smoke(server, output, dataset_root=None, real_weights=True, observation_factory=None):
    from robo_orchard_server.server.multiplex import PROTOCOL, wrap, unwrap
    from websockets.sync.server import serve
    fixtures = observations(dataset_root, observation_factory)
    spec = fixtures[0][1]['action_layout']['manipulators']['single_arm']
    joint_names = spec['arm_joint_names'] + spec['gripper_joint_names']
    cert, key = output / 'local-test-ca.pem', output / 'local-test-key.pem'
    subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
                    '-keyout', str(key), '-out', str(cert), '-days', '1',
                    '-subj', '/CN=127.0.0.1', '-addext', 'subjectAltName=IP:127.0.0.1'],
                   check=True, capture_output=True)
    os.chmod(key, 0o600)
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(cert, key)
    # Each test opens exactly one connection; TLS session resumption is unnecessary.
    # Avoid post-handshake ticket traffic in this in-process threaded TLS fixture.
    tls.num_tickets = 0
    original_ca = os.environ.get('SSL_CERT_FILE')
    original_no_proxy = {k: os.environ.get(k) for k in ('no_proxy', 'NO_PROXY')}
    for k in original_no_proxy:
        os.environ[k] = '127.0.0.1,localhost'
    os.environ['SSL_CERT_FILE'] = str(cert)
    # Only this test process trusts its generated test CA; verification stays on.
    results, latency = {}, []

    def supervisor(ws):
        ws.send(json.dumps({'type': 'policy_protocol', 'protocol': PROTOCOL}))
        assert json.loads(ws.recv(timeout=10)) == {'type': 'policy_protocol_ready', 'protocol': PROTOCOL}
        request_number = 0

        def send(session, request):
            nonlocal request_number
            request_number += 1
            identity = str(request_number)
            ws.send(wrap(encode(request), session, identity))
            return identity

        def receive(session, identity):
            got_session, got_request, payload = unwrap(ws.recv(timeout=30))
            assert (got_session, got_request) == (session, identity)
            reply = decode(payload)
            assert 'error' not in reply, reply
            return reply

        def exchange(session, request):
            return receive(session, send(session, request))

        def reset(session):
            assert exchange(session, {'type': 'reset'})['ok'] is True

        def infer(session, obs, kind='act_sequence'):
            return action_array(exchange(session, {'type': kind, 'obs_data': obs}),
                                1 if kind == 'act' else 15, joint_names)

        for i, (task, obs) in enumerate(fixtures):
            reset(f'task{i}')
            start = time.perf_counter()
            infer(f'task{i}', obs)
            latency.append({'task': task, 'seconds': time.perf_counter()-start})
        results['six_tasks_wire_passed'] = True
        results['fixture_source'] = 'training_observations' if dataset_root else 'synthetic_rgb'
        results['real_observation_tested'] = dataset_root is not None
        # Interleaved clients must have independent sampling RNG progression.
        for session in ['a', 'b', 'interleaved', 'd', 'e']:
            reset(session)
        obs = fixtures[0][1]
        np.testing.assert_array_equal(infer('a', obs), infer('b', obs))
        infer('interleaved', fixtures[1][1])
        a_second = infer('a', obs)
        infer('interleaved', fixtures[2][1])
        np.testing.assert_array_equal(a_second, infer('b', obs))
        results['cross_client_rng_isolation_passed'] = True
        # Reset discards 14 pending act() outputs without rewinding RNG.
        d_first = infer('d', obs, 'act')
        e_first = infer('e', obs)
        np.testing.assert_array_equal(d_first[0], e_first[0])
        reset('d')
        np.testing.assert_array_equal(infer('d', obs, 'act')[0], infer('e', obs)[0])
        results['reset_discards_pending_actions_passed'] = True
        burst_results = []
        for width in [4, 8]:
            for i in range(width):
                reset(f'burst{width}-{i}')
            for iteration in range(3):
                pending = []
                for i in range(width):
                    session = f'burst{width}-{i}'
                    begin = time.perf_counter()
                    identity = send(session, {'type': 'act_sequence', 'obs_data': fixtures[i % 6][1]})
                    pending.append((session, identity, begin))
                elapsed = []
                for session, identity, begin in pending:
                    action_array(receive(session, identity), joint_names=joint_names)
                    elapsed.append(time.perf_counter()-begin)
                burst_results.append({'clients': width, 'iteration': iteration,
                                      'request_queue_inclusive_seconds': elapsed,
                                      'max_seconds': max(elapsed)})
                assert max(elapsed) < 30, 'Local queue + inference exceeded 30 seconds'
        results['burst_results'] = burst_results
        # Close/recreate starts a clean logical session and reproduces first RNG.
        assert exchange('a', {'type': 'close_session'})['ok']
        reset('a')
        np.testing.assert_array_equal(infer('a', obs), e_first)
        results['close_recreate_clean_session_passed'] = True
        results['observation_timings'] = latency
        results['requests'] = request_number
        return results

    try:
        future = Future()

        def handle(ws):
            try:
                future.set_result(supervisor(ws))
            except BaseException as exc:
                future.set_exception(exc)

        with serve(handle, '127.0.0.1', 0, ssl=tls, max_size=200*1024*1024,
                   compression=None, close_timeout=1) as endpoint:
            thread = threading.Thread(target=endpoint.serve_forever, daemon=True)
            thread.start()
            try:
                server.run_reverse(f'wss://127.0.0.1:{endpoint.socket.getsockname()[1]}')
                result = future.result(timeout=5)
            finally:
                endpoint.shutdown()
                thread.join(timeout=5)
        return dict(status='REAL_WEIGHT_LOCAL_WSS_PASSED' if real_weights else 'MOCK_POLICY_LOCAL_WSS_PASSED',
                    weights_loaded=real_weights, tls_verification=True,
                    official_platform_connected=False, performance_not_task_score=True,
                    **result)
    finally:
        for k, value in original_no_proxy.items():
            if value is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = value
        if original_ca is None:
            os.environ.pop('SSL_CERT_FILE', None)
        else:
            os.environ['SSL_CERT_FILE'] = original_ca
