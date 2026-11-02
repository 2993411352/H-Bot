"""Independent protocol session; no evaluation RNG or episode is consumed."""
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'config'))
import numpy as np
from pi05_server_config import example_observation
from robo_orchard_server.server.policy_server import _encode_message, _decode_message
from websockets.sync.client import connect


def main():
    obs = example_observation()
    # Real camera dimensions do not affect the model's fixed 224x224 inputs.
    for camera in obs['cameras'].values():
        camera['rgb'] = np.zeros((1, 480, 640, 3), dtype=np.uint8)
    obs['format'] = 'canonical'
    timings = []
    with connect('ws://127.0.0.1:8766', max_size=200*1024*1024,
                 open_timeout=10, compression=None) as ws:
        ws.send(_encode_message({'type': 'reset'}))
        result = _decode_message(ws.recv(timeout=30))
        if result.get('ok') is not True:
            raise RuntimeError(str(result))
        for _ in range(3):
            started = time.monotonic()
            ws.send(_encode_message({'type': 'act_sequence', 'obs_data': obs}))
            result = _decode_message(ws.recv(timeout=300))
            actions = result.get('actions', [])
            if len(actions) != 15:
                raise RuntimeError('Warmup action count/error: ' + str(result))
            timings.append(time.monotonic() - started)
    print(json.dumps(dict(status='WARMUP_PASSED', independent_session=True,
                          action_steps=15, request_seconds=timings)), flush=True)


if __name__ == '__main__':
    main()
