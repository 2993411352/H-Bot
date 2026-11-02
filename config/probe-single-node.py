#!/usr/bin/env python3
"""Bounded, unprivileged probe; no changes to the main proxy or downloads."""
import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import tempfile
import time

import yaml

ROOT = Path('/home/wangmengfan/mihomo')
NODE_HOST = 'us1.oh-node.com'
NODE_IP = '54.176.102.50'
NODE_PORT = 443
ASSET = ('https://huggingface.co/datasets/HorizonRobotics/robo_orchard_sim_assets/'
         'resolve/542bec4a9c9fdb3d688e3bfa20cc1558cc016afc/'
         'PUBLIC_OBJECTS/evaluation_assets/tables/table_001/choice_textures/6040.png')


def selected_node():
    if os.getuid() != 1013:
        raise RuntimeError('Probe must run as wangmengfan (UID 1013), never root.')
    config = json.loads((ROOT / 'config.yaml').read_text())
    provider = config['proxy-providers']['Subscription']
    if provider.get('override'):
        raise RuntimeError('Unexpected provider override; refusing an ambiguous test.')
    payload = yaml.safe_load((ROOT / provider['path']).read_text())
    nodes = [p for p in payload['proxies'] if p.get('server') == NODE_HOST
             and p.get('port') == NODE_PORT and 'United States 01' in p.get('name', '')]
    if len(nodes) != 1:
        raise RuntimeError('Expected exactly one matching US01 node.')
    node = dict(nodes[0])
    if node.get('dialer-proxy') or node.get('routing-mark'):
        raise RuntimeError('Unexpected node route override; refusing an ambiguous test.')
    if not node.get('tls') or not (node.get('servername') or node.get('sni')):
        raise RuntimeError('Expected explicit TLS server name.')
    if node.get('skip-cert-verify'):
        raise RuntimeError('Refusing a node that disables certificate verification.')
    addresses = {result[4][0] for result in socket.getaddrinfo(
        NODE_HOST, NODE_PORT, socket.AF_INET, socket.SOCK_STREAM)}
    if NODE_IP not in addresses:
        raise RuntimeError('Node DNS changed; re-check the bypass destination first.')
    node['name'] = 'SingleNodeProbe'
    node['server'] = NODE_IP  # Exact match to the temporary /32 firewall exception.
    return node


def interrupted(signum, frame):
    raise KeyboardInterrupt


def run_probe(node):
    os.umask(0o077)
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        port = reservation.getsockname()[1]
    password = secrets.token_hex(24)
    config = {
        'mixed-port': port, 'allow-lan': False, 'bind-address': '127.0.0.1',
        'authentication': ['probe:' + password], 'skip-auth-prefixes': [],
        'mode': 'rule', 'log-level': 'warning', 'ipv6': False,
        'find-process-mode': 'off', 'tun': {'enable': False},
        'dns': {'enable': False}, 'sniffer': {'enable': False},
        'geo-auto-update': False, 'profile': {'store-selected': False},
        'proxies': [node], 'proxy-groups': [], 'rules': ['MATCH,SingleNodeProbe'],
    }
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, interrupted)
    with tempfile.TemporaryDirectory(prefix='single-node-probe-', dir=ROOT) as folder:
        folder = Path(folder)
        path = folder / 'config.json'
        path.write_text(json.dumps(config))
        process = None
        try:
            with (folder / 'core.log').open('wb') as log:
                process = subprocess.Popen([str(ROOT / 'bin/mihomo'), '-d', str(folder),
                                            '-f', str(path)], stdout=log, stderr=subprocess.STDOUT)
                for attempt in range(30):
                    if process.poll() is not None:
                        raise RuntimeError('Isolated probe core failed to start.')
                    try:
                        with socket.create_connection(('127.0.0.1', port), timeout=0.2):
                            break
                    except OSError:
                        time.sleep(0.1)
                else:
                    raise RuntimeError('Isolated probe listener did not become ready.')
                tests = [('huggingface_head_1', 'https://huggingface.co/', True),
                         ('huggingface_head_2', 'https://huggingface.co/', True),
                         ('github_head', 'https://github.com/', True),
                         ('asset_range_64KiB', ASSET, False)]
                for label, url, head in tests:
                    command = ['/usr/bin/curl', '--silent', '--show-error', '--location',
                               '--proto', '=https', '--proto-redir', '=https', '--max-redirs', '5',
                               '--noproxy', '', '--connect-timeout', '5', '--max-time', '10',
                               '--output', '/dev/null', '--write-out',
                               '%{http_code} %{ssl_verify_result} %{size_download} %{time_total}',
                               '--config', '-', url]
                    command[1:1] = ['--head'] if head else ['--range', '0-65535', '--max-filesize', '1048576']
                    settings = ('proxy = "http://127.0.0.1:%d"\nproxy-user = "probe:%s"\n'
                                % (port, password))
                    result = subprocess.run(command, input=settings.encode(), stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, timeout=13)
                    values = result.stdout.decode(errors='replace').strip().split()
                    valid = result.returncode == 0 and len(values) == 4
                    valid = valid and values[0].isdigit() and 200 <= int(values[0]) < 400 and values[1] == '0'
                    if not head:
                        valid = valid and float(values[2]) > 0
                    print(json.dumps({'test': label, 'curl_exit': result.returncode,
                                      'http_tls_bytes_seconds': values, 'passed': bool(valid)}), flush=True)
                    if not valid:
                        return 1
                return 0
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-config', action='store_true')
    args = parser.parse_args()
    node = selected_node()
    if args.check_config:
        print('Preflight OK: UID=1013 node=US01 pinned_destination=54.176.102.50:443')
        return 0
    return run_probe(node)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Probe interrupted; temporary core cleanup requested.', flush=True)
        raise SystemExit(130)
    except Exception as error:
        # Never print credentials, subscription payloads, or signed download URLs.
        print('Probe failed: ' + type(error).__name__, flush=True)
        raise SystemExit(2)
