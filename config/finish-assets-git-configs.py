#!/usr/bin/env python3
"""Fetch only missing non-LFS config blobs, then verify and install atomically."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
import time

ROOT = Path('/study/wangmengfan/hbot')
TARGET = ROOT / 'datasets/robo_orchard_sim_assets'
GIT = ROOT / 'tmp/assets-config-git'
OBJECTS_GIT = ROOT / 'tmp/assets-config-tree-only'
PLAN = ROOT / 'tmp/assets-config-git-plan.json'
REVISION = '542bec4a9c9fdb3d688e3bfa20cc1558cc016afc'


def log(message):
    text = f'[{datetime.now(timezone.utc).isoformat()}] {message}'
    print(text, flush=True)
    with (ROOT / 'logs/sim-assets-download.log').open('a') as stream:
        stream.write(text + '\n')


def destination(relative):
    parts = PurePosixPath(relative)
    if not relative.startswith('configs/') or parts.is_absolute() or '..' in parts.parts:
        raise ValueError('Not an allowed configuration path')
    p = TARGET / relative
    if not p.resolve().is_relative_to(TARGET.resolve()):
        raise ValueError('Configuration path escapes asset root')
    return p


def blob_hash(content):
    return hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def fetch_only():
    info = json.loads((ROOT / 'tmp/assets-info.json').read_text())
    assert info['sha'] == REVISION, 'Catalog revision differs'
    missing = []
    for item in info['siblings']:
        p = TARGET / item['rfilename']
        if p.is_file() and p.stat().st_size == item['size']:
            continue
        destination(item['rfilename'])
        if item.get('lfs') or item['size'] > 1024 * 1024 or not re.fullmatch('[0-9a-f]{40}', item['blobId']):
            raise ValueError('Unexpected remaining object; refusing bulk fetch')
        missing.append(item)
    tree = subprocess.check_output(['git', '-C', str(GIT), 'ls-tree', '-r', '-z', REVISION, '--', 'configs'])
    expected = {}
    for entry in tree.split(b'\0'):
        if not entry:
            continue
        header, name = entry.split(b'\t', 1)
        mode, kind, oid = header.split()
        if kind == b'blob' and mode in (b'100644', b'100755'):
            expected[name.decode()] = oid.decode()
    for item in missing:
        if expected.get(item['rfilename']) != item['blobId']:
            raise ValueError('Pinned Git tree differs from catalog')
    write_json(PLAN, {'revision': REVISION, 'files': missing})
    oids = sorted({item['blobId'] for item in missing})
    log(f'GIT_CONFIG_FETCH_START files={len(missing)} unique_blobs={len(oids)} bytes={sum(x["size"] for x in missing)}')
    started = time.monotonic()
    subtree = subprocess.check_output(['git', '-C', str(GIT), 'ls-tree', '-z', REVISION, '--', 'configs'])
    header, name = subtree.rstrip(b'\0').split(b'\t', 1)
    mode, kind, oid = header.split()
    assert mode == b'040000' and kind == b'tree' and name == b'configs'
    # One tree want is better supported than thousands of individual blob wants.
    # Traversal starts at the pinned configs tree only, excluding unrelated
    # large assets and repository history.
    # Keep this object store separate from the metadata-only partial clone.
    # Otherwise older Git may consider an existing tree complete and skip blobs.
    if not OBJECTS_GIT.exists():
        subprocess.run(['git', 'init', '--bare', '-q', str(OBJECTS_GIT)], check=True)
        subprocess.run(['git', '-C', str(OBJECTS_GIT), 'config', 'remote.origin.url',
                        'https://huggingface.co/datasets/HorizonRobotics/robo_orchard_sim_assets'], check=True)
    command = ['git', '-C', str(OBJECTS_GIT), '-c', 'protocol.version=2', '-c', 'http.version=HTTP/1.1',
               '-c', 'http.lowSpeedLimit=1024', '-c', 'http.lowSpeedTime=30',
               'fetch', '-q', '--no-tags', 'origin', oid.decode()]
    # No automatic rapid retry. In particular, a server 429 stops this job.
    result = subprocess.run(command, timeout=300)
    if result.returncode:
        raise SystemExit(f'Configuration tree fetch stopped: git exit {result.returncode}; old downloader unchanged')
    log(f'GIT_CONFIG_FETCH_TREE_READY elapsed={time.monotonic()-started:.1f}s')
    verify()
    log('GIT_CONFIG_FETCH_READY; existing downloader has not been changed')


def verify():
    plan = json.loads(PLAN.read_text())
    assert plan['revision'] == REVISION
    process = subprocess.Popen(['git', '-C', str(OBJECTS_GIT), 'cat-file', '--batch'],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for item in plan['files']:
            process.stdin.write((item['blobId'] + '\n').encode())
            process.stdin.flush()
            header = process.stdout.readline().decode().strip().split()
            if header != [item['blobId'], 'blob', str(item['size'])]:
                raise ValueError('Bulk transfer incomplete; old downloader must remain running')
            content = process.stdout.read(item['size'])
            if process.stdout.read(1) != b'\n' or blob_hash(content) != item['blobId']:
                raise ValueError('Downloaded config failed Git integrity check')
    finally:
        process.stdin.close()
        process.stdout.close()
        process.wait(timeout=10)
    log(f'GIT_CONFIG_LOCAL_VERIFIED files={len(plan["files"])}; no network requests needed')


def install():
    import fcntl
    with (ROOT / 'tmp/sim-assets-download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = json.loads(PLAN.read_text())
        assert plan['revision'] == REVISION
        # Disable lazy network fetch during local verification/install.
        subprocess.run(['git', '-C', str(OBJECTS_GIT), 'config', 'remote.origin.promisor', 'false'], check=True)
        process = subprocess.Popen(['git', '-C', str(OBJECTS_GIT), 'cat-file', '--batch'],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        added = reused = 0
        try:
            for item in plan['files']:
                process.stdin.write((item['blobId']+'\n').encode())
                process.stdin.flush()
                header = process.stdout.readline().decode().strip().split()
                if header != [item['blobId'], 'blob', str(item['size'])]:
                    raise ValueError('Requested local blob is absent or has wrong type/size')
                content = process.stdout.read(item['size'])
                if process.stdout.read(1) != b'\n' or blob_hash(content) != item['blobId']:
                    raise ValueError('Git blob integrity verification failed')
                path = destination(item['rfilename'])
                if path.exists():
                    if blob_hash(path.read_bytes()) != item['blobId']:
                        raise ValueError('Existing config mismatch; retained for inspection')
                    reused += 1
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(prefix='.hbot-git-', dir=path.parent, delete=False) as stream:
                    temp = Path(stream.name)
                    stream.write(content)
                try:
                    # Hard-linking fails if another writer created the destination.
                    temp.chmod(0o644)
                    os.link(temp, path)
                finally:
                    temp.unlink()
                added += 1
                if added % 1000 == 0:
                    log(f'GIT_CONFIG_INSTALL added={added} reused={reused}')
        finally:
            process.stdin.close()
            process.stdout.close()
            process.wait(timeout=10)
        info = json.loads((ROOT / 'tmp/assets-info.json').read_text())
        bad = []
        for item in info['siblings']:
            p = TARGET / item['rfilename']
            if not p.is_file() or p.stat().st_size != item['size']:
                bad.append(item['rfilename'])
        report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'revision': REVISION,
                  'catalog_files': len(info['siblings']), 'catalog_bytes': sum(x['size'] for x in info['siblings']),
                  'matching_files': len(info['siblings'])-len(bad), 'missing_or_wrong_size': bad,
                  'new_configs_installed': added, 'already_completed_configs_reused': reused,
                  'config_blobs_hash_verified': len(plan['files']),
                  'verification_scope': 'All catalog files: presence and size; supplemented configs: Git blob SHA-1'}
        write_json(ROOT / 'logs/sim-assets-verification.json', report)
        if bad:
            raise RuntimeError(f'Asset snapshot still incomplete: {len(bad)} files')
        log(f'Verification: remote_files={len(info["siblings"])} bytes={report["catalog_bytes"]} missing_or_wrong_size=0')
        log('Asset download and verification completed successfully (pinned manifest sizes; supplemented config Git hashes verified)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['fetch-only', 'verify', 'install'])
    args = parser.parse_args()
    {'fetch-only': fetch_only, 'verify': verify, 'install': install}[args.action]()
