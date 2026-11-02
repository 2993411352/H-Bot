#!/usr/bin/env python3
"""Pinned, resumable HTTP downloads; two jobs share one bandwidth budget.

Only writes the two new destinations and its own logs/lock. Does not import
torch/JAX, start training, alter the proxy, or interact with the asset job.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import threading
import time
from urllib.parse import quote
import zipfile

import google_crc32c
import requests


ROOT = Path('/study/wangmengfan/hbot')
HF_REPO = 'HorizonRobotics/InstructMove-Dataset'
HF_REVISION = '44e8964b83fbcf7575f66dddbfc36c3a8d06ec06'
HF_PREFIX = 'horizon_isaac/lerobot/'
GCS_PREFIX = 'checkpoints/pi05_base/'
PRINT_LOCK = threading.Lock()


def now():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    with PRINT_LOCK:
        print(f'[{now()}] {message}', flush=True)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def safe_path(root, relative):
    relative = PurePosixPath(relative)
    if relative.is_absolute() or '..' in relative.parts or '\\' in str(relative):
        raise ValueError('Unsafe relative path')
    path = root / str(relative)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Path escapes destination')
    return path


def get_json(url, **kwargs):
    for attempt in range(1, 13):
        try:
            response = requests.get(url, timeout=(10, 60), **kwargs)
            response.raise_for_status()
            return response.json(), response.links
        except (requests.RequestException, ValueError) as exc:
            # Never log proxy credentials or CDN signed URLs.
            log(f'METADATA_RETRY attempt={attempt} error={type(exc).__name__}')
            if attempt == 12:
                raise RuntimeError('Metadata retrieval failed') from None
            time.sleep(30)


def build_plan():
    data_files = []
    url = f'https://huggingface.co/api/datasets/{HF_REPO}/tree/{HF_REVISION}/{HF_PREFIX.rstrip("/")}?limit=100'
    while url:
        items, links = get_json(url)
        for item in items:
            if item['type'] != 'file' or not item['path'].endswith('.zip'):
                continue
            relative = item['path'].removeprefix(HF_PREFIX)
            if '/' in relative or not item.get('lfs', {}).get('oid'):
                raise ValueError('Unexpected dataset archive metadata')
            data_files.append(dict(path=relative, size=item['size'], sha256=item['lfs']['oid'],
                url=f'https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/{item["path"]}'))
        url = links.get('next', {}).get('url')
    if len(data_files) != 6:
        raise ValueError('Expected exactly six LeRobot archives')
    model_files = []
    params = {'prefix': GCS_PREFIX, 'maxResults': 1000}
    while True:
        data, _ = get_json('https://storage.googleapis.com/storage/v1/b/openpi-assets/o', params=params)
        for item in data.get('items', []):
            if item['name'].endswith('/'):
                continue
            relative = item['name'].removeprefix(GCS_PREFIX)
            if not item.get('crc32c'):
                raise ValueError('Missing GCS checksum')
            model_files.append(dict(path=relative, size=int(item['size']), crc32c=item['crc32c'],
                generation=item['generation'],
                url='https://storage.googleapis.com/download/storage/v1/b/openpi-assets/o/'
                    + quote(item['name'], safe='')
                    + '?alt=media&generation=' + item['generation']))
        if not data.get('nextPageToken'):
            break
        params['pageToken'] = data['nextPageToken']
    if not any(x['path'] == 'params/_METADATA' for x in model_files):
        raise ValueError('Model checkpoint metadata missing')
    return {'created_at': now(), 'jobs': [
        dict(name='lerobot', destination=str(ROOT / 'datasets/InstructMove_hbot/lerobot'),
             source=f'{HF_REPO}@{HF_REVISION}', files=sorted(data_files, key=lambda x: x['size'])),
        dict(name='pi05', destination=str(ROOT / 'models/pi05_base'),
             source='gs://openpi-assets/checkpoints/pi05_base',
             files=sorted(model_files, key=lambda x: (x['size'], x['path']))),
    ]}


class RateLimiter:
    def __init__(self, rate):
        self.rate = rate
        self.next_slot = 0.0
        self.lock = threading.Lock()

    def wait(self, count):
        with self.lock:
            start = max(time.monotonic(), self.next_slot)
            self.next_slot = start + count / self.rate
        time.sleep(max(0, start - time.monotonic()))


def verify(path, item):
    if not path.is_file() or path.stat().st_size != item['size']:
        return False
    digest = hashlib.sha256() if 'sha256' in item else google_crc32c.Checksum()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    actual = digest.hexdigest() if 'sha256' in item else base64.b64encode(digest.digest()).decode()
    return actual == item.get('sha256', item.get('crc32c'))


class State:
    def __init__(self, path, plan, rate):
        self.path = path
        self.lock = threading.Lock()
        self.data = {'started_at': now(), 'rate_bytes_per_second': rate, 'jobs': {
            j['name']: {'status': 'pending', 'total_bytes': sum(f['size'] for f in j['files']),
                        'verified_bytes': 0, 'verified_files': 0, 'total_files': len(j['files']),
                        'destination': j['destination']} for j in plan['jobs']}}

    def update(self, job, **fields):
        with self.lock:
            self.data['jobs'][job].update(fields)
            self.data['updated_at'] = now()
            write_json(self.path, self.data)


def download_file(item, root, limiter, progress):
    path = safe_path(root, item['path'])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if verify(path, item):
            return path
        raise RuntimeError('Existing final file failed verification; preserved for inspection')
    part = path.with_name(path.name + '.part')
    for attempt in range(1, 61):
        offset = part.stat().st_size if part.exists() else 0
        if offset > item['size']:
            raise RuntimeError('Oversized partial retained for inspection')
        try:
            if offset < item['size']:
                headers = {'Accept-Encoding': 'identity', 'Cache-Control': 'no-cache'}
                if offset:
                    headers['Range'] = f'bytes={offset}-'
                with requests.get(item['url'], headers=headers, stream=True, timeout=(10, 90)) as response:
                    response.raise_for_status()
                    if offset or response.status_code == 206:
                        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                        if response.status_code != 206 or not match or int(match[1]) != offset or int(match[3]) != item['size']:
                            raise RuntimeError('Range response mismatch; partial file left unchanged')
                    elif response.status_code != 200:
                        raise RuntimeError('Unexpected download status')
                    with part.open('ab') as stream:
                        last_report = 0.0
                        for block in response.iter_content(chunk_size=64 * 1024):
                            if not block:
                                continue
                            if offset + len(block) > item['size']:
                                raise RuntimeError('Response exceeds expected size')
                            limiter.wait(len(block))
                            stream.write(block)
                            offset += len(block)
                            if time.monotonic() - last_report >= 30:
                                stream.flush()
                                progress(offset, attempt)
                                last_report = time.monotonic()
            elif not part.exists():
                part.touch()
            if not verify(part, item):
                if part.stat().st_size == item['size']:
                    retained = part.with_name(part.name + f'.checksum-failed-{time.time_ns()}')
                    part.rename(retained)
                    log(f'CHECKSUM_FAILED retained={retained}')
                raise RuntimeError('Incomplete or checksum-mismatched file')
            if path.exists():
                raise RuntimeError('Final path appeared during download; preserved both files')
            part.rename(path)
            return path
        except (requests.RequestException, RuntimeError) as exc:
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            reason = str(exc) if isinstance(exc, RuntimeError) else f'HTTP={status}'
            log(f'RETRY file={item["path"]} attempt={attempt} error={type(exc).__name__} reason={reason}')
            progress(part.stat().st_size if part.exists() else 0, attempt, retry=True)
            if attempt == 60:
                raise RuntimeError('Download retries exhausted; partial retained') from None
            time.sleep(min(60, 5 * attempt))


def inspect_archive(path):
    # Inspect metadata only; leave archives intact and do not extract untrusted paths.
    with zipfile.ZipFile(path) as archive:
        infos = [i for i in archive.infolist() if i.filename.endswith('/meta/info.json') or i.filename == 'meta/info.json']
        result = {'archive': path.name, 'uncompressed_bytes': sum(x.file_size for x in archive.infolist())}
        result['datasets'] = []
        for member in infos:
            if member.file_size > 1024 * 1024:
                raise ValueError('Unexpectedly large dataset metadata')
            info = json.loads(archive.read(member))
            result['datasets'].append({k: info.get(k) for k in
                ('codebase_version', 'robot_type', 'total_episodes', 'total_frames', 'fps')})
        return result


def run_job(job, limiter, state):
    name, root = job['name'], Path(job['destination'])
    done, verified, archives = 0, [], []
    try:
        for item in job['files']:
            def progress(offset, attempt, retry=False):
                state.update(name, status='retrying' if retry else 'downloading', current_file=item['path'],
                    current_bytes=offset, current_size=item['size'], attempt=attempt,
                    verified_bytes=done, verified_files=len(verified))
                log(f'PROGRESS job={name} file={item["path"]} bytes={offset}/{item["size"]} verified_bytes={done}')
            progress(0, 0)
            path = download_file(item, root, limiter, progress)
            verified.append({k: v for k, v in item.items() if k != 'url'})
            done += item['size']
            if name == 'lerobot':
                archives.append(inspect_archive(path))
            log(f'FILE_VERIFIED job={name} file={item["path"]} bytes={item["size"]}')
            state.update(name, status='downloading', verified_bytes=done, verified_files=len(verified), current_bytes=0)
        report = dict(completed_at=now(), source=job['source'], destination=str(root),
                      total_bytes=done, files=verified, archives=archives)
        write_json(ROOT / f'logs/{name}-download-verified.json', report)
        state.update(name, status='complete', completed_at=now(), current_file=None, current_bytes=0)
        log(f'JOB_VERIFIED job={name} files={len(verified)} bytes={done}')
    except Exception as exc:
        state.update(name, status='failed', error_type=type(exc).__name__)
        log(f'JOB_FAILED job={name} error={type(exc).__name__}; existing files retained')
        raise RuntimeError(f'{name} download failed; see status and retained files') from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--rate-mib', type=float, default=2.0)
    args = parser.parse_args()
    if args.rate_mib <= 0:
        parser.error('--rate-mib must be positive')
    if args.plan_only:
        plan = build_plan()
        for job in plan['jobs']:
            print(json.dumps({k: v for k, v in job.items() if k != 'files'} | {
                'files': len(job['files']), 'bytes': sum(f['size'] for f in job['files'])}))
        return
    import fcntl
    lock_path = ROOT / 'tmp/pi05-lerobot-download.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = ROOT / 'config/pi05-lerobot-download-manifest.json'
        plan = json.loads(manifest.read_text()) if manifest.exists() else build_plan()
        if not manifest.exists():
            write_json(manifest, plan)
        rate = int(args.rate_mib * 1024 * 1024)
        limiter = RateLimiter(rate)
        state = State(ROOT / 'logs/pi05-lerobot-download-status.json', plan, rate)
        log(f'START jobs=2 shared_limit_mib={args.rate_mib} total_bytes={sum(f["size"] for j in plan["jobs"] for f in j["files"])}')
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_job, j, limiter, state) for j in plan['jobs']]
            for future in futures:
                future.result()
        log('ALL_DOWNLOADS_VERIFIED')


if __name__ == '__main__':
    main()
