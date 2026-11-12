#!/usr/bin/env python3
"""Host CLI; only Python's standard library required. No implicit GPU job."""
import argparse
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'runtime'))
from common import hardware, require, seed_for, sha, TASKS


def read_gpu(index):
    result = subprocess.check_output(['nvidia-smi', f'--id={index}',
        '--query-gpu=name,uuid,compute_cap,memory.total,memory.free,driver_version',
        '--format=csv,noheader,nounits'], text=True)
    rows = list(csv.reader(io.StringIO(result)))
    require(len(rows) == 1, 'Select exactly one physical GPU')
    name, uuid, arch, total, free, driver = [s.strip() for s in rows[0]]
    return dict(name=name, uuid=uuid, arch=arch, total=int(total), free=int(free),
                driver=driver, memory_profile=hardware(name, arch, int(total)))


def ensure_idle(gpu):
    procs = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True)
    require(not any(gpu['uuid'] in line for line in procs.splitlines()),
            'Selected GPU is running another compute task; nothing was stopped')
    require(gpu['free'] >= gpu['total'] - 1500, 'Selected GPU is not sufficiently free')


def container(cfg, image, work, *, gpu=None, asset_write=False, name=None):
    work.mkdir(parents=True, exist_ok=True)
    (work/'home').mkdir(exist_ok=True)
    cmd = ['docker', 'run', '--rm', '--init', '--shm-size=16g',
           '--user', f'{os.getuid()}:{os.getgid()}',
           '--mount', f'type=bind,src={work},dst=/work',
           '--mount', f'type=bind,src={cfg["assets"]},dst=/assets' + ('' if asset_write else ',readonly'),
           '-e', 'HOME=/work/home', '-e', 'ACCEPT_EULA=Y', '-e', 'PRIVACY_CONSENT=N',
           '-e', 'HF_HOME=/work/home/huggingface',
           '-e', 'XDG_CACHE_HOME=/work/home/.cache',
           '-e', 'TORCH_EXTENSIONS_DIR=/work/home/torch_extensions']
    # Upstream acquires an adjacent .lock even when reading the asset index.
    # Only this tiny index directory is writable; meshes/textures stay read-only.
    if not asset_write:
        index = Path(cfg['assets'])/'OBJECTS_v2/asset_indexes/asset_index.v7.parquet'
        require(index.is_file(), 'Download assets before running this command')
        index_dir = work/'asset_indexes'
        index_dir.mkdir(exist_ok=True)
        copy = index_dir/index.name
        if copy.exists():
            require(sha(copy) == sha(index), 'Asset index changed for existing run')
        else:
            shutil.copy2(index, copy)
        cmd += ['--mount', f'type=bind,src={index_dir},dst=/assets/OBJECTS_v2/asset_indexes']
    if name:
        cmd += ['--name', name, '--label', 'hbot.component=synthetic-collector']
    if gpu:
        cmd += ['--gpus', f'device={gpu["uuid"]}',
                '-e', 'NVIDIA_VISIBLE_DEVICES='+gpu['uuid'],
                '-e', 'HBOT_MEMORY_PROFILE='+gpu['memory_profile']]
    else:
        cmd += ['-e', 'CUDA_VISIBLE_DEVICES=', '-e', 'NVIDIA_VISIBLE_DEVICES=void']
    if cfg.get('reference') and Path(cfg['reference']).is_dir():
        cmd += ['--mount', f'type=bind,src={cfg["reference"]},dst=/reference,readonly']
    return cmd + [image]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=HERE/'local.json')
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('doctor')
    sub.add_parser('build')
    sub.add_parser('assets-check')
    down = sub.add_parser('download-assets')
    down.add_argument('--revision', default='instructmove_v2')
    run = sub.add_parser('collect')
    run.add_argument('--worker', type=int, required=True,
                     help='Globally unique assignment, never reused for another run')
    run.add_argument('--quota', type=int, default=400, help='Accepted quota PER task PER worker')
    run.add_argument('--max-attempts', type=int, default=2000, help='Attempt limit per task')
    run.add_argument('--resume', action='store_true')
    pause = sub.add_parser('pause')
    pause.add_argument('--worker', type=int, required=True)
    status = sub.add_parser('status')
    status.add_argument('--worker', type=int, required=True)
    export = sub.add_parser('export')
    export.add_argument('--worker', type=int, required=True)
    export.add_argument('--name', required=True)
    merge = sub.add_parser('merge-check')
    merge.add_argument('--workers', type=int, nargs='+', required=True)
    merge.add_argument('--name', required=True)
    args = p.parse_args()
    require(platform.system() == 'Linux' and platform.machine() == 'x86_64' and
            'microsoft' not in platform.release().lower(),
            'Use native Linux x86_64. Windows users SSH to a Linux collector.')
    cfg = json.loads(args.config.read_text())
    for key in ('assets', 'work', 'reference'):
        cfg[key] = str(Path(cfg[key]).expanduser().resolve())
        require(',' not in cfg[key], 'Docker mount paths must not contain commas')
    base = Path(cfg['work'])
    if args.cmd in ('pause', 'status'):
        path = base/f'worker_{args.worker:04d}'
        require(path.is_dir(), 'No such worker')
        if args.cmd == 'pause':
            (path/'PAUSE').touch()
            print('Pause requested. Current episode finishes before exit; no other job stopped.')
        else:
            s = json.loads((path/'status.json').read_text())
            print(json.dumps(dict(status=s['status'], accepted={t: sum(r['task']==t for r in s['accepted'])
                for t in TASKS}, attempts=len(s['batches']), error=s.get('error')), indent=2))
        return
    gpu = read_gpu(cfg['gpu'])
    image = cfg['image_prefix'] + ':sm' + gpu['arch'].replace('.', '')
    subprocess.run(['docker', 'info', '--format', '{{.ServerVersion}}'], check=True)
    print(json.dumps(gpu, indent=2), flush=True)
    if args.cmd == 'doctor':
        print('Detection only: no simulator launched. 16 GB uses compact IK; memory fit is not yet measured.')
        return
    require(cfg.get('accept_eula') is True, 'Review NVIDIA EULA, then set accept_eula=true in your local.json')
    Path(cfg['assets']).mkdir(parents=True, exist_ok=True)
    if args.cmd == 'build':
        subprocess.run(['docker', 'build', '--build-arg', 'CUDA_ARCH='+gpu['arch'],
                        '--tag', image, str(HERE)], check=True)
        return
    actual = subprocess.check_output(['docker', 'run', '--rm', '--entrypoint', 'cat', image,
                                     '/opt/hbot/build.json'], text=True)
    require(json.loads(actual)['architecture'] == gpu['arch'], 'Image architecture mismatch; rebuild')
    if args.cmd == 'download-assets':
        # A public dataset: token is not forwarded to the container.
        cmd = container(cfg, image, base/'asset_download', asset_write=True)
        subprocess.run(cmd + ['/opt/hbot/runtime/download_assets.py', args.revision], check=True)
    elif args.cmd == 'assets-check':
        subprocess.run(container(cfg, image, base/'asset_check') +
                       ['/opt/hbot/runtime/inventory.py'], check=True)
    elif args.cmd == 'collect':
        reference = Path(cfg['reference'])
        require((reference/'reference.json').is_file(), 'Obtain the shared coordinator reference first')
        expected = json.loads((reference/'reference.json').read_text())['sha256']
        require(sha(reference/'registry.sqlite3') == expected, 'Shared reference checksum mismatch')
        seed_for(args.worker, TASKS[0], 0)
        # A second terminal must not pass the same free-GPU check concurrently.
        gpu_lock = open(Path(tempfile.gettempdir()) / f'hbot-collector-{os.getuid()}-{gpu["uuid"]}.lock', 'a+')
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ensure_idle(gpu)
        work = base/f'worker_{args.worker:04d}'
        if (work/'plan.json').exists():
            require(args.resume, 'Worker ID already used. Use --resume only for a clean pause.')
        else:
            require(not args.resume and not work.exists(), 'Worker directory already exists; inspect first')
        if args.resume:
            require(not (work/'PAUSE').exists(), 'Remove this worker PAUSE file only when you want to resume')
        subprocess.run(container(cfg, image, work, gpu=gpu,
                        name=f'hbot-synthesis-worker-{args.worker}') +
                       ['/opt/hbot/runtime/worker.py', '--worker', str(args.worker),
                        '--quota', str(args.quota), '--max-attempts', str(args.max_attempts)] +
                       (['--resume'] if args.resume else []), check=True)
        gpu_lock.close()
    elif args.cmd == 'export':
        require(re.fullmatch(r'[A-Za-z0-9_-]+', args.name), 'Use a simple export name')
        work = base/f'worker_{args.worker:04d}'
        require((work/'manifest.json').is_file(), 'No accepted manifest')
        subprocess.run(container(cfg, image, work) +
                       ['/opt/hbot/runtime/export.py', '--name', args.name], check=True)
    elif args.cmd == 'merge-check':
        require(re.fullmatch(r'[A-Za-z0-9_-]+', args.name), 'Use a simple report name')
        subprocess.run(container(cfg, image, base) +
                       ['/opt/hbot/runtime/merge.py', '--name', args.name,
                        '--workers', *map(str, args.workers)], check=True)


if __name__ == '__main__':
    main()
