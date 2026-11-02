#!/usr/bin/env python3
"""Own one authorized two-client, three-task retest; never alter the old run."""
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess as sp
import sys
import time

ROOT = Path('/study/wangmengfan/hbot')
INSIDE = Path('/home/users/wangmengfan')
SOURCE_RUN = 'local_pi05_full_99999_parallel_20261101_191358'
SOURCE = ROOT / 'outputs/sim' / SOURCE_RUN
FROZEN = SOURCE / 'config_snapshot'
CHECKPOINT = ROOT / 'checkpoints/pi05_hbot_full/99999'
MARKER = ROOT / 'logs/full99999-supplement-20261102-start.json'
SESSION = 'hbot-full99999-supplement'
PREFIX = 'local_pi05_full_99999_supplement_'
WORKERS = 2
RETEST = ('pick_color', 'pick_spatial', 'place_a2b')


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def build_config(original, client_path):
    value = copy.deepcopy(original)
    assert value['defaults']['episode_num'] == 50 and value['defaults']['seed'] == 100000
    value['tasks'] = {k: v for k, v in value['tasks'].items() if k in RETEST}
    assert set(value['tasks']) == set(RETEST)
    assert value['tasks']['place_a2b']['shards'] == 2
    value['policy']['model_yaml'] = str(client_path)
    return value


def claim(path, status):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(status, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def check_worker_failures(path):
    if path.is_file():
        for match in re.finditer(r'\[done\][^\n]*?\brc=(-?\d+)\b', path.read_text(errors='replace')):
            if int(match.group(1)):
                raise RuntimeError('Owned worker failed: ' + match.group(0))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_stamp():
    return sorted((str(p.relative_to(CHECKPOINT)), p.stat().st_size, p.stat().st_mtime_ns)
                  for p in CHECKPOINT.rglob('*') if p.is_file())


def main():
    import yaml
    base = load_module(FROZEN / 'run-full99999-eval.py', 'owned_base')
    history = load_module(FROZEN / 'report-pi05-99999-comparison.py', 'history')
    report = load_module(Path(__file__).with_name('report-pi05-99999-supplement.py'), 'supplement_report')
    validator = ROOT / 'scripts/validate-99999-supplement.py'
    controller = str(INSIDE / 'scripts/hbot-eval-control.py')
    run = PREFIX + dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    output = ROOT / 'outputs/sim' / run
    config = ROOT / 'config' / (run + '.yaml')
    model_yaml = ROOT / 'config' / (run + '-model.yaml')
    client_yaml = ROOT / 'config' / (run + '-client.yaml')
    server_log = ROOT / 'logs' / (run + '-model.log')
    eval_log = ROOT / 'logs' / (run + '-eval.log')
    state_path = ROOT / 'logs' / (run + '-status.json')
    status = dict(run=run, output=str(output), checkpoint=str(CHECKPOINT),
                  source_run=SOURCE_RUN, started_at=base.timestamp(), state='preparing',
                  workers_per_gpu=WORKERS, tasks=list(RETEST), episodes_planned=150,
                  result_mode='150 new + 150 retained; mixed 2/3-client local evaluation')
    # All failure here is non-mutating: no claim, no model and no output directory.
    if MARKER.exists() or any((ROOT / 'outputs/sim').glob(PREFIX + '*')) or any((ROOT / 'logs').glob(PREFIX + '*-status.json')):
        raise RuntimeError('Supplement already attempted; inspect it, never resubmit automatically')
    previous = report.read(SOURCE / 'run_status.json')
    if previous.get('state') != 'failed' or not previous.get('finished_at') or previous.get('checkpoint') != str(CHECKPOINT):
        raise RuntimeError('Unexpected source run state/checkpoint')
    retained = {t: report.audit_task(SOURCE, t) for t in report.RETAINED}
    if not all(r['complete'] for r in retained.values()):
        raise RuntimeError('Retained 150 episodes failed raw audit: ' + json.dumps(retained))
    source_files = [SOURCE / 'run_status.json', SOURCE / 'run_manifest.json']
    source_files += [Path(p) for r in retained.values() for p in r['raw_files']]
    source_files += list(FROZEN.glob('*'))
    source_hashes = {str(p): sha(p) for p in source_files if p.is_file()}
    old_manifest = report.read(SOURCE / 'run_manifest.json')
    for p in (CHECKPOINT / 'params/_METADATA', CHECKPOINT / 'assets/hbot_panda_joint_delta_h15/norm_stats.json'):
        if old_manifest['sha256'].get(str(p)) != sha(p):
            raise RuntimeError('Checkpoint metadata/stats changed since source evaluation')
    stamp = checkpoint_stamp()
    source_started = dt.datetime.fromisoformat(previous['started_at']).timestamp()
    if not stamp or any(mtime / 1e9 > source_started for _, _, mtime in stamp):
        raise RuntimeError('Checkpoint changed after original run; do not combine results')
    preflight = sp.check_output(['bash', str(ROOT / 'scripts/run-pi05-server.sh'),
        '--profile', 'hbot-full', '--checkpoint', str(CHECKPOINT), '--model-yaml',
        str(FROZEN / (SOURCE_RUN + '-model.yaml')), '--gpu', '1', '--port', '8766', '--check'], text=True, timeout=180)
    if 'CPU_PREFLIGHT_PASSED' not in preflight or checkpoint_stamp() != stamp:
        raise RuntimeError('CPU check failed or checkpoint is still changing')
    for gpu in (0, 1):
        occupied = sp.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
        if occupied:
            raise RuntimeError(f'GPU {gpu} occupied; no other process stopped: {occupied}')
    initial = base.resources()
    base.check_resources(initial)
    if next(r for r in initial if r['gpu'] == 0)['memory_free_mib'] < 17408:
        raise RuntimeError('Two-worker startup needs at least 17408 MiB free on GPU0')
    if sp.check_output(['docker', 'inspect', '-f', '{{.State.Running}}', 'hbot-sim'], text=True).strip() != 'true':
        raise RuntimeError('hbot-sim is not running')
    active = json.loads(sp.check_output(['docker', 'exec', 'hbot-sim', 'python3', controller, 'status'], text=True, timeout=30))
    if active:
        raise RuntimeError('Another evaluation is active; leaving it alone')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 8766))
    for p in (output, config, model_yaml, client_yaml, server_log, eval_log, state_path):
        if p.exists():
            raise RuntimeError('Refusing to overwrite: ' + str(p))
    claim(MARKER, dict(status, tmux_session=SESSION, launcher_pid=os.getpid()))
    model = evaluation = exporter = None
    launched = False

    def save():
        base.write_json(state_path, status)
        if output.exists():
            base.write_json(output / 'run_status.json', status)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        shutil.copyfile(FROZEN / (SOURCE_RUN + '-client.yaml'), client_yaml)
        shutil.copyfile(FROZEN / (SOURCE_RUN + '-model.yaml'), model_yaml)
        original = yaml.safe_load((FROZEN / (SOURCE_RUN + '.yaml')).read_text())
        config.write_text(yaml.safe_dump(build_config(original, INSIDE / 'config' / client_yaml.name), sort_keys=False), encoding='utf-8')
        plan = json.loads(sp.check_output(['docker', 'exec', '-e', 'PYTHONDONTWRITEBYTECODE=1', 'hbot-sim',
            'python3', str(INSIDE / 'scripts' / validator.name), str(INSIDE / 'config' / config.name)], text=True, timeout=180))
        status['coverage'] = plan
        save()
        env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
        for key in ('HBOT_SIM_CPU_THREADS', 'HBOT_SIM_CPU_AFFINITY', 'HBOT_SIM_WORKERS_PER_GPU', 'HBOT_EVAL_PROFILE', 'HBOT_EVAL_DEEP_PROFILE'):
            env.pop(key, None)
        with server_log.open('x') as stream:
            model = sp.Popen(['bash', str(ROOT / 'scripts/run-pi05-server.sh'), '--profile', 'hbot-full',
                '--checkpoint', str(CHECKPOINT), '--model-yaml', str(model_yaml), '--gpu', '1', '--port', '8766'],
                stdout=stream, stderr=sp.STDOUT, env=env, start_new_session=True)
        status.update(state='loading_model', model_pid=model.pid)
        save()
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if model.poll() is not None:
                raise RuntimeError('Owned model exited during loading')
            listener = sp.check_output(['ss', '-H', '-lntp', '( sport = :8766 )'], text=True)
            if f'pid={model.pid},' in listener:
                break
            time.sleep(3)
        else:
            raise RuntimeError('Model failed to listen within 15 minutes')
        settings = dict(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', HBOT_EVAL_PROFILE='1', HBOT_EVAL_DEEP_PROFILE='0',
            HBOT_SIM_WORKERS_PER_GPU=str(WORKERS), HBOT_SIM_CPU_THREADS='8', HBOT_EVAL_NUM_RECORD='2', HBOT_EVAL_EXPORT_VIDEO='0',
            OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', MKL_NUM_THREADS='8', PXR_WORK_THREAD_LIMIT='8')
        command = ['docker', 'exec']
        for k, v in settings.items():
            command += ['-e', k + '=' + v]
        command += ['-w', str(INSIDE / 'robo_orchard_sim'), 'hbot-sim', 'taskset', '-c', '0-23,48-71',
                    'python3', controller, 'run', run, config.name]
        with eval_log.open('x') as stream:
            evaluation = sp.Popen(command, stdout=stream, stderr=sp.STDOUT, env=env, start_new_session=True)
        launched = True
        status.update(state='evaluating', evaluation_pid=evaluation.pid, evaluation_started_at=base.timestamp())
        save()
        deadline = time.monotonic() + 120
        while not output.exists():
            if evaluation.poll() is not None or model.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Owned evaluation failed during startup')
            time.sleep(1)
        archive = output / 'config_snapshot'
        archive.mkdir()
        for p in (config, model_yaml, client_yaml, Path(__file__), validator,
                  Path(__file__).with_name('report-pi05-99999-supplement.py'),
                  FROZEN / 'run-full99999-eval.py', FROZEN / 'report-pi05-99999-comparison.py', ROOT / 'scripts/hbot-eval-control.py'):
            shutil.copy2(p, archive / p.name)
        for p, name in ((server_log, 'model_server.log'), (eval_log, 'evaluation.log')):
            os.link(p, output / name)
        (output / 'cpu_preflight.log').write_text(preflight)
        base.write_json(output / 'run_manifest.json', dict(status, gpu_sim=0, gpu_model=1,
            cpu_threads_per_worker=8, cpu_affinity='0-23,48-71', recordings_per_task=2,
            settings=settings, source_sha256=source_hashes, checkpoint_stamp=stamp,
            power_limits_changed=False, initial_resources=initial,
            config_sha256={str(p): sha(p) for p in (config, model_yaml, client_yaml)}))
        report.generate(output, history=history)
        save()
        print('SUPPLEMENT_STARTED ' + str(output), flush=True)
        while evaluation.poll() is None:
            check_worker_failures(eval_log)
            if model.poll() is not None:
                raise RuntimeError('Owned model exited before evaluation completed')
            rows = base.resources()
            with (output / 'resource_samples.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(time=base.timestamp(), gpus=rows)) + '\n')
            base.check_resources(rows)
            time.sleep(15)
        status['evaluation_exit_code'] = evaluation.returncode
        check_worker_failures(eval_log)
        result = report.generate(output, history=history)
        status.update(complete_150_episode_supplement=result['supplement_complete'], complete_300_episode_combined_results=result['complete'])
        if evaluation.returncode or not result['complete']:
            raise RuntimeError('Supplement exit or raw-result integrity audit failed')
        base.terminate_owned(model)
        status.update(state='exporting_videos', model_stopped_at=base.timestamp())
        status['preview_count'] = history.prepare_previews(output, 2)
        save()
        with (output / 'video_export.log').open('x') as stream:
            exporter = sp.Popen(['docker', 'exec', '-e', 'OMP_NUM_THREADS=8', '-w', str(INSIDE / 'robo_orchard_sim'),
                'hbot-sim', 'python3', 'tools/mcap_to_video.py', str(INSIDE / 'outputs/sim' / run)],
                stdout=stream, stderr=sp.STDOUT, start_new_session=True)
        status['video_export_pid'] = exporter.pid
        save()
        while exporter.poll() is None:
            time.sleep(5)
        status['video_export_exit_code'] = exporter.returncode
        status['state'] = 'completed' if exporter.returncode == 0 else 'completed_with_video_error'
    except BaseException as exc:
        status.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        print(status['error'], flush=True)
    finally:
        if launched:
            try:
                remaining = json.loads(sp.check_output(['docker', 'exec', 'hbot-sim', 'python3', controller, 'status'], text=True, timeout=20))
                if (evaluation is not None and evaluation.poll() is None) or any(Path(p['output']).name == run for p in remaining):
                    sp.run(['docker', 'exec', 'hbot-sim', 'python3', controller, 'stop', run], timeout=60, check=True)
            except Exception as exc:
                status['cleanup_error'] = str(exc)
                status['state'] = 'failed'
        base.terminate_owned(evaluation)
        base.terminate_owned(model)
        base.terminate_owned(exporter)
        output.mkdir(parents=True, exist_ok=True)
        for p, name in ((server_log, 'model_server.log'), (eval_log, 'evaluation.log')):
            if p.exists() and not (output / name).exists():
                os.link(p, output / name)
        try:
            status['source_preserved'] = all(Path(p).is_file() and sha(Path(p)) == digest for p, digest in source_hashes.items())
            result = report.generate(output, history=history)
            status.update(complete_150_episode_supplement=result['supplement_complete'], complete_300_episode_combined_results=result['complete'])
            if not status['source_preserved'] or (status['state'].startswith('completed') and not result['complete']):
                status.update(state='failed', integrity_error='source changed or final audit failed')
        except Exception as exc:
            status.update(state='failed', report_error=str(exc))
        status['finished_at'] = base.timestamp()
        save()
        print(json.dumps(status), flush=True)
    return 0 if status['state'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
