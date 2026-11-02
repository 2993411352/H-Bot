"""One owned batch32/20k evaluation: GPU0 model+1 sim, GPU1 two sims."""
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess as sp
import sys
import time

ROOT = Path('/study/wangmengfan/hbot')
INSIDE = Path('/home/users/wangmengfan')
CHECKPOINT = ROOT / 'checkpoints/pi05_hbot_32b/20000'
PREFIX = 'local_pi05_b32_20000_'
SESSION = 'hbot-b32-20000-eval'
MARKER = ROOT / 'logs/b32-20000-start.json'
SCRIPT_NAMES = ('run-b32-20000.py', 'eval-b32-20000.py', 'control-b32-20000.py',
                'warmup-b32-20000.py', 'report-b32-20000.py')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def stamp():
    return sorted((str(p.relative_to(CHECKPOINT)), p.stat().st_size, p.stat().st_mtime_ns)
                  for p in CHECKPOINT.rglob('*') if p.is_file())


def guard(rows):
    for row in rows:
        if row['gpu'] in (0, 1):
            if row['temperature'] >= 90:
                raise RuntimeError(f"SAFETY_STOP: GPU{row['gpu']} reached 90 C")
            if row['memory_free_mib'] < 768:
                raise RuntimeError(f"SAFETY_STOP: GPU{row['gpu']} free VRAM below 768 MiB")


def startup_guard(rows, warmed=False):
    guard(rows)
    limits = {0: 10240 if warmed else 23000, 1: 17408}
    for index, minimum in limits.items():
        row = next(r for r in rows if r['gpu'] == index)
        if row['memory_free_mib'] < minimum:
            raise RuntimeError(f'GPU{index} requires {minimum} MiB free at this stage')


def claim(status):
    with MARKER.open('x', encoding='utf-8') as stream:
        json.dump(dict(status, tmux_session=SESSION, launcher_pid=os.getpid()), stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    import yaml
    base = module(ROOT / 'scripts/run-full99999-eval.py', 'base_helpers')
    report = module(ROOT / 'scripts/report-b32-20000.py', 'batch_report')
    previews = module(ROOT / 'scripts/report-pi05-99999-comparison.py', 'preview_helpers')
    controller = str(INSIDE / 'scripts/control-b32-20000.py')
    run = PREFIX + dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    output = ROOT / 'outputs/sim' / run
    config = ROOT / 'config' / (run + '.yaml')
    model_yaml = ROOT / 'config' / (run + '-model.yaml')
    client_yaml = ROOT / 'config' / (run + '-client.yaml')
    server_log = ROOT / 'logs' / (run + '-model.log')
    eval_log = ROOT / 'logs' / (run + '-eval.log')
    state_path = ROOT / 'logs' / (run + '-status.json')
    status = dict(run=run, output=str(output), checkpoint=str(CHECKPOINT),
        started_at=base.timestamp(), state='preparing', model_gpu=0,
        workers_per_gpu={'0': 1, '1': 2}, concurrency=3, episodes_planned=300)
    if MARKER.exists() or any((ROOT / 'outputs/sim').glob(PREFIX + '*')):
        raise RuntimeError('This evaluation has already been attempted; refusing duplicate start')
    before = stamp()
    if not before or any(time.time() - mtime / 1e9 < 120 for _, _, mtime in before):
        raise RuntimeError('Checkpoint missing or modified within the last 2 minutes')
    preflight = sp.check_output(['bash', str(ROOT / 'scripts/run-pi05-server.sh'),
        '--profile', 'hbot-full', '--checkpoint', str(CHECKPOINT), '--model-yaml',
        str(ROOT / 'scripts/b32-20000-model.template.yaml'), '--gpu', '0', '--port', '8766', '--check'],
        text=True, timeout=180)
    if 'CPU_PREFLIGHT_PASSED' not in preflight or stamp() != before:
        raise RuntimeError('Checkpoint structure check failed or files changed')
    for gpu in (0, 1):
        occupied = sp.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
        if occupied:
            raise RuntimeError(f'GPU{gpu} occupied; no other process stopped: {occupied}')
    initial = base.resources()
    startup_guard(initial)
    if sp.check_output(['docker', 'inspect', '-f', '{{.State.Running}}', 'hbot-sim-b32-20000'], text=True).strip() != 'true':
        raise RuntimeError('hbot-sim-b32-20000 not running')
    if json.loads(sp.check_output(['docker', 'exec', 'hbot-sim-b32-20000', 'python3', controller, 'status'], text=True, timeout=30)):
        raise RuntimeError('Other evaluation active')
    if json.loads(sp.check_output(['docker', 'exec', 'hbot-sim', 'python3',
        str(INSIDE / 'scripts/hbot-eval-control.py'), 'status'], text=True, timeout=30)):
        raise RuntimeError('Other evaluation active in the original container')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 8766))
    for p in (output, config, model_yaml, client_yaml, server_log, eval_log, state_path):
        if p.exists():
            raise RuntimeError('Refusing overwrite: ' + str(p))
    claim(status)
    model = evaluation = exporter = warmup = None
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
        shutil.copyfile(ROOT / 'scripts/b32-20000-client.template.yaml', client_yaml)
        shutil.copyfile(ROOT / 'scripts/b32-20000-model.template.yaml', model_yaml)
        value = yaml.safe_load((ROOT / 'scripts/b32-20000-eval.template.yaml').read_text())
        value['policy']['model_yaml'] = str(INSIDE / 'config' / client_yaml.name)
        config.write_text(yaml.safe_dump(value, sort_keys=False), encoding='utf-8')
        plan = json.loads(sp.check_output(['docker', 'exec', '-e', 'PYTHONDONTWRITEBYTECODE=1', 'hbot-sim-b32-20000',
            'python3', str(INSIDE / 'scripts/eval-b32-20000.py'), '--validate-config',
            str(INSIDE / 'config' / config.name)], text=True, timeout=180))
        status['coverage'] = plan
        save()
        env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
        for key in ('HBOT_SIM_CPU_THREADS', 'HBOT_SIM_CPU_AFFINITY', 'HBOT_SIM_WORKERS_PER_GPU', 'HBOT_EVAL_PROFILE', 'HBOT_EVAL_DEEP_PROFILE'):
            env.pop(key, None)
        with server_log.open('x') as stream:
            model = sp.Popen(['bash', str(ROOT / 'scripts/run-pi05-server.sh'), '--profile', 'hbot-full',
                '--checkpoint', str(CHECKPOINT), '--model-yaml', str(model_yaml), '--gpu', '0', '--port', '8766'],
                stdout=stream, stderr=sp.STDOUT, env=env, start_new_session=True)
        status.update(state='loading_model', model_pid=model.pid)
        save()
        deadline = time.monotonic() + 900
        while True:
            if model.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Model exited or timed out before listening')
            guard(base.resources())
            listener = sp.check_output(['ss', '-H', '-lntp', '( sport = :8766 )'], text=True)
            if f'pid={model.pid},' in listener:
                break
            time.sleep(3)
        status.update(state='warming_model')
        save()
        warmup_log = ROOT / 'logs' / (run + '-warmup.log')
        with warmup_log.open('x') as stream:
            warmup = sp.Popen([str(ROOT / 'openpi/.venv/bin/python'), str(ROOT / 'scripts/warmup-b32-20000.py')],
                stdout=stream, stderr=sp.STDOUT, env=env, start_new_session=True)
        deadline = time.monotonic() + 600
        while warmup.poll() is None:
            guard(base.resources())
            if model.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Model/warmup failed or timed out')
            time.sleep(3)
        if warmup.returncode or 'WARMUP_PASSED' not in warmup_log.read_text():
            raise RuntimeError('Warmup failed: ' + str(warmup_log))
        warmed = base.resources()
        startup_guard(warmed, warmed=True)
        settings = dict(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', CUDA_DEVICE_ORDER='PCI_BUS_ID',
            HBOT_EVAL_PROFILE='1', HBOT_EVAL_DEEP_PROFILE='0', HBOT_SIM_WORKERS_PER_GPU='1',
            HBOT_SIM_CPU_THREADS='8', HBOT_EVAL_NUM_RECORD='2', HBOT_EVAL_EXPORT_VIDEO='0',
            OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', MKL_NUM_THREADS='8', PXR_WORK_THREAD_LIMIT='8')
        command = ['docker', 'exec']
        for k, v in settings.items():
            command += ['-e', k + '=' + v]
        command += ['-w', str(INSIDE / 'robo_orchard_sim'), 'hbot-sim-b32-20000', 'taskset', '-c', '0-23,48-71',
                    'python3', controller, 'run', run, config.name]
        with eval_log.open('x') as stream:
            evaluation = sp.Popen(command, stdout=stream, stderr=sp.STDOUT, env=env, start_new_session=True)
        launched = True
        status.update(state='evaluating', evaluation_pid=evaluation.pid, evaluation_started_at=base.timestamp())
        save()
        deadline = time.monotonic() + 120
        while not output.exists():
            if evaluation.poll() is not None or model.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Evaluation failed during startup')
            time.sleep(1)
        archive = output / 'config_snapshot'
        archive.mkdir()
        for p in [config, model_yaml, client_yaml] + [ROOT / 'scripts' / n for n in SCRIPT_NAMES] + [
            ROOT / 'scripts/run-full99999-eval.py', ROOT / 'scripts/report-pi05-99999-comparison.py',
            ROOT / 'scripts/report-pi05-99999-supplement.py']:
            shutil.copy2(p, archive / p.name)
        for p, name in ((server_log, 'model_server.log'), (eval_log, 'evaluation.log'), (warmup_log, 'warmup.log')):
            os.link(p, output / name)
        (output / 'cpu_preflight.log').write_text(preflight)
        base.write_json(output / 'run_manifest.json', dict(status, cpu_threads_per_worker=8,
            cpu_affinity='0-23,48-71', recordings_per_task=2, settings=settings,
            initial_resources=initial, warmed_resources=warmed, checkpoint_stamp=before,
            power_limits_changed=False, inference_warmup_separate_session=True))
        report.generate(output)
        save()
        print('EVALUATION_STARTED ' + str(output), flush=True)
        while evaluation.poll() is None:
            base.check_worker_failures(eval_log)
            if model.poll() is not None:
                raise RuntimeError('Owned model exited before evaluation completed')
            rows = base.resources()
            with (output / 'resource_samples.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(time=base.timestamp(), gpus=rows)) + '\n')
            guard(rows)
            time.sleep(10)
        status['evaluation_exit_code'] = evaluation.returncode
        base.check_worker_failures(eval_log)
        status['complete_300_episode_evaluation'] = report.generate(output)
        if evaluation.returncode or not status['complete_300_episode_evaluation']:
            raise RuntimeError('Evaluation exit/raw audit failed')
        base.terminate_owned(model)
        status.update(state='exporting_videos', model_stopped_at=base.timestamp())
        status['preview_count'] = previews.prepare_previews(output, 2)
        save()
        with (output / 'video_export.log').open('x') as stream:
            exporter = sp.Popen(['docker', 'exec', '-e', 'OMP_NUM_THREADS=8', '-w', str(INSIDE / 'robo_orchard_sim'),
                'hbot-sim-b32-20000', 'python3', 'tools/mcap_to_video.py', str(INSIDE / 'outputs/sim' / run)],
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
                remaining = json.loads(sp.check_output(['docker', 'exec', 'hbot-sim-b32-20000', 'python3', controller, 'status'], text=True, timeout=20))
                if (evaluation is not None and evaluation.poll() is None) or any(Path(p['output']).name == run for p in remaining):
                    sp.run(['docker', 'exec', 'hbot-sim-b32-20000', 'python3', controller, 'stop', run], timeout=60, check=True)
            except Exception as exc:
                status.update(cleanup_error=str(exc), state='failed')
        for process in (evaluation, warmup, model, exporter):
            base.terminate_owned(process)
        output.mkdir(parents=True, exist_ok=True)
        for p, name in ((server_log, 'model_server.log'), (eval_log, 'evaluation.log')):
            if p.exists() and not (output / name).exists():
                os.link(p, output / name)
        try:
            status['complete_300_episode_evaluation'] = report.generate(output)
            status['checkpoint_unchanged'] = stamp() == before
            if not status['checkpoint_unchanged']:
                status.update(state='failed', integrity_error='checkpoint changed')
        except Exception as exc:
            status.update(state='failed', report_error=str(exc))
        status['finished_at'] = base.timestamp()
        save()
        print(json.dumps(status), flush=True)
    return 0 if status['state'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
