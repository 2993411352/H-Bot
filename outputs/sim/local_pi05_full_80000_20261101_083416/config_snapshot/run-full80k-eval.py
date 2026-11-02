#!/usr/bin/env python3
"""Own one 80k model and a two-worker evaluation; preserve all other jobs."""
import datetime as dt
import hashlib
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
CHECKPOINT = ROOT / 'checkpoints/pi05_hbot_full/80000'
CONTROLLER = str(INSIDE / 'scripts/hbot-eval-control.py')


def timestamp():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def terminate_owned(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except sp.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def resources():
    raw = sp.check_output([
        'nvidia-smi', '--query-gpu=index,temperature.gpu,memory.free,memory.used,utilization.gpu,power.draw',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    rows = []
    for line in raw.splitlines():
        index, temperature, free, used, utilization, power = line.split(',')
        rows.append(dict(gpu=int(index), temperature=int(temperature),
                         memory_free_mib=int(free), memory_used_mib=int(used),
                         utilization=int(utilization), power_w=float(power)))
    return rows


def check_resources(rows):
    for row in rows:
        if row['gpu'] in (0, 1) and row['temperature'] >= 90:
            raise RuntimeError(f"SAFETY_STOP: GPU {row['gpu']} reached 90 C; stop only this owned run")
        if row['gpu'] == 0 and row['memory_free_mib'] < 768:
            raise RuntimeError('SAFETY_STOP: simulation GPU has under 768 MiB free')


def main():
    run = 'local_pi05_full_80000_parallel_' + dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    output = ROOT / 'outputs/sim' / run
    config_path = ROOT / 'config' / (run + '.yaml')
    model_yaml = ROOT / 'config' / (run + '-model.yaml')
    client_yaml = ROOT / 'config' / (run + '-client.yaml')
    server_log = ROOT / 'logs' / (run + '-model.log')
    eval_log = ROOT / 'logs' / (run + '-eval.log')
    state_path = ROOT / 'logs' / (run + '-status.json')
    original_config = ROOT / 'config/hbot-local-pi05-parallel.yaml'
    spec = importlib.util.spec_from_file_location('comparison_report', ROOT / 'scripts/report-pi05-comparison.py')
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    model = evaluation = exporter = None
    launched = False
    status = dict(run=run, output=str(output), checkpoint=str(CHECKPOINT),
                  started_at=timestamp(), state='preparing', workers_per_gpu=2)

    def save():
        write_json(state_path, status)
        if output.exists():
            write_json(output / 'run_status.json', status)

    def stopped(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stopped)
    # Check collisions outside cleanup: never write into an existing run.
    for path in (output, config_path, model_yaml, client_yaml, server_log, eval_log, state_path):
        if path.exists():
            raise RuntimeError('Refusing to reuse run path: ' + str(path))
    try:
        for gpu in (0, 1):
            occupied = sp.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid',
                                       '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
            if occupied:
                raise RuntimeError(f'GPU {gpu} occupied; no process stopped: {occupied}')
        initial = resources()
        if next(r for r in initial if r['gpu'] == 0)['memory_free_mib'] < 17408:
            raise RuntimeError('Insufficient GPU0 memory for two workers')
        check_resources(initial)
        if sp.check_output(['docker', 'inspect', '-f', '{{.State.Running}}', 'hbot-sim'], text=True).strip() != 'true':
            raise RuntimeError('hbot-sim is not running')
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 8766))
        shutil.copyfile(ROOT / 'config/local_server_pi05.yaml', client_yaml)
        shutil.copyfile(ROOT / 'config/pi05-server.yaml', model_yaml)
        client_line = 'model_yaml: /home/users/wangmengfan/config/local_server_pi05.yaml'
        text = original_config.read_text()
        if text.count(client_line) != 1:
            raise RuntimeError('Unexpected client configuration reference')
        config_path.write_text(text.replace(client_line, 'model_yaml: ' + str(INSIDE / 'config' / client_yaml.name)))
        save()
        print(json.dumps(status), flush=True)
        env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
        for key in ('HBOT_SIM_CPU_THREADS', 'HBOT_SIM_CPU_AFFINITY', 'HBOT_SIM_WORKERS_PER_GPU',
                    'HBOT_EVAL_PROFILE', 'HBOT_EVAL_DEEP_PROFILE'):
            env.pop(key, None)
        with server_log.open('x') as stream:
            model = sp.Popen([
                'bash', str(ROOT / 'scripts/run-pi05-server.sh'), '--profile', 'hbot-full',
                '--checkpoint', str(CHECKPOINT), '--model-yaml', str(model_yaml),
                '--gpu', '1', '--port', '8766'], stdout=stream, stderr=sp.STDOUT,
                env=env, start_new_session=True)
        status.update(state='loading_model', model_pid=model.pid)
        save()
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if model.poll() is not None:
                raise RuntimeError(f'Model exited during load: {model.returncode}')
            listener = sp.check_output(['ss', '-H', '-lntp', '( sport = :8766 )'], text=True)
            if f'pid={model.pid},' in listener:
                break
            time.sleep(3)
        else:
            raise RuntimeError('Model listener not ready within 15 minutes')
        command = ['docker', 'exec']
        settings = {
            'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1',
            'HBOT_EVAL_PROFILE': '1', 'HBOT_EVAL_DEEP_PROFILE': '0',
            'HBOT_SIM_WORKERS_PER_GPU': '2', 'HBOT_SIM_CPU_THREADS': '8',
            'HBOT_EVAL_NUM_RECORD': '2', 'HBOT_EVAL_EXPORT_VIDEO': '0',
            'OMP_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8',
            'MKL_NUM_THREADS': '8', 'PXR_WORK_THREAD_LIMIT': '8',
        }
        for key, value in settings.items():
            command += ['-e', key + '=' + value]
        command += ['-w', str(INSIDE / 'robo_orchard_sim'), 'hbot-sim',
                    'taskset', '-c', '0-23,48-71', 'python3', CONTROLLER,
                    'run', run, config_path.name]
        with eval_log.open('x') as stream:
            evaluation = sp.Popen(command, stdout=stream, stderr=sp.STDOUT,
                                  env=env, start_new_session=True)
        launched = True
        status.update(state='evaluating', evaluation_pid=evaluation.pid,
                      evaluation_started_at=timestamp())
        save()
        startup_deadline = time.monotonic() + 120
        while not output.exists():
            if evaluation.poll() is not None or model.poll() is not None:
                raise RuntimeError('Evaluation/model exited during startup')
            if time.monotonic() > startup_deadline:
                raise RuntimeError('Output not created within 120 seconds')
            time.sleep(1)
        archive = output / 'config_snapshot'
        archive.mkdir()
        for source in (config_path, model_yaml, client_yaml, Path(__file__),
                       ROOT / 'scripts/report-pi05-comparison.py', ROOT / 'scripts/hbot-eval-control.py'):
            shutil.copy2(source, archive / source.name)
        for source, target in ((server_log, output / 'model_server.log'),
                               (eval_log, output / 'evaluation.log')):
            os.link(source, target)
        stats = CHECKPOINT / 'assets/hbot_panda_joint_delta_h15/norm_stats.json'
        manifest = dict(status, gpu_model=1, gpu_sim=0, cpu_threads_per_worker=8,
                        cpu_affinity_sim='0-23,48-71', recordings_per_task=2,
                        power_limits_changed=False, docker_environment=settings,
                        initial_resources=initial, concurrency_caveat='Independent client RNG streams; not exact serial trajectory parity.')
        manifest['sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (config_path, model_yaml, client_yaml, stats, CHECKPOINT / 'params/_METADATA')}
        write_json(output / 'run_manifest.json', manifest)
        save()
        print('EVALUATION_STARTED ' + str(output), flush=True)
        while evaluation.poll() is None:
            if model.poll() is not None:
                raise RuntimeError('Owned model exited before evaluation finished')
            rows = resources()
            with (output / 'resource_samples.jsonl').open('a') as stream:
                stream.write(json.dumps({'time': timestamp(), 'gpus': rows}) + '\n')
            check_resources(rows)
            time.sleep(15)
        status['evaluation_exit_code'] = evaluation.returncode
        status['complete_300_episode_evaluation'] = report.generate(output)
        terminate_owned(model)
        status['model_stopped_at'] = timestamp()
        status['state'] = 'exporting_videos'
        save()
        if (output / 'summary.json').is_file():
            status['preview_count'] = report.prepare_previews(output, 2)
            with (output / 'video_export.log').open('x') as stream:
                exporter = sp.Popen(['docker', 'exec', '-e', 'OMP_NUM_THREADS=8',
                                     '-w', str(INSIDE / 'robo_orchard_sim'), 'hbot-sim',
                                     'python3', 'tools/mcap_to_video.py', str(INSIDE / 'outputs/sim' / run)],
                                    stdout=stream, stderr=sp.STDOUT, start_new_session=True)
            status['video_export_pid'] = exporter.pid
            save()
            while exporter.poll() is None:
                time.sleep(5)
            status['video_export_exit_code'] = exporter.returncode
        status['state'] = 'completed' if evaluation.returncode == 0 and status['complete_300_episode_evaluation'] else 'evaluation_error'
        if status['state'] == 'completed' and status.get('video_export_exit_code') != 0:
            status['state'] = 'completed_with_video_error'
    except BaseException as exc:
        status.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        print(status['error'], flush=True)
    finally:
        if launched and evaluation is not None and evaluation.poll() is None:
            try:
                sp.run(['docker', 'exec', 'hbot-sim', 'python3', CONTROLLER, 'stop', run],
                       timeout=60, check=False)
            except Exception as exc:
                status['cleanup_error'] = str(exc)
            terminate_owned(evaluation)
        terminate_owned(model)
        # Evaluation scores are retained even if exporting is interrupted.
        terminate_owned(exporter)
        output.mkdir(parents=True, exist_ok=True)
        for source, target in ((server_log, output / 'model_server.log'),
                               (eval_log, output / 'evaluation.log')):
            if source.exists() and not target.exists():
                os.link(source, target)
        status['finished_at'] = timestamp()
        try:
            status['complete_300_episode_evaluation'] = report.generate(output)
        except Exception as exc:
            status['report_error'] = str(exc)
        save()
        print(json.dumps(status), flush=True)
    return 0 if status['state'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
