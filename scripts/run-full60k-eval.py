#!/usr/bin/env python3
"""Own one 60k model and one unchanged serial evaluation; archive one folder."""
import datetime as dt
import hashlib
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
CHECKPOINT = ROOT / 'checkpoints/pi05_hbot_full/60000'
REFERENCE = ROOT / 'outputs/sim/local_pi05_full_30000_20261030_170853_3664253_7601'
CONTROLLER = str(INSIDE / 'scripts/hbot-eval-control.py')


def timestamp():
    return dt.datetime.now().astimezone().isoformat()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def terminate_owned(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except sp.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def main():
    run = 'local_pi05_full_60000_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    output = ROOT / 'outputs/sim' / run
    config_name = run + '.yaml'
    snapshot_config = ROOT / 'config' / config_name
    model = evaluation = None
    launched = False
    status = {'run': run, 'output': str(output), 'checkpoint': str(CHECKPOINT),
              'started_at': timestamp(), 'state': 'preparing'}
    if output.exists() or snapshot_config.exists():
        raise RuntimeError('Refusing to reuse run paths')
    # Freeze values; change only the client-YAML path to its identical snapshot.
    original_config = ROOT / 'config/hbot-local-pi05.yaml'
    client_yaml = ROOT / 'config' / (run + '-client.yaml')
    shutil.copyfile(ROOT / 'config/local_server_pi05.yaml', client_yaml)
    original_text = original_config.read_text()
    client_line = 'model_yaml: /home/users/wangmengfan/config/local_server_pi05.yaml'
    if original_text.count(client_line) != 1:
        raise RuntimeError('Unexpected client configuration reference')
    snapshot_config.write_text(original_text.replace(
        client_line, 'model_yaml: ' + str(INSIDE / 'config' / client_yaml.name)
    ))
    model_yaml = ROOT / 'config' / (run + '-model.yaml')
    shutil.copyfile(ROOT / 'config/pi05-server.yaml', model_yaml)
    server_log = ROOT / 'logs' / (run + '-model.log')
    eval_log = ROOT / 'logs' / (run + '-eval.log')
    state_path = ROOT / 'logs' / (run + '-status.json')
    write_json(state_path, status)
    print(json.dumps(status), flush=True)
    env = dict(os.environ)
    for key in ('HBOT_SIM_CPU_THREADS', 'HBOT_SIM_CPU_AFFINITY',
                'HBOT_SIM_WORKERS_PER_GPU', 'HBOT_EVAL_PROFILE',
                'HBOT_EVAL_DEEP_PROFILE'):
        env.pop(key, None)

    def stopped(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stopped)
    try:
        gpu0 = sp.check_output([
            'nvidia-smi', '-i', '0', '--query-compute-apps=pid',
            '--format=csv,noheader,nounits'
        ], text=True).strip()
        if gpu0:
            raise RuntimeError('GPU0 is occupied; no evaluation started: ' + gpu0)
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 8766))
        with server_log.open('x') as stream:
            model = sp.Popen([
                'bash', str(ROOT / 'scripts/run-pi05-server.sh'),
                '--profile', 'hbot-full', '--checkpoint', str(CHECKPOINT),
                '--model-yaml', str(model_yaml), '--gpu', '1', '--port', '8766',
            ], stdout=stream, stderr=sp.STDOUT, env=env, start_new_session=True)
        status.update(state='loading_model', model_pid=model.pid)
        write_json(state_path, status)
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if model.poll() is not None:
                raise RuntimeError(f'Model exited during load: {model.returncode}')
            listener = sp.check_output(
                ['ss', '-H', '-lntp', '( sport = :8766 )'], text=True
            )
            if f'pid={model.pid},' in listener:
                break
            time.sleep(3)
        else:
            raise RuntimeError('Model listener was not ready within 15 minutes')
        with eval_log.open('x') as stream:
            evaluation = sp.Popen([
                'docker', 'exec', '-e', 'HBOT_EVAL_PROFILE=0',
                '-e', 'HBOT_SIM_WORKERS_PER_GPU=1', '-e', 'HBOT_EVAL_NUM_RECORD=2',
                '-w', str(INSIDE / 'robo_orchard_sim'), 'hbot-sim',
                'python3', CONTROLLER, 'run', run, config_name,
            ], stdout=stream, stderr=sp.STDOUT, env=env, start_new_session=True)
        launched = True
        status.update(state='evaluating', evaluation_pid=evaluation.pid,
                      evaluation_started_at=timestamp())
        write_json(state_path, status)
        startup_deadline = time.monotonic() + 120
        while not output.exists():
            if evaluation.poll() is not None:
                raise RuntimeError(f'Evaluation failed to start: {evaluation.returncode}')
            if model.poll() is not None:
                raise RuntimeError('Model exited while evaluation was starting')
            if time.monotonic() > startup_deadline:
                raise RuntimeError('Evaluation output was not created within 120 seconds')
            time.sleep(1)
        archive = output / 'config_snapshot'
        archive.mkdir()
        for source in (snapshot_config, model_yaml, client_yaml):
            shutil.copy2(source, archive / source.name)
        os.link(server_log, output / 'model_server.log')
        os.link(eval_log, output / 'evaluation.log')
        os.link(state_path, output / 'initial_status.json')
        shutil.copy2(REFERENCE / 'summary.json', output / 'reference_30000_summary.json')
        stats = CHECKPOINT / 'assets/hbot_panda_joint_delta_h15/norm_stats.json'
        manifest = {
            **status, 'reference': str(REFERENCE), 'gpu_model': 1, 'gpu_sim': 0,
            'workers_per_gpu': 1, 'record_per_task': 2,
            'original_eval_config_sha256': hashlib.sha256(original_config.read_bytes()).hexdigest(),
            'eval_config_sha256': hashlib.sha256(snapshot_config.read_bytes()).hexdigest(),
            'model_config_sha256': hashlib.sha256(model_yaml.read_bytes()).hexdigest(),
            'normalization_sha256': hashlib.sha256(stats.read_bytes()).hexdigest(),
            'checkpoint_metadata_sha256': hashlib.sha256((CHECKPOINT / 'params/_METADATA').read_bytes()).hexdigest(),
        }
        write_json(output / 'run_manifest.json', manifest)
        write_json(output / 'run_status.json', status)
        print('EVALUATION_STARTED ' + str(output), flush=True)
        while evaluation.poll() is None:
            if model.poll() is not None:
                raise RuntimeError('Owned model service exited before evaluation finished')
            time.sleep(5)
        status['evaluation_exit_code'] = evaluation.returncode
        status['state'] = 'completed' if evaluation.returncode == 0 else 'evaluation_error'
    except BaseException as exc:
        status.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        print(status['error'], flush=True)
    finally:
        # Only the exact evaluation launched by this script may be stopped.
        if launched and evaluation is not None and evaluation.poll() is None:
            try:
                sp.run(['docker', 'exec', 'hbot-sim', 'python3', CONTROLLER,
                        'stop', run], timeout=60, check=False)
            except Exception as exc:
                status['cleanup_error'] = str(exc)
            terminate_owned(evaluation)
        terminate_owned(model)
        output.mkdir(parents=True, exist_ok=True)
        for source, destination in ((server_log, output / 'model_server.log'),
                                    (eval_log, output / 'evaluation.log')):
            if source.exists() and not destination.exists():
                os.link(source, destination)
        status['finished_at'] = timestamp()
        try:
            from importlib.util import spec_from_file_location, module_from_spec
            spec = spec_from_file_location('eval_report', ROOT / 'scripts/report-pi05-eval.py')
            report = module_from_spec(spec)
            spec.loader.exec_module(report)
            complete = report.generate(output, REFERENCE)
            status['complete_300_episode_evaluation'] = complete
            if status['state'] == 'completed' and not complete:
                status['state'] = 'incomplete_results'
        except Exception as exc:
            status['report_error'] = str(exc)
        write_json(state_path, status)
        write_json(output / 'run_status.json', status)
        print(json.dumps(status), flush=True)
    return 0 if status['state'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
