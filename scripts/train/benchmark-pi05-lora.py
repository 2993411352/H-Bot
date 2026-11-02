#!/usr/bin/env python3
"""One bounded batch-size test of the real training launcher, without changing its algorithm."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import threading
import time


def stop_own_process(process, *, group=False):
    if process is None or process.poll() is not None:
        return
    try:
        if group:
            os.killpg(process.pid, signal.SIGINT)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        if group:
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--warmup-steps', type=int, default=10)
    parser.add_argument('--exp-name', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max-temperature-c', type=int, default=90,
                        help='Conservative test cutoff, not a manufacturer limit')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.exp_name):
        parser.error('Unsafe experiment name')
    if args.batch_size < 1 or args.gpu < 0 or not 1 <= args.warmup_steps < args.steps <= 100:
        parser.error('Use positive batch/GPU, 1 <= warmup < steps <= 100')
    if not 60 <= args.max_temperature_c <= 90:
        parser.error('Test temperature cutoff must be between 60 and 90 C')
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Benchmark received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    root = Path(os.environ.get('HBOT_HOME', '/study/wangmengfan/hbot')).resolve()
    checkpoint_root = root / 'checkpoints/pi05_hbot_lora' / args.exp_name
    if checkpoint_root.exists():
        parser.error('Experiment already exists; choose a new name')
    run_dir = root / 'logs' / ('pi05_bench_' + args.exp_name)
    run_dir.mkdir(exist_ok=False)
    env = dict(os.environ, PYTHONUNBUFFERED='1', HBOT_TRAIN_GPU=str(args.gpu))
    command = ['bash', str(root / 'scripts/run-pi05-lora.sh'), '--exp-name', args.exp_name,
               '--batch-size', str(args.batch_size), '--num-train-steps', str(args.steps),
               '--log-interval', '1']
    pattern = re.compile(r'^Step (\d+): grad_norm=([^,]+), loss=([^,]+), param_norm=(\S+)')
    steps = []
    monitor = trainer = None
    monitor_thread = None
    thermal_stop = threading.Event()
    monitor_failures = []
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    print(f'BENCHMARK directory={run_dir} batch={args.batch_size} steps={args.steps}', flush=True)
    try:
        with (run_dir / 'monitor.stderr').open('w') as monitor_error, (run_dir / 'train.log').open('w') as log:
            monitor = subprocess.Popen(
                ['stdbuf', '-oL', 'nvidia-smi', '-i', str(args.gpu),
                 '--query-gpu=timestamp,index,memory.used,utilization.gpu,temperature.gpu',
                 '--format=csv,noheader,nounits', '--loop-ms=500'],
                stdout=subprocess.PIPE, text=True, bufsize=1,
                stderr=monitor_error, start_new_session=True)
            trainer = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1, start_new_session=True)
            def watch_temperature():
                try:
                    with (run_dir / 'gpu.csv').open('w') as csv_file:
                        for sample in monitor.stdout:
                            csv_file.write(sample)
                            csv_file.flush()
                            row = next(csv.reader([sample]))
                            if float(row[4]) >= args.max_temperature_c and trainer.poll() is None:
                                thermal_stop.set()
                                print(f'THERMAL_CUTOFF {row[4].strip()} C: stopping only this benchmark', flush=True)
                                stop_own_process(trainer, group=True)
                                break
                        if trainer.poll() is None and not thermal_stop.is_set():
                            raise RuntimeError('GPU monitor ended while training was running')
                except Exception as error:
                    monitor_failures.append(str(error))
                    stop_own_process(trainer, group=True)
            monitor_thread = threading.Thread(target=watch_temperature, daemon=True)
            monitor_thread.start()
            for line in trainer.stdout:
                now = time.monotonic()
                log.write(line)
                log.flush()
                match = pattern.match(line.strip())
                if match:
                    steps.append(dict(step=int(match[1]), grad_norm=float(match[2]), loss=float(match[3]),
                                      param_norm=float(match[4]), elapsed_s=now - started))
                if match or any(token in line for token in ('Log:', 'FREEZE STRATEGY', 'TRAINING ',
                    'Initialized data loader', 'Initialized train state', 'Finished saving checkpoint',
                    'Traceback', 'RESOURCE_EXHAUSTED', 'Refusing', 'Error:')):
                    print(line.rstrip(), flush=True)
            returncode = trainer.wait()
    finally:
        # Only processes created by this runner; never discover/stop unrelated jobs.
        stop_own_process(trainer, group=True)
        stop_own_process(monitor)
        if monitor_thread is not None:
            monitor_thread.join(timeout=5)
    wall_s = time.monotonic() - started
    samples = []
    if (run_dir / 'gpu.csv').exists():
        with (run_dir / 'gpu.csv').open() as stream:
            samples = [dict(time=row[0].strip(), memory_mib=float(row[2]),
                            utilization=float(row[3]), temperature_c=float(row[4]))
                       for row in csv.reader(stream) if len(row) == 5]
    intervals = [steps[i]['elapsed_s'] - steps[i-1]['elapsed_s'] for i in range(1, len(steps))
                 if steps[i]['step'] >= args.warmup_steps]
    finite = all(math.isfinite(row[key]) for row in steps for key in ('loss', 'grad_norm', 'param_norm'))
    checkpoint = checkpoint_root / str(args.steps - 1)
    metadata = checkpoint / '_CHECKPOINT_METADATA'
    committed = metadata.exists() and json.loads(metadata.read_text()).get('commit_timestamp_nsecs') is not None
    complete_steps = [row['step'] for row in steps] == list(range(args.steps))
    success = (returncode == 0 and complete_steps and finite and committed and bool(samples)
               and not thermal_stop.is_set() and not monitor_failures)
    report = dict(status='PASSED' if success else 'FAILED', experiment=args.exp_name,
                  batch_size=args.batch_size, gpu=args.gpu, steps_requested=args.steps,
                  steps_completed=len(steps), warmup_steps_excluded=args.warmup_steps,
                  started_at=started_at, wall_s=wall_s, returncode=returncode,
                  peak_sampled_memory_mib=max((r['memory_mib'] for r in samples), default=None),
                  peak_sampled_temperature_c=max((r['temperature_c'] for r in samples), default=None),
                  sampling_interval_ms=500, sample_count=len(samples),
                  temperature_cutoff_c=args.max_temperature_c, thermal_cutoff_triggered=thermal_stop.is_set(),
                  monitor_errors=monitor_failures,
                  steady_step_intervals=len(intervals),
                  mean_step_s=statistics.mean(intervals) if intervals else None,
                  median_step_s=statistics.median(intervals) if intervals else None,
                  samples_per_s=args.batch_size / statistics.mean(intervals) if intervals else None,
                  all_logged_values_finite=finite, checkpoint_committed=committed,
                  checkpoint=str(checkpoint), metrics=steps,
                  timing_scope='Intervals between synchronized step logs; includes data loading/logging; '
                               'excludes first 10 steps for the default test and final checkpoint save. '
                               'Short-run throughput, not long-run convergence or performance.')
    (run_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('BENCHMARK_RESULT ' + json.dumps({key:value for key,value in report.items() if key != 'metrics'}), flush=True)
    raise SystemExit(0 if success else 1)


if __name__ == '__main__':
    main()
