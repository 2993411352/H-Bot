"""Real Linux process lifecycle tests; no model, simulation or GPU required."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

SUPERVISOR = Path(__file__).resolve().parents[1] / 'runtime/process_supervisor.py'


def command(report, worker, timeout=5):
    return [sys.executable, str(SUPERVISOR), '--report', str(report),
            '--timeout-seconds', str(timeout), '--cleanup-grace', '0.3',
            '--', sys.executable, '-c', worker]


def check(report):
    result = json.loads(report.read_text())
    assert result['ok'] and not result['remaining'], result
    return result


def main(cycles):
    with tempfile.TemporaryDirectory(prefix='hbot-supervisor-test-') as directory:
        root = Path(directory)
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
        try:
            for code in (0, 7):
                report = root / f'exit-{code}.json'
                worker = (
                    'import os,subprocess,sys,time; '
                    'subprocess.Popen([sys.executable,"-c",'
                    '"import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(120)"],start_new_session=True); '
                    f'time.sleep(.2);os._exit({code})')
                r = subprocess.run(command(report, worker), capture_output=True, text=True, timeout=15)
                assert r.returncode == code, r.stdout + r.stderr
                assert check(report)['descendants_seen'] >= 1
                assert unrelated.poll() is None
            report = root / 'timeout.json'
            r = subprocess.run(command(report, 'import time;time.sleep(120)', .3), capture_output=True, text=True, timeout=15)
            assert r.returncode == 124 and check(report)['reason'] == 'timeout'
            report = root / 'signal.json'
            with (root / 'signal.log').open('w') as log:
                p = subprocess.Popen(command(report, 'import time;time.sleep(120)'), stdout=log, stderr=log)
                time.sleep(.5);p.send_signal(signal.SIGTERM)
                assert p.wait(timeout=15) == 143
            assert check(report)['reason'] == 'signal'
            # A grandchild that exits first must be adopted and reaped as well.
            report = root / 'zombie.json'
            worker = 'import subprocess,sys,time,os;subprocess.Popen([sys.executable,"-c","pass"]);time.sleep(.2);os._exit(0)'
            r = subprocess.run(command(report, worker), capture_output=True, text=True, timeout=15)
            assert r.returncode == 0 and check(report)['reaped'] >= 1
            print('PROCESS_LIFECYCLE_TESTS_PASSED: hard exit, detached TERM-resistant child, failure code, timeout, signal, zombie, unrelated sibling', flush=True)
            for i in range(cycles):
                report = root / f'xvfb-{i}.json'
                worker = 'from pyvirtualdisplay import Display;import os;d=Display(backend="xvfb");d.start();os._exit(0)'
                r = subprocess.run(command(report, worker, 10), capture_output=True, text=True, timeout=20)
                assert r.returncode == 0, r.stdout + r.stderr
                assert check(report)['by_name'].get('Xvfb', 0) == 1
                assert unrelated.poll() is None
            print(f'XVFB_HARD_EXIT_CYCLES_PASSED: {cycles}', flush=True)
        finally:
            unrelated.terminate();unrelated.wait(timeout=5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xvfb-cycles', type=int, default=0)
    main(parser.parse_args().xvfb_cycles)
