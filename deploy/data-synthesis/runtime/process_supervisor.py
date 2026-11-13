"""Run one collector; reap ONLY its descendants even after os._exit/setsid.

Linux subreaping is essential in long-lived Docker exec containers whose PID 1
does not reap or terminate orphaned Xvfb / Isaac helper processes.
"""
import argparse
import ctypes
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def descendants():
    processes = {}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            raw = (directory / 'stat').read_text()
            fields = raw[raw.rfind(')') + 2:].split()
            processes[int(directory.name)] = dict(
                ppid=int(fields[1]), state=fields[0], start=fields[19],
                comm=(directory / 'comm').read_text().strip())
        except (OSError, ValueError, IndexError):
            continue
    owned = {os.getpid()}
    while True:
        found = {pid for pid, row in processes.items() if row['ppid'] in owned}
        if found.issubset(owned):
            break
        owned.update(found)
    return {pid: processes[pid] for pid in owned if pid != os.getpid()}


def send_owned(pid, row, sig):
    # Recheck ancestry AND birth time; never target a recycled/unrelated PID.
    fd = None
    try:
        if hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'):
            fd = os.pidfd_open(pid)
        current = descendants().get(pid)
        if current is None or current['start'] != row['start'] or current['state'] == 'Z':
            return
        if fd is not None:
            signal.pidfd_send_signal(fd, sig)
        else:
            os.kill(pid, sig)
    except ProcessLookupError:
        pass
    finally:
        if fd is not None:
            os.close(fd)


def reap():
    count = 0
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if not pid:
            break
        count += 1
    return count


def cleanup(grace):
    seen, sent = {}, set()
    reaped = 0
    for sig, duration in ((signal.SIGTERM, grace), (signal.SIGKILL, 5.0)):
        deadline = time.monotonic() + duration
        while True:
            rows = descendants()
            for pid, row in rows.items():
                key = (pid, row['start'])
                seen[key] = row
                request = key + (sig,)
                if request not in sent and row['state'] != 'Z':
                    send_owned(pid, row, sig)
                    sent.add(request)
            reaped += reap()
            if not descendants():
                return dict(ok=True, remaining=[], reaped=reaped,
                            descendants_seen=len(seen),
                            by_name=dict(Counter(r['comm'] for r in seen.values())))
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
    return dict(ok=False, remaining=descendants(), reaped=reaped,
                descendants_seen=len(seen),
                by_name=dict(Counter(r['comm'] for r in seen.values())))


def run(args):
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'Cannot enable child subreaper')
    interrupted = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda value, frame: interrupted.append(value))
    started = time.monotonic()
    proc = None
    error, reason, rc = None, 'worker_exit', 125
    try:
        proc = subprocess.Popen(args.command, start_new_session=True)
        print('SUPERVISOR_STARTED', json.dumps(dict(pid=os.getpid(), worker_pid=proc.pid)), flush=True)
        while proc.poll() is None:
            if interrupted:
                reason, rc = 'signal', 128 + interrupted[0]
                break
            if time.monotonic() - started >= args.timeout_seconds:
                reason, rc = 'timeout', 124
                break
            time.sleep(0.1)
        else:
            rc = proc.returncode if proc.returncode >= 0 else 128 - proc.returncode
    except BaseException as exc:
        error = repr(exc)
        reason = 'supervisor_error'
    finally:
        # poll() has already reaped a naturally exited worker; cleanup reaps
        # orphan descendants, including daemonized helpers and zombies.
        result = cleanup(args.cleanup_grace)
        if proc is not None and proc.returncode is None:
            proc.poll()
        if not result['ok']:
            rc = 125
        result.update(worker_exit_code=rc, reason=reason, error=error,
                      finished_at=datetime.now(timezone.utc).isoformat(),
                      supervisor_pid=os.getpid(),
                      worker_pid=proc.pid if proc else None,
                      seconds=round(time.monotonic() - started, 3))
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
        temp.write_text(json.dumps(result, indent=2))
        os.replace(temp, path)
        print('SUPERVISOR_CLEANUP', json.dumps(result), flush=True)
    return rc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    parser.add_argument('--timeout-seconds', type=float, default=3600)
    parser.add_argument('--cleanup-grace', type=float, default=8)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ['--']:
        args.command.pop(0)
    if not args.command or args.timeout_seconds <= 0 or args.cleanup_grace < 0:
        parser.error('Command and valid timeouts required')
    sys.exit(run(args))
