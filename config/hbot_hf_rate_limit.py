"""Conservative Hub request pacing for the existing HF 0.32 asset environment.

No package upgrades and no proxy changes. All downloader threads share a gate.
429 responses are retried in place after the server's reset interval (at least
10 minutes), so one rate limit does not immediately restart the repository scan.
"""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import re
import runpy
import sys
import threading
import time
from urllib.parse import urlsplit

import requests


def retry_delay(headers):
    delay = 600.0
    retry = headers.get('Retry-After', '')
    try:
        delay = max(delay, float(retry))
    except (ValueError, TypeError):
        try:
            delay = max(delay, (parsedate_to_datetime(retry) - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            pass
    for match in re.finditer(r'(?:^|[;,\s])t=(\d+)', headers.get('RateLimit', '')):
        delay = max(delay, float(match[1]) + 5)
    return delay


class RequestGate:
    def __init__(self, interval=5.0):
        self.interval = interval
        self.lock = threading.Lock()
        self.next_request = 0.0
        self.blocked_until = 0.0

    def wait(self):
        while True:
            with self.lock:
                now = time.monotonic()
                delay = max(self.next_request, self.blocked_until) - now
                if delay <= 0:
                    self.next_request = now + self.interval
                    return
            time.sleep(min(delay, 30))

    def defer(self, seconds):
        with self.lock:
            self.blocked_until = max(self.blocked_until, time.monotonic() + seconds)


GATE = RequestGate()


class PacedSession(requests.Session):
    def send(self, request, **kwargs):
        is_hub = urlsplit(request.url).hostname == 'huggingface.co'
        for attempt in range(1, 7):
            if is_hub or attempt > 1:
                GATE.wait()
            response = super().send(request, **kwargs)
            if response.status_code != 429:
                return response
            delay = retry_delay(response.headers)
            GATE.defer(delay)
            print(f'[{datetime.now(timezone.utc).isoformat()}] HF_RATE_LIMIT: HTTP 429; '
                  f'all Hub requests cooling down for {delay:.0f}s; attempt={attempt}', flush=True)
            if attempt == 6:
                return response
            response.close()
        raise AssertionError('Unreachable')


def configure():
    from huggingface_hub import configure_http_backend
    configure_http_backend(backend_factory=PacedSession)


if __name__ == '__main__':
    configure()
    if len(sys.argv) < 2:
        raise SystemExit('Usage: hbot_hf_rate_limit.py <HF CLI path> <arguments>')
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name='__main__')
