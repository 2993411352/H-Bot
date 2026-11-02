"""Offline tests; no network requests or real cooldown waits."""
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from unittest import TestCase, main
from unittest.mock import Mock, patch
import requests
import hbot_hf_rate_limit as rate


class RateLimitTests(TestCase):
    def test_default_spacing(self):
        self.assertEqual(rate.RequestGate().interval, 5.0)

    def test_minimum_and_server_deadlines(self):
        self.assertEqual(rate.retry_delay({}), 600)
        self.assertEqual(rate.retry_delay({'Retry-After': '1200'}), 1200)
        self.assertEqual(rate.retry_delay({'RateLimit': '"api";r=0;t=900'}), 905)
        self.assertEqual(rate.retry_delay({'Retry-After': 'bad'}), 600)
        future = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=1800))
        self.assertGreater(rate.retry_delay({'Retry-After': future}), 1790)

    def test_429_retries_in_place_with_shared_cooldown(self):
        request = requests.Request('GET', 'https://huggingface.co/api/example').prepare()
        limited = Mock(status_code=429, headers={'Retry-After': '900'})
        good = Mock(status_code=200)
        with patch.object(rate, 'GATE') as gate, patch.object(requests.Session, 'send', side_effect=[limited, good]) as send:
            self.assertIs(rate.PacedSession().send(request), good)
            self.assertEqual(send.call_count, 2)
            self.assertEqual(gate.wait.call_count, 2)
            gate.defer.assert_called_once_with(900)
            limited.close.assert_called_once()

    def test_cdn_success_does_not_spend_hub_request_budget(self):
        request = requests.Request('GET', 'https://storage.googleapis.com/example').prepare()
        with patch.object(rate, 'GATE') as gate, patch.object(requests.Session, 'send', return_value=Mock(status_code=200)):
            rate.PacedSession().send(request)
            gate.wait.assert_not_called()

    def test_gate_spacing_and_defer(self):
        gate = rate.RequestGate(interval=1.5)
        with patch.object(rate.time, 'monotonic', return_value=100):
            gate.wait()
            self.assertEqual(gate.next_request, 101.5)
            gate.defer(600)
            self.assertEqual(gate.blocked_until, 700)


if __name__ == '__main__':
    main()
