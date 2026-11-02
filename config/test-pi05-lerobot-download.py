"""Offline tests: no downloads, training, GPU initialization, or proxy changes."""
import base64
import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('downloader', Path(__file__).with_name('download-pi05-lerobot.py'))
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
CONTENT = b'resumable-model-and-dataset-test' * 50


class Response:
    def __init__(self, content, status=200, headers=None):
        self.content, self.status_code, self.headers = content, status, headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start:start + chunk_size]


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='hbot-download-test-', dir=os.getenv('TMPDIR'))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.item = dict(path='model.bin', size=len(CONTENT),
                         sha256=hashlib.sha256(CONTENT).hexdigest(), url='https://example.invalid/file')
        self.limiter = d.RateLimiter(100 * 1024 * 1024)

    def run_download(self):
        return d.download_file(self.item, self.root, self.limiter, lambda *a, **k: None)

    def test_fresh_download_and_verified_skip(self):
        with patch.object(d.requests, 'get', return_value=Response(CONTENT)) as request:
            result = self.run_download()
            self.assertEqual(result.read_bytes(), CONTENT)
            self.run_download()
            self.assertEqual(request.call_count, 1)

    def test_resume_exact_offset_and_crc32c(self):
        del self.item['sha256']
        self.item['crc32c'] = base64.b64encode(d.google_crc32c.Checksum(CONTENT).digest()).decode()
        offset = 333
        (self.root / 'model.bin.part').write_bytes(CONTENT[:offset])
        headers = {'Content-Range': f'bytes {offset}-{len(CONTENT)-1}/{len(CONTENT)}'}
        with patch.object(d.requests, 'get', return_value=Response(CONTENT[offset:], 206, headers)) as request:
            self.assertEqual(self.run_download().read_bytes(), CONTENT)
            self.assertEqual(request.call_args.kwargs['headers']['Range'], f'bytes={offset}-')

    def test_ignored_range_preserves_partial(self):
        part = self.root / 'model.bin.part'
        part.write_bytes(CONTENT[:123])
        with patch.object(d.requests, 'get', return_value=Response(CONTENT)), patch.object(d.time, 'sleep'), patch.object(d, 'log'):
            with self.assertRaises(RuntimeError):
                self.run_download()
        self.assertEqual(part.read_bytes(), CONTENT[:123])
        self.assertFalse((self.root / 'model.bin').exists())

    def test_existing_bad_final_is_not_overwritten(self):
        final = self.root / 'model.bin'
        final.write_bytes(b'existing-user-data')
        with patch.object(d.requests, 'get') as request:
            with self.assertRaises(RuntimeError):
                self.run_download()
            request.assert_not_called()
        self.assertEqual(final.read_bytes(), b'existing-user-data')

    def test_empty_object(self):
        self.item.update(size=0, sha256=hashlib.sha256(b'').hexdigest())
        with patch.object(d.requests, 'get') as request:
            self.assertEqual(self.run_download().read_bytes(), b'')
            request.assert_not_called()

    def test_path_escape_rejected(self):
        with self.assertRaises(ValueError):
            d.safe_path(self.root, '../outside')


if __name__ == '__main__':
    unittest.main()
