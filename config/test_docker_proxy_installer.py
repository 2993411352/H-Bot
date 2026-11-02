"""Read-only regression tests: no daemon/service changes or real credentials."""
import ast
import json
from pathlib import Path
import stat
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


script = Path(__file__).with_name('install-docker-mihomo-proxy.sh').read_text()
source = script.split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
tree = ast.parse(source)
# Load imports, constants and function definitions, but never execute main().
tree.body = [node for node in tree.body if not isinstance(node, ast.Try)]
module = {}
exec(compile(tree, 'installer-functions', 'exec'), module)
PROXY = 'http://hbot:' + 'a' * 64 + '@127.0.0.1:17890'
MASKED = 'http://xxxxx:xxxxx@127.0.0.1:17890'


class ProxyValidationTests(unittest.TestCase):
    def setUp(self):
        self.info = {'HttpProxy': MASKED, 'HttpsProxy': MASKED}
        self.environment = {'HTTP_PROXY': PROXY, 'HTTPS_PROXY': PROXY}

    def validate(self):
        module['validate_proxy_state'](self.info, PROXY, self.environment)

    def test_docker24_masked_credentials_and_json_keys(self):
        self.validate()

    def test_unmasked_exact_credentials(self):
        self.info = {'HttpProxy': PROXY, 'HttpsProxy': PROXY}
        self.validate()

    def test_wrong_uppercase_api_keys_rejected(self):
        self.info = {'HTTPProxy': MASKED, 'HTTPSProxy': MASKED}
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_old_port_rejected(self):
        self.info['HttpProxy'] = MASKED.replace('17890', '7891')
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_wrong_hostname_rejected(self):
        self.info['HttpsProxy'] = MASKED.replace('127.0.0.1', '127.0.0.2')
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_no_authentication_rejected(self):
        self.info['HttpProxy'] = 'http://127.0.0.1:17890'
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_masked_info_does_not_hide_wrong_actual_password(self):
        self.environment['HTTP_PROXY'] = PROXY.replace('a' * 64, 'b' * 64)
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_missing_actual_environment_rejected(self):
        self.environment.pop('HTTPS_PROXY')
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_verify_daemon_uses_validated_api_and_actual_environment(self):
        with patch.dict(module, {'run': Mock(return_value=json.dumps(self.info)),
                                 'daemon_environment': Mock(return_value=self.environment),
                                 'digest': Mock(return_value=module['DAEMON_HASH']),
                                 'print': Mock()}):
            module['verify_daemon'](PROXY)


class CredentialReuseTests(unittest.TestCase):
    def exercise(self, present=True, mode=0o600, actual=b'expected'):
        path = Mock()
        path.exists.return_value = present
        path.is_symlink.return_value = False
        path.read_bytes.return_value = actual
        checked = Mock(return_value=SimpleNamespace(st_mode=stat.S_IFREG | mode))
        with patch.dict(module, {'ENVFILE': path, 'regular': checked}):
            result = module['reusable_credentials'](b'expected')
            if present:
                checked.assert_called_once_with(path, 0)
            return result

    def test_absent_file_can_be_created(self):
        self.assertFalse(self.exercise(present=False))

    def test_exact_private_file_can_be_reused(self):
        self.assertTrue(self.exercise())

    def test_different_credentials_not_overwritten(self):
        with self.assertRaises(RuntimeError):
            self.exercise(actual=b'other')

    def test_unsafe_mode_not_reused(self):
        with self.assertRaises(RuntimeError):
            self.exercise(mode=0o644)


if __name__ == '__main__':
    unittest.main(verbosity=2)
