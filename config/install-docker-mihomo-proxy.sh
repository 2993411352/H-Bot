#!/usr/bin/env bash
# Switch only the Docker daemon proxy. Does not touch daemon.json or containers.
set -euo pipefail
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
exec /usr/bin/python3 - "$@" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit

CTL = '/usr/bin/mctes'
DROPIN = Path('/etc/systemd/system/docker.service.d/proxy.conf')
ENVFILE = Path('/etc/docker/hbot-mihomo-proxy.env')
DAEMON = Path('/etc/docker/daemon.json')
SECRET = Path('/home/wangmengfan/mihomo/proxy.secret')
OLD_HASH = 'bfc15e513f4d15ea2353c4992e18f8336f41d8c29e0e24b7c88fe532884a28a1'
DAEMON_HASH = '701dfe46bd36e91bd048a83303eafe590d80cc1ea8d01804641aa1df2bdc0f1b'
NEW_DROPIN = b'[Service]\nEnvironmentFile=/etc/docker/hbot-mihomo-proxy.env\n'

def fail(message):
    raise RuntimeError(message)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def regular(path, owner=None):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or (owner is not None and info.st_uid != owner):
        fail('Unexpected file type or owner: ' + str(path))
    return info

def run(args, timeout=25, env=None):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env)
    if result.returncode:
        # Do not echo command output: Docker diagnostics may contain proxy credentials.
        fail('Command failed: ' + Path(args[0]).name + ' ' + args[1] + ' (output withheld)')
    return result.stdout.strip()

def no_containers():
    if run(['/usr/bin/docker', 'ps', '-q']):
        fail('Running containers detected. Refusing to restart Docker.')

def atomic_write(path, content, mode):
    fd, temporary = tempfile.mkstemp(prefix='.hbot-proxy-stage.', dir=str(path.parent))
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)  # Only this invocation's exact temporary file.

def native_systemd():
    if hashlib.md5(Path(CTL).read_bytes()).hexdigest() != '98e9e7c9213bbd9987b015b512382076':
        fail('Native systemctl identity changed; re-inspection required.')

def restart_docker():
    # Recheck at the last possible point; never stop or remove any container.
    no_containers()
    run([CTL, 'daemon-reload'])
    no_containers()
    run([CTL, 'restart', 'docker'], timeout=55)
    run([CTL, 'is-active', '--quiet', 'docker'])

def registry_check(proxy):
    env = os.environ.copy()
    for key in ('ALL_PROXY', 'all_proxy', 'HTTP_PROXY', 'http_proxy', 'HTTPS_PROXY', 'https_proxy'):
        env.pop(key, None)
    env.update(HTTP_PROXY=proxy, http_proxy=proxy, HTTPS_PROXY=proxy, https_proxy=proxy,
               NO_PROXY='localhost,127.0.0.1,::1', no_proxy='localhost,127.0.0.1,::1')
    result = subprocess.run([
        '/usr/bin/curl', '--silent', '--show-error', '--head', '--output', '/dev/null',
        '--connect-timeout', '8', '--max-time', '20', '--write-out', '%{http_code}',
        'https://registry-1.docker.io/v2/'
    ], env=env, capture_output=True, text=True, timeout=23)
    # Registry's unauthenticated endpoint is expected to return 401, with valid TLS.
    if result.returncode or result.stdout.strip() != '401':
        fail('Registry HTTPS test failed (curl=%s, HTTP=%s).' %
             (result.returncode, result.stdout.strip() if re.fullmatch(r'\d{3}', result.stdout.strip()) else 'unknown'))
    print('REGISTRY_PROXY_OK: TLS verified, HTTP 401 (expected).', flush=True)

def daemon_environment():
    pid = run([CTL, 'show', 'docker', '--property=MainPID', '--value'])
    if not pid.isdigit() or int(pid) <= 1:
        fail('Docker daemon PID unavailable.')
    process = Path('/proc') / pid
    if (process / 'exe').resolve() != Path('/usr/bin/dockerd').resolve():
        fail('Docker main process identity is unexpected.')
    # Read locally as root; never print the environment or pass secrets in argv.
    entries = (process / 'environ').read_bytes().split(b'\0')
    return dict(entry.decode().split('=', 1) for entry in entries if b'=' in entry)

def validate_proxy_state(info, proxy, environment):
    expected = urlsplit(proxy)
    # Moby v24 masks BOTH username and password as xxxxx in the Info API.
    masked = expected._replace(netloc='xxxxx:xxxxx@' + expected.hostname + ':' + str(expected.port)).geturl()
    for api_key, env_key in (('HttpProxy', 'HTTP_PROXY'), ('HttpsProxy', 'HTTPS_PROXY')):
        if info.get(api_key) not in (proxy, masked):
            fail('Docker reported an unexpected ' + api_key + ' endpoint.')
        if environment.get(env_key) != proxy:
            fail('Docker process did not load the expected ' + env_key + ' credentials.')

def verify_daemon(proxy):
    info = json.loads(run(['/usr/bin/docker', 'info', '--format', '{{json .}}']))
    validate_proxy_state(info, proxy, daemon_environment())
    if digest(DAEMON) != DAEMON_HASH:
        fail('daemon.json changed during installation; manual review required.')
    print('DAEMON_PROXY_OK: http://127.0.0.1:17890, authentication=yes; daemon.json unchanged.', flush=True)

def reusable_credentials(content):
    if not ENVFILE.exists() and not ENVFILE.is_symlink():
        return False
    info = regular(ENVFILE, 0)
    if stat.S_IMODE(info.st_mode) != 0o600 or ENVFILE.read_bytes() != content:
        fail('Existing credential file differs or has unsafe permissions; no overwrite performed.')
    return True

def rollback(backup):
    backup = Path(backup)
    if (backup.parent != Path('/etc/docker') or not backup.name.startswith('hbot-proxy-backup.')
            or backup.is_symlink() or not backup.is_dir()):
        fail('Invalid rollback directory.')
    info = backup.stat()
    if info.st_uid != 0 or stat.S_IMODE(info.st_mode) != 0o700:
        fail('Backup directory must be root-owned mode 0700.')
    original = backup / 'proxy.conf'
    original_info = regular(original, 0)
    regular(DROPIN, 0)
    if digest(original) != OLD_HASH or DROPIN.read_bytes() != NEW_DROPIN:
        fail('Configuration differs from this migration; refusing to overwrite.')
    no_containers()
    atomic_write(DROPIN, original.read_bytes(), stat.S_IMODE(original_info.st_mode))
    restart_docker()
    print('ROLLED_BACK: original proxy.conf restored; private credential file retained, unused.')

def main():
    args = sys.argv[1:]
    if args not in ([], ['--check']) and not (len(args) == 2 and args[0] == '--rollback'):
        fail('Usage: script [--check | --rollback /etc/docker/hbot-proxy-backup.XXXXXX]')
    if args != ['--check'] and os.geteuid() != 0:
        fail('Run installation or rollback with sudo.')
    native_systemd()
    if args and args[0] == '--rollback':
        rollback(args[1])
        return
    regular(DROPIN, 0)
    regular(DAEMON, 0)
    if digest(DROPIN) != OLD_HASH or digest(DAEMON) != DAEMON_HASH:
        fail('Docker configuration changed since inspection; no changes made.')
    secret_info = regular(SECRET, 1013)
    if stat.S_IMODE(secret_info.st_mode) & 0o077:
        fail('Proxy secret must not be accessible to group/others.')
    password = SECRET.read_text().strip()
    if not re.fullmatch(r'[0-9a-fA-F]{64}', password):
        fail('Unexpected proxy secret format.')
    proxy = 'http://hbot:' + password + '@127.0.0.1:17890'
    no_containers()
    run([CTL, 'is-active', '--quiet', 'docker'])
    registry_check(proxy)
    if args == ['--check']:
        print('PREFLIGHT_OK: no running containers; all configuration files unchanged.')
        return
    content = ''.join('%s=%s\n' % (key, proxy) for key in
                      ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'))
    content += 'NO_PROXY=localhost,127.0.0.1,::1\nno_proxy=localhost,127.0.0.1,::1\n'
    content = content.encode()
    if reusable_credentials(content):
        print('REUSING_VERIFIED_CREDENTIAL_FILE: root-owned 0600, exact contents match.', flush=True)
    backup = Path(tempfile.mkdtemp(prefix='hbot-proxy-backup.', dir='/etc/docker'))
    os.chmod(backup, 0o700)
    shutil.copy2(DROPIN, backup / 'proxy.conf')
    print('BACKUP=' + str(backup), flush=True)
    changed = False
    try:
        no_containers()
        # Check again before commit to avoid overwriting concurrent administrator edits.
        if digest(DROPIN) != OLD_HASH or digest(DAEMON) != DAEMON_HASH:
            fail('Configuration changed during preflight.')
        if not reusable_credentials(content):
            atomic_write(ENVFILE, content, 0o600)
        atomic_write(DROPIN, NEW_DROPIN, 0o644)
        changed = True
        restart_docker()
        verify_daemon(proxy)
    except Exception:
        if changed:
            try:
                rollback(backup)
            except Exception:
                print('AUTO_ROLLBACK_NOT_COMPLETED: configuration preserved for review; no containers stopped.', flush=True)
                print('Backup location: ' + str(backup), flush=True)
        raise
    print('DOCKER_PROXY_SWITCHED_AND_VERIFIED')
    print('No images or containers removed; daemon.json and user services unchanged.')
    print('Rollback: sudo bash /study/wangmengfan/hbot/config/install-docker-mihomo-proxy.sh --rollback ' + str(backup))

try:
    main()
except Exception as exc:
    if isinstance(exc, RuntimeError):
        print('STOP: ' + str(exc), file=sys.stderr)
    else:
        print('STOP: ' + type(exc).__name__ + '; details withheld to protect credentials.', file=sys.stderr)
    sys.exit(1)
PY
