"""Read-only download status with proxy credentials and signed URLs omitted."""
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlsplit

base = Path('/study/wangmengfan/hbot')
log = base / 'logs/sim-assets-download.log'
stat = log.stat()
with log.open('rb') as stream:
    stream.seek(max(0, stat.st_size - 24000))
    text = stream.read().decode(errors='replace')
text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', text)
text = re.sub(r'https?://\S+', '[URL omitted]', text)
lines = text.replace('\r', '\n').splitlines()
events = [line for line in lines if line.startswith('[') and any(word in line for word in (
    'Asset proxy:', 'Starting/resuming', 'Downloader stopped', 'Download exited',
    'Proxy unhealthy', 'verification', 'completed successfully'))]
progress = [line for line in lines if line.startswith('Fetching ')]
completed = [line.replace(str(base / 'datasets/robo_orchard_sim_assets') + '/', '')
             for line in lines if line.startswith('Download complete.')]
errors = [line[:250] for line in lines if any(word in line for word in (
    'SSLError', 'ProxyError', 'ConnectTimeout', 'ReadTimeout', 'RuntimeError', 'Traceback'))]
print(json.dumps({'log_bytes': stat.st_size,
                  'log_mtime': datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
                  'events': events[-5:], 'latest_progress': progress[-1:] or [],
                  'recent_completed_count_in_tail': len(completed), 'last_completions': completed[-3:],
                  'recent_errors_in_tail': errors[-3:]}, ensure_ascii=False), flush=True)
for directory in Path('/proc').iterdir():
    if not directory.name.isdigit():
        continue
    try:
        if directory.stat().st_uid != os.getuid():
            continue
        args = (directory / 'cmdline').read_bytes().split(b'\0')
        target = str(base / 'openpi/.venv/bin/huggingface-cli').encode()
        if target not in args or b'HorizonRobotics/robo_orchard_sim_assets' not in args:
            continue
        environment = dict(item.split(b'=', 1) for item in (directory / 'environ').read_bytes().split(b'\0') if b'=' in item)
        proxy = urlsplit(environment.get(b'HTTPS_PROXY', b'').decode())
        print(json.dumps({'download_pid': int(directory.name), 'proxy_host': proxy.hostname,
                          'proxy_port': proxy.port, 'proxy_authenticated': bool(proxy.password),
                          'HF_HUB_DISABLE_XET': environment.get(b'HF_HUB_DISABLE_XET', b'').decode()}), flush=True)
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
sys.path.insert(0, '/home/wangmengfan/mihomo')
import manage
connections = manage.api('/connections')
safe = [{'host': c.get('metadata', {}).get('host'), 'chains': c.get('chains'),
         'download': c.get('download')} for c in connections.get('connections', [])
        if 'HBot' in c.get('chains', [])]
print(json.dumps({'mihomo_download_total': connections.get('downloadTotal'),
                  'asset_proxy_connections': safe[:6]}, ensure_ascii=False), flush=True)
