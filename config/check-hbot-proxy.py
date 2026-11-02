"""Small HTTPS and asset-range check; credentials are never printed or passed in argv."""
import json
import os
import subprocess
import sys
from urllib.parse import urlsplit

proxy = urlsplit(os.environ.get('HTTPS_PROXY', ''))
if proxy.scheme != 'http' or proxy.hostname != '127.0.0.1' or proxy.port != 17890 or not proxy.password:
    print('Expected the authenticated server-local Mihomo proxy, not the SSH tunnel.')
    raise SystemExit(2)
print('Checking authenticated Mihomo at 127.0.0.1:17890; secret omitted.', flush=True)
asset = ('https://huggingface.co/datasets/HorizonRobotics/robo_orchard_sim_assets/'
         'resolve/542bec4a9c9fdb3d688e3bfa20cc1558cc016afc/'
         'PUBLIC_OBJECTS/evaluation_assets/tables/table_001/choice_textures/6040.png')
for name, url, head in [('huggingface', 'https://huggingface.co/', True),
                        ('asset_range_64KiB', asset, False)]:
    command = ['curl', '--silent', '--show-error', '--location', '--noproxy', '',
               '--proto', '=https', '--proto-redir', '=https', '--max-redirs', '5',
               '--connect-timeout', '8', '--max-time', '20', '--output', '/dev/null',
               '--write-out', '%{http_code} %{ssl_verify_result} %{size_download} %{time_total}']
    command += ['--head'] if head else ['--range', '0-65535', '--max-filesize', '1048576']
    result = subprocess.run(command + [url], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=25)
    fields = result.stdout.decode(errors='replace').strip().split()
    okay = result.returncode == 0 and len(fields) == 4 and fields[0].isdigit()
    okay = okay and 200 <= int(fields[0]) < 400 and fields[1] == '0'
    if not head:
        okay = okay and float(fields[2]) > 0
    print(json.dumps({'test': name, 'curl_exit': result.returncode,
                      'http_tls_bytes_seconds': fields, 'passed': bool(okay)}), flush=True)
    if not okay:
        raise SystemExit(1)
print('MAIN_PROXY_AND_ASSET_OK', flush=True)
