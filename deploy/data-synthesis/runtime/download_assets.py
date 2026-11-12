"""Explicit full snapshot download, resolved revision pinned across retries."""
import json
from pathlib import Path
import sys
from huggingface_hub import HfApi, snapshot_download
from common import require, save

repo = 'HorizonRobotics/robo_orchard_sim_assets'
request = sys.argv[1]
receipt = Path('/assets/.hbot-download.json')
if receipt.exists():
    record = json.loads(receipt.read_text())
    require(record['requested_revision'] == request, 'Existing download revision differs')
else:
    info = HfApi().repo_info(repo, repo_type='dataset', revision=request)
    record = dict(repo=repo, requested_revision=request, resolved_revision=info.sha,
                  complete=False)
    save(receipt, record)
snapshot_download(repo, repo_type='dataset', revision=record['resolved_revision'],
                  local_dir='/assets', max_workers=2)
record['complete'] = True
save(receipt, record)
print('Snapshot downloaded:', record['resolved_revision'])
