"""Build GPU-specific extensions inside the new image, never on the host."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0, '/opt/hbot/runtime')
from common import require, sha

arch = sys.argv[1]
require(arch in ('8.6', '8.9', '12.0'), 'Unsupported build architecture')
base, dst = Path('/opt/curobo'), Path('/opt/hbot/curobo')
commit = subprocess.check_output(['git', '-c', 'safe.directory=/opt/curobo',
                                '-C', str(base), 'rev-parse', 'HEAD'], text=True).strip()
require(commit == '2fbffc35225398cf9d5f382804faa9de2608753b', 'Unexpected cuRobo source')
dst.mkdir()
shutil.copytree(base/'src', dst/'src', ignore=shutil.ignore_patterns(
    '*.so', '__pycache__', '*.pyc', '*.egg-info'))
for name in ('setup.py', 'setup.cfg', 'pyproject.toml', 'README.md',
             'MANIFEST.in', 'LICENSE', 'LICENSE_ASSETS'):
    shutil.copy2(base/name, dst/name)
mesh = dst/'src/curobo/geom/sdf/world_mesh.py'
text = mesh.read_text()
require(text.count('wp.torch.device_from_torch') == 1, 'Warp patch no longer applies')
mesh.write_text(text.replace('wp.torch.device_from_torch', 'wp.device_from_torch'))
env = dict(os.environ, TORCH_CUDA_ARCH_LIST=arch, MAX_JOBS='2',
           SETUPTOOLS_SCM_PRETEND_VERSION='0.7.6+hbotsynth')
subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'],
               cwd=dst, env=env, check=True)
extensions = sorted((dst/'src/curobo/curobolib').glob('*.so'))
require(len(extensions) == 5, 'cuRobo must have five fresh extensions')
for f in extensions:
    elf = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump', '--list-elf', str(f)], text=True)
    require('sm_' + arch.replace('.', '') in elf, f'Wrong architecture: {f}')
Path('/opt/hbot/build.json').write_text(json.dumps(dict(
    sim_commit='f2e041e5fabe354c1398ea97c934f603bb65d14d', curobo_commit=commit,
    architecture=arch, extensions={f.name: sha(f) for f in extensions},
    collector_patch_sha256=sha('/opt/hbot/patches/collector.patch')), indent=2))
