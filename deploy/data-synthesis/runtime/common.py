"""Portable identity, paths and immutable run plans; no GPU imports."""
import hashlib
import json
from pathlib import Path

TASKS = ('pick_attribute', 'spatial_directions', 'place_a2b', 'stack_cubes')
LIMITS = dict(zip(TASKS, (200, 200, 300, 900)))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def seed_for(worker, task, attempt):
    require(1 <= worker <= 1999, 'Worker ID must be assigned uniquely, 1..1999')
    require(task in TASKS and 0 <= attempt < 250000, 'Seed partition exhausted')
    return 1000000000 + worker * 1000000 + TASKS.index(task) * 250000 + attempt


def contained(root, relative):
    root = Path(root).resolve()
    p = (root / relative).resolve()
    require(p.is_relative_to(root) and not Path(relative).is_absolute(),
            'Manifest path escapes its run root')
    return p


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def hardware(name, arch, memory_mib):
    # Rendering support is independent of CUDA compute capability.
    require('RTX' in name.upper(),
            'This renderer requires an RTX GPU; A100/H100/H20 are not supported by this package')
    require(arch in ('8.6', '8.9', '12.0'),
            'Architecture has not been ported to this pinned CUDA 12.8 build')
    require(memory_mib >= 15000, 'This package requires a 16 GB or larger RTX GPU')
    return 'standard' if memory_mib >= 22000 else 'compact-ik'
