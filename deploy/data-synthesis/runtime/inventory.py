"""Read only the official seen split and immutable asset index."""
from pathlib import Path
import pyarrow.parquet as pq
import yaml
from common import require, sha


def inventory(root=Path('/assets/OBJECTS_v2')):
    split = root / 'splits/hbot_panda_droid_split_seen.yaml'
    index = root / 'asset_indexes/asset_index.v7.parquet'
    allowed = {r['uuid'] for r in yaml.safe_load(split.read_text())['seen']}
    rows = [r for r in pq.read_table(index).to_pylist() if r['uuid'] in allowed]
    require({r['uuid'] for r in rows} == allowed, 'Seen assets missing from index')
    for r in rows:
        for key in ('usd_path', 'urdf_path'):
            p = root / r[key]
            require(p.resolve().is_relative_to(root.resolve()) and p.is_file(),
                    f'Missing asset or external asset path: {p}')
    categories = sorted({r['category'] for r in rows
                         if 'has_interactive_poses' in (r['tags'] or [])
                         and 'is_container' not in (r['tags'] or [])})
    containers = {r['category'] for r in rows if 'is_container' in (r['tags'] or [])}
    require(set(('bowl', 'basket', 'tray', 'plate')) <= containers, 'Seen containers incomplete')
    require(bool(categories), 'No graspable seen categories')
    return dict(training_categories=categories, seen_count=len(rows),
                split_sha256=sha(split), index_sha256=sha(index))


if __name__ == '__main__':
    import json
    print(json.dumps(inventory(), indent=2))
