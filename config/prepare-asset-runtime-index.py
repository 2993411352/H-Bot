#!/usr/bin/env python3
"""Resolve portable index paths in a derived copy; never edit downloaded data."""
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HOME = Path('/home/users/wangmengfan')
ROOT = Path('/assets/OBJECTS_v2')
SOURCE = HOME / 'datasets/robo_orchard_sim_assets/OBJECTS_v2/asset_indexes/asset_index.v7.parquet'
OUTPUT = HOME / 'cache/sim-asset-index/asset_index.v7.parquet'
PATH_COLUMNS = ('usd_path', 'urdf_path', 'interaction_path', 'caption_path', 'metadata_path')


def main():
    original = pq.read_table(SOURCE)
    table = original
    normalized = 0
    for column in PATH_COLUMNS:
        values = []
        for value in original[column].to_pylist():
            if not value:
                values.append(value)
                continue
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = ROOT / candidate
                normalized += 1
            candidate = candidate.resolve()
            if not candidate.is_relative_to(ROOT.resolve()):
                raise ValueError(f'Index path outside asset library: {value}')
            if column in ('usd_path', 'urdf_path') and not candidate.is_file():
                raise FileNotFoundError(candidate)
            values.append(str(candidate))
        i = table.schema.get_field_index(column)
        table = table.set_column(i, table.schema.field(i), pa.array(values, type=table.schema.field(i).type))
    unchanged = [name for name in table.column_names if name not in PATH_COLUMNS]
    assert original.select(unchanged).equals(table.select(unchanged), check_metadata=True)
    assert original.schema.metadata == table.schema.metadata
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix('.parquet.tmp')
    pq.write_table(table, temporary)
    assert pq.read_table(temporary).equals(table, check_metadata=True)
    temporary.replace(OUTPUT)
    report = {
        'source': str(SOURCE), 'output': str(OUTPUT),
        'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'output_sha256': hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        'rows': table.num_rows, 'normalized_paths': normalized,
        'changed_columns': PATH_COLUMNS,
        'all_other_columns_and_metadata_unchanged': True,
        'all_usd_and_urdf_files_exist': True,
    }
    OUTPUT.with_suffix('.report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
