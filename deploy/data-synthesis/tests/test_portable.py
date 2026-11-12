import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from common import TASKS, contained, hardware, seed_for


class PortableTests(unittest.TestCase):
    def test_team_seed_intervals_never_overlap(self):
        intervals = sorted((seed_for(w,t,0), seed_for(w,t,249999))
                           for w in (1,2,101,102,1999) for t in TASKS)
        self.assertLess(intervals[-1][1], 2**32)
        for (_, end), (start, _) in zip(intervals, intervals[1:]):
            self.assertLess(end, start)
        for bad in (0,2000):
            with self.assertRaises(ValueError):
                seed_for(bad, TASKS[0], 0)
        with self.assertRaises(ValueError):
            seed_for(1, TASKS[0], 250000)

    def test_expected_gpu_modes_and_unsupported_compute(self):
        self.assertEqual(hardware('NVIDIA RTX A5000','8.6',24564), 'standard')
        self.assertEqual(hardware('NVIDIA GeForce RTX 5070 Ti','12.0',16303), 'compact-ik')
        with self.assertRaises(ValueError):
            hardware('NVIDIA H20','9.0',96000)
        with self.assertRaises(ValueError):
            hardware('NVIDIA RTX 5070','12.0',12000)

    def test_external_manifest_paths_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(contained(d,'foo/bar'), Path(d)/'foo/bar')
            for escape in ('../elsewhere', str(Path(d).parent/'outside')):
                with self.assertRaises(ValueError):
                    contained(d, escape)

    def test_reference_snapshot_preserves_original_and_scrubs_paths(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); src=root/'source.db'; out=root/'reference'
            with sqlite3.connect(src) as db:
                db.executescript('CREATE TABLE records(source TEXT PRIMARY KEY,kind TEXT);'
                                 'CREATE TABLE windows(digest BLOB,source TEXT);'
                                 'CREATE TABLE decisions(source TEXT,result_json TEXT);')
                db.execute('INSERT INTO records VALUES (?,?)',('/private/sample','baseline'))
                db.execute('INSERT INTO windows VALUES (?,?)',(b'abc','/private/sample'))
                db.execute('INSERT INTO decisions VALUES (?,?)',('/private/sample','{}'))
            script=Path(__file__).resolve().parents[1]/'snapshot-reference.py'
            subprocess.run([sys.executable,str(script),'--registry',str(src),'--output',str(out)],check=True)
            with sqlite3.connect(src) as db:
                self.assertEqual(db.execute('SELECT source FROM records').fetchone()[0],'/private/sample')
            with sqlite3.connect(out/'registry.sqlite3') as db:
                a=db.execute('SELECT source FROM records').fetchone()[0]
                self.assertTrue(a.startswith('reference:'))
                self.assertEqual(db.execute('SELECT source FROM windows').fetchone()[0],a)
                self.assertEqual(db.execute('SELECT count(*) FROM decisions').fetchone()[0],0)
            self.assertEqual(json.loads((out/'reference.json').read_text())['records'],1)
            result=subprocess.run([sys.executable,str(script),'--registry',str(src),'--output',str(out)],capture_output=True)
            self.assertNotEqual(result.returncode,0)


if __name__ == '__main__':
    unittest.main()
