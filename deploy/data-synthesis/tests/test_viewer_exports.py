import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from app import create_app
from build_index import build_index, freeze_index
from viewer_fixture import make_export, write_json


class ExportViewerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root/'source'
        self.export = make_export(self.source,101,('stack_cubes','pick_attribute','spatial_directions','place_a2b'))
        make_export(self.source,102)
        self.rows = build_index(self.source,[102,101],'export_v1')
        self.index = freeze_index(self.root/'review/episodes.json',self.rows)
        self.app = create_app(self.source,self.index,self.root/'review/labels')
        self.app.testing = True
        self.client = self.app.test_client()
        self.boot = self.client.get('/api/bootstrap').get_json()

    def tearDown(self): self.tmp.cleanup()

    def hashes(self):
        return {p.relative_to(self.source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.source.rglob('*') if p.is_file()}

    def test_worker_identity_review_and_unchanged_source(self):
        before = self.hashes()
        self.assertEqual(len(self.rows),5)
        self.assertEqual([r['task'] for r in self.rows[:4]],['stack_cubes','pick_color','spatial_pick','place_a2b'])
        self.assertNotEqual(self.rows[0]['id'],self.rows[-1]['id'])
        body = dict(revision=0,status='reviewed',reviewer='fixture',notes='',segments=[
            dict(start=0,end=29,phase='reach',quality='exclude',cycle=0,confidence='high',tags=[],critical=False)])
        response = self.client.post('/api/annotation/'+self.rows[0]['id'],json=body,
                                    headers={'X-CSRF-Token':self.boot['csrf']})
        self.assertEqual(response.status_code,200)
        self.assertEqual(self.app.config['WORKSPACE'].label(self.rows[-1]['id'])['revision'],0)
        summary = self.client.get('/api/summary').get_json()
        self.assertEqual((summary['total'],summary['total_reviewed']),(5,1))
        self.assertEqual(len(summary['tasks']),4)
        self.assertEqual(summary['tasks'][0]['target'],2)
        self.assertEqual(self.client.get('/api/export').get_json()['annotations'][0]['segments'][0]['quality'],'exclude')
        self.assertEqual(before,self.hashes())

    def test_video_range_exact_frame_and_signals(self):
        eid = self.rows[0]['id']
        response = self.client.get('/api/episode/'+eid)
        self.assertEqual(response.status_code,200)
        self.assertEqual(len(response.get_json()['signals']['gripper_action']),30)
        for camera in ('ext1_camera','ext2_camera','wrist_camera'):
            response = self.client.get(f'/media/{eid}/{camera}.mp4',headers={'Range':'bytes=0-99'})
            self.assertEqual(response.status_code,206)
            self.assertEqual(len(response.data),100)
            response.close()
            response = self.client.get(f'/frame/{eid}/{camera}/17.jpg')
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.headers['X-Frame-Index'],'17')
            with Image.open(io.BytesIO(response.data)) as im:
                self.assertLess(abs(im.getpixel((80,48))[0]-85),5)
        self.assertEqual(self.client.get(f'/frame/{eid}/ext1_camera/30.jpg').status_code,400)

    def test_frozen_mapping_survives_collection_resume(self):
        write_json(self.source/'worker_0101/manifest.json',[])
        self.assertEqual(build_index(self.source,[101,102],'export_v1'),self.rows)
        self.assertEqual(freeze_index(self.index,self.rows),self.index)
        changed = copy.deepcopy(self.rows); changed[0]['seed'] += 1
        with self.assertRaisesRegex(ValueError,'inputs changed'): freeze_index(self.index,changed)

    def test_legacy_export_and_wrong_provenance(self):
        (self.export/'source_manifest.json').unlink()
        self.assertEqual(build_index(self.source,[101,102],'export_v1'),self.rows)
        entries = json.loads((self.export/'converter_manifest.json').read_text(encoding='utf-8'))
        entries.reverse(); write_json(self.export/'converter_manifest.json',entries)
        with self.assertRaisesRegex(ValueError,'mapping mismatch'): build_index(self.source,[101],'export_v1')

    def test_incomplete_and_outside_source_rejected(self):
        write_json(self.export/'report.json',dict(status='converting'))
        with self.assertRaisesRegex(ValueError,'not complete'): build_index(self.source,[101],'export_v1')
        write_json(self.export/'report.json',dict(status='format_verified',accepted_episodes=4))
        info_path = self.export/'lerobot_v21/meta/info.json'
        info = json.loads(info_path.read_text(encoding='utf-8')); info['data_path']='../../../../../review/episodes.json'
        write_json(info_path,info)
        with self.assertRaisesRegex(ValueError,'outside source'): build_index(self.source,[101],'export_v1')

    def test_modified_parquet_detected(self):
        path = self.source/self.rows[0]['parquet']
        path.write_bytes(path.read_bytes()+b'changed')
        self.assertEqual(self.client.get('/api/episode/'+self.rows[0]['id']).status_code,400)


if __name__ == '__main__': unittest.main()
