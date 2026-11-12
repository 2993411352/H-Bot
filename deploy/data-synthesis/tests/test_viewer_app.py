import copy
import json
from pathlib import Path
import tempfile
import unittest
from app import create_app, TASKS, validate_annotation

class AppTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        source=self.root/'source'; source.mkdir(); index=self.root/'episodes.json'
        rows=[]
        for task in TASKS:
            for i in range(12):
                rows.append(dict(task=task,episode_index=i,length=20,instruction='test instruction',source_id=f'{task}-{i}',
                            parquet_sha256='a'*64,asset_id=str(i%3),attribute='open' if i%2 else 'close'))
        index.write_text(json.dumps(rows));self.app=create_app(source,index,self.root/'labels',True)
        self.app.config['TESTING']=True;self.client=self.app.test_client()
        b=self.client.get('/api/bootstrap').get_json();self.csrf=b['csrf'];self.boot=b
        self.id='stack_cubes--000000'
        self.body=dict(revision=0,status='reviewed',reviewer='unit-test',notes='',segments=[
          dict(start=0,end=19,phase='reach',quality='usable',cycle=1,confidence='high',critical=False,tags=[],note='')])
    def tearDown(self): self.tmp.cleanup()
    def post(self,body):return self.client.post('/api/annotation/'+self.id,json=body,headers={'X-CSRF-Token':self.csrf})
    def test_samples(self):
        self.assertEqual(sum(e['suggested'] for e in self.boot['episodes']),60)
        self.assertFalse(self.boot['summary']['ready_for_auto'])
    def test_save_history_and_conflict(self):
        r=self.post(self.body);self.assertEqual(r.status_code,200);self.assertEqual(r.get_json()['revision'],1)
        self.assertEqual(self.post(self.body).status_code,409)
        b=copy.deepcopy(self.body);b['revision']=1;b['notes']='updated'
        self.assertEqual(self.post(b).status_code,200)
        history=self.client.get('/api/history/'+self.id).get_json();self.assertEqual(len(history),2)
        self.assertEqual(history[1]['notes'],'')
        summary=self.client.get('/api/summary').get_json();self.assertEqual(summary['total_reviewed'],1)
    def test_bad_segments(self):
        for change in ({'end':20},{'start':-1},{'start':2},{'end':18},{'phase':'unknown'},{'cycle':4},{'start':True}):
            b=copy.deepcopy(self.body);b['segments'][0].update(change)
            self.assertEqual(self.post(b).status_code,400,change)
        b=copy.deepcopy(self.body);b['segments'].append(copy.deepcopy(b['segments'][0]));self.assertEqual(self.post(b).status_code,400)
        self.assertEqual(self.post([]).status_code,400)
    def test_draft_allows_gaps_not_overlap(self):
        b=copy.deepcopy(self.body);b['status']='draft';b['segments'][0]['start']=5
        self.assertEqual(self.post(b).status_code,200)
        self.assertEqual(self.client.get('/api/summary').get_json()['total_reviewed'],0)
    def test_security(self):
        self.assertEqual(self.client.post('/api/annotation/'+self.id,json=self.body).status_code,403)
        self.assertEqual(self.client.post('/api/annotation/'+self.id,json=self.body,headers={'X-CSRF-Token':self.csrf,'Origin':'https://evil.test'}).status_code,403)
        self.assertEqual(self.client.get('/api/bootstrap',headers={'Host':'evil.test'}).status_code,400)
        self.assertEqual(self.client.get('/media/'+self.id+'/bad.mp4').status_code,404)
        with self.assertRaises(ValueError): self.app.config['WORKSPACE'].safe_source('../episodes.json')
    def test_export(self):
        self.post(self.body);data=self.client.get('/api/export').get_json()
        self.assertEqual(data['annotations'][0]['origin'],'human');self.assertEqual(data['fps'],30)
    def test_unknown_needs_review(self):
        b=copy.deepcopy(self.body);b['segments'][0].update(phase='unknown',quality='uncertain')
        self.assertEqual(self.post(b).status_code,200)
        summary=self.client.get('/api/summary').get_json()
        self.assertEqual(summary['tasks'][0]['uncertain_reviewed'],1)
    def test_durable_restart(self):
        self.post(self.body)
        other=create_app(self.root/'source',self.root/'episodes.json',self.root/'labels',True)
        self.assertEqual(other.config['WORKSPACE'].label(self.id)['revision'],1)

if __name__=='__main__':unittest.main(verbosity=2)
