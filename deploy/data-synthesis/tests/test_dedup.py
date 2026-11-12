from pathlib import Path
import tempfile
import unittest
import numpy as np
from dedup_registry import initialize, connect, describe, compare, insert


class DedupTests(unittest.TestCase):
    def test_seed_and_filename_changes_do_not_evade_dedup(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            a,b=root/'a.mcap',root/'b.mcap'
            a.write_bytes(b'a'); b.write_bytes(b'b')
            state=np.arange(80*8).reshape(80,8)/1000
            with connect(root/'index.db') as db:
                initialize(db)
                old=describe(a,'baseline','pick_attribute',1,state,state,{})
                insert(db,old)
                new=describe(b,'new','pick_attribute',999,state,state,{})
                result=compare(db,new)
                self.assertFalse(result['pass_check'])
                self.assertEqual(result['duplicate_matches'][0]['reason'],
                                 'same_command_trajectory_after_idle_collapse')

    def test_assets_and_prompt_alone_do_not_reject_different_scene(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); a=root/'a'; b=root/'b'
            a.touch(); b.touch()
            state=np.arange(80*8).reshape(80,8)/1000
            def meta(x):
                return dict(actors={'p':{'actor_uuid':'same','actor_category':'spoon'}},
                            init_position={'p':[x,0,0,1,0,0,0]},
                            role_bindings={'pick':{'scene_name':'p'}})
            with connect(root/'index.db') as db:
                initialize(db)
                insert(db,describe(a,'baseline','pick_attribute',1,state,state,meta(.1)))
                self.assertTrue(compare(db,describe(b,'new','pick_attribute',2,
                    state+.2,state+.4,meta(.2)))['pass_check'])
                self.assertFalse(compare(db,describe(b,'new','pick_attribute',2,
                    state+.2,state+.4,meta(.1001)))['pass_check'])


if __name__ == '__main__':
    unittest.main()
