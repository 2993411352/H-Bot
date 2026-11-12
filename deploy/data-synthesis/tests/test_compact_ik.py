"""Exercise all-candidate preservation without CUDA or Isaac imports."""
import copy
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from compact_ik import install


class Tensor:
    def __init__(self, data): self.data = np.asarray(data)
    @property
    def shape(self): return self.data.shape
    def dim(self): return self.data.ndim
    def __len__(self): return len(self.data)
    def __getitem__(self, index): return Tensor(self.data[index])
    def reshape(self, *shape): return Tensor(self.data.reshape(*shape))


class CompactIKTests(unittest.TestCase):
    def test_chunked_solver_retains_every_pose_in_original_order(self):
        class Config:
            ik_solver = SimpleNamespace(cuda_grasp_batch_size=1280)
            def model_copy(self, deep): return copy.deepcopy(self)
        class Planner:
            def __init__(self, cfg, env_nums=1): self.cfg=cfg
            def graph_ik(self, candidates):
                # Boundary fake models the upstream finite batch capacity.
                self.assert_size = len(candidates) <= self.cfg.ik_solver.cuda_grasp_batch_size
                if not self.assert_size: raise RuntimeError('Would truncate')
                return SimpleNamespace(success=Tensor(np.ones((len(candidates),1),dtype=bool)),
                    solution=candidates.reshape(len(candidates),1,7))
        fake = SimpleNamespace(ArticulationJointCuroboTrajPlanner=Planner, IkResult=SimpleNamespace)
        torch = SimpleNamespace(cat=lambda values: Tensor(np.concatenate([v.data for v in values])))
        with patch.dict('sys.modules', {'torch':torch,
            'robo_orchard_sim.controllers.curobo_planner.curobo':fake}), \
             patch.dict(os.environ, {'HBOT_MEMORY_PROFILE':'compact-ik'}):
            install()
            solver=Planner(Config())
            inputs=np.arange(3*451*7).reshape(3,451,7)
            result=solver.graph_ik(Tensor(inputs))
            np.testing.assert_array_equal(result.solution.data, inputs)
            np.testing.assert_array_equal(result.success.data,np.ones((3,451),dtype=bool))


if __name__ == '__main__': unittest.main()
