"""Bound IK batch allocation without dropping candidate grasps.

Opt-in per process. Physics, images, planner tolerances and IK seed count stay
unchanged. Chunk boundaries can still change numerical/RNG results.
"""
import os


def install():
    if os.environ.get('HBOT_MEMORY_PROFILE') != 'compact-ik':
        return
    import torch
    from robo_orchard_sim.controllers.curobo_planner.curobo import (
        ArticulationJointCuroboTrajPlanner as Planner, IkResult)
    original_init, original_graph = Planner.__init__, Planner.graph_ik

    def init(self, cfg, env_nums=1):
        cfg = cfg.model_copy(deep=True)
        cfg.ik_solver.cuda_grasp_batch_size = 128
        original_init(self, cfg, env_nums)

    def graph(self, candidates):
        if candidates.dim() == 2:
            b, n = candidates.shape[0], 1
        elif candidates.dim() == 3:
            b, n = candidates.shape[:2]
        else:
            raise ValueError('Expected [B,7] or [B,N,7]')
        flat = candidates.reshape(-1, 7)
        if not len(flat):
            raise ValueError('No grasp candidates')
        successes, solutions = [], []
        size = self.cfg.ik_solver.cuda_grasp_batch_size
        for start in range(0, len(flat), size):
            result = original_graph(self, flat[start:start+size])
            successes.append(result.success.reshape(-1))
            solutions.append(result.solution.reshape(-1, result.solution.shape[-1]))
        return IkResult(goal_poses=candidates,
                        success=torch.cat(successes).reshape(b, n),
                        solution=torch.cat(solutions).reshape(b, n, -1))

    Planner.__init__, Planner.graph_ik = init, graph
    print('COMPACT_IK: 128 candidates per graph batch, all candidates retained', flush=True)
